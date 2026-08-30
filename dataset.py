from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from scipy.signal import find_peaks
from torch.utils.data import Dataset

from config import cfg
from utils import compute_vpg_apg, zscore_normalize, get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Low level .mat loading
# --------------------------------------------------------------------------- #
def _load_mat_file(path: str) -> np.ndarray:
    """
    Load a single .mat file and return the cell array `data["p"]` as an
    object array of shape (1, 1000), where each element is a (3, 61000)
    float array. Falls back to h5py for MATLAB v7.3 files.
    """
    try:
        import scipy.io as sio

        mat = sio.loadmat(path)
        p = mat["p"]
        return p
    except NotImplementedError:
        # MATLAB v7.3 (HDF5-based) files are not supported by scipy.io
        import h5py

        with h5py.File(path, "r") as f:
            refs = f["p"][:]
            # h5py stores cell arrays as arrays of object references
            cells = np.empty(refs.shape, dtype=object)
            for idx in np.ndindex(refs.shape):
                cells[idx] = np.array(f[refs[idx][0] if refs[idx].ndim else refs[idx]])
            return cells


def _iter_signals_in_file(mat_p: np.ndarray):
    """
    Yield each (3, 61000) signal array contained in the (1, 1000) cell
    array loaded from one .mat file.
    """
    flat = mat_p.reshape(-1)
    for cell in flat:
        arr = np.asarray(cell)
        if arr.ndim == 2 and arr.shape[0] == 3:
            yield arr
        elif arr.ndim == 2 and arr.shape[1] == 3:
            yield arr.T
        else:
            # Skip malformed / empty cells rather than crash the whole run
            continue


# --------------------------------------------------------------------------- #
#  SBP/DBP labels from an ABP window (peak / trough averaging)
# --------------------------------------------------------------------------- #
def compute_sbp_dbp(abp_window: np.ndarray) -> Tuple[float, float]:
    """
    SBP = mean of all detected systolic peaks in the window.
    DBP = mean of all detected diastolic troughs in the window.

    Falls back to the plain max/min if peak detection finds nothing
    (e.g. a very flat or unusually short window), so every window still
    gets a usable label.
    """
    a = cfg.assumptions
    min_distance = max(1, int(round(a.peak_min_distance_sec * a.sampling_rate_hz)))

    peak_idx, _ = find_peaks(abp_window, distance=min_distance)
    trough_idx, _ = find_peaks(-abp_window, distance=min_distance)

    sbp = float(np.mean(abp_window[peak_idx])) if len(peak_idx) > 0 else float(np.max(abp_window))
    dbp = float(np.mean(abp_window[trough_idx])) if len(trough_idx) > 0 else float(np.min(abp_window))
    return sbp, dbp


# --------------------------------------------------------------------------- #
#  Windowing
# --------------------------------------------------------------------------- #
@dataclass
class WindowSample:
    x: np.ndarray  # (3, window_length) -> PPG, VPG, APG (z-scored)
    y: np.ndarray  # (2,) -> [SBP, DBP]


def _windows_from_signal(
    signal_3ch: np.ndarray,
    window_length: int,
    stride: int,
) -> List[WindowSample]:
    """
    signal_3ch: (3, L) raw array, channel order PPG/ABP/ECG as per
    `cfg.assumptions`. Returns a list of WindowSample built only from
    PPG (network input) and ABP (label).
    """
    a = cfg.assumptions
    ppg_full = signal_3ch[a.ppg_channel_idx]
    abp_full = signal_3ch[a.abp_channel_idx]

    total_len = ppg_full.shape[-1]
    samples: List[WindowSample] = []

    start = 0
    while start + window_length <= total_len:
        end = start + window_length
        ppg_win = ppg_full[start:end].astype(np.float64)
        abp_win = abp_full[start:end].astype(np.float64)

        # Drop windows containing NaN/Inf in either signal.
        if not np.all(np.isfinite(ppg_win)) or not np.all(np.isfinite(abp_win)):
            start += stride
            continue

        sbp, dbp = compute_sbp_dbp(abp_win)

        if not _label_is_valid(sbp, dbp):
            start += stride
            continue

        # Derivatives computed inside this window.
        vpg_win, apg_win = compute_vpg_apg(ppg_win)
        mode = cfg.model.input_mode

        if mode == "ppg":
            x = np.stack([ppg_win], axis=0)

        elif mode == "ppg_vpg":
            x = np.stack([ppg_win, vpg_win], axis=0)

        else:
            x = np.stack([ppg_win, vpg_win, apg_win], axis=0)

        x = zscore_normalize(x)

        # x = np.stack([ppg_win, vpg_win, apg_win], axis=0)  # (3, window_length)
        # x = zscore_normalize(x)  # per-channel: (x - mean) / (std + 1e-8)

        samples.append(WindowSample(x=x.astype(np.float32),
                                     y=np.array([sbp, dbp], dtype=np.float32)))
        start += stride

    return samples


def _label_is_valid(sbp: float, dbp: float) -> bool:
    """Discard windows whose SBP/DBP fall outside configurable valid ranges."""
    a = cfg.assumptions
    if not (a.sbp_valid_range[0] <= sbp <= a.sbp_valid_range[1]):
        return False
    if not (a.dbp_valid_range[0] <= dbp <= a.dbp_valid_range[1]):
        return False
    return True


# --------------------------------------------------------------------------- #
#  Building the full window set (with on-disk caching)
# --------------------------------------------------------------------------- #
def _cache_path() -> str:
    return os.path.join(
        cfg.data.cache_dir,
        f"windows_cache_{cfg.model.input_mode}.npz"
    )
    # return os.path.join(cfg.data.cache_dir, "windows_cache.npz")


def build_all_windows(force_rebuild: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X, Y, recording_id) where:
      X: (N, 3, window_length) float32
      Y: (N, 2) float32
      recording_id: (N,) int -- index of the source .mat file, used to
                     build a recording-level split.
    """
    cache_file = _cache_path()
    if cfg.data.use_cache and not force_rebuild and os.path.exists(cache_file):
        logger.info(f"Loading cached windows from {cache_file}")
        cached = np.load(cache_file)
        return cached["X"], cached["Y"], cached["rid"]

    mat_files = sorted(glob.glob(os.path.join(cfg.data.data_dir, cfg.data.file_glob)))
    if len(mat_files) == 0:
        raise FileNotFoundError(
            f"No .mat files found in {cfg.data.data_dir} "
            f"(pattern '{cfg.data.file_glob}')."
        )

    all_x, all_y, all_rid = [], [], []
    recording_counter = 0

    for file_path in mat_files:
        logger.info(f"Processing {file_path}")
        mat_p = _load_mat_file(file_path)
        for signal_3ch in _iter_signals_in_file(mat_p):
            windows = _windows_from_signal(
                signal_3ch,
                window_length=cfg.data.window_length,
                stride=cfg.data.stride,
            )
            for w in windows:
                all_x.append(w.x)
                all_y.append(w.y)
                all_rid.append(recording_counter)
            recording_counter += 1

    X = np.stack(all_x, axis=0)
    Y = np.stack(all_y, axis=0)
    rid = np.array(all_rid, dtype=np.int64)

    logger.info(f"Built {X.shape[0]} windows from {recording_counter} recordings.")

    if cfg.data.use_cache:
        os.makedirs(cfg.data.cache_dir, exist_ok=True)
        np.savez_compressed(cache_file, X=X, Y=Y, rid=rid)
        logger.info(f"Cached windows to {cache_file}")

    return X, Y, rid


def split_windows_random(
    X: np.ndarray, Y: np.ndarray, seed: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Pools all windows together, shuffles them, and splits randomly into
    train/val/test according to cfg.data.{train_frac,val_frac,test_frac}
    (default 70% / 15% / 15%).

    Note: this splits at the *window* level, so overlapping windows from
    the same recording can end up in different splits. This matches the
    requested preprocessing spec; if you need strict subject-level
    separation instead, use `split_by_recording` below.
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = int(round(n * cfg.data.train_frac))
    n_val = int(round(n * cfg.data.val_frac))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        (X[train_idx], Y[train_idx]),
        (X[val_idx], Y[val_idx]),
        (X[test_idx], Y[test_idx]),
    )


def split_by_recording(
    X: np.ndarray, Y: np.ndarray, rid: np.ndarray, seed: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Alternative to `split_windows_random`: splits windows into
    train/val/test at the *recording* level so that overlapping windows
    from the same subject never leak across splits. Not used by default
    (see `get_dataloaders`), but kept available since it's a common and
    often preferable choice for subject-based physiological data.
    """
    rng = np.random.RandomState(seed)
    unique_recordings = np.unique(rid)
    rng.shuffle(unique_recordings)

    n = len(unique_recordings)
    n_train = int(round(n * cfg.data.train_frac))
    n_val = int(round(n * cfg.data.val_frac))

    train_ids = set(unique_recordings[:n_train].tolist())
    val_ids = set(unique_recordings[n_train:n_train + n_val].tolist())
    test_ids = set(unique_recordings[n_train + n_val:].tolist())

    train_mask = np.isin(rid, list(train_ids))
    val_mask = np.isin(rid, list(val_ids))
    test_mask = np.isin(rid, list(test_ids))

    return (
        (X[train_mask], Y[train_mask]),
        (X[val_mask], Y[val_mask]),
        (X[test_mask], Y[test_mask]),
    )


# --------------------------------------------------------------------------- #
#  Dataset class
# --------------------------------------------------------------------------- #
class CufflessBPDataset(Dataset):
    """
    A thin tensor-wrapping Dataset. Expects pre-built numpy arrays
    (X: (N,3,L), Y: (N,2)) -- use `build_all_windows` + `split_by_recording`
    to construct these once and then pass the appropriate split to each
    of the three Dataset instances (train/val/test).
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert X.shape[0] == Y.shape[0], "X and Y must have matching length"
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


def get_dataloaders(seed: int = 42):
    """
    Convenience function: builds windows (or loads cache), shuffles and
    splits them randomly 70/15/15 into train/val/test, and returns
    (train_loader, val_loader, test_loader). Only the train DataLoader
    shuffles on each epoch; val/test are always served in a fixed order.
    """
    from torch.utils.data import DataLoader

    X, Y, rid = build_all_windows()
    (Xtr, Ytr), (Xval, Yval), (Xte, Yte) = split_windows_random(X, Y, seed=seed)

    train_ds = CufflessBPDataset(Xtr, Ytr)
    val_ds = CufflessBPDataset(Xval, Yval)
    test_ds = CufflessBPDataset(Xte, Yte)

    logger.info(
        f"Split sizes -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    return train_loader, val_loader, test_loader
