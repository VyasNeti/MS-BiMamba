# MS-BiMamba

A multi-scale Bidirectional Mamba (state-space) network for **cuffless blood pressure estimation** — predicting Systolic (SBP) and Diastolic (DBP) blood pressure directly from PPG (photoplethysmogram) signals.

## Overview

The model takes a PPG waveform (optionally with its derivatives, VPG/APG) and regresses SBP/DBP using a multi-branch architecture:

1. **Channel Mixing** — a residual MLP mixes the input channels (PPG / VPG / APG).
2. **Multi-Scale Conv Embedding** — three parallel strided Conv1D branches tokenize the signal at different temporal resolutions.
3. **BiMamba Encoder** — each branch is encoded by a stack of bidirectional selective state-space blocks (forward + backward Mamba, gated fusion, feed-forward).
4. **Multi-Scale Fusion** — branch outputs are aligned to a common length and fused.
5. **Attention Pooling** — a learnable attention layer pools the fused sequence into a single embedding.
6. **Regression Head** — an MLP head outputs `[SBP, DBP]`.

The `SelectiveSSM` module is a pure PyTorch re-implementation of the Mamba S6 mechanism, so no custom CUDA kernel or `mamba-ssm` package is required.

## Repository Structure

```
.
├── config.py               # All configuration (data, model, training) as dataclasses
├── dataset.py               # .mat loading, windowing, SBP/DBP labeling, train/val/test splits
├── model.py                  # MS-BiMamba model definition
├── modules.py                 # Building blocks: BiMambaBlock, SelectiveSSM, fusion, pooling, etc.
├── losses.py                   # Weighted MSE loss for SBP/DBP
├── trainer.py                   # Training loop, checkpointing, early stopping
├── train.py                      # Training entry point (CLI)
├── test.py                        # Evaluation entry point — MAE/RMSE metrics
├── evaluate_standards.py           # AAMI and BHS standard compliance evaluation
├── utils.py                         # Seeding, VPG/APG computation, normalization, metrics
├── checkpoint/                       # Saved model checkpoints
└── results/                           # Saved evaluation results (JSON)
```

## Dataset

This project uses the **UCI Cuff-Less Blood Pressure Estimation Data Set**, derived from the MIMIC-II waveform database (~12,000 patient records). Each `.mat` file contains a cell array `data["p"]` of `(3, 61000)` signals with channels ordered `[PPG, ABP, ECG]`, sampled at 125 Hz. Windows of PPG are extracted (default length 1024, stride 512), and SBP/DBP labels are derived from the corresponding ABP window via systolic-peak / diastolic-trough detection.

- **Dataset URL:** https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation
- **Source:** MIMIC-II Waveform Database (via PhysioNet)
- **Citation:**

  > M. Kachuee, M. M. Kiani, H. Mohammadzade, and M. Shabany, "Cuff-Less High-Accuracy Calibration-Free Blood Pressure Estimation Using Pulse Transit Time," *2015 IEEE International Symposium on Circuits and Systems (ISCAS)*, 2015, pp. 1006–1009.

Set the dataset location via `cfg.data.data_dir` in [config.py](config.py) or with `--data-dir` on the CLI.

## Installation

```bash
pip install torch numpy scipy h5py tqdm
```

## Usage

### Train

```bash
python train.py --data-dir /path/to/mat/files --epochs 100 --batch-size 256
```

Useful flags: `--lr`, `--seed`, `--no-cache`, `--no-amp`, `--multi-gpu`.

### Evaluate (MAE / RMSE)

```bash
python test.py --checkpoint checkpoint/best_model.pth
```

Saves `test_metrics.json` and raw predictions to the configured output directory.

### Evaluate against AAMI / BHS standards

```bash
python evaluate_standards.py
```

Computes mean/std error against the AAMI criterion (mean ≤ 5 mmHg, std ≤ 8 mmHg) and the BHS cumulative error grade (A–D), saving results to `results/AAMI_BHS_results.json`.

## Results

Current test-set performance (see [results/](results/)):

| Metric | SBP | DBP |
|---|---|---|
| MAE (mmHg) | 5.72 | 3.60 |
| RMSE (mmHg) | 8.72 | 5.71 |
| AAMI Pass | ✗ | ✓ |
| BHS Grade | B | A |

## Configuration

All hyperparameters (data windowing, model dimensions, Mamba block settings, optimizer, scheduler, training) live in [config.py](config.py) as a single `Config` dataclass, and can be overridden via environment variables (`EXP_NAME`, `INPUT_MODE`, `NUM_MAMBA_BLOCKS`) or CLI flags on `train.py` / `test.py`.
