import os
import json
import numpy as np
import torch

from config import cfg
from dataset import get_dataloaders
from model import build_model
from utils import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading test data...")
_, _, test_loader = get_dataloaders(seed=cfg.train.seed)

print("Loading model...")
model = build_model(cfg).to(device)

checkpoint = os.path.join(
    cfg.train.checkpoint_dir,
    cfg.train.best_ckpt_name
)

load_checkpoint(
    checkpoint,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location=device,
)

model.eval()

preds = []
targets = []

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(device)

        pred = model(x)

        preds.append(pred.cpu().numpy())
        targets.append(y.numpy())

preds = np.concatenate(preds, axis=0)
targets = np.concatenate(targets, axis=0)


errors = preds - targets

sbp_err = errors[:,0]
dbp_err = errors[:,1]


def aami(errors):

    mean = np.mean(errors)
    std = np.std(errors)

    passed = (abs(mean) <= 5) and (std <= 8)

    return mean, std, passed


def bhs(errors):

    abs_err = np.abs(errors)

    p5 = np.mean(abs_err <= 5)*100
    p10 = np.mean(abs_err <= 10)*100
    p15 = np.mean(abs_err <= 15)*100

    if p5>=60 and p10>=85 and p15>=95:
        grade="A"
    elif p5>=50 and p10>=75 and p15>=90:
        grade="B"
    elif p5>=40 and p10>=65 and p15>=85:
        grade="C"
    else:
        grade="D"

    return grade,p5,p10,p15


sbp_mean,sbp_std,sbp_pass = aami(sbp_err)
dbp_mean,dbp_std,dbp_pass = aami(dbp_err)

sbp_grade,sbp5,sbp10,sbp15 = bhs(sbp_err)
dbp_grade,dbp5,dbp10,dbp15 = bhs(dbp_err)

results = {

"SBP":{

"AAMI Mean Error":float(sbp_mean),
"AAMI Std":float(sbp_std),
"AAMI Pass":bool(sbp_pass),

"BHS Grade":sbp_grade,
"<=5":float(sbp5),
"<=10":float(sbp10),
"<=15":float(sbp15)

},

"DBP":{

"AAMI Mean Error":float(dbp_mean),
"AAMI Std":float(dbp_std),
"AAMI Pass":bool(dbp_pass),

"BHS Grade":dbp_grade,
"<=5":float(dbp5),
"<=10":float(dbp10),
"<=15":float(dbp15)

}

}

print(json.dumps(results,indent=4))

os.makedirs(cfg.train.output_dir,exist_ok=True)

with open(
    os.path.join(cfg.train.output_dir,"AAMI_BHS_results.json"),
    "w"
) as f:

    json.dump(results,f,indent=4)

print("Saved results.")
