"""
run_temperature_scaling.py
==========================
Post-training temperature scaling calibration.
Run after training completes:
    python run_temperature_scaling.py
"""

import os, sys
from pathlib import Path

_root   = os.path.dirname(os.path.abspath(__file__))
_tb_sys = os.path.join(_root, "tb_system")
sys.path.insert(0, _root)
sys.path.insert(0, _tb_sys)

import torch
from torch.utils.data import DataLoader

from core.model         import MultiScaleTBModel, TemperatureScaler
from core.preprocessing import TBDataset

CKPT_DIR  = os.path.join(_root, "checkpoints")
DATA_ROOT = os.path.join(_root, "data")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("\n" + "="*52)
    print("  Temperature Scaling - Post-Training Calibration")
    print("="*52)
    print(f"  Device : {DEVICE}")
    ckpt_path = os.path.join(CKPT_DIR, "best_model.pth")
    print(f"  Model  : {ckpt_path}\n")

    # ── Load model ────────────────────────────────────────────
    print("  Loading model weights...")
    model = MultiScaleTBModel(
        num_classes  = 2,
        pretrained   = False,
        dropout      = 0.4,
        fusion_dim   = 256,
        use_metadata = False,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print("  Model loaded OK")

    # ── Validation set ────────────────────────────────────────
    print("  Loading validation set...")
    val_ds = TBDataset(DATA_ROOT, split="val", size=224)
    val_ld = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"  {len(val_ds)} validation images")

    # ── Fit temperature scaler ────────────────────────────────
    print("\n  Fitting temperature scaling...")
    scaler = TemperatureScaler(model)
    scaler.fit(val_ld, device=DEVICE)

    T = scaler.temperature.item()
    print(f"  Temperature T = {T:.4f}")

    # ── Save ──────────────────────────────────────────────────
    out_path = os.path.join(CKPT_DIR, "temperature_scaler.pth")
    torch.save(scaler.state_dict(), out_path)
    print(f"  Saved → {out_path}")

    print("\n" + "="*52)
    print(f"  Done. T = {T:.4f}")
    if T > 1.5:
        print("  Model was overconfident — scaling will help significantly")
    elif T < 0.8:
        print("  Model was underconfident — scaling will sharpen probabilities")
    else:
        print("  Model calibration is reasonable")
    print("="*52 + "\n")


if __name__ == "__main__":
    main()