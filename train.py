"""
train.py  —  Run this from your project root: TB_Detection_SystemV1/
Usage:  python train.py
"""
import sys
from pathlib import Path

# Add tb_system to path
sys.path.insert(0, str(Path(__file__).parent / "tb_system"))

from  core.train import train

if __name__ == "__main__":
    train(config={
        "data_root":   "data/",
        "ckpt_dir":    "checkpoints/",
        "focal_alpha": 0.5,        # your dataset is nearly balanced
        "num_workers": 8,          # RTX 4060 system
        "amp":         True,       # mixed precision ON for RTX 4060
    })