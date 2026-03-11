"""
1_lower_threshold.py
====================
Lowers the detection threshold from 0.4616 to 0.30.
Run from project root:
    python 1_lower_threshold.py

This pushes sensitivity from 68.9% to ~80%+ at the cost of
some specificity (94.2% → ~87%). For PHC screening, catching
more TB is more important than avoiding false alarms.

You can also run with --evaluate to see the exact metric
change before committing.
"""

import os, sys, json, argparse
import numpy as np
from pathlib import Path

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "tb_system"))

CKPT_DIR    = os.path.join(_root, "checkpoints")
THRESH_FILE = os.path.join(CKPT_DIR, "optimal_threshold.json")
NEW_THRESHOLD = 0.30


def evaluate_threshold(threshold, probs, labels):
    preds = (probs >= threshold).astype(int)
    TP = ((preds == 1) & (labels == 1)).sum()
    TN = ((preds == 0) & (labels == 0)).sum()
    FP = ((preds == 1) & (labels == 0)).sum()
    FN = ((preds == 0) & (labels == 1)).sum()
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (TP + TN) / len(labels)
    return dict(sensitivity=sensitivity, specificity=specificity,
                precision=precision, f1=f1, accuracy=accuracy,
                TP=int(TP), TN=int(TN), FP=int(FP), FN=int(FN))


def load_val_probs():
    """Run inference on validation set and return (probs, labels)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from core.model import TBModel, TemperatureScaler
    from core.preprocessing import TBDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TBModel(num_classes=2, pretrained=False, use_metadata=False)
    model.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, "best_model.pth"), map_location=device))
    model.to(device).eval()

    t_path = os.path.join(CKPT_DIR, "temperature_scaler.pth")
    if os.path.exists(t_path):
        scaler = TemperatureScaler(model)
        scaler.load_state_dict(torch.load(t_path, map_location=device))
        scaler.to(device).eval()
        infer = scaler
        print("  Using temperature-calibrated probabilities")
    else:
        infer = model

    val_ds = TBDataset(os.path.join(_root, "data"), split="val", size=224)
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, *_ in loader:
            logits, _ = infer(imgs.to(device), None)
            probs = F.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true",
                        help="Show metric comparison before writing")
    parser.add_argument("--threshold", type=float, default=NEW_THRESHOLD,
                        help=f"New threshold (default: {NEW_THRESHOLD})")
    args = parser.parse_args()

    # Read current threshold
    with open(THRESH_FILE) as f:
        current = json.load(f)
    old_thresh = current["optimal_threshold"]
    print(f"\nCurrent threshold : {old_thresh:.4f}")
    print(f"New threshold     : {args.threshold:.4f}")

    if args.evaluate:
        print("\nRunning validation inference to compare metrics...")
        probs, labels = load_val_probs()

        old_m = evaluate_threshold(old_thresh,     probs, labels)
        new_m = evaluate_threshold(args.threshold, probs, labels)

        print(f"\n{'Metric':<15} {'Old':>10} {'New':>10} {'Change':>10}")
        print("-" * 48)
        for k in ["sensitivity", "specificity", "f1", "accuracy", "precision"]:
            delta = new_m[k] - old_m[k]
            sign  = "+" if delta >= 0 else ""
            print(f"{k:<15} {old_m[k]:>9.1%}  {new_m[k]:>9.1%}  {sign}{delta:.1%}")

        print(f"\n{'':15} {'Old':>10} {'New':>10}")
        for k in ["TP","TN","FP","FN"]:
            print(f"{k:<15} {old_m[k]:>10}  {new_m[k]:>10}")

        confirm = input("\nWrite new threshold? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted. No changes made.")
            return

    # Write new threshold
    current["optimal_threshold"] = args.threshold
    current["previous_threshold"] = old_thresh
    current["threshold_method"] = "manual_sensitivity_tuned"
    current["note"] = (
        f"Lowered from {old_thresh:.4f} (Youden's J) to {args.threshold:.4f} "
        f"to increase sensitivity for PHC screening. "
        f"Missing TB is worse than a false alarm."
    )

    with open(THRESH_FILE, "w") as f:
        json.dump(current, f, indent=2)

    print(f"\nDone. Threshold updated: {old_thresh:.4f} → {args.threshold:.4f}")
    print(f"File: {THRESH_FILE}")
    print("\nRestart the Streamlit app to apply the new threshold.")


if __name__ == "__main__":
    main()