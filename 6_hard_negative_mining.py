"""
6_hard_negative_mining.py
=========================
Finds images the model is MOST WRONG about:
  - Hard negatives: Normal X-rays the model thinks are TB (false alarms)
  - Hard positives: TB X-rays the model misses (false negatives)

These are your most valuable training examples.

Run from project root:
    python 6_hard_negative_mining.py

What it does:
  1. Runs inference on the full validation set
  2. Finds the top-N worst errors in each category
  3. Saves them to results/hard_cases/ with GradCAM overlays
  4. Generates a report showing WHY the model was wrong

Use the output to:
  a) Add more images like the hard negatives to your Normal class
  b) Add more images like the hard positives to your TB class
  c) Understand what patterns confuse the model (pneumonia? fibrosis?)
"""

import os, sys, json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "tb_system"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from core.model         import TBModel, TemperatureScaler, GradCAMPP
from core.preprocessing import TBDataset, infer_transforms

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = os.path.join(_root, "checkpoints")
OUT_DIR  = os.path.join(_root, "results", "hard_cases")
os.makedirs(OUT_DIR, exist_ok=True)

TOP_N    = 20   # How many hard cases to save per category
BG      = "#0A0F1A"


# ── LOAD MODEL ────────────────────────────────────────────────
def load_model():
    model = TBModel(num_classes=2, pretrained=False, use_metadata=False)
    model.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, "best_model.pth"), map_location=DEVICE))
    model.to(DEVICE).eval()

    t_path = os.path.join(CKPT_DIR, "temperature_scaler.pth")
    if os.path.exists(t_path):
        scaler = TemperatureScaler(model)
        scaler.load_state_dict(torch.load(t_path, map_location=DEVICE))
        scaler.to(DEVICE).eval()
        return scaler, model
    return model, model


# ── DATASET WITH PATHS ────────────────────────────────────────
class TBDatasetWithPaths(TBDataset):
    """Extends TBDataset to also return the image file path."""
    def __getitem__(self, idx):
        tensor, label = super().__getitem__(idx)
        path = self.samples[idx][0] if hasattr(self, "samples") else f"image_{idx}"
        return tensor, label, str(path)


# ── COMPUTE HEATMAP ───────────────────────────────────────────
def make_heatmap(base_model, image_tensor):
    try:
        gradcam = GradCAMPP(base_model)
        cam = gradcam(image_tensor.unsqueeze(0).to(DEVICE), None, cls=1)
        return cam.detach().cpu().numpy()
    except Exception:
        return np.zeros((224, 224))


def overlay_heatmap(orig_np, heatmap):
    H, W = orig_np.shape[:2]
    hmap = cv2.resize(heatmap, (W, H))
    colored = cv2.applyColorMap((hmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(orig_np, 0.55, colored, 0.45, 0)


# ── SAVE HARD CASE ────────────────────────────────────────────
def save_hard_case(idx, img_tensor, true_label, pred_prob, img_path,
                   base_model, category, rank):
    """Save a single hard case with GradCAM overlay."""
    # Convert tensor to numpy for display
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_np = (img_np * std + mean)
    img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

    heatmap = make_heatmap(base_model, img_tensor)
    overlay = overlay_heatmap(img_np, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
    fig.suptitle(
        f"HARD {category.upper()}  #{rank+1}\n"
        f"True: {'TB' if true_label == 1 else 'Normal'}  |  "
        f"Pred: {pred_prob:.1%} TB  |  "
        f"File: {os.path.basename(img_path)}",
        color="white", fontsize=12, y=1.02)

    for ax in axes:
        ax.set_facecolor(BG)
        ax.axis("off")

    axes[0].imshow(img_np);  axes[0].set_title("Original",     color="white")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("GradCAM++", color="white")
    axes[2].imshow(overlay); axes[2].set_title("Overlay",     color="white")

    fn = f"{category}_{rank+1:02d}_prob{pred_prob:.2f}.png"
    fig.savefig(os.path.join(OUT_DIR, fn), dpi=100,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ── MAIN ──────────────────────────────────────────────────────
def main():
    print(f"\n{'='*55}")
    print(f"  Hard Case Mining")
    print(f"{'='*55}")

    infer_model, base_model = load_model()

    # Load threshold
    thresh_path = os.path.join(CKPT_DIR, "optimal_threshold.json")
    with open(thresh_path) as f:
        threshold = json.load(f)["optimal_threshold"]
    print(f"Threshold: {threshold:.4f}")

    # Load validation set
    print("\nLoading validation set...")
    val_ds = TBDataset(os.path.join(_root, "data"), split="val", size=224)
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"  {len(val_ds)} images")

    # Collect all predictions
    print("\nRunning inference...")
    all_probs, all_labels, all_tensors, all_paths = [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(val_ld):
            imgs, labels, *rest = batch
            path = rest[0] if rest else f"image_{i}"

            logits, _ = infer_model(imgs.to(DEVICE), None)
            prob = F.softmax(logits, dim=1)[0, 1].item()

            all_probs.append(prob)
            all_labels.append(labels[0].item())
            all_tensors.append(imgs[0])
            all_paths.append(path if isinstance(path, str) else f"image_{i}")

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(val_ds)}", end="\r")

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Identify hard cases
    preds = (all_probs >= threshold).astype(int)

    # False Negatives (hard positives) — TB cases model missed
    # Sorted by how WRONG the model was (lowest prob for TB cases)
    fn_mask  = (all_labels == 1) & (preds == 0)
    fn_idx   = np.where(fn_mask)[0]
    fn_sorted = fn_idx[np.argsort(all_probs[fn_idx])]  # lowest TB prob first

    # False Positives (hard negatives) — Normal cases model flagged
    # Sorted by how WRONG the model was (highest prob for Normal cases)
    fp_mask  = (all_labels == 0) & (preds == 1)
    fp_idx   = np.where(fp_mask)[0]
    fp_sorted = fp_idx[np.argsort(-all_probs[fp_idx])]  # highest TB prob first

    print(f"\n\nResults:")
    print(f"  False Negatives (missed TB)    : {fn_mask.sum()} cases")
    print(f"  False Positives (false alarms) : {fp_mask.sum()} cases")
    print(f"  Total errors                   : {fn_mask.sum() + fp_mask.sum()}")
    print(f"\nSaving top {TOP_N} hard cases each...")

    # Save hard positives (missed TB)
    for rank, idx in enumerate(fn_sorted[:TOP_N]):
        save_hard_case(
            idx, all_tensors[idx], all_labels[idx],
            all_probs[idx], all_paths[idx],
            base_model, "missed_TB", rank)
        print(f"  Missed TB #{rank+1}: prob={all_probs[idx]:.3f}", end="\r")

    print(f"\n  Saved {min(len(fn_sorted), TOP_N)} missed TB cases")

    # Save hard negatives (false alarms)
    for rank, idx in enumerate(fp_sorted[:TOP_N]):
        save_hard_case(
            idx, all_tensors[idx], all_labels[idx],
            all_probs[idx], all_paths[idx],
            base_model, "false_alarm", rank)
        print(f"  False alarm #{rank+1}: prob={all_probs[idx]:.3f}", end="\r")

    print(f"\n  Saved {min(len(fp_sorted), TOP_N)} false alarm cases")

    # Summary report
    report = {
        "total_val_cases":     len(val_ds),
        "threshold":           float(threshold),
        "false_negatives":     int(fn_mask.sum()),
        "false_positives":     int(fp_mask.sum()),
        "fn_prob_range":       [float(all_probs[fn_idx].min()),
                                float(all_probs[fn_idx].max())] if len(fn_idx) else [],
        "fp_prob_range":       [float(all_probs[fp_idx].min()),
                                float(all_probs[fp_idx].max())] if len(fp_idx) else [],
        "fn_mean_prob":        float(all_probs[fn_idx].mean()) if len(fn_idx) else 0,
        "fp_mean_prob":        float(all_probs[fp_idx].mean()) if len(fp_idx) else 0,
        "output_dir":          OUT_DIR,
        "recommendation": (
            "Add images similar to missed_TB cases to your TB training set. "
            "Add images similar to false_alarm cases as hard Normal examples. "
            "Source: pneumonia, pleural effusion, fibrosis X-rays from NIH CXR14 dataset."
        )
    }

    report_path = os.path.join(OUT_DIR, "hard_cases_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Hard case analysis complete!")
    print(f"  Images saved → {OUT_DIR}/")
    print(f"  Report saved → {report_path}")
    print(f"\n  NEXT STEPS:")
    print(f"  1. Look at the missed_TB images — what pattern made the model miss?")
    print(f"     (miliary TB? early nodules? poor image quality?)")
    print(f"  2. Look at false_alarm images — what confused the model?")
    print(f"     (pneumonia? pleural effusion? old scars?)")
    print(f"  3. Source 200-500 similar images from NIH CXR14 or similar")
    print(f"  4. Add to training set and retrain")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()