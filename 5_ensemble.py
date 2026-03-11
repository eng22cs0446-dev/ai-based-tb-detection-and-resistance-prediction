"""
5_ensemble.py
=============
Ensemble multiple trained model checkpoints.
No retraining needed — works with your existing best_model.pth.

Two modes:
  A) Single-model ensemble: average predictions from multiple TTA passes
     (you already have this with TTA x8)

  B) Multi-model ensemble: average predictions from 2-3 different checkpoints
     (better — needs you to save checkpoints from different training runs)

     To get multiple checkpoints:
       1. Train v2 again with a different seed → save as best_model_v2b.pth
       2. Train with DenseNet-121 → save as densenet_model.pth
       3. Ensemble all three

Run from project root:
    python 5_ensemble.py --evaluate        # show metrics for ensemble vs single
    python 5_ensemble.py --export          # export ensemble as single artifact

Expected improvement: +2-4% AUC, more stable confidence on borderline cases
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "tb_system"))

from core.model import TBModel, TemperatureScaler
from core.preprocessing import TBDataset

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = os.path.join(_root, "checkpoints")


# ── LOAD ONE MODEL ────────────────────────────────────────────
def load_model(ckpt_path: str, calibrated: bool = True):
    model = TBModel(num_classes=2, pretrained=False, use_metadata=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    # Check for temperature scaler
    t_path = ckpt_path.replace(".pth", "_temperature.pth")
    if not os.path.exists(t_path):
        t_path = os.path.join(CKPT_DIR, "temperature_scaler.pth")

    if calibrated and os.path.exists(t_path):
        scaler = TemperatureScaler(model)
        scaler.load_state_dict(torch.load(t_path, map_location=DEVICE))
        scaler.to(DEVICE).eval()
        return scaler, True
    return model, False


# ── TTA INFERENCE FOR ONE MODEL ───────────────────────────────
def tta_predict(model, image_tensor: torch.Tensor, n_aug: int = 8) -> np.ndarray:
    """
    Run TTA and return averaged TB probability.
    image_tensor: (1, 3, H, W)
    """
    import torchvision.transforms.functional as TF

    augmented = [
        image_tensor,
        TF.hflip(image_tensor),
        TF.rotate(image_tensor,  5),
        TF.rotate(image_tensor, -5),
        TF.adjust_brightness(image_tensor, 1.1),
        TF.adjust_brightness(image_tensor, 0.9),
        TF.adjust_contrast(image_tensor, 1.1),
        TF.adjust_contrast(image_tensor, 0.9),
    ][:n_aug]

    probs_list = []
    with torch.no_grad():
        for aug in augmented:
            logits, _ = model(aug.to(DEVICE), None)
            p = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs_list.append(p)

    return np.mean(probs_list, axis=0)


# ── ENSEMBLE INFERENCE ────────────────────────────────────────
class ModelEnsemble:
    """
    Weighted average ensemble of multiple TB detection models.

    Usage:
        ensemble = ModelEnsemble()
        ensemble.add("checkpoints/best_model.pth",   weight=1.0)
        ensemble.add("checkpoints/best_model_v2b.pth", weight=0.8)
        prob = ensemble.predict_single(image_tensor)
    """

    def __init__(self):
        self.models  = []
        self.weights = []
        self.names   = []

    def add(self, ckpt_path: str, weight: float = 1.0, name: str = None):
        if not os.path.exists(ckpt_path):
            print(f"  SKIP (not found): {ckpt_path}")
            return
        model, calibrated = load_model(ckpt_path)
        self.models.append(model)
        self.weights.append(weight)
        self.names.append(name or os.path.basename(ckpt_path))
        cal_str = "calibrated" if calibrated else "raw"
        print(f"  Loaded: {self.names[-1]} (weight={weight}, {cal_str})")

    def predict_single(self, image_tensor: torch.Tensor, use_tta: bool = True) -> float:
        """Predict TB probability for a single image tensor (1, 3, H, W)."""
        if not self.models:
            raise RuntimeError("No models loaded. Call .add() first.")

        total_weight = sum(self.weights)
        weighted_sum = 0.0

        for model, weight in zip(self.models, self.weights):
            if use_tta:
                prob = tta_predict(model, image_tensor).item()
            else:
                with torch.no_grad():
                    logits, _ = model(image_tensor.to(DEVICE), None)
                    prob = F.softmax(logits, dim=1)[0, 1].item()
            weighted_sum += weight * prob

        return weighted_sum / total_weight

    def predict_batch(self, loader) -> tuple:
        """Run ensemble on a DataLoader. Returns (probs, labels)."""
        all_probs, all_labels = [], []

        for imgs, labels, *_ in loader:
            batch_probs = []
            for model, weight in zip(self.models, self.weights):
                with torch.no_grad():
                    logits, _ = model(imgs.to(DEVICE), None)
                    p = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    batch_probs.append(p * weight)

            total_w = sum(self.weights)
            ensemble_p = sum(batch_probs) / total_w
            all_probs.append(ensemble_p)
            all_labels.append(labels.numpy())

        return np.concatenate(all_probs), np.concatenate(all_labels)

    def find_optimal_weights(self, loader) -> list:
        """
        Grid search for optimal ensemble weights.
        Tries all combinations and returns weights that maximise AUC.
        Only useful with 2-3 models.
        """
        print("\nOptimising ensemble weights...")
        n     = len(self.models)
        grids = np.arange(0.2, 1.1, 0.2)

        # Collect individual model predictions first
        individual_probs = []
        for model in self.models:
            probs, labels = [], []
            with torch.no_grad():
                for imgs, lbls, *_ in loader:
                    logits, _ = model(imgs.to(DEVICE), None)
                    p = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    probs.append(p)
                    labels.append(lbls.numpy())
            individual_probs.append(np.concatenate(probs))
        all_labels = np.concatenate(labels)

        # Grid search
        best_auc, best_weights = 0.0, self.weights
        from itertools import product
        for weights in product(grids, repeat=n):
            weights = list(weights)
            total   = sum(weights)
            combined = sum(w * p for w, p in zip(weights, individual_probs)) / total
            auc = roc_auc_score(all_labels, combined)
            if auc > best_auc:
                best_auc     = auc
                best_weights = [w / total for w in weights]

        print(f"  Best AUC: {best_auc:.4f} with weights: {best_weights}")
        return best_weights


# ── EVALUATION ────────────────────────────────────────────────
def print_metrics(name, probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    TP = ((preds == 1) & (labels == 1)).sum()
    TN = ((preds == 0) & (labels == 0)).sum()
    FP = ((preds == 1) & (labels == 0)).sum()
    FN = ((preds == 0) & (labels == 1)).sum()
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0
    auc  = roc_auc_score(labels, probs)
    f1   = f1_score(labels, preds)

    print(f"\n  {name}")
    print(f"  {'AUC':<15} {auc:.4f}")
    print(f"  {'Sensitivity':<15} {sens:.4f}  ({sens:.1%})")
    print(f"  {'Specificity':<15} {spec:.4f}  ({spec:.1%})")
    print(f"  {'F1':<15} {f1:.4f}")
    print(f"  {'TP/TN/FP/FN':<15} {TP}/{TN}/{FP}/{FN}")
    return auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true",
                        help="Compare ensemble vs single model")
    parser.add_argument("--export", action="store_true",
                        help="Export ensemble config for app.py")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Model Ensemble")
    print(f"{'='*55}")

    # Load all available checkpoints
    ensemble = ModelEnsemble()
    for fname, weight in [
        ("best_model.pth",       1.0),
        ("best_model_v2b.pth",   0.8),   # if you train a second run
        ("best_model_v3.pth",    0.9),   # if you train v3
    ]:
        path = os.path.join(CKPT_DIR, fname)
        ensemble.add(path, weight=weight)

    if len(ensemble.models) == 0:
        print("\nNo models found. Using single model from best_model.pth")
        ensemble.add(os.path.join(CKPT_DIR, "best_model.pth"), weight=1.0)

    if args.evaluate:
        print("\nLoading validation set...")
        val_ds = TBDataset(os.path.join(_root, "data"), split="val", size=224)
        val_ld = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

        # Load threshold
        thresh_path = os.path.join(CKPT_DIR, "optimal_threshold.json")
        with open(thresh_path) as f:
            threshold = json.load(f)["optimal_threshold"]
        print(f"Threshold: {threshold:.4f}")

        # Single model
        single_model, _ = load_model(os.path.join(CKPT_DIR, "best_model.pth"))
        s_probs, s_labels = [], []
        with torch.no_grad():
            for imgs, labels, *_ in val_ld:
                logits, _ = single_model(imgs.to(DEVICE), None)
                p = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                s_probs.append(p)
                s_labels.append(labels.numpy())
        s_probs  = np.concatenate(s_probs)
        s_labels = np.concatenate(s_labels)

        # Ensemble
        e_probs, e_labels = ensemble.predict_batch(val_ld)

        print("\n" + "=" * 55)
        print("  COMPARISON")
        print("=" * 55)
        single_auc   = print_metrics("Single Model", s_probs, s_labels, threshold)
        ensemble_auc = print_metrics("Ensemble",     e_probs, e_labels, threshold)
        print(f"\n  AUC gain from ensemble: +{ensemble_auc - single_auc:.4f}")

    if args.export:
        config = {
            "ensemble_models": [
                {"path": os.path.join(CKPT_DIR, n),
                 "weight": w}
                for n, w in zip(ensemble.names, ensemble.weights)
            ],
            "note": "Load all models and average weighted predictions in app.py"
        }
        out = os.path.join(CKPT_DIR, "ensemble_config.json")
        with open(out, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\nExported ensemble config → {out}")
        print("Add ModelEnsemble to app.py and load this config at startup.")

    print("\nDone.")


if __name__ == "__main__":
    main()