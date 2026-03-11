# DATA USAGE GUIDE
# TB Detection System — Where and How to Use Your 19k Images
# ═══════════════════════════════════════════════════════════════

# ── STEP 1: WHAT STRUCTURE DOES YOUR DATA FOLDER LOOK LIKE? ───
#
# The TBDataset in core/preprocessing.py expects this layout:
#
#   data/
#   ├── train/
#   │   ├── Normal/    ← normal X-rays for training
#   │   └── TB/        ← TB X-rays for training
#   ├── val/
#   │   ├── Normal/
#   │   └── TB/
#   └── test/
#       ├── Normal/
#       └── TB/
#
# If your folders already look like this → skip to STEP 3.
# If not → run the reorganise script below (STEP 2).


# ── STEP 2: REORGANISE YOUR DATA (if needed) ──────────────────
# Run this ONCE to move your images into the right structure.

import os, shutil, random
from pathlib import Path

def reorganise_data(
    source_dir: str,       # your current data root
    dest_dir:   str = "data",
    train_pct:  float = 0.70,
    val_pct:    float = 0.15,
    # test gets the rest (0.15)
    seed:       int   = 42,
):
    """
    Handles these common input layouts:

    Layout A — already split, class subfolders:
        source/train/Normal/  source/train/TB/
        source/val/Normal/    source/val/TB/

    Layout B — already split, flat (images named by class):
        source/train/  source/val/  source/test/
        (images named like Normal_001.jpg / TB_001.jpg)

    Layout C — not split yet, just two class folders:
        source/Normal/  source/TB/
    """
    random.seed(seed)
    src = Path(source_dir)
    dst = Path(dest_dir)

    # ── Detect layout ─────────────────────────────────────────
    has_class_subfolders = (src / "train" / "Normal").exists() or \
                           (src / "train" / "TB").exists()
    has_split_flat       = (src / "train").exists() and \
                           not (src / "train" / "Normal").exists()
    has_only_classes     = (src / "Normal").exists() or (src / "TB").exists()

    if has_class_subfolders:
        print("✓ Layout A detected — already in correct format.")
        print(f"  Just set data_root = '{source_dir}' in your config.")
        return

    if has_only_classes:
        print("✓ Layout C detected — splitting into train/val/test…")
        _split_from_classes(src, dst, train_pct, val_pct)
        return

    if has_split_flat:
        print("✓ Layout B detected — reorganising into class subfolders…")
        _reorganise_flat(src, dst)
        return

    print("⚠ Could not detect layout. Check your folder structure.")


def _split_from_classes(src: Path, dst: Path, train_pct, val_pct):
    """Source has Normal/ and TB/ folders. Split into train/val/test."""
    for cls in ["Normal", "TB"]:
        images = list((src / cls).glob("*.[jp][pn]g")) + \
                 list((src / cls).glob("*.jpeg"))
        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_pct)
        n_val   = int(n * val_pct)
        splits  = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:],
        }
        for split, files in splits.items():
            out_dir = dst / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, out_dir / f.name)
        print(f"  {cls}: {n} images → "
              f"train={len(splits['train'])} / "
              f"val={len(splits['val'])} / "
              f"test={len(splits['test'])}")
    print(f"\n✓ Data reorganised into: {dst}/")


def _reorganise_flat(src: Path, dst: Path):
    """Source has train/val/test flat — infer class from filename prefix."""
    for split in ["train", "val", "test"]:
        split_dir = src / split
        if not split_dir.exists():
            continue
        for img in split_dir.glob("*.[jp][pn]g"):
            name_lower = img.name.lower()
            if "normal" in name_lower or "healthy" in name_lower:
                cls = "Normal"
            elif "tb" in name_lower or "tuberculosis" in name_lower:
                cls = "TB"
            else:
                print(f"  ⚠ Can't infer class for: {img.name} — skipping")
                continue
            out = dst / split / cls
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out / img.name)
    print(f"✓ Reorganised into: {dst}/")


# ── STEP 3: VERIFY YOUR DATA ──────────────────────────────────

def verify_data(data_root: str = "data"):
    """Run this to confirm everything is in order before training."""
    root = Path(data_root)
    print(f"\n{'='*50}")
    print(f"DATA VERIFICATION: {data_root}/")
    print(f"{'='*50}")
    total = 0
    for split in ["train", "val", "test"]:
        for cls in ["Normal", "TB"]:
            d = root / split / cls
            if d.exists():
                n = len(list(d.glob("*.[jp][pn]g")))
                total += n
                print(f"  {split:5s}/{cls:6s} → {n:5d} images")
            else:
                print(f"  {split:5s}/{cls:6s} → ✗ MISSING")
    print(f"{'─'*50}")
    print(f"  TOTAL              → {total:5d} images")
    print(f"{'='*50}\n")


# ── STEP 4: PLUG INTO TRAINING ────────────────────────────────
#
# In core/train.py, the data_root is set in DEFAULTS:
#
#   DEFAULTS = dict(
#       data_root = "data/",    ← this is where your images live
#       ...
#   )
#
# Or override at runtime:
#
#   from core.train import train
#   train(config={"data_root": "/absolute/path/to/your/data"})
#
# The training pipeline does the rest:
#   • TBDataset loads images from train/ val/ test/
#   • CLAHE enhancement applied automatically
#   • Lung segmentation applied if weights provided
#   • Balanced sampler handles Normal/TB imbalance automatically
#   • 3-stage progressive training runs (warmup → finetune → highres)


# ── STEP 5: PLUG INTO THE APP ─────────────────────────────────
#
# The app uses your TRAINED model weights, not raw images directly.
# After training completes, a best_model.pth is saved to checkpoints/.
#
# In app/app.py, replace the mock run_tb_model() function with:
#
#   from core.model import TBModel
#   from core.preprocessing import infer_transforms
#   import torch
#
#   @st.cache_resource
#   def load_tb_model():
#       model = TBModel(num_classes=2, pretrained=False)
#       model.load_state_dict(torch.load("checkpoints/best_model.pth",
#                                         map_location="cpu"))
#       model.eval()
#       return model
#
#   def run_tb_model(image: PIL.Image.Image) -> dict:
#       model = load_tb_model()
#       tfm   = infer_transforms(size=384)
#       x     = tfm(image.convert("RGB")).unsqueeze(0)
#       gradcam = GradCAMPP(model)
#       with torch.no_grad():
#           logits, features = model(x, metadata=None)
#           probs = torch.softmax(logits, dim=1).squeeze()
#       heatmap = gradcam(x).cpu().numpy()
#       return {
#           "tb_detected":    probs[1].item() > 0.5,
#           "tb_prob":        probs[1].item(),
#           "normal_prob":    probs[0].item(),
#           "image_features": features.squeeze().numpy(),
#           "heatmap":        heatmap,
#           "finding":        "Model inference result",
#       }


# ── QUICK START ───────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Verify your data is structured correctly
    verify_data("data")

    # 2. If not structured yet, reorganise:
    # reorganise_data(source_dir="data", dest_dir="data")

    # 3. Then train:
    # from core.train import train
    # train()
