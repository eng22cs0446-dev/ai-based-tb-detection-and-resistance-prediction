"""
core/preprocessing.py
X-ray preprocessing pipeline — IMPROVED version
Changes:
  - X-ray specific augmentations (gamma, elastic, grid distortion)
  - Label smoothing in dataset
  - Better CLAHE parameters
"""

from __future__ import annotations
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple
import random


# ══════════════════════════════════════════════════════════════
#  CLAHE ENHANCEMENT
# ══════════════════════════════════════════════════════════════

def apply_clahe(img: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """Enhance TB-related opacities via CLAHE on LAB L-channel."""
    clahe    = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab      = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    blur     = cv2.GaussianBlur(enhanced, (0, 0), 3)
    return np.clip(cv2.addWeighted(enhanced, 1.4, blur, -0.4, 0), 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════
#  X-RAY SPECIFIC AUGMENTATIONS
# ══════════════════════════════════════════════════════════════

def random_gamma(img: np.ndarray, gamma_range=(0.7, 1.5)) -> np.ndarray:
    """
    Random gamma correction — simulates different X-ray machine exposures.
    Different PHC machines produce different image brightness/contrast.
    """
    gamma = random.uniform(*gamma_range)
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def elastic_deformation(img: np.ndarray, alpha: float = 20.0, sigma: float = 5.0) -> np.ndarray:
    """
    Elastic deformation — simulates patient breathing/movement during scan.
    Slight warping makes model robust to positional variation.
    """
    H, W = img.shape[:2]
    dx   = cv2.GaussianBlur((np.random.rand(H, W) * 2 - 1).astype(np.float32),
                            (0, 0), sigma) * alpha
    dy   = cv2.GaussianBlur((np.random.rand(H, W) * 2 - 1).astype(np.float32),
                            (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = np.clip(x + dx, 0, W - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, H - 1).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)


def random_clahe_strength(img: np.ndarray) -> np.ndarray:
    """Random CLAHE clip limit — augments contrast variation between machines."""
    clip = random.uniform(1.5, 5.0)
    return apply_clahe(img, clip=clip)


def xray_augment(img: np.ndarray, p: float = 0.5) -> np.ndarray:
    """
    Apply X-ray specific augmentations with probability p each.
    Applied during training only.
    """
    if random.random() < p:
        img = random_gamma(img)
    if random.random() < p * 0.5:     # less frequent — more aggressive
        img = elastic_deformation(img)
    if random.random() < p * 0.4:
        img = random_clahe_strength(img)
    return img


# ══════════════════════════════════════════════════════════════
#  LUNG SEGMENTATION
# ══════════════════════════════════════════════════════════════

class _DConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
        )
    def forward(self, x): return self.b(x)


class LungUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1, self.e2, self.e3 = _DConv(1,32), _DConv(32,64), _DConv(64,128)
        self.bn = _DConv(128, 256)
        self.u3, self.d3 = nn.ConvTranspose2d(256,128,2,2), _DConv(256,128)
        self.u2, self.d2 = nn.ConvTranspose2d(128, 64,2,2), _DConv(128, 64)
        self.u1, self.d1 = nn.ConvTranspose2d( 64, 32,2,2), _DConv( 64, 32)
        self.out = nn.Conv2d(32, 1, 1)
        self.mp  = nn.MaxPool2d(2)

    def forward(self, x):
        e1=self.e1(x); e2=self.e2(self.mp(e1)); e3=self.e3(self.mp(e2))
        b =self.bn(self.mp(e3))
        d3=self.d3(torch.cat([self.u3(b),  e3],1))
        d2=self.d2(torch.cat([self.u2(d3), e2],1))
        d1=self.d1(torch.cat([self.u1(d2), e1],1))
        return self.out(d1)


class LungSegmenter:
    def __init__(self, weights: Optional[str] = None, device: str = "cpu"):
        self.dev = torch.device(device)
        self.net = None
        if weights and Path(weights).exists():
            self.net = LungUNet().to(self.dev)
            self.net.load_state_dict(torch.load(weights, map_location=self.dev))
            self.net.eval()
            print(f"[LungSegmenter] Loaded {weights}")
        else:
            print("[LungSegmenter] No weights — using heuristic crop.")

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        return self._nn_segment(rgb) if self.net else self._heuristic(rgb)

    def _nn_segment(self, rgb):
        H, W = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        t    = torch.tensor(cv2.resize(gray,(256,256))/255.,
                            dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.dev)
        with torch.no_grad():
            mask = torch.sigmoid(self.net(t)).squeeze().cpu().numpy()
        mask = (cv2.resize(mask,(W,H)) > 0.5).astype(np.uint8)
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k),cv2.MORPH_OPEN,k)
        return rgb * np.stack([mask]*3,-1)

    def _heuristic(self, rgb):
        H, W = rgb.shape[:2]
        out  = np.zeros_like(rgb)
        out[int(H*.08):int(H*.95), int(W*.04):int(W*.96)] = \
            rgb[int(H*.08):int(H*.95), int(W*.04):int(W*.96)]
        return out


# ══════════════════════════════════════════════════════════════
#  TRANSFORMS
# ══════════════════════════════════════════════════════════════

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def train_transforms(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size + 32, size + 32)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),                          # was 10
        transforms.ColorJitter(brightness=0.25, contrast=0.25), # was 0.2
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def val_transforms(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def infer_transforms(size: int = 224) -> transforms.Compose:
    return val_transforms(size)


# ══════════════════════════════════════════════════════════════
#  MIXUP / CUTMIX
# ══════════════════════════════════════════════════════════════

def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    return lam*x+(1-lam)*x[idx], y, y[idx], lam

def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    B,_,H,W = x.shape
    idx = torch.randperm(B)
    r   = np.sqrt(1-lam)
    ch, cw = int(H*r), int(W*r)
    cx, cy = random.randint(0,W), random.randint(0,H)
    x1,x2  = max(cx-cw//2,0), min(cx+cw//2,W)
    y1,y2  = max(cy-ch//2,0), min(cy+ch//2,H)
    xm = x.clone(); xm[:,:,y1:y2,x1:x2] = x[idx,:,y1:y2,x1:x2]
    lam = 1-(y2-y1)*(x2-x1)/(H*W)
    return xm, y, y[idx], lam

def mix_loss(criterion, pred, ya, yb, lam):
    return lam*criterion(pred,ya)+(1-lam)*criterion(pred,yb)


# ══════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════

class TBDataset(Dataset):
    CLASSES = ["Normal", "TB"]

    def __init__(
        self,
        root:          str,
        split:         str            = "train",
        size:          int            = 224,
        segmenter:     Optional[LungSegmenter] = None,
        label_smooth:  float          = 0.0,   # 0.0 = off, 0.1 = smooth
        xray_aug:      bool           = False,  # X-ray specific augmentations
    ):
        self.split        = split
        self.segmenter    = segmenter
        self.label_smooth = label_smooth
        self.xray_aug     = xray_aug and (split == "train")
        self.tfm          = train_transforms(size) if split == "train" else val_transforms(size)
        self.samples: list[tuple[str,int]] = []

        for label, cls in enumerate(self.CLASSES):
            d = Path(root) / split / cls
            if d.exists():
                self.samples += [(str(f), label) for f in d.glob("*.[jp][pn]g")]

        n_normal = sum(1 for _,l in self.samples if l==0)
        n_tb     = sum(1 for _,l in self.samples if l==1)
        print(f"[TBDataset] {split}: {len(self.samples)} images  "
              f"(Normal={n_normal}, TB={n_tb})"
              + (f"  label_smooth={label_smooth}" if label_smooth > 0 else "")
              + (f"  xray_aug=ON" if self.xray_aug else ""))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        img = apply_clahe(img)
        if self.xray_aug:
            img = xray_augment(img, p=0.5)
        if self.segmenter:
            img = self.segmenter.segment(img)
        img = self.tfm(Image.fromarray(img))

        # Label smoothing — soft targets instead of hard 0/1
        if self.label_smooth > 0 and self.split == "train":
            smooth = self.label_smooth / 2
            target = torch.zeros(2)
            target[label]   = 1.0 - self.label_smooth + smooth
            target[1-label] = smooth
            return img, target   # returns soft label tensor

        return img, torch.tensor(label, dtype=torch.long)