"""
drug_resistance/assessor.py
Drug Resistance Assessment — fuses X-ray image features + clinical metadata.
Predicts: Drug-Sensitive | MDR-TB | XDR-TB
"""

from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

# Import patient metadata tools
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metadata.patient import Patient, PatientEncoder, RiskScorer


# ══════════════════════════════════════════════════════════════
#  FUSION NETWORK
# ══════════════════════════════════════════════════════════════

class DRFusionNet(nn.Module):
    """
    Late-fusion: image features (512-dim) + metadata (19-dim) → 3 classes.
    Classes: [Drug-Sensitive, MDR-TB, XDR-TB]
    """
    def __init__(self, img_dim: int = 512, meta_dim: int = 16,
                 hidden: int = 256, dropout: float = 0.4):
        super().__init__()
        self.img_branch = nn.Sequential(
            nn.Linear(img_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.meta_branch = nn.Sequential(
            nn.Linear(meta_dim, 64), nn.GELU(),
            nn.Linear(64, 64),       nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + 64, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, img_feat: torch.Tensor, meta_feat: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.img_branch(img_feat),
                                    self.meta_branch(meta_feat)], dim=1))


# ══════════════════════════════════════════════════════════════
#  RESULT
# ══════════════════════════════════════════════════════════════

@dataclass
class DRResult:
    patient_id     : str
    prediction     : str           # "Drug-Sensitive" | "MDR-TB" | "XDR-TB"
    confidence     : float
    probabilities  : dict[str, float]
    risk_band      : str
    risk_score     : int
    risk_factors   : list[str]
    warnings       : list[str]
    recommendations: list[str]

    def to_dict(self) -> dict:
        return {
            "patient_id":      self.patient_id,
            "prediction":      self.prediction,
            "confidence":      self.confidence,
            "probabilities":   self.probabilities,
            "risk_band":       self.risk_band,
            "risk_score":      self.risk_score,
            "risk_factors":    self.risk_factors,
            "warnings":        self.warnings,
            "recommendations": self.recommendations,
        }


# ══════════════════════════════════════════════════════════════
#  ASSESSOR  (main interface)
# ══════════════════════════════════════════════════════════════

CLASSES = ["Drug-Sensitive", "MDR-TB", "XDR-TB"]


class DrugResistanceAssessor:

    def __init__(self, weights: Optional[str] = None,
                 img_dim: int = 512, device: str = "cpu"):
        self.device = torch.device(device)
        self.model  = DRFusionNet(img_dim=img_dim).to(self.device)

        if weights and os.path.exists(weights):
            self.model.load_state_dict(torch.load(weights, map_location=self.device))
            print(f"[DrugResistance] Loaded weights: {weights}")
        else:
            print("[DrugResistance] No weights — demo mode (random outputs).")

        self.model.eval()

    # ──────────────────────────────────────────────────────────
# REPLACE WITH:
def assess(
    patient,
    image_features: Optional[np.ndarray] = None,
    meta_vec_override: Optional[np.ndarray] = None,
) -> DRResult:
    # Risk score
    _hiv     = getattr(patient, "hiv_status", "Unknown") == "Positive"
    _prev_tb = getattr(patient, "previous_tb", "Never")  != "Never"
    _factors_map = {
        "Previous TB":      _prev_tb,
        "HIV positive":     _hiv,
        "Diabetes":         getattr(patient, "diabetes",         False),
        "Treatment failed": getattr(patient, "previously_failed", False),
        "Alcoholism":       getattr(patient, "alcoholism",        False),
        "Immunosuppressed": getattr(patient, "immunosuppressed",  False),
    }
    _score = sum(_factors_map.values())
    risk = {
        "band":    "HIGH" if _score >= 2 else "MODERATE" if _score == 1 else "LOW",
        "score":   _score,
        "factors": [f for f, v in _factors_map.items() if v],
    }

    # DR model not yet trained — run in demo mode with risk-based heuristic
    _xpert_rif = getattr(patient, "xpert_rif", "Not done")
    _dst       = getattr(patient, "dst_result", "Not done")

    if _dst in ("MDR-TB", "XDR-TB", "Pre-XDR"):
        prediction, confidence = _dst if _dst != "Pre-XDR" else "MDR-TB", 0.91
    elif _xpert_rif == "Resistant":
        prediction, confidence = "MDR-TB", 0.85
    elif _score >= 3:
        prediction, confidence = "MDR-TB", 0.62
    elif _score >= 1:
        prediction, confidence = "Drug-Sensitive", 0.55
    else:
        prediction, confidence = "Drug-Sensitive", 0.78

    prob_dict = {
        "Drug-Sensitive": round(confidence if prediction == "Drug-Sensitive" else 1 - confidence, 4),
        "MDR-TB":         round(confidence if prediction == "MDR-TB"         else 1 - confidence, 4),
        "XDR-TB":         0.0,
    }

    return DRResult(
        patient_id      = getattr(patient, "patient_id", ""),
        prediction      = prediction,
        confidence      = round(confidence, 4),
        probabilities   = prob_dict,
        risk_band       = risk["band"],
        risk_score      = risk["score"],
        risk_factors    = risk["factors"],
        warnings        = ["⚠ DR model not yet trained — result based on clinical risk factors only"],
        recommendations = [],
    )


# ══════════════════════════════════════════════════════════════
#  CLINICAL LOGIC
# ══════════════════════════════════════════════════════════════

def _warnings(p: Patient, pred: str, risk: dict) -> list[str]:
    w = []
    if p.hiv == 1:
        w.append("⚠ HIV positive — TB risk significantly elevated. Ensure ART status reviewed.")
    if p.prev_tb == 1:
        w.append("⚠ Previous TB history — higher probability of acquired drug resistance.")
    if pred in ("MDR-TB", "XDR-TB") and p.hiv == 1:
        w.append("🚨 MDR/XDR + HIV co-infection — urgent specialist referral required.")
    if p.haemoptysis == 1:
        w.append("⚠ Haemoptysis present — consider advanced or cavitary disease.")
    if risk["band"] in ("HIGH","VERY HIGH") and pred == "Drug-Sensitive":
        w.append("ℹ High clinical risk despite Sensitive prediction — confirm with full DST.")
    if p.immunocompromised == 1:
        w.append("⚠ Immunocompromised patient — atypical TB presentation possible.")
    return w


def _recommendations(p: Patient, pred: str, risk: dict) -> list[str]:
    r = []
    if pred == "Drug-Sensitive":
        r += [
            "Initiate standard first-line regimen: 2HRZE / 4HR.",
            "Sputum smear, culture, and DST recommended to confirm sensitivity.",
            "Monthly follow-up sputum during intensive phase.",
        ]
    elif pred == "MDR-TB":
        r += [
            "Do NOT use standard first-line drugs.",
            "Refer immediately to MDR-TB specialist unit.",
            "Full DST panel required (fluoroquinolones, injectables, newer agents).",
            "Consider BPaL regimen (Bedaquiline + Pretomanid + Linezolid) per WHO 2022 guidelines.",
            "Notify national TB programme — MDR-TB is notifiable.",
        ]
    elif pred == "XDR-TB":
        r += [
            "🚨 XDR-TB requires highly specialised management centre.",
            "Notify public health authority immediately.",
            "Negative pressure isolation — strict infection control.",
            "Individualised regimen based on complete resistance profile.",
            "Palliative care discussion may be appropriate if no effective drugs remain.",
        ]

    if p.hiv == 1:
        r.append("Coordinate with HIV/ID team — check current ART, drug interactions with TB treatment.")
    if p.diabetic == 1:
        r.append("Monitor blood glucose closely — hyperglycaemia impairs TB treatment outcomes.")
    if p.immunocompromised == 1:
        r.append("Review immunosuppression dose if clinically safe — reduce if possible during TB treatment.")
    if risk["band"] in ("HIGH", "VERY HIGH"):
        r.append("High-risk patient — consider twice-monthly clinic review during intensive phase.")

    return r
