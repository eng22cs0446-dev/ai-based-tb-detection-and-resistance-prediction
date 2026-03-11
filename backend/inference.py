"""
inference.py — Bridge between FastAPI and tb_system model code.
Loaded once at startup via load_model(), called per request via run_inference().
"""
from __future__ import annotations
import os, sys, json, base64, uuid
from datetime import datetime
from io import BytesIO
from typing import Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────────────
_backend = os.path.dirname(os.path.abspath(__file__))
_root    = os.path.dirname(_backend)
_tb_sys  = os.path.join(_root, "tb_system")
sys.path.insert(0, _root)
sys.path.insert(0, _tb_sys)

from core.model         import MultiScaleTBModel, MultiScaleGradCAMPP, TemperatureScaler, predict_with_tta
from core.preprocessing import infer_transforms
from drug_resistance.assessor import assess as dr_assess
from app.drug_resistance_form import DetailedPatient


# ── Constants ─────────────────────────────────────────────────────────────────
CKPT_DIR      = os.path.join(_root, "checkpoints")
HEATMAP_DIR   = os.path.join(_root, "heatmaps")
os.makedirs(HEATMAP_DIR, exist_ok=True)

ZONE_COLORS = {
    "DETECTED":     "#f85149",
    "BORDERLINE":   "#e3b341",
    "NOT_DETECTED": "#3fb950",
}


# ── Model loader (called once at startup) ────────────────────────────────────
def load_model() -> dict:
    """Load model, GradCAM engine, temperature scaler and threshold into a state dict."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[inference] Loading model on {device}...")

    # TB model
    ckpt = os.path.join(CKPT_DIR, "best_model.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model = MultiScaleTBModel(
        num_classes=2, pretrained=False,
        dropout=0.4, fusion_dim=256, use_metadata=False,
    )
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device)
    model.eval()

    # GradCAM engine — created once, hooks stay registered
    gradcam = MultiScaleGradCAMPP(model)

    # Temperature scaler
    scaler = None
    t_path = os.path.join(CKPT_DIR, "temperature_scaler.pth")
    if os.path.exists(t_path):
        scaler = TemperatureScaler(model)
        scaler.load_state_dict(torch.load(t_path, map_location=device))
        scaler.eval()
        print("[inference] Temperature scaler loaded — probabilities calibrated")

    # Threshold
    thresh_path = os.path.join(CKPT_DIR, "optimal_threshold.json")
    threshold   = 0.35
    model_auc   = 0.0
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            data      = json.load(f)
            threshold = data.get("optimal_threshold", 0.35)
            model_auc = data.get("final_auc", 0.0)

    print(f"[inference] Model loaded. Threshold={threshold:.3f}  AUC={model_auc:.4f}")
    return {
        "model":     model,
        "gradcam":   gradcam,
        "scaler":    scaler,
        "threshold": threshold,
        "model_auc": model_auc,
        "device":    device,
    }


# ── Per-request inference ─────────────────────────────────────────────────────
def run_inference(image_b64: str, patient: DetailedPatient, state: dict) -> dict:
    """
    Full inference pipeline.
    Returns a dict ready to be stored in DB and returned to frontend.
    """
    model     = state["model"]
    gradcam   = state["gradcam"]
    scaler    = state["scaler"]
    threshold = state["threshold"]
    device    = state["device"]

    # ── Decode image ──────────────────────────────────────────
    img_bytes  = base64.b64decode(image_b64)
    image_pil  = Image.open(BytesIO(img_bytes)).convert("RGB")
    orig_np    = np.array(image_pil)
    tfm        = infer_transforms(size=224)
    tensor     = tfm(image_pil).unsqueeze(0)

    # ── TTA prediction ────────────────────────────────────────
    tb_prob, normal_prob = predict_with_tta(model, tensor, n_aug=8)

    # ── Temperature scaling ───────────────────────────────────
    calibrated = False
    if scaler is not None:
        with torch.no_grad():
            cal_logits, _ = scaler(tensor.to(device), None)
            cal_probs     = torch.softmax(cal_logits, dim=1).squeeze()
            tb_prob       = float(cal_probs[1].item())
        calibrated = True

    # ── Zone classification ───────────────────────────────────
    borderline_lo = threshold - 0.12
    if tb_prob >= threshold:
        zone = "DETECTED"
        finding = (f"TB pattern detected ({tb_prob:.1%}). "
                   f"Probability exceeds threshold ({threshold:.2f}). "
                   f"Recommend clinical correlation and sputum tests.")
    elif tb_prob >= borderline_lo:
        zone = "BORDERLINE"
        finding = (f"Borderline TB probability ({tb_prob:.1%}). "
                   f"Clinical correlation required. Consider repeat X-ray in 4–6 weeks.")
    else:
        zone = "NOT_DETECTED"
        finding = (f"No TB pattern detected ({tb_prob:.1%}). "
                   f"Probability below threshold ({threshold:.2f}).")

    # ── Image features for DR model ───────────────────────────
    with torch.no_grad():
        _, image_features = model(tensor.to(device), None)
    image_features_np = image_features.squeeze().cpu().numpy()

    # ── Drug resistance ───────────────────────────────────────
    dr = dr_assess(patient, image_features_np)

    # ── Clinical risk ─────────────────────────────────────────
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
    band_colors = {"HIGH": "#f85149", "MODERATE": "#e3b341", "LOW": "#3fb950"}

    # ── Recommendations ───────────────────────────────────────
    recs = []
    if zone == "DETECTED":
        recs += ["Initiate sputum smear microscopy and GeneXpert MTB/RIF",
                 "Notify district TB officer within 24 hours",
                 "Begin contact tracing for household members"]
        if _hiv:
            recs.append("Urgent HIV/TB co-management — coordinate with ART clinic")
    elif zone == "BORDERLINE":
        recs += ["Repeat chest X-ray in 4–6 weeks",
                 "Sputum culture if clinically indicated",
                 "Monitor symptoms closely"]
    else:
        recs += ["Continue routine monitoring if symptomatic",
                 "Consider alternative diagnosis if symptoms persist"]
    if dr.prediction in ("MDR-TB", "XDR-TB"):
        recs.insert(0, f"⚠ {dr.prediction} risk — refer to DR-TB centre immediately")

    # ── GradCAM heatmap ───────────────────────────────────────
    tensor_for_cam = tensor.clone().detach().requires_grad_(True)
    cam, _         = gradcam(tensor_for_cam, metadata=None, cls=1)
    heatmap_np     = cam.detach().cpu().numpy()

    original_b64, heatmap_b64, overlay_b64 = _make_heatmap_images(
        orig_np, heatmap_np, patient.patient_id
    )

    # ── Assemble result ───────────────────────────────────────
    case_id = str(uuid.uuid4())[:8].upper()

    return {
        "case_id":       case_id,
        "patient_id":    patient.patient_id,
        "timestamp":     datetime.now().isoformat(),

        "tb_detection": {
            "probability":    round(tb_prob, 4),
            "zone":           zone,
            "zone_color":     ZONE_COLORS[zone],
            "threshold_used": threshold,
            "calibrated":     calibrated,
        },
        "drug_resistance": {
            "prediction":    dr.prediction,
            "probabilities": dr.probabilities,
            "is_demo_mode":  True,
            "demo_warning":  "DR model not yet trained — risk-factor based result",
        },
        "clinical_risk": {
            "band":       risk["band"],
            "band_color": band_colors.get(risk["band"], "#8b949e"),
            "score":      risk["score"],
            "factors":    risk["factors"],
        },
        "finding_text":    finding,
        "recommendations": recs,
        "warnings":        dr.warnings,

        "heatmap": {
            "original_base64":     original_b64,
            "heatmap_only_base64": heatmap_b64,
            "overlay_base64":      overlay_b64,
        },
        "model_meta": {
            "model_version": "v3",
            "auc":           state["model_auc"],
            "threshold":     threshold,
            "device":        device,
        },

        # For DB storage
        "_dr_confidence":   dr.confidence,
        "_dr_probabilities": json.dumps(dr.probabilities),
        "_risk_factors":    json.dumps(risk["factors"]),
        "_image_features":  image_features_np,
    }


# ── Heatmap helpers ───────────────────────────────────────────────────────────
def _make_heatmap_images(orig_np: np.ndarray, heatmap_np: np.ndarray,
                          patient_id: str) -> tuple[str, str, str]:
    import cv2

    orig_resized = cv2.resize(orig_np, (224, 224))
    h_resized    = cv2.resize(heatmap_np, (224, 224))
    h_uint8      = (h_resized * 255).astype(np.uint8)
    heatmap_rgb  = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
    heatmap_rgb  = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)

    if len(orig_resized.shape) == 2:
        orig_resized = cv2.cvtColor(orig_resized, cv2.COLOR_GRAY2RGB)

    overlay = cv2.addWeighted(orig_resized, 0.6, heatmap_rgb, 0.4, 0)

    def to_b64(arr):
        pil = Image.fromarray(arr.astype(np.uint8))
        buf = BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    return to_b64(orig_resized), to_b64(heatmap_rgb), to_b64(overlay)
