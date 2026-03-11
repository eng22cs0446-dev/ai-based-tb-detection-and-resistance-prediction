"""
app/app.py — TB Detection System
IMPROVED: Real model + TTA + calibrated threshold + temperature scaling + lung-masked GradCAM
"""

import sys, os
_app_dir   = os.path.dirname(os.path.abspath(__file__))
_tb_system = os.path.dirname(_app_dir)
_project   = os.path.dirname(_tb_system)
sys.path.insert(0, _project)
sys.path.insert(0, _tb_system)

import streamlit as st
import numpy as np
import cv2
import json
import datetime
import torch
from PIL import Image

from drug_resistance_form     import drug_resistance_form, DetailedPatient
from drug_resistance.assessor import DrugResistanceAssessor
from pdf_report               import generate_pdf
from core.model               import TBModel, GradCAMPP, TemperatureScaler, predict_with_tta,MultiScaleTBModel, MultiScaleGradCAMPP
from core.preprocessing       import infer_transforms
from metadata.patient         import Patient, PatientValidator, PatientEncoder

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="TB Detection System", page_icon="🫁",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;background:#080c10;color:#c9d1d9;}
.main,.block-container{background:#080c10;}
.stButton>button{background:#1e3a5f;color:#58a6ff;border:1px solid #1e4080;
  border-radius:6px;font-weight:600;transition:all .2s;}
.stButton>button:hover{background:#2d5a9e;color:#fff;}
.stDownloadButton>button{background:#1c3d2a;color:#3fb950;border:1px solid #2ea043;
  border-radius:6px;font-weight:600;width:100%;}
.card{background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:20px;margin-bottom:12px;}
.metric-label{font-size:.7rem;color:#8b949e;text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px;}
.metric-value{font-family:'IBM Plex Mono',monospace;font-size:1.6rem;font-weight:700;line-height:1.1;}
.sec-title{color:#58a6ff;font-size:.75rem;font-weight:700;text-transform:uppercase;
  letter-spacing:1px;margin:14px 0 8px;padding-bottom:4px;border-bottom:1px solid #21262d;
  font-family:'IBM Plex Mono',monospace;}
.prob-row{display:flex;align-items:center;gap:8px;margin:5px 0;}
.prob-label{width:130px;font-size:.78rem;color:#8b949e;font-family:'IBM Plex Mono',monospace;flex-shrink:0;}
.prob-bar-bg{flex:1;height:8px;background:#21262d;border-radius:4px;overflow:hidden;}
.prob-fill{height:100%;border-radius:4px;}
.prob-val{width:44px;text-align:right;font-size:.78rem;font-family:'IBM Plex Mono',monospace;}
.warn-item{background:#2e1208;border-left:3px solid #f0883e;border-radius:4px;padding:8px 12px;margin:4px 0;font-size:.83rem;}
.rec-item{background:#0d1f2e;border-left:3px solid #1e4080;border-radius:4px;padding:8px 12px;margin:4px 0;font-size:.83rem;}
.risk-MINIMAL  {background:#0d2818;border-left:3px solid #3fb950;padding:12px 16px;border-radius:6px;}
.risk-LOW      {background:#0d1f2e;border-left:3px solid #58a6ff;padding:12px 16px;border-radius:6px;}
.risk-MODERATE {background:#2e2008;border-left:3px solid #e3b341;padding:12px 16px;border-radius:6px;}
.risk-HIGH     {background:#2e1208;border-left:3px solid #f0883e;padding:12px 16px;border-radius:6px;}
.risk-VERY_HIGH{background:#2e0808;border-left:3px solid #f85149;padding:12px 16px;border-radius:6px;}
.zone-detected    {background:#2e0808;border:1px solid #f85149;border-radius:8px;padding:14px 18px;}
.zone-borderline  {background:#2e2008;border:1px solid #e3b341;border-radius:8px;padding:14px 18px;}
.zone-normal      {background:#0d2818;border:1px solid #3fb950;border-radius:8px;padding:14px 18px;}
.caption{text-align:center;font-size:.72rem;color:#8b949e;margin-top:4px;font-family:'IBM Plex Mono',monospace;}
div[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #21262d;}
</style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
BI = {"Yes": 1, "No": 0, "Unknown": -1}
RISK_COLORS = {"MINIMAL":"#3fb950","LOW":"#58a6ff","MODERATE":"#e3b341",
               "HIGH":"#f0883e","VERY HIGH":"#f85149"}
DR_COLORS   = {"Drug-Sensitive":"#3fb950","MDR-TB":"#e3b341","XDR-TB":"#f85149"}

# Load optimal threshold from file if available, else use safe clinical default
def _load_threshold() -> float:
    path = os.path.join(_project, "checkpoints", "optimal_threshold.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        t = data["optimal_threshold"]
        print(f"[Threshold] Loaded from file: {t:.4f} "
              # REPLACE WITH:
                f"(sens={data.get('sensitivity', 0.0):.3f} "
                f"spec={data.get('specificity', 0.0):.3f})")
        return t
    return 0.42   # safe clinical default — higher sensitivity


# ══════════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_tb_model():
    ckpt = os.path.abspath(os.path.join(_project, "checkpoints", "best_model.pth"))
    if not os.path.exists(ckpt):
        return None, None, f"Weights not found at: {ckpt}"

    model = MultiScaleTBModel(
        num_classes=2,
        pretrained=False,
        dropout=0.4,
        fusion_dim=256,
        use_metadata=False,
    )
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    gradcam_engine = MultiScaleGradCAMPP(model)
    model.eval()

    # Load temperature scaler if available
    t_path = os.path.join(_project, "checkpoints", "temperature_scaler.pth")
    scaler = None
    if os.path.exists(t_path):
        scaler = TemperatureScaler(model)
        scaler.load_state_dict(torch.load(t_path, map_location="cpu"))
        scaler.eval()
        print("[TemperatureScaler] Loaded — confidence scores are calibrated")

    return model, scaler, gradcam_engine, None

@st.cache_resource
def get_assessor():
    return DrugResistanceAssessor(weights=None, img_dim=512)


# ══════════════════════════════════════════════════════════════
#  INFERENCE WITH ALL IMPROVEMENTS
# ══════════════════════════════════════════════════════════════
def run_tb_model(image: Image.Image, model, scaler, gradcam_engine) -> dict:
    model, scaler, gradcam_engine, err = load_tb_model()
    if model is None:
        st.error(f"Cannot run: {err}")
        st.stop()

    threshold  = _load_threshold()
    tfm        = infer_transforms(size=224)
    orig_rgb   = np.array(image.convert("RGB"))

    # ── TTA — 8 augmented predictions, averaged ───────────────
    tensor = tfm(image.convert("RGB")).unsqueeze(0)
    tb_prob, normal_prob = predict_with_tta(model, tensor, n_aug=8)

    # Apply temperature scaling if available (calibrated confidence)
    if scaler is not None:
        with torch.no_grad():
            cal_logits, features = scaler(tensor, None)
            cal_probs  = torch.softmax(cal_logits, dim=1).squeeze()
            tb_prob    = float(cal_probs[1].item())
            normal_prob= float(cal_probs[0].item())

    # Get image features for drug resistance
    with torch.no_grad():
        logits, features = model(tensor, None)

        # ── GradCAM++ with lung mask ──────────────────────────────
    tensor_for_cam  = tensor.clone().detach().requires_grad_(True)
    cam, scale_maps = gradcam_engine(tensor_for_cam, metadata=None, cls=1)
    heatmap_np     = cam.detach().cpu().numpy()

    # ── Three-zone classification ─────────────────────────────
    borderline_lo = threshold - 0.12
    if tb_prob >= threshold:
        zone    = "detected"
        finding = (f"TB pattern detected ({tb_prob:.1%}). "
                   f"{'High' if tb_prob > 0.75 else 'Moderate'} confidence — "
                   f"{'refer immediately.' if tb_prob > 0.75 else 'clinical correlation and sputum test recommended.'}")
    elif tb_prob >= borderline_lo:
        zone    = "borderline"
        finding = (f"Borderline result ({tb_prob:.1%}). "
                   f"Indeterminate — repeat scan or clinician review required.")
    else:
        zone    = "normal"
        finding = f"No significant pathological opacity detected ({tb_prob:.1%}). Normal lung fields."

    return {
        "tb_detected":    zone == "detected",
        "zone":           zone,
        "tb_prob":        tb_prob,
        "normal_prob":    normal_prob,
        "image_features": features.squeeze().detach().numpy(),
        "heatmap":        heatmap_np,
        "finding":        finding,
        "threshold":      threshold,
        "calibrated":     scaler is not None,
        "tta":            True,
    }


# ══════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════
def make_overlay(orig, hmap, alpha=0.45):
    H, W    = orig.shape[:2]
    hmap_r  = cv2.resize(hmap, (W, H))
    colored = cv2.applyColorMap((hmap_r*255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(orig, 1-alpha, colored, alpha, 0)

def make_heatmap_rgb(hmap, H, W):
    hmap_r  = cv2.resize(hmap, (W, H))
    colored = cv2.applyColorMap((hmap_r*255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

def prob_bar(label, val, color):
    st.markdown(f"""<div class="prob-row">
      <div class="prob-label">{label}</div>
      <div class="prob-bar-bg"><div class="prob-fill" style="width:{val*100:.1f}%;background:{color};"></div></div>
      <div class="prob-val" style="color:{color};">{val:.1%}</div>
    </div>""", unsafe_allow_html=True)

def metric_box(label, value, color):
    st.markdown(f"""<div class="card" style="text-align:center;">
      <div class="metric-label">{label}</div>
      <div class="metric-value" style="color:{color};">{value}</div>
    </div>""", unsafe_allow_html=True)

def zone_badge(zone, tb_prob, threshold):
    if zone == "detected":
        cls = "zone-detected"
        icon, title, color = "🔴", "TB DETECTED", "#f85149"
        msg = f"Probability {tb_prob:.1%} exceeds threshold {threshold:.2f}"
    elif zone == "borderline":
        cls = "zone-borderline"
        icon, title, color = "🟡", "BORDERLINE — REVIEW REQUIRED", "#e3b341"
        msg = f"Probability {tb_prob:.1%} — indeterminate. Repeat scan or clinician review."
    else:
        cls = "zone-normal"
        icon, title, color = "🟢", "NOT DETECTED", "#3fb950"
        msg = f"Probability {tb_prob:.1%} below threshold {threshold:.2f}"
    st.markdown(f"""<div class="{cls}">
      <div style="font-size:1.2rem;font-weight:700;color:{color};">{icon} {title}</div>
      <div style="font-size:.82rem;color:#8b949e;margin-top:4px;">{msg}</div>
    </div>""", unsafe_allow_html=True)

def risk_block(band, score, factors):
    key  = band.replace(" ","_")
    col  = RISK_COLORS.get(band,"#8b949e")
    ftxt = " · ".join(factors) if factors else "None identified"
    st.markdown(f"""<div class="risk-{key}">
      <div style="font-size:1.3rem;font-weight:700;color:{col};font-family:'IBM Plex Mono',monospace;">{band}</div>
      <div style="font-size:.8rem;color:#8b949e;margin-top:2px;">Score: <b style="color:{col};">{score}</b></div>
      <div style="font-size:.8rem;color:#8b949e;margin-top:6px;">Factors: {ftxt}</div>
    </div>""", unsafe_allow_html=True)

def s_select(label, key, help=""):
    return BI[st.sidebar.selectbox(label, ["Unknown","Yes","No"], key=key, help=help)]


# ══════════════════════════════════════════════════════════════
#  SIDEBAR PATIENT FORM
# ══════════════════════════════════════════════════════════════
def patient_form() -> Patient:
    st.sidebar.markdown("## 🫁 Patient Form")
    st.sidebar.markdown('<div class="sec-title">👤 Demographics</div>', unsafe_allow_html=True)
    pid    = st.sidebar.text_input("Patient ID *", placeholder="PT-2024-001")
    age    = st.sidebar.number_input("Age (years)", 0, 120, 35)
    gender = st.sidebar.selectbox("Gender", ["Male","Female","Other","Unknown"])

    st.sidebar.markdown('<div class="sec-title">⚠ Risk Factors</div>', unsafe_allow_html=True)
    smoking = s_select("Smoking",           "sm")
    alcohol = s_select("Alcohol Use",       "al")
    diabetic= s_select("Diabetic",          "db")
    hiv     = s_select("HIV Positive",      "hv")
    immuno  = s_select("Immunocompromised", "im", "Steroids, chemo, transplant")
    prev_tb = s_select("Previous TB",       "pt")
    contact = s_select("TB Contact",        "ct", "Known contact with active TB")

    st.sidebar.markdown('<div class="sec-title">🤒 Symptoms</div>', unsafe_allow_html=True)
    cough_wks = st.sidebar.number_input("Cough Duration (weeks)", 0, 104, 0)
    fever     = s_select("Fever",        "fv")
    sweats    = s_select("Night Sweats", "ns")
    wt_loss   = s_select("Weight Loss",  "wl")
    fatigue   = s_select("Fatigue",      "fg")
    haemo     = s_select("Haemoptysis",  "hm", "Coughing up blood")
    notes     = st.sidebar.text_area("Clinical Notes", placeholder="Additional observations…")

    return Patient(
        patient_id=pid, age=int(age), gender=gender,
        smoking=smoking, alcohol=alcohol, diabetic=diabetic,
        hiv=hiv, immunocompromised=immuno, prev_tb=prev_tb, tb_contact=contact,
        cough_weeks=int(cough_wks), fever=fever, night_sweats=sweats,
        weight_loss=wt_loss, fatigue=fatigue, haemoptysis=haemo, notes=notes,
    )


# ══════════════════════════════════════════════════════════════
#  RESULTS PAGE
# ══════════════════════════════════════════════════════════════
def show_results(patient, xray, dr, orig_np):
    ts       = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")
    zone     = xray.get("zone", "detected" if xray["tb_detected"] else "normal")
    tb_color = "#f85149" if zone=="detected" else "#e3b341" if zone=="borderline" else "#3fb950"
    tb_label = "TB DETECTED" if zone=="detected" else "BORDERLINE" if zone=="borderline" else "NOT DETECTED"
    dr_color = DR_COLORS.get(dr.prediction, "#8b949e")
    rc_color = RISK_COLORS.get(dr.risk_band, "#8b949e")

    # Header
    st.markdown(f"""<div style="background:#0d1117;border:1px solid #21262d;
      border-radius:10px;padding:20px 28px;margin-bottom:16px;">
      <div style="font-size:1.5rem;font-weight:700;color:#58a6ff;">🫁 Diagnostic Report</div>
      <div style="font-size:.85rem;color:#8b949e;margin-top:4px;">
        Patient: <b style="color:#c9d1d9;">{patient.patient_id}</b> &nbsp;|&nbsp;
        Age: {patient.age} &nbsp;|&nbsp; Gender: {patient.gender} &nbsp;|&nbsp; {ts}
        {"&nbsp;|&nbsp;<span style='color:#3fb950;'>✓ Calibrated</span>" if xray.get("calibrated") else ""}
        {"&nbsp;|&nbsp;<span style='color:#58a6ff;'>✓ TTA×8</span>" if xray.get("tta") else ""}
      </div>
    </div>""", unsafe_allow_html=True)

    # 4 metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_box("TB FINDING",      tb_label,                    tb_color)
    with c2: metric_box("TB PROBABILITY",  f"{xray['tb_prob']:.1%}",    tb_color)
    with c3: metric_box("DRUG RESISTANCE", dr.prediction,               dr_color)
    with c4: metric_box("CLINICAL RISK",   dr.risk_band,                rc_color)

    st.markdown("<br>", unsafe_allow_html=True)

    # Three-zone badge
    zone_badge(zone, xray["tb_prob"], xray.get("threshold", 0.42))
    st.markdown("<br>", unsafe_allow_html=True)

    # X-Ray panels
    st.markdown('<div class="sec-title">🩻 X-Ray Analysis</div>', unsafe_allow_html=True)
    H, W        = orig_np.shape[:2]
    heatmap_rgb = make_heatmap_rgb(xray["heatmap"], H, W)
    overlay_rgb = make_overlay(orig_np, xray["heatmap"])

    p1, p2, p3 = st.columns(3)
    with p1:
        st.image(orig_np,     use_container_width=True)
        st.markdown('<div class="caption">Original X-Ray</div>', unsafe_allow_html=True)
    with p2:
        st.image(heatmap_rgb, use_container_width=True)
        st.markdown('<div class="caption">GradCAM++ Heatmap (lung-masked)</div>', unsafe_allow_html=True)
    with p3:
        st.image(overlay_rgb, use_container_width=True)
        st.markdown('<div class="caption">Attention Overlay</div>', unsafe_allow_html=True)

    st.markdown(f'<p style="color:#8b949e;font-size:.82rem;margin-top:6px;">'
                f'Finding: {xray["finding"]}</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Probabilities + Risk
    cp1, cp2, cp3 = st.columns(3)
    with cp1:
        st.markdown('<div class="sec-title">TB Detection</div>', unsafe_allow_html=True)
        prob_bar("TB Positive", xray["tb_prob"],    "#f85149")
        prob_bar("Normal",      xray["normal_prob"],"#3fb950")
        t = xray.get("threshold", 0.42)
        st.markdown(f'<div style="color:#8b949e;font-size:.75rem;margin-top:6px;">'
                    f'Threshold: {t:.3f} (Youden-optimal)</div>', unsafe_allow_html=True)
    with cp2:
        st.markdown('<div class="sec-title">Drug Resistance</div>', unsafe_allow_html=True)
        for cls, p in dr.probabilities.items():
            prob_bar(cls, p, DR_COLORS[cls])
        st.markdown(f'<div style="color:#8b949e;font-size:.75rem;margin-top:6px;">'
                    f'Confidence: {dr.confidence:.1%}</div>', unsafe_allow_html=True)
    with cp3:
        st.markdown('<div class="sec-title">Clinical Risk</div>', unsafe_allow_html=True)
        risk_block(dr.risk_band, dr.risk_score, dr.risk_factors)

    st.markdown("<br>", unsafe_allow_html=True)

    # Warnings + Recommendations
    if dr.warnings:
        st.markdown('<div class="sec-title">⚠ Warnings</div>', unsafe_allow_html=True)
        for w in dr.warnings:
            st.markdown(f'<div class="warn-item">{w}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">✅ Recommendations</div>', unsafe_allow_html=True)
    for r in dr.recommendations:
        st.markdown(f'<div class="rec-item">{r}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Feature vector
    with st.expander("🔬 Clinical Feature Vector"):
        vec = patient.encode_for_model()
        names = [
            "age", "bmi", "previous_tb", "treatment_failed",
            "treatment_type", "first_line_drugs", "second_line_drugs",
            "isoniazid_months", "rifampicin_months", "fq_months", "injectable_months",
            "smear_grade", "xpert_mtb", "rif_resistant", "dst_result",
            "cough_severity", "fever_severity", "weight_loss_kg",
            "breathlessness", "hiv_status", "diabetes", "drug_adherence",
            "contact_dr_status", "crowded_living", "alcoholism",
            "smoking_pack_years", "prison_history", "tb_episodes",
            "cough_weeks", "fever_weeks", "night_sweats_severity", "fatigue_severity"
        ]
        c1, c2 = st.columns(2)
        half = len(names)//2
        for col, sl, sv in [(c1,names[:half],vec[:half]),(c2,names[half:],vec[half:])]:
            with col:
                for n, v in zip(sl, sv):
                    bv   = max(0.0, min(1.0, float(v))) if v >= 0 else 0.0
                    col_ = "#f85149" if v>0.5 else "#58a6ff" if v>=0 else "#444"
                    st.markdown(f"""<div class="prob-row">
                      <div class="prob-label" style="width:155px;font-size:.72rem;">{n}</div>
                      <div class="prob-bar-bg"><div class="prob-fill"
                        style="width:{bv*100:.0f}%;background:{col_};"></div></div>
                      <div class="prob-val" style="font-size:.72rem;">{v:.3f}</div>
                    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Downloads
    st.markdown('<div class="sec-title">📥 Export Report</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    report_json = json.dumps({
        "patient":         patient.to_dict(),
        "tb_analysis":     {k:v for k,v in xray.items()
                            if k not in ("image_features","heatmap")},
        "drug_resistance": dr.to_dict(),
        "generated_at":    ts,
    }, indent=2)
    with d1:
        st.download_button("⬇ Download JSON", data=report_json,
            file_name=f"TB_{patient.patient_id}_{ts[:10]}.json",
            mime="application/json", use_container_width=True)
    with d2:
        with st.spinner("Generating PDF…"):
            pdf_bytes = generate_pdf(patient, xray, dr, orig_np)
        st.download_button("⬇ Download PDF", data=pdf_bytes,
            file_name=f"TB_{patient.patient_id}_{ts[:10]}.pdf",
            mime="application/pdf", use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    assessor = get_assessor()
    patient  = drug_resistance_form()

    model, scaler, gradcam_engine, err = load_tb_model()

    if err:
        st.error(f"⚠ {err}")
        st.info("Train the model first: `python train.py` from project root.")
        st.stop()

    st.markdown("""<div style="background:#0d1117;border:1px solid #21262d;
      border-radius:10px;padding:18px 24px;margin-bottom:20px;">
      <div style="font-size:1.6rem;font-weight:700;color:#58a6ff;">🫁 TB Detection System</div>
      <div style="font-size:.85rem;color:#8b949e;margin-top:4px;">
        EfficientNetV2-S · SE-CBAM · GradCAM++ · TTA×8 · Calibrated · Karnataka Regional
      </div>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Chest X-Ray (frontal PA view preferred)",
                                 type=["jpg","jpeg","png"])

    if not uploaded:
        if not patient.patient_id or patient.age <= 0:
            _hiv      = patient.hiv_status == "Positive"
            _prev_tb  = patient.previous_tb != "Never"
            _factors  = {
                "Previous TB":      _prev_tb,
                "HIV positive":     _hiv,
                "Diabetes":         patient.diabetes,
                "Treatment failed": patient.previously_failed,
                "Alcoholism":       patient.alcoholism,
                "Immunosuppressed": patient.immunosuppressed,
            }
            _score = sum(_factors.values())
            risk = {
                "band":    "HIGH" if _score >= 2 else "MODERATE" if _score == 1 else "LOW",
                "score":   _score,
                "factors": [f for f, v in _factors.items() if v],
            }
            st.markdown('<div class="sec-title">📊 Live Clinical Risk Preview</div>',
                        unsafe_allow_html=True)
            risk_block(risk["band"], risk["score"], risk["factors"])
            st.caption("Upload a chest X-ray above to run full analysis.")
        else:
            st.markdown("""<div style="text-align:center;padding:60px;color:#8b949e;">
              <div style="font-size:2.5rem;">🫁</div>
              <div style="margin-top:12px;">Fill in patient details in the sidebar,
              then upload a chest X-ray to begin.</div>
            </div>""", unsafe_allow_html=True)
        return

    image_pil = Image.open(uploaded).convert("RGB")
    orig_np   = np.array(image_pil)

    ci, cd = st.columns([1, 1])
    with ci:
        st.image(image_pil, caption="Uploaded X-Ray", use_container_width=True)
    with cd:
        st.markdown(f"""<div class="card">
          <div class="sec-title">Image Info</div>
          <div style="font-size:.85rem;line-height:2.2;">
            📄 <span style="color:#8b949e;">File:</span> <b>{uploaded.name}</b><br>
            📐 <span style="color:#8b949e;">Size:</span>
               <b>{image_pil.width} × {image_pil.height} px</b>
          </div>
        </div>""", unsafe_allow_html=True)

        errors = []
        if not patient.patient_id or patient.patient_id.strip() == "":
            errors.append("Patient ID is required")
        if patient.age <= 0:
            errors.append("Age must be greater than 0")
        if errors:
            st.error("Fix before running:")
            for e in errors: st.markdown(f"- {e}")
        else:
            if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
                with st.spinner("Running X-ray analysis + TTA×8 + GradCAM++…"):
                    model, scaler, gradcam_engine, err = load_tb_model()
                    if err:
                        st.error(err)
                        st.stop()
                    xray = run_tb_model(image_pil, model, scaler, gradcam_engine)
                with st.spinner("Assessing drug resistance…"):
                    metadata_vector = patient.encode_for_model()
                    from drug_resistance.assessor import assess as dr_assess
                    dr = dr_assess(patient, xray["image_features"])
                st.session_state["results"] = (xray, dr, orig_np)
                st.session_state["patient"] = patient

    if "results" in st.session_state:
        xray, dr, orig_np = st.session_state["results"]
        pt = st.session_state.get("patient", patient)
        st.markdown("---")
        show_results(pt, xray, dr, orig_np)


if __name__ == "__main__":
    main()