"""
main.py — FastAPI application with all TB Detection API endpoints.

Run with:
    cd D:\\TB_Detection_SystemV1
    uvicorn backend.main:app --reload --port 8000

Base URL: http://localhost:8000/api/v1
"""
from __future__ import annotations
import json, os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func as sqlfunc

import sys
_backend = os.path.dirname(os.path.abspath(__file__))
_root    = os.path.dirname(_backend)
_tb_sys  = os.path.join(_root, "tb_system")
sys.path.insert(0, _root)
sys.path.insert(0, _tb_sys)

from backend.database  import engine, get_db, Base
from backend.models    import Case
from backend.schemas   import (
    PredictRequest, PredictResponse,
    CaseListResponse, CaseSummary,
    DashboardStats, DistrictStat, WeeklyStat,
    HealthResponse, ErrorResponse,
)
from backend.inference import load_model, run_inference
from app.drug_resistance_form import DetailedPatient


# ── App setup ─────────────────────────────────────────────────────────────────
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title       = "TB Detection System API",
    description = "AI-powered TB detection for Karnataka NTEP",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # tighten in production
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Load model once at startup ────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    try:
        app.state.model_state = load_model()
        print("[startup] Model loaded successfully")
    except Exception as e:
        print(f"[startup] WARNING: Model failed to load — {e}")
        app.state.model_state = None


# ══════════════════════════════════════════════════════════════════════════════
#  HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health():
    """Check if the API and model are ready."""
    ms = app.state.model_state
    if ms is None:
        return HealthResponse(
            status="degraded", model_loaded=False,
            threshold=0.0, device="cpu",
            model_auc=0.0, model_version="unknown",
        )
    return HealthResponse(
        status        = "ok",
        model_loaded  = True,
        threshold     = ms["threshold"],
        device        = ms["device"],
        model_auc     = ms["model_auc"],
        model_version = "v3",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest, db: Session = Depends(get_db)):
    """
    Full TB analysis — X-ray + patient data → all results.
    Saves result to database and returns complete response.
    """
    if app.state.model_state is None:
        raise HTTPException(503, "Model not loaded. Check server logs.")

    # Convert PatientIn → DetailedPatient dataclass
    patient = DetailedPatient(**request.patient.dict())

    # Run inference
    try:
        result = run_inference(request.xray_base64, patient, app.state.model_state)
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {str(e)}")

    # Save to DB
    case = Case(
        case_id          = result["case_id"],
        patient_id       = result["patient_id"],
        timestamp        = result["timestamp"],
        tb_probability   = result["tb_detection"]["probability"],
        tb_zone          = result["tb_detection"]["zone"],
        threshold_used   = result["tb_detection"]["threshold_used"],
        calibrated       = result["tb_detection"]["calibrated"],
        dr_prediction    = result["drug_resistance"]["prediction"],
        dr_confidence    = result["_dr_confidence"],
        dr_probabilities = result["_dr_probabilities"],
        dr_demo_mode     = True,
        risk_band        = result["clinical_risk"]["band"],
        risk_score       = result["clinical_risk"]["score"],
        risk_factors     = json.dumps(result["clinical_risk"]["factors"]),
        finding_text     = result["finding_text"],
        recommendations  = json.dumps(result["recommendations"]),
        warnings         = json.dumps(result["warnings"]),
        district         = request.patient.district,
        age              = request.patient.age,
        gender           = request.patient.gender,
        patient_json     = json.dumps(request.patient.dict()),
        model_version    = result["model_meta"]["model_version"],
        model_auc        = result["model_meta"]["auc"],
    )
    db.add(case)
    db.commit()

    return PredictResponse(**{k: v for k, v in result.items() if not k.startswith("_")})


# ══════════════════════════════════════════════════════════════════════════════
#  CASES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/cases", response_model=CaseListResponse, tags=["Cases"])
def list_cases(
    page:     int            = Query(1,    ge=1),
    limit:    int            = Query(20,   ge=1, le=100),
    district: Optional[str] = Query(None),
    zone:     Optional[str] = Query(None),
    date_from: Optional[str] = Query(None, description="ISO date YYYY-MM-DD"),
    date_to:   Optional[str] = Query(None, description="ISO date YYYY-MM-DD"),
    db:       Session        = Depends(get_db),
):
    """List cases with optional filters."""
    q = db.query(Case)
    if district:
        q = q.filter(Case.district == district)
    if zone:
        q = q.filter(Case.tb_zone == zone)
    if date_from:
        q = q.filter(Case.timestamp >= date_from)
    if date_to:
        q = q.filter(Case.timestamp <= date_to + "T23:59:59")

    total  = q.count()
    cases  = q.order_by(Case.timestamp.desc()).offset((page - 1) * limit).limit(limit).all()

    return CaseListResponse(
        cases = [CaseSummary(
            case_id        = c.case_id,
            patient_id     = c.patient_id,
            timestamp      = c.timestamp,
            tb_zone        = c.tb_zone,
            tb_probability = c.tb_probability,
            dr_prediction  = c.dr_prediction,
            risk_band      = c.risk_band,
            district       = c.district,
        ) for c in cases],
        total = total,
        page  = page,
        limit = limit,
    )


@app.get("/api/v1/cases/{case_id}", tags=["Cases"])
def get_case(case_id: str, db: Session = Depends(get_db)):
    """Get full result for a single case."""
    case = db.query(Case).filter(Case.case_id == case_id).first()
    if not case:
        raise HTTPException(404, f"Case {case_id} not found")

    return {
        "case_id":        case.case_id,
        "patient_id":     case.patient_id,
        "timestamp":      case.timestamp,
        "patient":        json.loads(case.patient_json),
        "tb_detection": {
            "probability":    case.tb_probability,
            "zone":           case.tb_zone,
            "threshold_used": case.threshold_used,
            "calibrated":     case.calibrated,
        },
        "drug_resistance": {
            "prediction":    case.dr_prediction,
            "probabilities": json.loads(case.dr_probabilities),
            "is_demo_mode":  case.dr_demo_mode,
        },
        "clinical_risk": {
            "band":    case.risk_band,
            "score":   case.risk_score,
            "factors": json.loads(case.risk_factors),
        },
        "finding_text":    case.finding_text,
        "recommendations": json.loads(case.recommendations),
        "warnings":        json.loads(case.warnings),
        "model_meta": {
            "model_version": case.model_version,
            "auc":           case.model_auc,
        },
    }


@app.delete("/api/v1/cases/{case_id}", tags=["Cases"])
def delete_case(case_id: str, db: Session = Depends(get_db)):
    """Delete a case."""
    case = db.query(Case).filter(Case.case_id == case_id).first()
    if not case:
        raise HTTPException(404, f"Case {case_id} not found")
    db.delete(case)
    db.commit()
    return {"deleted": True, "case_id": case_id}


@app.get("/api/v1/cases/{case_id}/pdf", tags=["Cases"])
def download_pdf(case_id: str, db: Session = Depends(get_db)):
    """Download PDF report for a case."""
    case = db.query(Case).filter(Case.case_id == case_id).first()
    if not case:
        raise HTTPException(404, f"Case {case_id} not found")

    # Reconstruct objects and generate PDF
    try:
        from app.pdf_report import generate_pdf
        from app.drug_resistance_form import DetailedPatient
        from drug_resistance.assessor import DRResult
        import tempfile, io
        from PIL import Image
        import numpy as np

        patient = DetailedPatient(**json.loads(case.patient_json))

        dr = DRResult(
            patient_id      = case.patient_id,
            prediction      = case.dr_prediction,
            confidence      = case.dr_confidence,
            probabilities   = json.loads(case.dr_probabilities),
            risk_band       = case.risk_band,
            risk_score      = case.risk_score,
            risk_factors    = json.loads(case.risk_factors),
            warnings        = json.loads(case.warnings),
            recommendations = json.loads(case.recommendations),
        )

        xray = {
            "tb_prob":       case.tb_probability,
            "zone":          case.tb_zone,
            "finding":       case.finding_text,
            "image_features": np.zeros(512),
        }

        orig_np = np.zeros((224, 224, 3), dtype=np.uint8)

        pdf_bytes = generate_pdf(patient, xray, dr, orig_np)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf_bytes)
        tmp.close()

        return FileResponse(
            tmp.name,
            media_type    = "application/pdf",
            filename      = f"TB_Report_{case.patient_id}_{case.case_id}.pdf",
        )
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
#  PATIENTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/patients/{patient_id}/cases", tags=["Patients"])
def get_patient_cases(patient_id: str, db: Session = Depends(get_db)):
    """Get all cases for a patient."""
    cases = db.query(Case).filter(
        Case.patient_id == patient_id
    ).order_by(Case.timestamp.desc()).all()

    if not cases:
        raise HTTPException(404, f"No cases found for patient {patient_id}")

    return {
        "patient_id": patient_id,
        "total":      len(cases),
        "cases": [CaseSummary(
            case_id        = c.case_id,
            patient_id     = c.patient_id,
            timestamp      = c.timestamp,
            tb_zone        = c.tb_zone,
            tb_probability = c.tb_probability,
            dr_prediction  = c.dr_prediction,
            risk_band      = c.risk_band,
            district       = c.district,
        ) for c in cases],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/dashboard/stats", response_model=DashboardStats, tags=["Dashboard"])
def dashboard_stats(
    days: int = Query(30, ge=1, le=365, description="Stats for last N days"),
    db:   Session = Depends(get_db),
):
    """Aggregated statistics for the dashboard."""
    since = (datetime.now() - timedelta(days=days)).isoformat()
    q     = db.query(Case).filter(Case.timestamp >= since)
    cases = q.all()

    total       = len(cases)
    tb_detected = sum(1 for c in cases if c.tb_zone == "DETECTED")
    avg_prob    = sum(c.tb_probability for c in cases) / total if total else 0.0

    # By district
    district_map: dict = {}
    for c in cases:
        d = c.district or "Unknown"
        if d not in district_map:
            district_map[d] = {"total": 0, "tb": 0}
        district_map[d]["total"] += 1
        if c.tb_zone == "DETECTED":
            district_map[d]["tb"] += 1

    by_district = [
        DistrictStat(
            district    = d,
            total       = v["total"],
            tb_detected = v["tb"],
            rate        = round(v["tb"] / v["total"], 3) if v["total"] else 0.0,
        )
        for d, v in sorted(district_map.items(), key=lambda x: -x[1]["total"])
    ]

    # By week
    week_map: dict = {}
    for c in cases:
        try:
            dt   = datetime.fromisoformat(c.timestamp)
            week = dt.strftime("%Y-W%W")
        except Exception:
            week = "Unknown"
        if week not in week_map:
            week_map[week] = {"total": 0, "tb": 0}
        week_map[week]["total"] += 1
        if c.tb_zone == "DETECTED":
            week_map[week]["tb"] += 1

    by_week = [
        WeeklyStat(week=w, total=v["total"], tb_detected=v["tb"])
        for w, v in sorted(week_map.items())
    ]

    # DR breakdown
    dr_map: dict = {}
    for c in cases:
        dr_map[c.dr_prediction] = dr_map.get(c.dr_prediction, 0) + 1

    return DashboardStats(
        total_cases     = total,
        tb_detected     = tb_detected,
        detection_rate  = round(tb_detected / total, 3) if total else 0.0,
        avg_probability = round(avg_prob, 3),
        by_district     = by_district,
        by_week         = by_week,
        dr_breakdown    = dr_map,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ROOT
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["System"])
def root():
    return {
        "name":    "TB Detection System API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/api/v1/health",
    }
