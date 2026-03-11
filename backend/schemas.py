"""
schemas.py — Pydantic request/response models for the TB Detection API
"""
from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ── REQUEST ───────────────────────────────────────────────────────────────────

class PatientIn(BaseModel):
    # Demographics
    patient_id:             str
    age:                    int
    gender:                 str   = "Unknown"
    weight_kg:              float = 0.0
    height_cm:              float = 0.0
    district:               str   = ""
    occupation:             str   = ""
    # TB History
    previous_tb:            str   = "Never"
    previous_tb_years:      int   = 0
    previous_tb_count:      int   = 0
    previous_treatment:     str   = "None"
    treatment_duration_months: int = 0
    treatment_completed:    str   = "N/A"
    reason_stopped:         str   = "N/A"
    previously_failed:      bool  = False
    # Symptoms
    cough_present:          bool  = False
    cough_weeks:            int   = 0
    cough_severity:         int   = 0
    cough_character:        str   = "Dry"
    sputum_volume_ml:       str   = "None"
    haemoptysis:            bool  = False
    fever:                  bool  = False
    fever_weeks:            int   = 0
    fever_severity:         int   = 0
    night_sweats:           bool  = False
    night_sweats_severity:  int   = 0
    weight_loss:            bool  = False
    weight_loss_kg:         float = 0.0
    fatigue_severity:       int   = 0
    breathlessness:         bool  = False
    chest_pain:             bool  = False
    # Lab results
    sputum_smear:           str   = "Not done"
    xpert_mtb:              str   = "Not done"
    xpert_rif:              str   = "Not done"
    dst_result:             str   = "Not done"
    hiv_status:             str   = "Unknown"
    hba1c:                  float = 0.0
    # Drug exposure
    isoniazid_months:       int   = 0
    rifampicin_months:      int   = 0
    fluoroquinolone_months: int   = 0
    injectable_months:      int   = 0
    drug_adherence:         str   = "Unknown"
    # Comorbidities
    diabetes:               bool  = False
    hiv_on_art:             bool  = False
    alcoholism:             bool  = False
    smoking:                bool  = False
    smoking_pack_years:     float = 0.0
    immunosuppressed:       bool  = False
    previously_failed:      bool  = False
    # Social
    tb_contact:             bool  = False
    contact_dr_status:      str   = "Unknown"


class PredictRequest(BaseModel):
    patient:      PatientIn
    xray_base64:  str = Field(..., description="Base64-encoded X-ray image (JPEG or PNG)")


# ── RESPONSE ──────────────────────────────────────────────────────────────────

class TBDetection(BaseModel):
    probability:    float
    zone:           str    # DETECTED | BORDERLINE | NOT_DETECTED
    zone_color:     str    # hex color
    threshold_used: float
    calibrated:     bool


class DrugResistance(BaseModel):
    prediction:    str
    probabilities: Dict[str, float]
    is_demo_mode:  bool
    demo_warning:  Optional[str] = None


class ClinicalRisk(BaseModel):
    band:       str
    band_color: str
    score:      int
    factors:    List[str]


class Heatmap(BaseModel):
    original_base64:    str
    heatmap_only_base64: str
    overlay_base64:     str


class ModelMeta(BaseModel):
    model_version:  str
    auc:            float
    threshold:      float
    device:         str


class PredictResponse(BaseModel):
    case_id:       str
    patient_id:    str
    timestamp:     str
    tb_detection:  TBDetection
    drug_resistance: DrugResistance
    clinical_risk: ClinicalRisk
    finding_text:  str
    recommendations: List[str]
    heatmap:       Heatmap
    model_meta:    ModelMeta
    warnings:      List[str] = []


# ── CASE LIST ─────────────────────────────────────────────────────────────────

class CaseSummary(BaseModel):
    case_id:        str
    patient_id:     str
    timestamp:      str
    tb_zone:        str
    tb_probability: float
    dr_prediction:  str
    risk_band:      str
    district:       str


class CaseListResponse(BaseModel):
    cases:  List[CaseSummary]
    total:  int
    page:   int
    limit:  int


# ── DASHBOARD ─────────────────────────────────────────────────────────────────

class DistrictStat(BaseModel):
    district:    str
    total:       int
    tb_detected: int
    rate:        float


class WeeklyStat(BaseModel):
    week:        str
    total:       int
    tb_detected: int


class DashboardStats(BaseModel):
    total_cases:       int
    tb_detected:       int
    detection_rate:    float
    avg_probability:   float
    by_district:       List[DistrictStat]
    by_week:           List[WeeklyStat]
    dr_breakdown:      Dict[str, int]


# ── ERROR ─────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error:   bool = True
    code:    int
    message: str
    detail:  Optional[str] = None


# ── HEALTH ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:          str
    model_loaded:    bool
    threshold:       float
    device:          str
    model_auc:       float
    model_version:   str
