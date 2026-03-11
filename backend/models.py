"""
models.py — SQLAlchemy ORM table definitions
"""
from sqlalchemy import Column, String, Float, Integer, Boolean, Text, DateTime
from sqlalchemy.sql import func
from backend.database import Base


class Case(Base):
    __tablename__ = "cases"

    case_id         = Column(String, primary_key=True, index=True)
    patient_id      = Column(String, index=True, nullable=False)
    timestamp       = Column(String, nullable=False)

    # TB detection
    tb_probability  = Column(Float,   nullable=False)
    tb_zone         = Column(String,  nullable=False)   # DETECTED | BORDERLINE | NOT_DETECTED
    threshold_used  = Column(Float,   nullable=False)
    calibrated      = Column(Boolean, default=False)

    # Drug resistance
    dr_prediction   = Column(String,  nullable=False)
    dr_confidence   = Column(Float,   nullable=False)
    dr_probabilities = Column(Text,   nullable=False)   # JSON string
    dr_demo_mode    = Column(Boolean, default=True)

    # Clinical risk
    risk_band       = Column(String,  nullable=False)
    risk_score      = Column(Integer, nullable=False)
    risk_factors    = Column(Text,    nullable=False)   # JSON string

    # Finding & recommendations
    finding_text    = Column(Text,    nullable=False)
    recommendations = Column(Text,    nullable=False)   # JSON string
    warnings        = Column(Text,    default="[]")     # JSON string

    # Patient snapshot (key fields for filtering)
    district        = Column(String,  default="")
    age             = Column(Integer, default=0)
    gender          = Column(String,  default="Unknown")

    # Full patient JSON + heatmap paths
    patient_json    = Column(Text,    nullable=False)   # full DetailedPatient as JSON
    heatmap_original_path  = Column(String, default="")
    heatmap_overlay_path   = Column(String, default="")
    heatmap_only_path      = Column(String, default="")

    # Model metadata
    model_version   = Column(String,  default="v3")
    model_auc       = Column(Float,   default=0.0)
