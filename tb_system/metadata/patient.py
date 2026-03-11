"""
metadata/patient.py
Single source of truth for patient clinical data.
Fields: Demographics + Risk Factors + Symptoms only (no lab results).
"""

from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class Patient:
    patient_id        : str = ""
    age               : int = 0
    gender            : str = "Unknown"
    smoking           : int = -1
    alcohol           : int = -1
    diabetic          : int = -1
    hiv               : int = -1
    immunocompromised : int = -1
    prev_tb           : int = -1
    tb_contact        : int = -1
    cough_weeks       : int = 0
    fever             : int = -1
    night_sweats      : int = -1
    weight_loss       : int = -1
    fatigue           : int = -1
    haemoptysis       : int = -1
    notes             : str = ""

    def to_dict(self):  return asdict(self)
    def to_json(self):  return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PatientValidator:
    GENDERS = {"male","female","other","unknown"}
    BINARY  = {-1, 0, 1}

    @classmethod
    def validate(cls, p: Patient) -> list[str]:
        errors = []
        if not p.patient_id.strip():      errors.append("Patient ID is required.")
        if not (0 <= p.age <= 120):       errors.append(f"Age must be 0-120 (got {p.age}).")
        if p.gender.lower() not in cls.GENDERS: errors.append("Invalid gender value.")
        for f in ["smoking","alcohol","diabetic","hiv","immunocompromised","prev_tb",
                  "tb_contact","fever","night_sweats","weight_loss","fatigue","haemoptysis"]:
            if getattr(p, f) not in cls.BINARY:
                errors.append(f"'{f}' must be -1/0/1.")
        if p.cough_weeks < 0: errors.append("Cough weeks cannot be negative.")
        return errors


class PatientEncoder:
    DIM = 16

    @staticmethod
    def encode(p: Patient) -> np.ndarray:
        v = np.zeros(16, dtype=np.float32)
        v[0] = min(p.age / 100.0, 1.0)
        v[1] = float(p.gender.lower() == "male")
        v[2] = float(p.gender.lower() == "female")
        for i, f in enumerate(["smoking","alcohol","diabetic","hiv",
                                "immunocompromised","prev_tb","tb_contact"], 3):
            v[i] = float(getattr(p, f))
        v[10] = min(p.cough_weeks / 52.0, 1.0)
        for i, f in enumerate(["fever","night_sweats","weight_loss","fatigue","haemoptysis"], 11):
            v[i] = float(getattr(p, f))
        return v

    @staticmethod
    def feature_names() -> list[str]:
        return ["age","gender_male","gender_female","smoking","alcohol","diabetic",
                "hiv","immunocompromised","prev_tb","tb_contact","cough_weeks",
                "fever","night_sweats","weight_loss","fatigue","haemoptysis"]


class RiskScorer:
    WEIGHTS = {"hiv":3,"prev_tb":3,"tb_contact":2,"immunocompromised":2,
               "haemoptysis":2,"diabetic":1,"smoking":1,"alcohol":1,
               "night_sweats":1,"weight_loss":1,"fever":1,"fatigue":1}

    @classmethod
    def score(cls, p: Patient) -> dict:
        total, factors = 0, []
        for field, w in cls.WEIGHTS.items():
            if getattr(p, field) == 1:
                total += w
                factors.append(field.replace("_"," ").title())
        if p.cough_weeks >= 3:
            total += 2
            factors.append(f"Persistent Cough ({p.cough_weeks} wks)")
        band = ("VERY HIGH" if total>=8 else "HIGH" if total>=5
                else "MODERATE" if total>=3 else "LOW" if total>=1 else "MINIMAL")
        return {"score": total, "band": band, "factors": factors}
