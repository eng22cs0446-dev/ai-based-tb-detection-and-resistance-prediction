"""
7_drug_resistance_form.py
=========================
Complete drug resistance assessment form with detailed clinical questions.
Drop-in replacement for the current patient sidebar in app.py.

Sections:
  1. Demographics & Anthropometrics
  2. TB History & Treatment History
  3. Current Symptoms (with severity scales)
  4. Lab & Sputum Results
  5. Drug Exposure History
  6. Risk Factors & Comorbidities
  7. Social & Epidemiological Factors

Replaces the simple Yes/No form with:
  - Numeric scales (severity 1-10, duration in weeks)
  - Multi-choice dropdowns (which drugs, how long, reason stopped)
  - Quantitative fields (weight loss in kg, cough frequency)
  - Free text (clinical notes per section)

PASTE THIS INTO app.py — replace the patient_form() function entirely.
"""

import streamlit as st
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict


# ── DETAILED PATIENT DATACLASS ────────────────────────────────
@dataclass
class DetailedPatient:
    # -- Section 1: Demographics
    patient_id:         str   = ""
    age:                int   = 0
    gender:             str   = "Unknown"
    weight_kg:          float = 0.0
    height_cm:          float = 0.0
    district:           str   = ""
    occupation:         str   = ""

    # -- Section 2: TB History
    previous_tb:        str   = "Never"          # Never / Yes-completed / Yes-incomplete / Unknown
    previous_tb_years:  int   = 0
    previous_tb_count:  int   = 0                # How many episodes
    previous_treatment: str   = "None"           # DOTS / Self-medicated / Private / Unknown
    treatment_duration_months: int = 0
    treatment_completed: str  = "N/A"            # Completed / Stopped early / Ongoing / N/A
    reason_stopped:     str   = "N/A"            # Side effects / Felt better / Cost / Unavailability / N/A
    previously_failed:  bool  = False            # Treatment failure

    # -- Section 3: Current Symptoms
    cough_present:      bool  = False
    cough_weeks:        int   = 0
    cough_severity:     int   = 0                # 1-10
    cough_character:    str   = "Dry"            # Dry / Productive / Bloody
    sputum_volume_ml:   str   = "None"           # None / <5ml / 5-30ml / >30ml
    haemoptysis:        bool  = False
    haemoptysis_amount: str   = "None"           # None / Streaks / Teaspoon / >Teaspoon
    fever:              bool  = False
    fever_weeks:        int   = 0
    fever_severity:     int   = 0
    fever_pattern:      str   = "Unknown"        # Continuous / Intermittent / Evening rise / Unknown
    night_sweats:       bool  = False
    night_sweats_severity: int = 0
    weight_loss:        bool  = False
    weight_loss_kg:     float = 0.0
    weight_loss_weeks:  int   = 0
    fatigue_severity:   int   = 0
    breathlessness:     bool  = False
    breathlessness_grade: str = "None"           # None / On exertion / At rest / At rest with orthopnoea
    chest_pain:         bool  = False
    chest_pain_type:    str   = "None"           # None / Pleuritic / Dull ache / Sharp

    # -- Section 4: Lab & Sputum Results
    sputum_smear:       str   = "Not done"       # Not done / Negative / 1+ / 2+ / 3+
    sputum_culture:     str   = "Not done"       # Not done / Negative / Positive / Pending
    xpert_mtb:          str   = "Not done"       # Not done / Not detected / Detected / Trace
    xpert_rif:          str   = "Not done"       # Not done / Sensitive / Resistant / Indeterminate
    dst_performed:      bool  = False
    dst_result:         str   = "Not done"       # Not done / DS-TB / MDR-TB / XDR-TB / Pre-XDR
    dst_drugs_resistant: List[str] = field(default_factory=list)
    line_probe_assay:   str   = "Not done"       # Not done / Sensitive / INH resistant / RIF resistant / Both resistant
    hiv_status:         str   = "Unknown"        # Positive / Negative / Unknown
    cd4_count:          int   = 0                # if HIV+
    viral_load:         str   = "Unknown"
    hba1c:              float = 0.0              # if diabetic
    serum_creatinine:   float = 0.0

    # -- Section 5: Drug Exposure
    first_line_drugs:   List[str] = field(default_factory=list)   # HRZE taken
    second_line_drugs:  List[str] = field(default_factory=list)   # Fluoroquinolones, injectables
    isoniazid_months:   int   = 0
    rifampicin_months:  int   = 0
    fluoroquinolone_months: int = 0
    injectable_months:  int   = 0
    drug_adherence:     str   = "Unknown"        # Good (>90%) / Moderate (70-90%) / Poor (<70%) / Unknown
    adverse_reactions:  List[str] = field(default_factory=list)

    # -- Section 6: Comorbidities
    diabetes:           bool  = False
    diabetes_duration_years: int = 0
    diabetes_controlled: str  = "Unknown"
    hiv_on_art:         bool  = False
    art_regimen:        str   = ""
    chronic_kidney:     bool  = False
    liver_disease:      bool  = False
    malnutrition:       bool  = False
    alcoholism:         bool  = False
    alcoholism_units_week: int = 0
    smoking:            bool  = False
    smoking_pack_years: float = 0.0
    immunosuppressed:   bool  = False
    immunosuppression_reason: str = "None"

    # -- Section 7: Social & Epidemiological
    tb_contact:         bool  = False
    contact_type:       str   = "None"           # None / Household / Workplace / Healthcare
    contact_dr_status:  str   = "Unknown"        # DS-TB / MDR-TB / XDR-TB / Unknown
    healthcare_worker:  bool  = False
    prison_history:     bool  = False
    migrant:            bool  = False
    migrant_from:       str   = ""
    homeless:           bool  = False
    crowded_living:     bool  = False
    household_size:     int   = 1
    income_level:       str   = "Unknown"        # Below poverty / Low / Middle / Unknown

    # -- Clinical notes
    clinical_notes:     str   = ""

    @property
    def bmi(self) -> float:
        if self.height_cm > 0 and self.weight_kg > 0:
            return self.weight_kg / ((self.height_cm / 100) ** 2)
        return 0.0

    def to_dict(self):
        d = asdict(self)
        d["bmi"] = round(self.bmi, 1)
        return d

    def encode_for_model(self) -> np.ndarray:
        """
        Encode patient data into a numeric vector for DRFusionNet.
        Returns a 32-dim float32 vector (expanded from original 16-dim).
        """
        def safe(v, lo=0, hi=1):
            return float(np.clip(v, lo, hi))

        # Previous TB / treatment features
        prev_tb_enc = {"Never": 0, "Yes-completed": 0.5,
                       "Yes-incomplete": 1.0, "Unknown": 0}.get(self.previous_tb, 0)
        failed_enc  = 1.0 if self.previously_failed else 0.0
        treatment_enc = {"None": 0, "DOTS": 0.3, "Private": 0.5,
                         "Self-medicated": 0.8, "Unknown": 0}.get(self.previous_treatment, 0)

        # Drug exposure
        fl_any = 1.0 if self.first_line_drugs else 0.0
        sl_any = 1.0 if self.second_line_drugs else 0.0
        inh_dur  = safe(self.isoniazid_months / 24)
        rif_dur  = safe(self.rifampicin_months / 24)
        fq_dur   = safe(self.fluoroquinolone_months / 24)
        inj_dur  = safe(self.injectable_months / 24)

        # Lab results
        smear_enc = {"Not done": 0, "Negative": 0, "1+": 0.33,
                     "2+": 0.67, "3+": 1.0}.get(self.sputum_smear, 0)
        xpert_enc = {"Not done": 0, "Not detected": 0, "Trace": 0.3,
                     "Detected": 1.0}.get(self.xpert_mtb, 0)
        rif_res   = 1.0 if self.xpert_rif == "Resistant" else 0.0
        dst_enc   = {"Not done": 0, "DS-TB": 0, "MDR-TB": 0.7,
                     "XDR-TB": 1.0, "Pre-XDR": 0.85}.get(self.dst_result, 0)

        # Symptom severity
        cough_sev = safe(self.cough_severity / 10)
        fever_sev = safe(self.fever_severity / 10)
        wt_loss_n = safe(self.weight_loss_kg / 20)
        breath_enc = {"None": 0, "On exertion": 0.33,
                      "At rest": 0.67, "At rest with orthopnoea": 1.0}.get(
                          self.breathlessness_grade, 0)

        # Risk factors
        hiv_enc = {"Positive": 1, "Negative": 0, "Unknown": 0.3}.get(self.hiv_status, 0)
        diab_enc = safe(1.0 if self.diabetes else 0)
        adherence_enc = {"Good (>90%)": 0, "Moderate (70-90%)": 0.5,
                         "Poor (<70%)": 1.0, "Unknown": 0.5}.get(self.drug_adherence, 0.5)
        contact_dr_enc = {"None": 0, "DS-TB": 0, "MDR-TB": 0.7,
                          "XDR-TB": 1.0, "Unknown": 0.3}.get(self.contact_dr_status, 0)

        # Social
        crowded_enc = safe(min(self.household_size / 8, 1.0))
        alc_enc     = safe(1.0 if self.alcoholism else 0)
        smoke_enc   = safe(min(self.smoking_pack_years / 30, 1.0))
        prison_enc  = safe(1.0 if self.prison_history else 0)

        # Demographics
        age_norm    = safe(self.age / 80)
        bmi_norm    = safe((self.bmi - 14) / 26)   # BMI range 14-40

        return np.array([
            age_norm,       bmi_norm,       prev_tb_enc,    failed_enc,
            treatment_enc,  fl_any,         sl_any,         inh_dur,
            rif_dur,        fq_dur,         inj_dur,        smear_enc,
            xpert_enc,      rif_res,        dst_enc,        cough_sev,
            fever_sev,      wt_loss_n,      breath_enc,     hiv_enc,
            diab_enc,       adherence_enc,  contact_dr_enc, crowded_enc,
            alc_enc,        smoke_enc,      prison_enc,     float(self.previous_tb_count / 5),
            float(self.cough_weeks / 52),   float(self.fever_weeks / 52),
            float(self.night_sweats_severity / 10),
            float(self.fatigue_severity / 10),
        ], dtype=np.float32)


# ── THE FULL STREAMLIT FORM ───────────────────────────────────
def drug_resistance_form() -> DetailedPatient:
    """
    Full drug resistance assessment form for the Streamlit sidebar.

    Usage in app.py:
        from drug_resistance_form import drug_resistance_form
        patient = drug_resistance_form()
    """

    st.sidebar.markdown("## 🫁 TB Patient Assessment")
    st.sidebar.markdown("---")

    # ─────────────────────────────────────────────────────────
    # SECTION 1 — Demographics
    # ─────────────────────────────────────────────────────────
    with st.sidebar.expander("👤 Section 1: Demographics", expanded=True):
        pid     = st.text_input("Patient ID *", placeholder="KA-2025-001")
        col1, col2 = st.columns(2)
        age     = col1.number_input("Age (years)", 0, 120, 35)
        gender  = col2.selectbox("Gender", ["Male", "Female", "Other", "Unknown"])
        col3, col4 = st.columns(2)
        weight  = col3.number_input("Weight (kg)", 20.0, 200.0, 55.0, step=0.5)
        height  = col4.number_input("Height (cm)", 100.0, 220.0, 162.0, step=0.5)

        # Auto BMI
        bmi_val = weight / ((height / 100) ** 2) if height > 0 else 0
        st.caption(f"BMI: {bmi_val:.1f} — "
                   f"{'Severely underweight' if bmi_val < 16 else 'Underweight' if bmi_val < 18.5 else 'Normal' if bmi_val < 25 else 'Overweight'}")

        district = st.selectbox("District", [
            "Select district...",
            "Bagalkot", "Ballari", "Belagavi", "Bengaluru Urban", "Bengaluru Rural",
            "Bidar", "Chamarajanagar", "Chikballapur", "Chikkamagaluru", "Chitradurga",
            "Dakshina Kannada", "Davanagere", "Dharwad", "Gadag", "Hassan",
            "Haveri", "Kalaburagi", "Kodagu", "Kolar", "Koppal",
            "Mandya", "Mysuru", "Raichur", "Ramanagara", "Shivamogga",
            "Tumakuru", "Udupi", "Uttara Kannada", "Vijayapura", "Yadgir", "Other"
        ])
        occupation = st.selectbox("Occupation", [
            "Agricultural worker", "Construction worker", "Healthcare worker",
            "Domestic worker", "Factory/Industrial", "Mining", "Student",
            "Unemployed", "Self-employed", "Other"
        ])

    # ─────────────────────────────────────────────────────────
    # SECTION 2 — TB History
    # ─────────────────────────────────────────────────────────
    with st.sidebar.expander("📋 Section 2: TB History & Treatment"):
        prev_tb = st.selectbox("Previous TB diagnosis?", [
            "Never", "Yes — treatment completed", "Yes — treatment incomplete",
            "Yes — treatment ongoing", "Unknown"
        ])
        prev_tb_map = {
            "Never": "Never",
            "Yes — treatment completed": "Yes-completed",
            "Yes — treatment incomplete": "Yes-incomplete",
            "Yes — treatment ongoing": "Yes-incomplete",
            "Unknown": "Unknown"
        }
        prev_tb_enc = prev_tb_map[prev_tb]

        prev_tb_count, prev_tb_years, treatment_type = 0, 0, "None"
        treatment_duration, treatment_completed = 0, "N/A"
        reason_stopped, previously_failed = "N/A", False

        if prev_tb_enc != "Never":
            prev_tb_count = st.number_input(
                "How many previous TB episodes?", 0, 10, 1)
            prev_tb_years = st.number_input(
                "Year of most recent TB episode", 2000, 2025, 2023)
            treatment_type = st.selectbox("Type of treatment received", [
                "DOTS (Government / RNTCP)",
                "Private practitioner",
                "Self-medicated (bought drugs directly)",
                "Traditional/herbal medicine",
                "Combination of above",
                "Unknown"
            ])
            treatment_duration = st.number_input(
                "Total duration of treatment received (months)", 0, 48, 6)
            treatment_completed = st.selectbox("Was treatment completed?", [
                "Yes — completed full course",
                "No — stopped before completing",
                "No — changed regimen mid-course",
                "Ongoing",
                "Unknown"
            ])
            if "stopped" in treatment_completed or "changed" in treatment_completed:
                reason_stopped = st.selectbox("Primary reason for stopping/changing", [
                    "Side effects / adverse reactions",
                    "Felt better, thought cured",
                    "Could not afford drugs",
                    "Drug unavailability",
                    "Lost to follow up",
                    "Transferred out",
                    "Provider decision",
                    "Unknown"
                ])
            previously_failed = st.checkbox(
                "Declared treatment failure by provider?",
                help="Treatment failure = positive sputum at 5 months or later during treatment")

    # ─────────────────────────────────────────────────────────
    # SECTION 3 — Current Symptoms
    # ─────────────────────────────────────────────────────────
    with st.sidebar.expander("🤒 Section 3: Current Symptoms"):
        st.markdown("**Cough**")
        cough_present = st.checkbox("Cough present")
        cough_weeks, cough_severity, cough_char, sputum_volume = 0, 0, "Dry", "None"
        haemoptysis, haemo_amount = False, "None"

        if cough_present:
            cough_weeks    = st.number_input("Cough duration (weeks)", 0, 260, 3)
            cough_severity = st.slider(
                "Cough severity (1 = mild, 10 = disabling)", 1, 10, 3)
            cough_char = st.selectbox("Character of cough", [
                "Dry / non-productive",
                "Productive (clear/white sputum)",
                "Productive (yellow/green sputum)",
                "Productive (blood-tinged sputum)"
            ])
            sputum_volume = st.selectbox("Sputum volume per day", [
                "None", "Scant (<5 ml)", "Moderate (5–30 ml)", "Copious (>30 ml)"
            ])
            haemoptysis = st.checkbox("Haemoptysis (coughing blood)?")
            if haemoptysis:
                haemo_amount = st.selectbox("Amount of blood", [
                    "Streaks in sputum", "Teaspoon (<5ml)", "Tablespoon (5–30ml)",
                    "Large (>30ml)", "Massive haemoptysis"
                ])

        st.markdown("**Fever & Sweats**")
        fever = st.checkbox("Fever")
        fever_weeks, fever_severity, fever_pattern, night_sweats, sweats_severity = 0, 0, "Unknown", False, 0

        if fever:
            fever_weeks    = st.number_input("Fever duration (weeks)", 0, 104, 2)
            fever_severity = st.slider(
                "Fever severity (1 = low-grade, 10 = high/rigors)", 1, 10, 4)
            fever_pattern  = st.selectbox("Fever pattern", [
                "Continuous (present all day)",
                "Intermittent (comes and goes)",
                "Evening rise (higher in evenings)",
                "Unknown"
            ])

        night_sweats = st.checkbox("Night sweats")
        if night_sweats:
            sweats_severity = st.slider(
                "Night sweats severity (1 = mild dampness, 10 = drenching)", 1, 10, 5)

        st.markdown("**Weight & Appetite**")
        weight_loss = st.checkbox("Significant weight loss")
        wt_loss_kg, wt_loss_weeks = 0.0, 0
        if weight_loss:
            wt_loss_kg    = st.number_input(
                "Weight lost (kg)", 0.0, 50.0, 5.0, step=0.5)
            wt_loss_weeks = st.number_input(
                "Over how many weeks?", 1, 104, 8)

        fatigue_severity = st.slider(
            "Fatigue severity (0 = none, 10 = cannot work)", 0, 10, 3)

        st.markdown("**Respiratory**")
        breathlessness = st.checkbox("Breathlessness / shortness of breath")
        breath_grade   = "None"
        if breathlessness:
            breath_grade = st.selectbox("Breathlessness grade", [
                "Only on significant exertion (climbing stairs, running)",
                "On moderate exertion (walking on flat ground)",
                "At rest (present even lying down)",
                "Orthopnoea (worse lying flat, needs pillows)"
            ])

        chest_pain = st.checkbox("Chest pain")
        chest_pain_type = "None"
        if chest_pain:
            chest_pain_type = st.selectbox("Type of chest pain", [
                "Pleuritic (sharp, worse on breathing/coughing)",
                "Dull central ache",
                "Sharp stabbing",
                "Tight/constricting"
            ])

    # ─────────────────────────────────────────────────────────
    # SECTION 4 — Lab Results
    # ─────────────────────────────────────────────────────────
    with st.sidebar.expander("🔬 Section 4: Lab & Sputum Results"):
        sputum_smear = st.selectbox("Sputum smear microscopy (AFB)", [
            "Not done", "Negative (0 AFB)", "Scanty (1–9/100 fields)",
            "1+ (10–99/100 fields)", "2+ (1–10/field)", "3+ (>10/field)"
        ])
        sputum_culture = st.selectbox("Sputum culture (Mycobacterium TB)", [
            "Not done", "Negative", "Positive — culture confirmed",
            "Contaminated", "Pending"
        ])

        st.markdown("**Molecular Tests**")
        xpert_mtb = st.selectbox("GeneXpert MTB/RIF — MTB result", [
            "Not done", "MTB NOT detected", "MTB Detected (High)",
            "MTB Detected (Medium)", "MTB Detected (Low)", "MTB Detected (Trace)"
        ])
        xpert_rif = st.selectbox("GeneXpert MTB/RIF — RIF resistance", [
            "Not done", "Rifampicin SENSITIVE", "Rifampicin RESISTANT",
            "Rifampicin INDETERMINATE"
        ])
        lpa = st.selectbox("Line Probe Assay (LPA) result", [
            "Not done", "Sensitive to all tested",
            "INH resistant only (low-level rpoB mutation)",
            "RIF resistant only",
            "Both INH + RIF resistant (MDR)",
            "Fluoroquinolone resistant",
            "Injectable resistant",
            "MDR + Fluoroquinolone resistant (pre-XDR)"
        ])

        st.markdown("**Drug Susceptibility Testing (DST)**")
        dst_done = st.checkbox("DST performed?")
        dst_result, dst_drugs = "Not done", []
        if dst_done:
            dst_result = st.selectbox("DST final classification", [
                "Drug-Sensitive TB (DS-TB)",
                "Mono-resistant TB",
                "Poly-resistant TB",
                "Multi-Drug Resistant TB (MDR-TB)",
                "Pre-Extensively Drug Resistant (pre-XDR)",
                "Extensively Drug Resistant TB (XDR-TB)",
                "Pending"
            ])
            dst_drugs = st.multiselect("Drugs confirmed resistant", [
                "Isoniazid (H)", "Rifampicin (R)", "Ethambutol (E)",
                "Pyrazinamide (Z)", "Streptomycin (S)",
                "Levofloxacin", "Moxifloxacin", "Bedaquiline",
                "Linezolid", "Clofazimine", "Amikacin", "Kanamycin", "Capreomycin"
            ])

        st.markdown("**Blood Tests**")
        hiv_status = st.selectbox("HIV status", ["Unknown", "Negative", "Positive"])
        cd4_count, viral_load, hba1c, creatinine = 0, "Unknown", 0.0, 0.0

        if hiv_status == "Positive":
            cd4_count   = st.number_input("CD4 count (cells/μL)", 0, 2000, 350)
            viral_load  = st.selectbox("Viral load", [
                "Undetectable (<50 copies/mL)",
                "Low (50–1000 copies/mL)",
                "Detectable (>1000 copies/mL)",
                "Not tested"
            ])

        hba1c      = st.number_input("HbA1c % (if diabetic, else 0)", 0.0, 20.0, 0.0,
                                     step=0.1)
        creatinine = st.number_input("Serum creatinine mg/dL (0 if not done)",
                                     0.0, 15.0, 0.0, step=0.1)

    # ─────────────────────────────────────────────────────────
    # SECTION 5 — Drug Exposure
    # ─────────────────────────────────────────────────────────
    with st.sidebar.expander("💊 Section 5: Drug Exposure History"):
        fl_drugs = st.multiselect(
            "First-line drugs previously taken",
            ["Isoniazid (H)", "Rifampicin (R)", "Ethambutol (E)",
             "Pyrazinamide (Z)", "Streptomycin (S)"],
            help="Select all drugs the patient has been exposed to"
        )
        sl_drugs = st.multiselect(
            "Second-line drugs previously taken",
            ["Levofloxacin", "Moxifloxacin", "Ofloxacin", "Amikacin",
             "Kanamycin", "Capreomycin", "Bedaquiline", "Linezolid",
             "Clofazimine", "Delamanid", "Pretomanid"],
            help="Any second-line drugs — even if taken briefly"
        )

        if fl_drugs:
            inh_months = st.slider(
                "Isoniazid — months of exposure", 0, 36, 0) if "Isoniazid (H)" in fl_drugs else 0
            rif_months = st.slider(
                "Rifampicin — months of exposure", 0, 36, 0) if "Rifampicin (R)" in fl_drugs else 0
        else:
            inh_months, rif_months = 0, 0

        if sl_drugs:
            fq_any = any(d in sl_drugs for d in ["Levofloxacin","Moxifloxacin","Ofloxacin"])
            inj_any = any(d in sl_drugs for d in ["Amikacin","Kanamycin","Capreomycin"])
            fq_months  = st.slider("Fluoroquinolone — total months", 0, 36, 0) if fq_any else 0
            inj_months = st.slider("Injectable — total months", 0, 24, 0) if inj_any else 0
        else:
            fq_months, inj_months = 0, 0

        drug_adherence = st.selectbox("Self-reported drug adherence", [
            "Good — took >90% of doses",
            "Moderate — took 70–90% of doses",
            "Poor — took <70% of doses",
            "Very poor — frequently missed",
            "Unknown / difficult to assess"
        ])
        adherence_map = {
            "Good — took >90% of doses": "Good (>90%)",
            "Moderate — took 70–90% of doses": "Moderate (70-90%)",
            "Poor — took <70% of doses": "Poor (<70%)",
            "Very poor — frequently missed": "Poor (<70%)",
            "Unknown / difficult to assess": "Unknown"
        }

        adverse_reactions = st.multiselect(
            "Adverse drug reactions experienced",
            ["Nausea/vomiting", "Hepatotoxicity (jaundice)", "Peripheral neuropathy",
             "Visual disturbance (Ethambutol)", "Hearing loss (Aminoglycosides)",
             "Rash/allergic reaction", "Arthralgia/joint pain", "Nephrotoxicity",
             "Psychiatric effects (Cycloserine)", "QT prolongation", "Hypothyroidism",
             "Anaemia", "Other"]
        )

    # ─────────────────────────────────────────────────────────
    # SECTION 6 — Comorbidities
    # ─────────────────────────────────────────────────────────
    with st.sidebar.expander("🩺 Section 6: Comorbidities"):
        diabetes = st.checkbox("Diabetes mellitus")
        diab_years, diab_controlled = 0, "Unknown"
        if diabetes:
            diab_years     = st.number_input("Diabetes duration (years)", 0, 60, 5)
            diab_controlled = st.selectbox("Diabetes control", [
                "Well controlled (HbA1c <7%)",
                "Moderately controlled (HbA1c 7–9%)",
                "Poorly controlled (HbA1c >9%)",
                "Unknown"
            ])

        hiv_on_art = st.checkbox("HIV — currently on ART?")
        art_regimen = ""
        if hiv_on_art:
            art_regimen = st.selectbox("ART regimen", [
                "TDF + 3TC + DTG (standard)", "TDF + 3TC + EFV",
                "AZT + 3TC + DTG", "Other first-line", "Second-line ART",
                "Unknown"
            ])

        chronic_kidney = st.checkbox("Chronic kidney disease")
        liver_disease  = st.checkbox("Liver disease (cirrhosis, hepatitis B/C)")
        malnutrition   = st.checkbox("Severe malnutrition (BMI < 16 or MUAC < 18.5cm)")

        alcohol = st.checkbox("Active alcoholism / heavy alcohol use")
        alc_units = 0
        if alcohol:
            alc_units = st.number_input(
                "Estimated units per week (1 unit = 10ml ethanol)", 0, 200, 20)

        smoking = st.checkbox("Current/former smoker")
        pack_years = 0.0
        if smoking:
            pack_years = st.number_input(
                "Pack-years (packs/day × years smoked)", 0.0, 100.0, 10.0, step=0.5)

        immunosuppressed = st.checkbox(
            "Immunosuppressed (other than HIV)")
        immuno_reason = "None"
        if immunosuppressed:
            immuno_reason = st.selectbox("Reason for immunosuppression", [
                "Corticosteroids (>15mg/day prednisolone)",
                "Chemotherapy / cancer",
                "Organ transplant (immunosuppressants)",
                "Biological therapy (TNF-alpha inhibitors)",
                "Haematological malignancy",
                "Other"
            ])

    # ─────────────────────────────────────────────────────────
    # SECTION 7 — Epidemiological
    # ─────────────────────────────────────────────────────────
    with st.sidebar.expander("🌍 Section 7: Social & Epidemiological Factors"):
        tb_contact = st.checkbox("Known TB contact in last 2 years")
        contact_type, contact_dr = "None", "Unknown"
        if tb_contact:
            contact_type = st.selectbox("Type of contact", [
                "Household member (lives together)",
                "Close workplace contact",
                "Healthcare setting exposure",
                "Social contact (friend/relative not living together)",
                "Unknown type"
            ])
            contact_dr = st.selectbox("Drug resistance status of contact", [
                "Unknown", "Drug-sensitive TB (DS-TB)",
                "MDR-TB confirmed", "XDR-TB confirmed"
            ])

        hcw          = st.checkbox("Healthcare worker?")
        prison_hist  = st.checkbox("History of incarceration?")
        migrant      = st.checkbox("Internal/international migrant?")
        migrant_from = ""
        if migrant:
            migrant_from = st.text_input("Migrated from (state/country)")

        homeless     = st.checkbox("Homeless / no fixed address?")
        crowded      = st.checkbox("Living in crowded conditions?")
        household_sz = st.number_input("Household size (number of people)", 1, 30, 4)
        income       = st.selectbox("Household income level", [
            "Below poverty line (BPL card holder)",
            "Low income (just above BPL)",
            "Lower middle income",
            "Middle income / above",
            "Unknown / prefer not to say"
        ])
        income_map = {
            "Below poverty line (BPL card holder)":  "Below poverty",
            "Low income (just above BPL)":           "Low",
            "Lower middle income":                   "Low",
            "Middle income / above":                 "Middle",
            "Unknown / prefer not to say":           "Unknown"
        }

    # ─────────────────────────────────────────────────────────
    # SECTION 8 — Clinical Notes
    # ─────────────────────────────────────────────────────────
    with st.sidebar.expander("📝 Clinical Notes"):
        notes = st.text_area(
            "Free text observations",
            placeholder="e.g. Patient reports irregular pill-taking, lives 40km from PHC...",
            height=100
        )

    # ─────────────────────────────────────────────────────────
    # BUILD AND RETURN PATIENT OBJECT
    # ─────────────────────────────────────────────────────────
    return DetailedPatient(
        patient_id=pid, age=int(age), gender=gender,
        weight_kg=float(weight), height_cm=float(height),
        district=district, occupation=occupation,
        previous_tb=prev_tb_enc, previous_tb_years=int(prev_tb_years),
        previous_tb_count=int(prev_tb_count), previous_treatment=treatment_type,
        treatment_duration_months=int(treatment_duration),
        treatment_completed=treatment_completed,
        reason_stopped=reason_stopped, previously_failed=previously_failed,
        cough_present=cough_present, cough_weeks=int(cough_weeks),
        cough_severity=int(cough_severity), cough_character=cough_char,
        sputum_volume_ml=sputum_volume, haemoptysis=haemoptysis,
        haemoptysis_amount=haemo_amount,
        fever=fever, fever_weeks=int(fever_weeks), fever_severity=int(fever_severity),
        fever_pattern=fever_pattern, night_sweats=night_sweats,
        night_sweats_severity=int(sweats_severity),
        weight_loss=weight_loss, weight_loss_kg=float(wt_loss_kg),
        weight_loss_weeks=int(wt_loss_weeks), fatigue_severity=int(fatigue_severity),
        breathlessness=breathlessness, breathlessness_grade=breath_grade,
        chest_pain=chest_pain, chest_pain_type=chest_pain_type,
        sputum_smear=sputum_smear, sputum_culture=sputum_culture,
        xpert_mtb=xpert_mtb, xpert_rif=xpert_rif, dst_performed=dst_done,
        dst_result=dst_result, dst_drugs_resistant=dst_drugs, line_probe_assay=lpa,
        hiv_status=hiv_status, cd4_count=int(cd4_count), viral_load=viral_load,
        hba1c=float(hba1c), serum_creatinine=float(creatinine),
        first_line_drugs=fl_drugs, second_line_drugs=sl_drugs,
        isoniazid_months=int(inh_months), rifampicin_months=int(rif_months),
        fluoroquinolone_months=int(fq_months), injectable_months=int(inj_months),
        drug_adherence=adherence_map.get(drug_adherence, "Unknown"),
        adverse_reactions=adverse_reactions,
        diabetes=diabetes, diabetes_duration_years=int(diab_years),
        diabetes_controlled=diab_controlled, hiv_on_art=hiv_on_art,
        art_regimen=art_regimen, chronic_kidney=chronic_kidney,
        liver_disease=liver_disease, malnutrition=malnutrition,
        alcoholism=alcohol, alcoholism_units_week=int(alc_units),
        smoking=smoking, smoking_pack_years=float(pack_years),
        immunosuppressed=immunosuppressed, immunosuppression_reason=immuno_reason,
        tb_contact=tb_contact, contact_type=contact_type,
        contact_dr_status=contact_dr,
        healthcare_worker=hcw, prison_history=prison_hist,
        migrant=migrant, migrant_from=migrant_from, homeless=homeless,
        crowded_living=crowded, household_size=int(household_sz),
        income_level=income_map.get(income, "Unknown"),
        clinical_notes=notes,
    )