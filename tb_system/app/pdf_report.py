"""
app/pdf_report.py
Clean PDF report - fixed probability bars, proper heatmap rendering.
"""
from __future__ import annotations
import io, datetime
import numpy as np
import cv2
import PIL.Image

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether,
)

# ── Colours ────────────────────────────────────────────────────
C_BLUE   = colors.HexColor("#1E4080")
C_LBLUE  = colors.HexColor("#D6E4F7")
C_GREEN  = colors.HexColor("#1A5C2E")
C_LGREEN = colors.HexColor("#D6F0DF")
C_YELLOW = colors.HexColor("#B8860B")
C_LYELLO = colors.HexColor("#FFF8DC")
C_ORANGE = colors.HexColor("#C25A00")
C_LORNG  = colors.HexColor("#FFF0DC")
C_RED    = colors.HexColor("#8B0000")
C_LRED   = colors.HexColor("#FFE4E4")
C_GREY   = colors.HexColor("#555555")
C_LGREY  = colors.HexColor("#F5F5F5")
C_DARK   = colors.HexColor("#1A1A2E")
C_BORDER = colors.HexColor("#CCCCCC")
C_TEXT   = colors.HexColor("#222222")
C_WHITE  = colors.white

RISK_COLORS = {
    "MINIMAL":   C_GREEN,
    "LOW":       C_BLUE,
    "MODERATE":  C_YELLOW,
    "HIGH":      C_ORANGE,
    "VERY HIGH": C_RED,
}
DR_COLORS = {
    "Drug-Sensitive": C_GREEN,
    "MDR-TB":         C_YELLOW,
    "XDR-TB":         C_RED,
}

W = 9360  # content width in DXA (letter, 1" margins)
from reportlab.lib.units import inch

def S(name, fontSize=9, fontName="Helvetica", **kwargs):
    return ParagraphStyle(name, fontName=fontName, fontSize=fontSize, **kwargs)

def bdr(c=C_BORDER):
    return {"style": "SINGLE", "width": 0.5, "color": c}

def cell_border():
    from reportlab.platypus.tables import TableStyle
    b = colors.HexColor("#CCCCCC")
    return [
        ("BOX",      (0,0),(-1,-1), 0.5, b),
        ("INNERGRID",(0,0),(-1,-1), 0.5, b),
    ]

# ── Image helpers ──────────────────────────────────────────────
def _img_bytes(arr: np.ndarray) -> io.BytesIO:
    buf = io.BytesIO()
    PIL.Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    buf.seek(0)
    return buf

def _make_panels(heatmap: np.ndarray, orig: np.ndarray):
    H, W_img = orig.shape[:2]
    hmap_r   = cv2.resize(heatmap, (W_img, H))

    # Heatmap panel — JET colormap on white background
    colored  = cv2.applyColorMap((hmap_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored  = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # Overlay panel
    overlay  = cv2.addWeighted(orig, 0.55, colored, 0.45, 0)

    return orig, colored, overlay


# ── Simple text probability bar using table ────────────────────
def _prob_row(label: str, pct: float, fill_color, row_bg=C_WHITE):
    """Returns a table row showing label | bar | pct%"""
    bar_total = 200  # total bar width in points
    filled    = max(int(bar_total * pct), 2)
    empty     = bar_total - filled

    # Bar as nested table
    bar_table = Table(
        [[Paragraph("", S("x")), Paragraph("", S("x"))]],
        colWidths=[filled, empty],
        rowHeights=[8],
    )
    bar_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(0,0), fill_color),
        ("BACKGROUND", (1,0),(1,0), colors.HexColor("#E0E0E0")),
        ("TOPPADDING",    (0,0),(-1,-1), 0),
        ("BOTTOMPADDING", (0,0),(-1,-1), 0),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 0),
    ]))

    return [
        Paragraph(label, ParagraphStyle("pl", fontName="Helvetica",
                  fontSize=8, textColor=C_GREY)),
        bar_table,
        Paragraph(f"<b>{pct:.1%}</b>", ParagraphStyle("pv", fontName="Helvetica-Bold",
                  fontSize=8, textColor=fill_color, alignment=TA_RIGHT)),
    ]


# ══════════════════════════════════════════════════════════════
#  MAIN GENERATOR
# ══════════════════════════════════════════════════════════════
def generate_pdf(patient, xray_result: dict, dr_result, orig_image: np.ndarray) -> bytes:

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm,  bottomMargin=2*cm)
    W_body = A4[0] - 4*cm
    story  = []
    ts     = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")

    tb_color  = C_RED   if xray_result["tb_detected"] else C_GREEN
    tb_label  = "TB DETECTED" if xray_result["tb_detected"] else "NOT DETECTED"
    dr_color  = DR_COLORS.get(dr_result.prediction, C_GREY)
    rc_color  = RISK_COLORS.get(dr_result.risk_band, C_GREY)

    # ── HEADER ────────────────────────────────────────────────
    hdr = Table([[
        Paragraph("<b>🫁 TB DETECTION SYSTEM</b>",
                  ParagraphStyle("ht", fontName="Helvetica-Bold", fontSize=16,
                                 textColor=C_WHITE, leading=20)),
        Paragraph(
            f"<b>Patient:</b> {patient.patient_id}  "
            f"<b>Age:</b> {patient.age}  <b>Gender:</b> {patient.gender}<br/>"
            f"<b>Generated:</b> {ts}",
            ParagraphStyle("hi", fontName="Helvetica", fontSize=8,
                           textColor=colors.HexColor("#AAAAAA"),
                           leading=13, alignment=TA_RIGHT)),
    ]], colWidths=[W_body*0.55, W_body*0.45])
    hdr.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), C_DARK),
        ("ROWPADDING", (0,0),(-1,-1), 14),
        ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
        ("BOX",        (0,0),(-1,-1), 1, C_BLUE),
    ]))
    story += [hdr, Spacer(1, 0.3*cm)]

    # ── 4 KEY METRICS ─────────────────────────────────────────
    def metric_cell(label, value, color):
        return [
            Paragraph(label, ParagraphStyle("ml", fontName="Helvetica", fontSize=7,
                      textColor=C_GREY, alignment=TA_CENTER)),
            Paragraph(f"<b>{value}</b>",
                      ParagraphStyle("mv", fontName="Helvetica-Bold", fontSize=13,
                                     textColor=color, alignment=TA_CENTER, leading=18)),
        ]

    metrics = Table([[
        metric_cell("TB FINDING",      tb_label,                    tb_color),
        metric_cell("TB PROBABILITY",  f"{xray_result['tb_prob']:.1%}", tb_color),
        metric_cell("DRUG RESISTANCE", dr_result.prediction,        dr_color),
        metric_cell("CLINICAL RISK",   dr_result.risk_band,         rc_color),
    ]], colWidths=[W_body/4]*4)
    metrics.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), C_LGREY),
        ("BOX",        (0,0),(-1,-1), 0.5, C_BORDER),
        ("INNERGRID",  (0,0),(-1,-1), 0.5, C_BORDER),
        ("ROWPADDING", (0,0),(-1,-1), 10),
        ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
    ]))
    story += [metrics, Spacer(1, 0.4*cm)]

    # ── X-RAY PANELS ──────────────────────────────────────────
    story.append(Paragraph("<b>X-Ray Analysis</b>",
                           S("h2", fontSize=11, textColor=C_BLUE, spaceBefore=4, spaceAfter=6)))

    orig_panel, heat_panel, overlay_panel = _make_panels(xray_result["heatmap"], orig_image)
    img_w = (W_body - 0.6*cm) / 3
    img_h = img_w * 0.85

    panels = []
    for arr, cap in [(orig_panel,"Original X-Ray"),
                     (heat_panel,"GradCAM++ Heatmap"),
                     (overlay_panel,"Attention Overlay")]:
        ri = RLImage(_img_bytes(arr), width=img_w, height=img_h)
        cp = Paragraph(cap, ParagraphStyle("cap", fontName="Helvetica", fontSize=7,
                                           textColor=C_GREY, alignment=TA_CENTER))
        panels.append((ri, cp))

    img_tbl = Table(
        [[p[0] for p in panels],
         [p[1] for p in panels]],
        colWidths=[img_w]*3
    )
    img_tbl.setStyle(TableStyle([
        ("BOX",        (0,0),(-1,-1), 0.5, C_BORDER),
        ("INNERGRID",  (0,0),(-1,-1), 0.5, C_BORDER),
        ("ROWPADDING", (0,0),(-1,-1), 6),
        ("ALIGN",      (0,0),(-1,-1), "CENTER"),
        ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
        ("BACKGROUND", (0,0),(-1,-1), C_LGREY),
    ]))
    story.append(img_tbl)
    story.append(Paragraph(f"Finding: {xray_result.get('finding','N/A')}",
                           S("sm", fontSize=8, textColor=C_GREY, spaceBefore=4)))
    story.append(Spacer(1, 0.4*cm))

    # ── PROBABILITY BARS ──────────────────────────────────────
    story.append(Paragraph("<b>Detection Probabilities</b>",
                           S("h2", fontSize=11, textColor=C_BLUE, spaceAfter=6)))

    prob_data = [
        _prob_row("TB Positive",    xray_result["tb_prob"],    C_RED),
        _prob_row("Normal",         xray_result["normal_prob"],C_GREEN),
        _prob_row("Drug-Sensitive", dr_result.probabilities.get("Drug-Sensitive",0), C_GREEN),
        _prob_row("MDR-TB",         dr_result.probabilities.get("MDR-TB",0),         C_YELLOW),
        _prob_row("XDR-TB",         dr_result.probabilities.get("XDR-TB",0),         C_RED),
    ]
    prob_tbl = Table(prob_data, colWidths=[3.5*cm, W_body-6.5*cm, 2.5*cm])
    prob_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_LGREY),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [C_WHITE, C_LGREY]),
        ("BOX",           (0,0),(-1,-1), 0.5, C_BORDER),
        ("INNERGRID",     (0,0),(-1,-1), 0.5, C_BORDER),
        ("ROWPADDING",    (0,0),(-1,-1), 7),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    story += [prob_tbl, Spacer(1, 0.4*cm)]

    # ── CLINICAL RISK ─────────────────────────────────────────
    story.append(Paragraph("<b>Clinical Risk Assessment</b>",
                           S("h2", fontSize=11, textColor=C_BLUE, spaceAfter=6)))
    factors_str = ", ".join(dr_result.risk_factors) if dr_result.risk_factors else "None identified"
    risk_tbl = Table([
        [Paragraph("<b>Risk Band</b>", S("rl", textColor=C_GREY, fontSize=8)),
         Paragraph(f"<b>{dr_result.risk_band}</b>",
                   ParagraphStyle("rv", fontName="Helvetica-Bold",
                                  fontSize=13, textColor=rc_color))],
        [Paragraph("<b>Score</b>",     S("rl", textColor=C_GREY, fontSize=8)),
         Paragraph(str(dr_result.risk_score), S("rb", fontSize=9))],
        [Paragraph("<b>Factors</b>",   S("rl", textColor=C_GREY, fontSize=8)),
         Paragraph(factors_str, S("rf", fontSize=8.5))],
    ], colWidths=[3.5*cm, W_body-3.5*cm])
    risk_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), C_LGREY),
        ("BOX",        (0,0),(-1,-1), 0.5, C_BORDER),
        ("INNERGRID",  (0,0),(-1,-1), 0.5, C_BORDER),
        ("ROWPADDING", (0,0),(-1,-1), 8),
        ("VALIGN",     (0,0),(-1,-1), "TOP"),
    ]))
    story += [risk_tbl, Spacer(1, 0.4*cm)]

    # ── PATIENT SUMMARY ───────────────────────────────────────
    story.append(Paragraph("<b>Patient Clinical Summary</b>",
                           S("h2", fontSize=11, textColor=C_BLUE, spaceAfter=6)))

    def yn(v): return "Yes" if v==1 else "No" if v==0 else "Unknown"
    def lbl(t): return Paragraph(t, S("ml", fontName="Helvetica-Bold",
                                       fontSize=8, textColor=C_GREY))
    def val(t): return Paragraph(t, S("mv", fontSize=8.5))

    meta_rows = [
# REPLACE WITH:
        [lbl("Smoking"),           val(yn(patient.smoking)),
         lbl("Alcohol Use"),       val(yn(patient.alcoholism))],
        [lbl("Diabetic"),          val(yn(patient.diabetes)),
         lbl("HIV Status"),        val(patient.hiv_status)],
        [lbl("Immunosuppressed"),  val(yn(patient.immunosuppressed)),
         lbl("Previous TB"),       val(patient.previous_tb)],
        [lbl("TB Contact"),        val(yn(patient.tb_contact)),
         lbl("Cough Duration"),    val(f"{patient.cough_weeks} weeks" if patient.cough_weeks else "None")],
        [lbl("Fever"),             val(yn(patient.fever)),
         lbl("Night Sweats"),      val(yn(patient.night_sweats))],
        [lbl("Weight Loss"),       val(yn(patient.weight_loss)),
         lbl("Fatigue"),           val(f"Severity {patient.fatigue_severity}/10")],
        [lbl("Haemoptysis"),       val(yn(patient.haemoptysis)),
         lbl("Breathlessness"),    val(yn(patient.breathlessness))],
    ]
    cw = W_body / 4
    meta_tbl = Table(meta_rows, colWidths=[cw]*4)
    meta_tbl.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0,0),(-1,-1), [C_LGREY, C_WHITE]),
        ("BOX",            (0,0),(-1,-1), 0.5, C_BORDER),
        ("INNERGRID",      (0,0),(-1,-1), 0.5, C_BORDER),
        ("ROWPADDING",     (0,0),(-1,-1), 7),
        ("VALIGN",         (0,0),(-1,-1), "TOP"),
    ]))
    story += [meta_tbl, Spacer(1, 0.4*cm)]

    # ── WARNINGS ──────────────────────────────────────────────
    if dr_result.warnings:
        story.append(Paragraph("<b>Clinical Warnings</b>",
                               S("h2", fontSize=11, textColor=C_ORANGE, spaceAfter=6)))
        for w in dr_result.warnings:
            story.append(Paragraph(f"• {w}",
                         S("w", fontSize=8.5, textColor=C_ORANGE, leading=13,
                           leftIndent=10, spaceAfter=4)))
        story.append(Spacer(1, 0.3*cm))

    # ── RECOMMENDATIONS ───────────────────────────────────────
    story.append(Paragraph("<b>Recommendations</b>",
                           S("h2", fontSize=11, textColor=C_BLUE, spaceAfter=6)))
    for r in dr_result.recommendations:
        story.append(Paragraph(f"• {r}",
                     S("r", fontSize=8.5, textColor=C_TEXT, leading=13,
                       leftIndent=10, spaceAfter=4)))
    story.append(Spacer(1, 0.5*cm))

    # ── DISCLAIMER ────────────────────────────────────────────
    story.append(HRFlowable(width=W_body, color=C_BORDER))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "<b>DISCLAIMER:</b> This report is generated by an AI-assisted clinical decision "
        "support tool. All findings must be reviewed and confirmed by a qualified clinician. "
        "This tool is not a substitute for professional medical diagnosis or treatment. "
        "Drug resistance predictions are probabilistic and require confirmatory DST.",
        S("disc", fontSize=7, textColor=C_GREY, leading=10)
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()