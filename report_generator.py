"""
PDF Report Generator for RetinaAI
Uses reportlab to create professional diagnostic reports.
Version: 1.1.0 - Fixed ReportLab dependency
"""

import os
import uuid
import json
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_OK = True
except ImportError as e:
    REPORTLAB_OK = False
    print(f"[WARN] ReportLab import failed: {e}")
except Exception as e:
    REPORTLAB_OK = False
    print(f"[ERROR] report_generator init error: {e}")

REPORTS_DIR = "static/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

GRADE_NAMES  = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
GRADE_COLORS_HEX = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#a855f7"]

RECOMMENDATIONS = {
    0: "No diabetic retinopathy detected. Continue annual dilated eye examinations. Maintain good glycaemic control.",
    1: "Mild non-proliferative DR. Optimize blood glucose and blood pressure. Schedule follow-up in 12 months.",
    2: "Moderate non-proliferative DR. Refer to ophthalmologist within 6 months. Intensify medical management.",
    3: "Severe non-proliferative DR. Urgent ophthalmology referral within 1 month. Consider laser therapy.",
    4: "Proliferative DR. Immediate ophthalmology referral. Pan-retinal photocoagulation or anti-VEGF therapy indicated.",
}


def rl_color(hex_str):
    """Convert hex color string to reportlab Color."""
    h = hex_str.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return colors.Color(r / 255, g / 255, b / 255)


def generate_diagnostic_report(scan: dict, patient: dict, profile: dict,
                                probs: dict, risk_factors: list) -> tuple:
    """
    Generate a PDF diagnostic report for a scan.
    Returns (file_path, file_url) or (None, None) on failure.
    """
    if not REPORTLAB_OK:
        return None, None

    filename = f"report_{scan['id']}_{uuid.uuid4().hex[:8]}.pdf"
    filepath = os.path.join(REPORTS_DIR, filename)
    file_url = f"/static/reports/{filename}"

    doc = SimpleDocTemplate(
        filepath, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Header ──────────────────────────────────────────────────────────────────
    header_style = ParagraphStyle(
        "Header", parent=styles["Title"],
        fontSize=22, textColor=colors.HexColor("#6366f1"),
        spaceAfter=4, fontName="Helvetica-Bold"
    )
    sub_style = ParagraphStyle(
        "Sub", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#64748b"),
        spaceAfter=2
    )
    story.append(Paragraph("RetinaAI Diagnostic Report", header_style))
    story.append(Paragraph("AI-Assisted Diabetic Retinopathy Screening", sub_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}  |  Scan ID: #{scan['id']}",
        sub_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor("#6366f1"), spaceAfter=12))

    # ── Patient Info ─────────────────────────────────────────────────────────────
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"],
        fontSize=12, textColor=colors.HexColor("#818cf8"),
        spaceBefore=12, spaceAfter=6, fontName="Helvetica-Bold"
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#1e293b"), leading=14
    )
    story.append(Paragraph("Patient Information", section_style))

    dob  = profile.get("date_of_birth", "—") if profile else "—"
    gen  = profile.get("gender", "—") if profile else "—"
    dtyp = profile.get("diabetes_type", "—") if profile else "—"
    ddur = f"{int(profile['diabetes_duration'])} years" if profile and profile.get("diabetes_duration") else "—"

    patient_data = [
        ["Patient Name", patient.get("name", "—"),
         "Date of Birth", dob],
        ["Email", patient.get("email", "—"),
         "Gender", gen],
        ["Diabetes Type", dtyp,
         "Duration", ddur],
    ]
    pt = Table(patient_data, colWidths=[3.5*cm, 6*cm, 3.5*cm, 4.5*cm])
    pt.setStyle(TableStyle([
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",  (0, 0), (0, -1), colors.HexColor("#64748b")),
        ("TEXTCOLOR",  (2, 0), (2, -1), colors.HexColor("#64748b")),
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",   (2, 0), (2, -1), "Helvetica-Bold"),
        ("GRID",       (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [colors.HexColor("#f8fafc"), colors.HexColor("#f1f5f9")]),
        ("PADDING",    (0, 0), (-1, -1), 6),
    ]))
    story.append(pt)

    # ── AI Diagnosis ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 10))
    story.append(Paragraph("AI Diagnosis", section_style))

    grade_id   = scan.get("ai_class_id", 0) or 0
    grade_name = scan.get("ai_class_name", "Unknown")
    confidence = scan.get("ai_confidence", 0) or 0
    g_color    = rl_color(GRADE_COLORS_HEX[grade_id] if grade_id < 5 else "#818cf8")

    diag_style = ParagraphStyle(
        "Diag", parent=styles["Normal"],
        fontSize=16, textColor=g_color, fontName="Helvetica-Bold"
    )
    story.append(Paragraph(f"{grade_name}  —  Grade {grade_id}", diag_style))
    story.append(Paragraph(f"AI Confidence: {confidence:.1f}%", body_style))
    story.append(Spacer(1, 6))

    # Probability table
    prob_rows = [["Grade", "Classification", "Confidence (%)"]]
    gnames = GRADE_NAMES
    gcols  = GRADE_COLORS_HEX
    for i, (name, prob) in enumerate(probs.items()):
        prob_rows.append([str(i), name, f"{prob:.1f}%"])
    pt2 = Table(prob_rows, colWidths=[2*cm, 7*cm, 5*cm])
    pt2.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#6366f1")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f8fafc"), colors.HexColor("#f1f5f9")]),
        ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
        ("PADDING",     (0, 0), (-1, -1), 6),
        ("FONTNAME",    (0, grade_id+1), (-1, grade_id+1), "Helvetica-Bold"),
    ]))
    story.append(pt2)

    # ── Risk Score ───────────────────────────────────────────────────────────────
    if scan.get("risk_score") is not None:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Clinical Risk Assessment", section_style))
        risk_level = scan.get("risk_level", "—")
        risk_score = scan.get("risk_score", 0)
        risk_color_map = {
            "Low": "#22c55e", "Moderate": "#facc15",
            "High": "#f59e0b", "Very High": "#ef4444"
        }
        rc = rl_color(risk_color_map.get(risk_level, "#818cf8"))
        rs = ParagraphStyle("RS", parent=styles["Normal"],
                            fontSize=14, textColor=rc, fontName="Helvetica-Bold")
        story.append(Paragraph(f"Risk Score: {risk_score}/100  ({risk_level} Risk)", rs))
        if risk_factors:
            story.append(Spacer(1, 4))
            story.append(Paragraph("Contributing Risk Factors:", body_style))
            for f in risk_factors:
                story.append(Paragraph(f"  •  {f}", body_style))

    # ── Clinical Profile ─────────────────────────────────────────────────────────
    if profile:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Clinical Parameters", section_style))
        fields = [
            ("HbA1c",          f"{profile['hba1c']}%" if profile.get("hba1c") else "—"),
            ("Blood Pressure",  f"{profile.get('systolic_bp','—')}/{profile.get('diastolic_bp','—')} mmHg"),
            ("Cholesterol",     f"{profile['cholesterol']} mmol/L" if profile.get("cholesterol") else "—"),
            ("BMI",             f"{profile['bmi']}" if profile.get("bmi") else "—"),
            ("Smoking",         profile.get("smoking", "—")),
            ("Insulin Use",     "Yes" if profile.get("insulin_use") else "No"),
        ]
        clin_data = []
        for i in range(0, len(fields), 2):
            row = [fields[i][0], fields[i][1]]
            if i + 1 < len(fields):
                row += [fields[i+1][0], fields[i+1][1]]
            else:
                row += ["", ""]
            clin_data.append(row)
        ct = Table(clin_data, colWidths=[3.5*cm, 5*cm, 3.5*cm, 5.5*cm])
        ct.setStyle(TableStyle([
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("TEXTCOLOR",  (0, 0), (0, -1), colors.HexColor("#64748b")),
            ("TEXTCOLOR",  (2, 0), (2, -1), colors.HexColor("#64748b")),
            ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME",   (2, 0), (2, -1), "Helvetica-Bold"),
            ("GRID",       (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [colors.HexColor("#f8fafc"), colors.HexColor("#f1f5f9")]),
            ("PADDING",    (0, 0), (-1, -1), 6),
        ]))
        story.append(ct)

    # ── Doctor Review ────────────────────────────────────────────────────────────
    if scan.get("status") != "pending" and scan.get("doctor_name"):
        story.append(Spacer(1, 10))
        story.append(Paragraph("Medical Review", section_style))
        final_grade = scan.get("doctor_class_name") or grade_name
        story.append(Paragraph(f"Reviewing Physician: {scan['doctor_name']}", body_style))
        story.append(Paragraph(f"Final Diagnosis: <b>{final_grade}</b>", body_style))
        story.append(Paragraph(f"Review Status: {scan.get('status', '—').title()}", body_style))
        if scan.get("doctor_notes"):
            story.append(Spacer(1, 4))
            story.append(Paragraph("Clinical Notes:", body_style))
            story.append(Paragraph(scan["doctor_notes"], body_style))
        if scan.get("treatment"):
            story.append(Spacer(1, 4))
            story.append(Paragraph("Treatment Recommendations:", body_style))
            story.append(Paragraph(scan["treatment"], body_style))
        if scan.get("followup_weeks"):
            story.append(Spacer(1, 4))
            story.append(Paragraph(
                f"Follow-up Scheduled: {scan['followup_weeks']} weeks", body_style
            ))

    # ── Recommendation ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 10))
    story.append(Paragraph("Clinical Recommendation", section_style))
    rec_text = RECOMMENDATIONS.get(grade_id, "Please consult a specialist.")
    rec_style = ParagraphStyle(
        "Rec", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#334155"),
        backColor=colors.HexColor("#f1f5f9"),
        borderPadding=(8, 10, 8, 10),
        borderColor=colors.HexColor("#6366f1"),
        borderWidth=1, borderRadius=4, leading=16
    )
    story.append(Paragraph(rec_text, rec_style))

    # ── Disclaimer ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#e2e8f0"), spaceAfter=6))
    disc_style = ParagraphStyle(
        "Disc", parent=styles["Normal"],
        fontSize=7, textColor=colors.HexColor("#94a3b8"), leading=10
    )
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by RetinaAI, an AI-assisted clinical decision support tool. "
        "It is intended to support, not replace, the judgement of qualified medical professionals. "
        "All diagnoses and treatment decisions should be made by a licensed physician. "
        "This report is confidential and intended solely for the named patient and their healthcare team.",
        disc_style
    ))

    try:
        doc.build(story)
        return filepath, file_url
    except Exception as e:
        print(f"[ERROR] PDF build failed for scan {scan.get('id')}: {e}")
        import traceback
        traceback.print_exc()
        return None, None
