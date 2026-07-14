import os
import sys
import io
import json

import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable,
)
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Path resolution ────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config

# Import sibling modules from core/ via the project root (sys.path includes _ROOT)
import core.safetrak_db as db
from core.safetrak_core import (
    AR_LABELS, SCENARIO_NAMES_AR, display_label, feature_label, generate_traffic_report,
)

LOGO_PATH = str(config.LOGO_PATH)
BRAND_NAME = "SAFE TRAK EGY"
BRAND_TAGLINE = "SEE RISK. ACT FAST."
BRAND_NAME_AR = "SAFE TRAK EGY"
BRAND_TAGLINE_AR = "ارصد الخطر، تحرك بسرعة"

# ============================================================================
# ARABIC FONT REGISTRATION -- Unicode TTF loaded from assets/ (downloaded by
# the notebook's install cell). Falls back gracefully (report still builds)
# if the font files aren't present, but Arabic PDFs are only ever rendered
# in Arabic -- there is no English-language fallback.
# ============================================================================
_AR_FONT_REGULAR = str(config.AR_FONT_REGULAR)
_AR_FONT_BOLD = str(config.AR_FONT_BOLD)
ARABIC_FONT_NAME = "Helvetica"
ARABIC_FONT_BOLD = "Helvetica-Bold"
_ARABIC_FONT_READY = False

try:
    if os.path.exists(_AR_FONT_REGULAR):
        pdfmetrics.registerFont(TTFont("NotoNaskhArabic", _AR_FONT_REGULAR))
        ARABIC_FONT_NAME = "NotoNaskhArabic"
        if os.path.exists(_AR_FONT_BOLD):
            pdfmetrics.registerFont(TTFont("NotoNaskhArabic-Bold", _AR_FONT_BOLD))
            ARABIC_FONT_BOLD = "NotoNaskhArabic-Bold"
        else:
            ARABIC_FONT_BOLD = "NotoNaskhArabic"
        _ARABIC_FONT_READY = True
except Exception:
    ARABIC_FONT_NAME, ARABIC_FONT_BOLD, _ARABIC_FONT_READY = "Helvetica", "Helvetica-Bold", False

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _AR_SHAPING_READY = True
except Exception:
    _AR_SHAPING_READY = False


def _ar(text) -> str:
    """Reshape + reorder Arabic text for correct glyph joining and RTL
    rendering inside a reportlab Paragraph. Passes non-Arabic / empty text
    through untouched. No-op (best effort) if the shaping libraries are
    unavailable, so the report still builds."""
    text = "" if text is None else str(text)
    if not text or not _AR_SHAPING_READY:
        return text
    try:
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


# ============================================================================
# BILINGUAL SECTION LABELS -- report chrome (headers, field names) only.
# Uses the same closed-vocabulary AR_LABELS translation table as the
# dashboard for prediction/decision/agent values, so the PDF and the UI
# always agree on Arabic wording.
# ============================================================================
STR = {
    "en": {
        "report_title": "Traffic Intelligence Report",
        "timestamp": "Timestamp", "scenario": "Scenario", "language": "Language", "report_id": "Report ID",
        "exec_summary": "Executive Summary",
        "current_risk": "Current Risk", "predicted_severity": "Predicted Severity",
        "main_risk_factors": "Main Risk Factors", "recommended_action": "Recommended Action",
        "emergency_response": "Emergency Response", "citizen_advice": "Citizen Advice",
        "prediction_summary": "Prediction Summary",
        "severity": "Severity", "risk_score": "Risk Score", "risk_level": "Risk Level",
        "confidence": "Confidence", "contributing_factors": "Contributing Factors",
        "recommended_actions": "Recommended Actions", "no_actions": "No specific actions recorded.",
        "operational_actions": "Operational Actions",
        "priority": "Priority", "emergency_level": "Emergency Level",
        "signal_optimization": "Signal Optimization", "traffic_diversion": "Traffic Diversion",
        "congestion_management": "Congestion Management", "police_dispatch": "Police Dispatch",
        "response_time": "Estimated Response Time", "citizen_notification": "Citizen Notification",
        "safety_recommendation": "Safety Recommendation", "alt_route": "Alternative Route",
        "ai_explanation": "AI Explanation",
        "footer_note": ("This report was generated automatically by the SAFE TRAK EGY Traffic "
                         "Intelligence platform and is intended for operational decision support."),
        "footer_bar": f"{BRAND_NAME} -- {BRAND_TAGLINE}   |   Confidential Traffic Intelligence Report   |   Page",
    },
    "ar": {
        "report_title": "تقرير استخبارات المرور",
        "timestamp": "التوقيت", "scenario": "السيناريو", "language": "اللغة", "report_id": "رقم التقرير",
        "exec_summary": "ملخص تنفيذي",
        "current_risk": "الخطر الحالي", "predicted_severity": "مستوى الحادث المتوقع",
        "main_risk_factors": "عوامل الخطر الرئيسية", "recommended_action": "الإجراء الموصى به",
        "emergency_response": "الاستجابة الطارئة", "citizen_advice": "إرشادات السلامة",
        "prediction_summary": "ملخص التنبؤ",
        "severity": "مستوى الحادث", "risk_score": "مؤشر الخطورة", "risk_level": "مستوى الخطر",
        "confidence": "الثقة", "contributing_factors": "العوامل المؤثرة",
        "recommended_actions": "الإجراءات الموصى بها", "no_actions": "لا توجد إجراءات محددة مسجلة.",
        "operational_actions": "الإجراءات التشغيلية",
        "priority": "الأولوية", "emergency_level": "مستوى الطوارئ",
        "signal_optimization": "تحسين الإشارات", "traffic_diversion": "تحويل المرور",
        "congestion_management": "إدارة الازدحام", "police_dispatch": "الإيفاد الشرطي",
        "response_time": "وقت الاستجابة المقدر", "citizen_notification": "إشعار المواطنين",
        "safety_recommendation": "نصيحة السلامة", "alt_route": "المسار البديل",
        "ai_explanation": "التفسير الذكي",
        "footer_note": ("تم إنشاء هذا التقرير تلقائيًا بواسطة منصة SAFE TRAK EGY لاستخبارات المرور، "
                         "وهو مخصص لدعم القرار التشغيلي."),
        "footer_bar": f"{BRAND_NAME_AR} -- {BRAND_TAGLINE_AR}   |   تقرير سري لاستخبارات المرور   |   صفحة",
    },
}

ALL_LABELS_AR = {**AR_LABELS, **SCENARIO_NAMES_AR}

LANGUAGE_NAME = {
    "en": {"en": "English", "ar": "Arabic"},
    "ar": {"en": "الإنجليزية", "ar": "العربية"},
}


def _language_display(lang_code: str, report_lang: str) -> str:
    """Full language name (never a bare 'EN'/'AR' code) for the report's own
    metadata row, shown in whichever language the report itself is in."""
    code = "ar" if (lang_code or "en") == "ar" else "en"
    return LANGUAGE_NAME[report_lang][code]


def _tr(value, lang: str) -> str:
    """Translate a closed-vocabulary pipeline value (categorical inputs,
    severities, risk levels, decision/agent phrases) for the PDF, reusing
    the same lookup table as the dashboard."""
    if lang != "ar" or value is None:
        return str(value) if value is not None else "-"
    value = str(value)
    return ALL_LABELS_AR.get(value, value.strip())


# ============================================================================
# LOGO -- loaded directly from assets/logo.png, aspect ratio preserved.
# Never Base64-encoded and never modified.
# ============================================================================
def _logo_flowable(max_width=1.05 * inch, max_height=0.75 * inch):
    """Return a reportlab Image flowable for assets/logo.png, scaled to fit
    within max_width/max_height while preserving its original aspect ratio.
    Returns None if the logo file isn't present (report still builds)."""
    if not os.path.exists(LOGO_PATH):
        return None
    reader = ImageReader(LOGO_PATH)
    iw, ih = reader.getSize()
    if not iw or not ih:
        return None
    aspect = ih / float(iw)
    width, height = max_width, max_width * aspect
    if height > max_height:
        height = max_height
        width = height / aspect
    return Image(LOGO_PATH, width=width, height=height)


def _severity_chart_flowable(probabilities: dict, lang: str, width=4.6 * inch, height=2.3 * inch):
    """Render the severity-probability breakdown as a small bar chart,
    built at call time and kept only in memory. Returns None if there is
    nothing to chart."""
    if not probabilities:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [_tr(k, lang) if lang == "ar" else str(k) for k in probabilities.keys()]
    if lang == "ar" and _AR_SHAPING_READY:
        labels = [_ar(l) for l in labels]
    values = [float(v) for v in probabilities.values()]
    palette = ["#60a5fa", "#f59e0b", "#ef4444", "#34d399", "#a78bfa"]

    fig, ax = plt.subplots(figsize=(width / inch, height / inch), dpi=150)
    bars = ax.bar(labels, values, color=[palette[i % len(palette)] for i in range(len(labels))])
    ax.set_ylabel("Probability (%)", fontsize=8)
    ax.set_ylim(0, max(100.0, max(values) * 1.15 if values else 100.0))
    ax.tick_params(axis="both", labelsize=8)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width, height=height)


def _footer_factory(lang: str, font_name: str):
    """Build a page-footer callback for the given language/font: brand,
    tagline, confidentiality note, page number."""
    label = STR[lang]["footer_bar"]

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.line(0.75 * inch, 0.62 * inch, doc.pagesize[0] - 0.75 * inch, 0.62 * inch)
        canvas.setFont(font_name, 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        footer_text = f"{label} {doc.page}" if lang == "en" else _ar(f"{label} {doc.page}")
        canvas.drawCentredString(doc.pagesize[0] / 2.0, 0.42 * inch, footer_text)
        canvas.restoreState()

    return _footer


# ============================================================================
# PDF REPORT -- built in memory, only when explicitly requested.
# ============================================================================
def _regenerate_ai_report(record: dict, lang: str) -> str:
    """Re-run only the AI narrative (Prompt Engineering step) in `lang`,
    reusing the already-computed Prediction / Decision / Agent JSON stored
    in the record -- prediction, preprocessing, and decision logic are never
    re-run. Used only when the PDF language the user picked differs from
    the language the prediction was originally made in, so the AI report
    section always matches the rest of the PDF and never mixes languages."""
    raw = record.get("raw") or {}
    prediction = raw.get("prediction") or record.get("prediction") or {}
    decision = raw.get("decision") or record.get("decision") or {}
    dtma = raw.get("dtma") or {}
    sraa = raw.get("sraa") or {}
    caa = raw.get("caa") or {}
    context = raw.get("context")
    if not (prediction and decision and dtma and sraa and caa):
        # Not enough pipeline context stored to safely regenerate -- fall
        # back to whatever narrative was originally saved rather than error.
        return record.get("ai_report") or ""
    try:
        return generate_traffic_report(prediction, decision, dtma, sraa, caa, context, lang)
    except Exception:
        return record.get("ai_report") or ""


def generate_pdf_report(record: dict, pdf_lang: str = None) -> bytes:
    """Build a professional PDF Traffic Intelligence Report for one stored
    prediction record (as returned by safetrak_db.load_history /
    filter_history / search_history). Returns raw PDF bytes; nothing is
    written to disk.

    `pdf_lang` ("en" or "ar") is chosen independently by the user at export
    time and is NOT tied to the language the underlying prediction/UI was
    originally made in (record["language"]). If omitted, falls back to the
    record's original language for backward compatibility. Whichever
    language is selected, the PDF -- including the AI Explanation section --
    renders entirely in that language: Arabic gets a fully Arabic,
    right-to-left PDF on a Unicode Arabic font; English gets a fully
    English PDF on Helvetica. The two languages are never mixed."""
    lang = "ar" if (pdf_lang or record.get("language") or "en") == "ar" else "en"
    stored_lang = "ar" if (record.get("language") or "en") == "ar" else "en"
    s = STR[lang]
    is_ar = lang == "ar"
    font = ARABIC_FONT_NAME if is_ar else "Helvetica"
    font_bold = ARABIC_FONT_BOLD if is_ar else "Helvetica-Bold"
    align = TA_RIGHT if is_ar else TA_LEFT

    def T(text):
        return _ar(text) if is_ar else str(text)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.7 * inch, bottomMargin=0.9 * inch,
        title=f"{BRAND_NAME_AR if is_ar else BRAND_NAME} {s['report_title']}",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("SafetrakTitle", fontName=font_bold, fontSize=20,
                                  textColor=colors.HexColor("#0f172a"), alignment=align)
    h2 = ParagraphStyle("SafetrakH2", fontName=font_bold, fontSize=12.5,
                         textColor=colors.HexColor("#1d4ed8"), spaceBefore=12, spaceAfter=5, alignment=align)
    body = ParagraphStyle("SafetrakBody", fontName=font, fontSize=10, leading=15, alignment=align)
    brand_sub = ParagraphStyle("SafetrakBrandSub", fontName=font, fontSize=9,
                                textColor=colors.HexColor("#64748b"), alignment=align)
    small = ParagraphStyle("SafetrakSmall", fontName=font, fontSize=8.5,
                            textColor=colors.HexColor("#64748b"), alignment=align)
    bullet_style = ParagraphStyle("SafetrakBullet", fontName=font, fontSize=10.5, leading=16, alignment=align)

    story = []

    # ---- Header: logo (native aspect ratio, loaded from assets/logo.png) ----
    brand_name_text = BRAND_NAME_AR if is_ar else BRAND_NAME
    brand_tag_text = BRAND_TAGLINE_AR if is_ar else BRAND_TAGLINE
    logo = _logo_flowable()
    if logo is not None:
        brand_block = [
            Paragraph(f"<b>{T(brand_name_text)}</b>", ParagraphStyle(
                "SafetrakBrandName", fontName=font_bold, fontSize=13,
                textColor=colors.HexColor("#0f172a"), alignment=align)),
            Paragraph(T(brand_tag_text), brand_sub),
        ]
        cols = [logo, brand_block] if not is_ar else [brand_block, logo]
        col_widths = [1.25 * inch, 4.95 * inch] if not is_ar else [4.95 * inch, 1.25 * inch]
        header_table = Table([cols], colWidths=col_widths)
        header_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.append(header_table)
    else:
        story.append(Paragraph(T(brand_name_text), title_style))
        story.append(Paragraph(T(brand_tag_text), brand_sub))

    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#cbd5e1")))
    story.append(Spacer(1, 10))
    story.append(Paragraph(T(s["report_title"]), title_style))
    story.append(Spacer(1, 6))

    prediction = record.get("prediction", {}) or {}
    decision = record.get("decision", {}) or {}
    probabilities = record.get("probabilities", {}) or {}
    inputs = record.get("inputs", {}) or {}
    recommendations = record.get("recommendations", []) or []
    raw = record.get("raw", {}) or {}
    dtma = raw.get("dtma", {}) or {}
    sraa = raw.get("sraa", {}) or {}
    caa = raw.get("caa", {}) or {}

    def two_col_table(rows, col_widths, header_col=True):
        rows = [[T(a), T(b)] for a, b in rows]
        if is_ar:
            rows = [[b, a] for a, b in rows]
            col_widths = list(reversed(col_widths))
        table = Table(rows, colWidths=col_widths)
        label_col = 1 if is_ar else 0
        style = [
            ("FONTNAME", (0, 0), (-1, -1), font),
            ("ALIGN", (0, 0), (-1, -1), "RIGHT" if is_ar else "LEFT"),
        ]
        if header_col:
            style += [
                ("FONTNAME", (label_col, 0), (label_col, -1), font_bold),
                ("TEXTCOLOR", (label_col, 0), (label_col, -1), colors.HexColor("#64748b")),
            ]
        table.setStyle(TableStyle(style))
        return table

    # ---- Report metadata ----
    meta_rows = [
        [s["timestamp"], str(record.get("timestamp", "-"))],
        [s["scenario"], _tr(record.get("scenario", "-"), lang) if lang == "ar" else str(record.get("scenario", "-"))],
        [s["language"], _language_display(record.get("language"), lang)],
        [s["report_id"], str(record.get("id", "-"))],
    ]
    meta_table = two_col_table(meta_rows, [1.6 * inch, 4.6 * inch])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # ---- Executive Summary (mirrors the in-app default view) ----
    story.append(Paragraph(T(s["exec_summary"]), h2))
    risk_factor_keys = ["Weather_conditions", "Road_surface_conditions", "Light_conditions"]
    risk_factors = ", ".join(
        display_label(inputs.get(k), lang) for k in risk_factor_keys if inputs.get(k)
    ) or "-"
    action_bits = ", ".join(_tr(a, lang) for a in recommendations[:3]) or "-"
    exec_lines = [
        f"{s['current_risk']}: {record.get('risk_score', prediction.get('risk_score', '-'))} / 100 "
        f"({_tr(record.get('risk_level', prediction.get('risk_level')), lang)})",
        f"{s['predicted_severity']}: {_tr(record.get('severity', prediction.get('severity')), lang)}",
        f"{s['main_risk_factors']}: {risk_factors}",
        f"{s['recommended_action']}: {action_bits}",
        f"{s['emergency_response']}: {_tr(sraa.get('police_dispatch'), lang)} "
        f"({_tr(sraa.get('estimated_response_time'), lang)})",
        f"{s['citizen_advice']}: {_tr(caa.get('safety_recommendation'), lang)}",
    ]
    for line in exec_lines:
        bullet = "•" if not is_ar else "•"
        story.append(Paragraph(f"{bullet} {T(line)}", bullet_style))
    story.append(Spacer(1, 8))

    # ---- Prediction Summary: Risk Score, Severity ----
    story.append(Paragraph(T(s["prediction_summary"]), h2))
    summary_rows = [
        [s["severity"], _tr(record.get("severity", prediction.get("severity", "-")), lang)],
        [s["risk_score"], f"{record.get('risk_score', prediction.get('risk_score', '-'))} / 100"],
        [s["risk_level"], _tr(record.get("risk_level", prediction.get("risk_level", "-")), lang)],
    ]
    summary_table = two_col_table(summary_rows, [1.6 * inch, 4.6 * inch])
    summary_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (1 if is_ar else 0, 0), (1 if is_ar else 0, -1), colors.HexColor("#f8fafc")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 8))

    # ---- Confidence (severity probabilities) + chart ----
    if probabilities:
        story.append(Paragraph(T(s["confidence"]), h2))
        conf_rows = [[_tr(k, lang), f"{v}%"] for k, v in probabilities.items()]
        conf_table = two_col_table(conf_rows, [2.8 * inch, 1.4 * inch], header_col=False)
        conf_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ]))
        story.append(conf_table)
        story.append(Spacer(1, 6))
        chart = _severity_chart_flowable(probabilities, lang)
        if chart is not None:
            story.append(chart)
        story.append(Spacer(1, 6))

    # ---- Contributing factors (inputs used) ----
    if inputs:
        story.append(Paragraph(T(s["contributing_factors"]), h2))
        factor_rows = [
            [feature_label(str(k), lang), display_label(v, lang) if lang == "ar" else str(v)]
            for k, v in inputs.items()
        ]
        factor_table = two_col_table(factor_rows, [2.3 * inch, 3.9 * inch], header_col=False)
        factor_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#eef2f7")),
        ]))
        story.append(factor_table)
        story.append(Spacer(1, 8))

    # ---- Recommendations ----
    story.append(Paragraph(T(s["recommended_actions"]), h2))
    if recommendations:
        for action in recommendations:
            story.append(Paragraph(f"&bull; {T(_tr(action, lang))}", body))
    else:
        story.append(Paragraph(T(s["no_actions"]), body))
    story.append(Spacer(1, 8))

    # ---- Operational Actions (Decision Engine + Agents) ----
    story.append(Paragraph(T(s["operational_actions"]), h2))
    op_rows = [
        [s["priority"], _tr(decision.get("priority", "-"), lang)],
        [s["emergency_level"], _tr(decision.get("emergency_level", "-"), lang)],
        [s["signal_optimization"], _tr(dtma.get("signal_optimization", "-"), lang)],
        [s["traffic_diversion"], _tr(dtma.get("traffic_diversion", "-"), lang)],
        [s["congestion_management"], _tr(dtma.get("congestion_management", "-"), lang)],
        [s["police_dispatch"], _tr(sraa.get("police_dispatch", "-"), lang)],
        [s["response_time"], _tr(sraa.get("estimated_response_time", "-"), lang)],
        [s["citizen_notification"], _tr(caa.get("citizen_notification", "-"), lang)],
        [s["safety_recommendation"], _tr(caa.get("safety_recommendation", "-"), lang)],
        [s["alt_route"], _tr(caa.get("alternative_route_suggestion", "-"), lang)],
    ]
    op_table = two_col_table(op_rows, [1.9 * inch, 4.3 * inch], header_col=True)
    op_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#eef2f7")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(op_table)
    story.append(Spacer(1, 10))

    # ---- AI Explanation (full report, same text shown in "View Full AI Report") ----
    ai_report = record.get("ai_report") or "" if lang == stored_lang else _regenerate_ai_report(record, lang)
    if ai_report:
        story.append(Paragraph(T(s["ai_explanation"]), h2))
        for para in ai_report.split("\n"):
            if para.strip():
                story.append(Paragraph(T(para.strip()), body))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 4))
    story.append(Paragraph(T(s["footer_note"]), small))

    footer_fn = _footer_factory(lang, font)
    doc.build(story, onFirstPage=footer_fn, onLaterPages=footer_fn)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================================
# CSV EXPORT -- every stored prediction, flattened.
# ============================================================================
def export_all_to_csv() -> bytes:
    """Flatten every prediction stored in SQLite into a single CSV, built
    in memory. Returns CSV bytes ready for a download button."""
    rows = db.load_history()
    records = []
    for r in rows:
        inputs = r.get("inputs", {}) or {}
        decision = r.get("decision", {}) or {}
        records.append({
            "id": r.get("id"),
            "timestamp": r.get("timestamp"),
            "language": r.get("language"),
            "scenario": r.get("scenario"),
            "severity": r.get("severity"),
            "risk_score": r.get("risk_score"),
            "risk_level": r.get("risk_level"),
            "priority": decision.get("priority"),
            "emergency_level": decision.get("emergency_level"),
            "weather": inputs.get("Weather_conditions"),
            "vehicle": inputs.get("Type_of_vehicle"),
            "road_surface": inputs.get("Road_surface_conditions"),
            "light": inputs.get("Light_conditions"),
            "location": inputs.get("Area_accident_occured"),
            "recommended_actions": "; ".join(r.get("recommendations", []) or []),
        })
    df = pd.DataFrame(records)
    return df.to_csv(index=False).encode("utf-8-sig")


# JSON EXPORT
def export_record_to_json(record: dict) -> bytes:
    """Export one prediction record's full raw pipeline JSON as
    pretty-printed JSON bytes, ready for a download button."""
    payload = record.get("raw") or record
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
