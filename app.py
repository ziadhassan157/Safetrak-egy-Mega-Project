import os
import sys

# ── Ensure project root is on sys.path so config.py and core/ are importable ──
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Load .env and path config before anything else
import config  # noqa: E402 -- intentional early import

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

from core.safetrak_core import (
    run_pipeline, SCENARIO_INPUTS, SCENARIOS, display_label, field_label,
    scenario_display_name, scenario_display_context, AR_LABELS, SCENARIO_NAMES_AR,
)
import core.safetrak_db as db
import core.safetrak_reports as reports

st.set_page_config(page_title="SAFE TRAK EGY", page_icon="🚦", layout="wide",
                    initial_sidebar_state="expanded")

APP_VERSION = "v2.1.0"


# ============================================================================
# PRODUCTION DATA LAYER (SQLite)
# ============================================================================
@st.cache_resource
def init_data_layer():
    """Create database/, reports/, exports/, assets/ and predictions.db
    (with its schema) exactly once per running server process."""
    db.ensure_folders()
    db.init_db()
    return True


init_data_layer()


@st.cache_data(ttl=5)
def _fetch_history_rows():
    """Cached read of every stored prediction. Cleared whenever a new
    prediction is saved so the UI always reflects the latest data."""
    return db.load_history()

# ============================================================================
# TRANSLATIONS  (full UI + data-display localization. The AI report itself is
# also generated in the selected language -- see safetrak_core.generate_traffic_report.)
# ============================================================================
T = {
    "en": {
        "app_name": "SAFE TRAK EGY", "tagline": "SEE RISK. ACT FAST.",
        "nav_dashboard": "🏠 Dashboard", "nav_manual": "🚦 Manual Prediction",
        "nav_scenarios": "📋 Example Scenarios", "nav_history": "📈 History Analytics",
        "nav_reports": "🗂️ Reports",
        "nav_settings": "⚙ Settings",
        "nav_group_label": "Navigate",
        "language": "Language", "theme": "Theme",
        "theme_dark": "Dark", "theme_light": "Light",
        "dashboard_title": "Command Center Overview",
        "dashboard_sub": "Live snapshot of today's traffic-risk intelligence.",
        "kpi_total": "Today's Predictions", "kpi_avg": "Average Risk",
        "kpi_high": "Highest Risk", "kpi_latest": "Latest Prediction",
        "kpi_level": "Current Risk Level",
        "dist_title": "📊 Prediction Distribution", "recent_title": "🕓 Recent Activity",
        "no_data": "No predictions yet. Run a Manual Prediction or an Example Scenario to populate the dashboard.",
        "manual_title": "Manual Prediction", "manual_sub": "Provide the key situational factors.",
        "manual_scenario_label": "Manual",
        "context_label": "Additional context (optional)",
        "run_btn": "Run Analysis", "analyzing": "Analyzing traffic scenario...",
        "scenarios_title": "Example Scenarios", "scenarios_sub": "One click runs the full pipeline.",
        "choose_scenario": "Choose a scenario", "view_inputs": "View scenario inputs",
        "history_title": "History Analytics", "history_sub": "Operational insight from every stored prediction.",
        "reports_title": "Reports", "reports_sub": "Search, filter, and export every stored prediction.",
        "search_label": "Search reports",
        "search_placeholder": "Search by scenario, severity, risk level, or report text...",
        "filter_scenario": "Scenario", "filter_risk": "Risk Level", "filter_severity": "Severity",
        "sort_label": "Sort by", "sort_newest": "Newest first", "sort_oldest": "Oldest first",
        "sort_risk_desc": "Highest risk first", "sort_risk_asc": "Lowest risk first",
        "export_all_csv": "⬇ Export All to CSV",
        "reports_count_suffix": "report(s) found",
        "no_reports": "No predictions saved yet. Run a Manual Prediction or Example Scenario first.",
        "no_reports_match": "No reports match your search/filter.",
        "preview_report": "👁 Preview", "open_report": "📂 Open",
        "download_pdf": "📄 Generate PDF", "save_pdf": "⬇ Save PDF", "download_json": "⬇ JSON",
        "delete_report": "🗑 Delete", "report_deleted": "Report deleted.",
        "pdf_language": "PDF Language",
        "settings_title": "Settings", "settings_sub": "Preferences and system status.",
        "settings_lang": "Language", "settings_theme": "Theme",
        "settings_db": "Database Status", "settings_hist": "History Management",
        "settings_predictions": "Prediction Count", "settings_version": "Application Version",
        "db_placeholder": "No persistent database connected. Predictions live in this session only.",
        "hist_placeholder": "Stored in SQLite; persists across restarts.",
        "clear_history": "Clear session history",
        "history_cleared": "Session history cleared.",
        "tab_overview": "Overview", "tab_recommendations": "Recommendations",
        "tab_ai": "AI Analysis", "tab_raw": "Raw JSON",
        "exec_summary": "Executive Summary", "prediction": "Prediction", "risk_score": "Risk Score",
        "severity": "Severity", "confidence": "Confidence", "contributing_factors": "Contributing Factors",
        "recommended_actions": "Recommended Actions", "emergency_response": "Emergency Response",
        "citizen_advisory": "Citizen Advisory", "ai_explanation": "AI Explanation",
        "priority": "Priority", "emergency_level": "Emergency Level",
        "dispatch": "Dispatch", "response_time": "Estimated Response Time",
        "notification": "Notification", "safety_tip": "Safety Recommendation", "alt_route": "Alternative Route",
        "signal": "Signal Optimization", "diversion": "Traffic Diversion", "congestion": "Congestion Management",
        "weather": "Weather", "road_surface": "Road Surface", "light": "Light Conditions",
        "location_type": "Location Type",
        "risk_word": "Risk",
        "agent_dtma": "Dynamic Traffic Management (DTMA)",
        "agent_sraa": "Safety Response (SRAA)",
        "agent_caa": "Citizen Advisory (CAA)",
        "risk_trend": "Risk Score Trend Over Time", "severity_pie": "Severity Distribution",
        "risk_dist": "Risk Distribution",
        "predictions_timeline": "Predictions Timeline",
        "avg_risk_weather": "Average Risk by Weather", "avg_risk_vehicle": "Average Risk by Vehicle Type",
        "avg_risk_location": "Average Risk by Location", "avg_risk_experience": "Average Risk by Driver Experience",
        "prediction_frequency": "Prediction Frequency", "top_high_risk": "Top High-Risk Scenarios",
        "highest_risk_factors": "Highest Risk Factors", "avg_daily_risk": "Average Daily Risk",
        "not_enough_data": "Not enough historical data for analytics.",
        "avg_risk_chart": "Average Risk Over Time", "recent_predictions": "Recent Predictions",
        "col_time": "Time", "col_scenario": "Scenario", "col_severity": "Severity",
        "col_risk_score": "Risk Score", "col_risk_level": "Risk Level", "col_weather": "Weather",
        "col_vehicle": "Vehicle", "col_location": "Location", "col_count": "Count",
        "col_experience": "Driver Experience", "col_date": "Date",
        "current_risk": "Current Risk", "predicted_severity": "Predicted Severity",
        "main_risk_factors": "Main Risk Factors", "recommended_action": "Recommended Action",
        "citizen_advice": "Citizen Advice", "view_full_report": "📖 View Full AI Report",
        "brand_line": "Traffic Intelligence Dashboard",
    },
    "ar": {
        "app_name": "SAFE TRAK EGY", "tagline": "ارصد الخطر، تحرك بسرعة",
        "nav_dashboard": "🏠 لوحة التحكم", "nav_manual": "🚦 تنبؤ يدوي",
        "nav_scenarios": "📋 سيناريوهات توضيحية", "nav_history": "📈 تحليل السجل",
        "nav_reports": "🗂️ التقارير",
        "nav_settings": "⚙ الإعدادات",
        "nav_group_label": "التنقل",
        "language": "اللغة", "theme": "المظهر",
        "theme_dark": "داكن", "theme_light": "فاتح",
        "dashboard_title": "نظرة عامة على مركز القيادة",
        "dashboard_sub": "لقطة حية لذكاء مخاطر المرور اليوم.",
        "kpi_total": "تنبؤات اليوم", "kpi_avg": "متوسط الخطر",
        "kpi_high": "أعلى خطر", "kpi_latest": "آخر تنبؤ",
        "kpi_level": "مستوى الخطر الحالي",
        "dist_title": "📊 توزيع التنبؤات", "recent_title": "🕓 النشاط الأخير",
        "no_data": "لا توجد تنبؤات بعد. قم بتشغيل تنبؤ يدوي أو سيناريو توضيحي لملء لوحة التحكم.",
        "manual_title": "تنبؤ يدوي", "manual_sub": "أدخل العوامل الرئيسية للموقف.",
        "manual_scenario_label": "يدوي",
        "context_label": "سياق إضافي (اختياري)",
        "run_btn": "تشغيل التحليل", "analyzing": "جارٍ تحليل سيناريو المرور...",
        "scenarios_title": "سيناريوهات توضيحية", "scenarios_sub": "ضغطة واحدة تشغّل المسار الكامل.",
        "choose_scenario": "اختر سيناريو", "view_inputs": "عرض مدخلات السيناريو",
        "history_title": "تحليل السجل", "history_sub": "رؤى تشغيلية من كل تنبؤ محفوظ.",
        "reports_title": "التقارير", "reports_sub": "ابحث وصفّي وصدّر كل تنبؤ محفوظ.",
        "search_label": "بحث في التقارير",
        "search_placeholder": "ابحث حسب السيناريو أو مستوى الحادث أو مستوى الخطر أو نص التقرير...",
        "filter_scenario": "السيناريو", "filter_risk": "مستوى الخطر", "filter_severity": "مستوى الحادث",
        "sort_label": "ترتيب حسب", "sort_newest": "الأحدث أولاً", "sort_oldest": "الأقدم أولاً",
        "sort_risk_desc": "الأعلى خطورة أولاً", "sort_risk_asc": "الأقل خطورة أولاً",
        "export_all_csv": "⬇ تصدير الكل CSV",
        "reports_count_suffix": "تقرير موجود",
        "no_reports": "لا توجد تنبؤات محفوظة بعد. شغّل تنبؤ يدوي أو سيناريو توضيحي أولاً.",
        "no_reports_match": "لا توجد تقارير مطابقة للبحث/التصفية.",
        "preview_report": "👁 معاينة", "open_report": "📂 فتح",
        "download_pdf": "📄 إنشاء PDF", "save_pdf": "⬇ حفظ PDF", "download_json": "⬇ JSON",
        "delete_report": "🗑 حذف", "report_deleted": "تم حذف التقرير.",
        "pdf_language": "لغة تقرير PDF",
        "settings_title": "الإعدادات", "settings_sub": "التفضيلات وحالة النظام.",
        "settings_lang": "اللغة", "settings_theme": "المظهر",
        "settings_db": "حالة قاعدة البيانات", "settings_hist": "إدارة السجل",
        "settings_predictions": "عدد التنبؤات", "settings_version": "إصدار التطبيق",
        "db_placeholder": "لا توجد قاعدة بيانات دائمة متصلة. تبقى التنبؤات ضمن الجلسة فقط.",
        "hist_placeholder": "مخزّن في SQLite ويبقى بعد إعادة التشغيل.",
        "clear_history": "مسح سجل الجلسة",
        "history_cleared": "تم مسح سجل الجلسة.",
        "tab_overview": "نظرة عامة", "tab_recommendations": "التوصيات",
        "tab_ai": "التحليل الذكي", "tab_raw": "JSON خام",
        "exec_summary": "ملخص تنفيذي",
        "prediction": "التنبؤ",
        "risk_score": "مؤشر الخطورة",
        "severity": "مستوى الحادث", "confidence": "الثقة", "contributing_factors": "العوامل المؤثرة",
        "recommended_actions": "الإجراءات الموصى بها", "emergency_response": "الاستجابة الطارئة",
        "citizen_advisory": "إرشادات السلامة", "ai_explanation": "التفسير الذكي",
        "priority": "الأولوية", "emergency_level": "مستوى الطوارئ",
        "dispatch": "الإيفاد", "response_time": "وقت الاستجابة المقدر",
        "notification": "الإشعار", "safety_tip": "نصيحة السلامة", "alt_route": "المسار البديل",
        "signal": "تحسين الإشارات", "diversion": "تحويل المرور", "congestion": "إدارة الازدحام",
        "weather": "الطقس", "road_surface": "سطح الطريق", "light": "حالة الإضاءة",
        "location_type": "نوع الموقع",
        "risk_word": "خطر",
        "agent_dtma": "إدارة المرور الديناميكية (DTMA)",
        "agent_sraa": "الاستجابة الأمنية (SRAA)",
        "agent_caa": "إرشاد المواطنين (CAA)",
        "risk_trend": "اتجاه مؤشر الخطورة عبر الزمن", "severity_pie": "توزيع مستوى الحادث",
        "risk_dist": "توزيع الخطر",
        "predictions_timeline": "خط زمني للتنبؤات",
        "avg_risk_weather": "متوسط الخطر حسب الطقس", "avg_risk_vehicle": "متوسط الخطر حسب نوع المركبة",
        "avg_risk_location": "متوسط الخطر حسب الموقع", "avg_risk_experience": "متوسط الخطر حسب خبرة السائق",
        "prediction_frequency": "تكرار التنبؤات", "top_high_risk": "أعلى السيناريوهات خطورة",
        "highest_risk_factors": "أعلى عوامل الخطر", "avg_daily_risk": "متوسط الخطر اليومي",
        "not_enough_data": "لا توجد بيانات تاريخية كافية للتحليلات.",
        "avg_risk_chart": "متوسط الخطر عبر الزمن", "recent_predictions": "أحدث التنبؤات",
        "col_time": "الوقت", "col_scenario": "السيناريو", "col_severity": "مستوى الحادث",
        "col_risk_score": "مؤشر الخطورة", "col_risk_level": "مستوى الخطر", "col_weather": "الطقس",
        "col_vehicle": "المركبة", "col_location": "الموقع", "col_count": "العدد",
        "col_experience": "خبرة السائق", "col_date": "التاريخ",
        "current_risk": "الخطر الحالي", "predicted_severity": "مستوى الحادث المتوقع",
        "main_risk_factors": "عوامل الخطر الرئيسية", "recommended_action": "الإجراء الموصى به",
        "citizen_advice": "نصيحة للمواطنين", "view_full_report": "📖 عرض التقرير الكامل",
        "brand_line": "لوحة استخبارات المرور",
    },
}

# Combined closed-vocabulary lookup used to translate any raw pipeline value
# (categorical inputs, severities, risk levels, decision/agent phrases, and
# scenario names) for display -- charts, tables, cards, and badges.
ALL_LABELS_AR = {**AR_LABELS, **SCENARIO_NAMES_AR}


def tr_value(text, lang):
    """Translate a closed-vocabulary value coming out of safetrak_core
    (Prediction / Decision Engine / Agents / scenario names). Passes through
    unchanged if no translation is registered, or if English is selected."""
    if lang == "en" or text is None:
        return text
    return ALL_LABELS_AR.get(text, text)


def tr_scenario_label(label, lang):
    """Translate a history 'scenario' cell, which is either 'Manual' or one
    of the SCENARIOS keys."""
    if lang == "en":
        return label
    if label == "Manual":
        return T["ar"]["manual_scenario_label"]
    return scenario_display_name(label, lang)


COLUMN_LABEL_KEYS = {
    "time": "col_time", "scenario": "col_scenario", "severity": "col_severity",
    "risk_score": "col_risk_score", "risk_level": "col_risk_level", "weather": "col_weather",
    "vehicle": "col_vehicle", "location": "col_location", "count": "col_count",
    "experience": "col_experience", "date": "col_date",
}


# ============================================================================
# SESSION STATE
# ============================================================================
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"
if "page" not in st.session_state:
    st.session_state.page = "nav_dashboard"

L = st.session_state.lang
S = T[L]
RTL = (L == "ar")


def t(key):
    return S.get(key, T["en"].get(key, key))


def col_label(col):
    return t(COLUMN_LABEL_KEYS.get(col, col))


def localized_df(df, columns):
    """Return a copy of df restricted to `columns`, with every categorical
    cell value and every column header translated into the active language.
    The underlying session history (df) always stays in the canonical
    English vocabulary produced by the pipeline; this is a display-only
    transform applied fresh on every render."""
    out = df[columns].copy()
    for c in ("severity", "risk_level", "weather", "vehicle", "location", "experience"):
        if c in out.columns:
            out[c] = out[c].apply(lambda v: tr_value(v, L))
    if "scenario" in out.columns:
        out["scenario"] = out["scenario"].apply(lambda v: tr_scenario_label(v, L))
    out = out.rename(columns={c: col_label(c) for c in out.columns})
    return out


def localized_value_counts(df, col):
    """value_counts() for a categorical column, translated for display."""
    vc = df[col].value_counts().reset_index()
    vc.columns = [col, "count"]
    vc[col] = vc[col].apply(lambda v: tr_value(v, L))
    vc = vc.rename(columns={col: col_label(col), "count": col_label("count")})
    return vc, col_label(col), col_label("count")


# ============================================================================
# THEME
# ============================================================================
if st.session_state.theme == "Dark":
    THEME = dict(bg="#0b1120", panel="#111827", border="#1f2937", text="#e5e7eb",
                 subtext="#94a3b8", accent="#60a5fa", accent_soft="#93c5fd",
                 sidebar="#0f172a", input_bg="#111827", table_header="#1f2937",
                 shadow="0 4px 14px rgba(0,0,0,0.35)", btn_text="#0b1120")
else:
    THEME = dict(bg="#f8fafc", panel="#ffffff", border="#dbe3ec", text="#0f172a",
                 subtext="#475569", accent="#2563eb", accent_soft="#1d4ed8",
                 sidebar="#ffffff", input_bg="#ffffff", table_header="#eef2f7",
                 shadow="0 2px 10px rgba(15,23,42,0.08)", btn_text="#ffffff")

BADGE_COLORS = {
    "Low": ("#14532d", "#bbf7d0") if st.session_state.theme == "Dark" else ("#dcfce7", "#166534"),
    "Medium": ("#713f12", "#fde68a") if st.session_state.theme == "Dark" else ("#fef9c3", "#854d0e"),
    "High": ("#7f1d1d", "#fecaca") if st.session_state.theme == "Dark" else ("#fee2e2", "#991b1b"),
}

direction_css = "rtl" if RTL else "ltr"
text_align = "right" if RTL else "left"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; direction: {direction_css}; }}

.stApp {{ background-color: {THEME['bg']}; color: {THEME['text']}; }}

/* ---- Base text readability (fixes invisible/low-contrast text in Light theme) ---- */
.stApp, .stApp p, .stApp span, .stApp label, .stApp li,
.stMarkdown, .stCaption, [data-testid="stMarkdownContainer"] {{ color: {THEME['text']}; }}
.stApp .stCaption, [data-testid="stCaptionContainer"] {{ color: {THEME['subtext']}; }}
h1, h2, h3, h4, h5 {{ letter-spacing: -0.01em; color: {THEME['text']}; text-align: {text_align}; }}
[data-testid="stHeader"], [data-testid="stMarkdownContainer"] > h1,
[data-testid="stMarkdownContainer"] > h2, [data-testid="stMarkdownContainer"] > h3 {{ text-align: {text_align}; }}
.stApp .stCaption, [data-testid="stCaptionContainer"] {{ text-align: {text_align}; direction: {direction_css}; }}

/* ---- Alerts (info/success/warning/error): full RTL support ---- */
[data-testid="stAlert"] {{ direction: {direction_css}; text-align: {text_align}; }}
[data-testid="stAlert"] * {{ text-align: {text_align}; }}

/* ---- Buttons / selectboxes: RTL text flow ---- */
.stButton, .stDownloadButton {{ direction: {direction_css}; }}
div[data-baseweb="select"] {{ direction: {direction_css}; text-align: {text_align}; }}

/* ---- Sidebar radio (navigation) RTL ---- */
section[data-testid="stSidebar"] .stRadio > div {{ direction: {direction_css}; }}
section[data-testid="stSidebar"] label {{ text-align: {text_align}; }}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {{
    background-color: {THEME['sidebar']}; border-right: 1px solid {THEME['border']};
}}
section[data-testid="stSidebar"] * {{ color: {THEME['text']}; }}

/* ---- Responsive container spacing ---- */
.block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1280px; }}
[data-testid="column"] {{ padding: 0.35rem; }}
@media (max-width: 900px) {{
    .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
    .st-kpi {{ padding: 0.85rem 1rem; }}
}}

/* ---- KPI cards ---- */
.st-kpi {{
    background-color: {THEME['panel']}; border: 1px solid {THEME['border']}; border-radius: 16px;
    padding: 1.1rem 1.3rem; box-shadow: {THEME['shadow']}; text-align: {text_align};
    height: 100%;
}}
.st-kpi .kpi-label {{ color: {THEME['subtext']}; font-size: 0.82rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.35rem; }}
.st-kpi .kpi-value {{ color: {THEME['text']}; font-size: 1.65rem; font-weight: 800; }}
.st-kpi .kpi-icon {{ font-size: 1.4rem; }}

/* ---- Info / content cards ---- */
.safetrak-card {{
    background-color: {THEME['panel']}; border: 1px solid {THEME['border']}; border-radius: 16px;
    padding: 1.15rem 1.35rem; margin-bottom: 1rem; box-shadow: {THEME['shadow']};
    text-align: {text_align}; direction: {direction_css}; width: 100%;
}}
.safetrak-card h4 {{ margin-top: 0; margin-bottom: 0.6rem; color: {THEME['accent_soft']};
    font-weight: 700; letter-spacing: 0.01em; }}
.safetrak-card, .safetrak-card * {{ color: {THEME['text']}; }}
.safetrak-card h4 {{ color: {THEME['accent_soft']}; }}

.badge {{ display: inline-block; padding: 0.3rem 0.85rem; border-radius: 999px;
    font-weight: 700; font-size: 0.83rem; letter-spacing: 0.02em; }}

.report-box {{
    background-color: {THEME['panel']}; border: 1px solid {THEME['border']}; border-radius: 16px;
    padding: 1.4rem 1.6rem; direction: {direction_css}; text-align: {text_align}; line-height: 1.75;
    font-size: 1.0rem; white-space: pre-wrap; box-shadow: {THEME['shadow']};
    color: {THEME['text']};
}}

.brand-box {{ text-align: center; padding: 0.6rem 0 1rem 0; }}
.brand-title {{ font-size: 1.35rem; font-weight: 800; color: {THEME['accent_soft']}; letter-spacing: 0.02em; }}
.brand-tag {{ font-size: 0.85rem; color: {THEME['subtext']}; font-style: italic; }}
.brand-logo-img {{ max-width: 96px; height: auto; margin: 0 auto 0.35rem auto; display: block; }}

/* ---- Buttons: readable in both themes ---- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
    background-color: {THEME['accent']} !important; color: {THEME['btn_text']} !important; border: 1px solid {THEME['accent']} !important;
    border-radius: 10px; font-weight: 600;
}}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {{
    background-color: {THEME['accent_soft']} !important; border-color: {THEME['accent_soft']} !important; color: {THEME['btn_text']} !important;
}}

/* ---- Header ---- */
header[data-testid="stHeader"], [data-testid="stHeader"] {{ background-color: {THEME['bg']} !important; }}

/* ---- Inputs / selects / text areas: fix white-on-white & low contrast ---- */
input, textarea, select, 
[data-baseweb="select"] > div, 
[data-baseweb="input"] > div, 
[data-baseweb="input"] input,
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] > div > div > div,
[data-testid="stTextInput"] > div > div,
[data-testid="stNumberInput"] > div > div {{
    background-color: {THEME['input_bg']} !important; 
    border-color: {THEME['border']} !important;
    color: {THEME['text']} !important;
}}

/* Force text colors inside selectboxes and inputs */
[data-baseweb="select"] *, [data-baseweb="input"] *, [data-testid="stSelectbox"] *, [data-testid="stTextInput"] * {{
    color: {THEME['text']} !important;
}}

/* Popovers and dropdown menus */
div[data-baseweb="popover"], div[data-baseweb="popover"] > div, ul[data-testid="stSelectboxVirtualDropdown"] {{ 
    background-color: {THEME['panel']} !important; 
}}
div[data-baseweb="popover"] *, ul[data-testid="stSelectboxVirtualDropdown"] li {{ 
    color: {THEME['text']} !important; 
}}
ul[data-testid="stSelectboxVirtualDropdown"] li:hover {{ 
    background-color: {THEME['bg']} !important; 
}}

/* ---- Tables / dataframes ---- */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    background-color: {THEME['panel']}; border: 1px solid {THEME['border']}; border-radius: 12px;
}}
[data-testid="stDataFrame"] div[role="columnheader"] {{
    background-color: {THEME['table_header']}; color: {THEME['text']}; font-weight: 700;
}}
[data-testid="stDataFrame"] * {{ color: {THEME['text']}; }}

/* ---- Metrics ---- */
[data-testid="stMetric"] {{
    background-color: {THEME['panel']}; border: 1px solid {THEME['border']}; border-radius: 12px;
    padding: 0.6rem 0.9rem; direction: {direction_css}; text-align: {text_align};
}}
[data-testid="stMetric"] * {{ color: {THEME['text']}; text-align: {text_align}; }}

/* ---- Tabs / expanders / forms: full RTL + theme support ---- */
.stTabs [data-baseweb="tab-list"] {{ direction: {direction_css}; }}
.stTabs [data-baseweb="tab"] {{ color: {THEME['subtext']}; }}
.stTabs [aria-selected="true"] {{ color: {THEME['accent_soft']}; }}

[data-testid="stExpander"] {{
    background-color: {THEME['panel']}; border: 1px solid {THEME['border']}; border-radius: 12px;
    direction: {direction_css};
}}
[data-testid="stExpander"] summary {{ color: {THEME['text']}; text-align: {text_align}; }}
[data-testid="stExpander"] * {{ color: {THEME['text']}; }}

[data-testid="stForm"] {{
    background-color: {THEME['panel']}; border: 1px solid {THEME['border']}; border-radius: 14px;
    padding: 1rem 1.2rem; direction: {direction_css};
}}
.stRadio, .stSelectbox, .stMultiSelect, .stTextInput, .stTextArea {{ direction: {direction_css}; text-align: {text_align}; }}
.stRadio > label, .stSelectbox > label, .stMultiSelect > label,
.stTextInput > label, .stTextArea > label {{ color: {THEME['text']}; text-align: {text_align}; width: 100%; }}

/* ---- Multiselect chips ---- */
span[data-baseweb="tag"] {{ background-color: {THEME['accent']} !important; }}

/* ---- Full page direction for the whole app when Arabic is active ---- */
[data-testid="stAppViewContainer"] {{ direction: {direction_css}; }}
.main .block-container {{ direction: {direction_css}; text-align: {text_align}; }}
</style>
""", unsafe_allow_html=True)


def badge_class_inline(level):
    bg, fg = BADGE_COLORS[level]
    return f"background-color:{bg}; color:{fg};"


def risk_badge_text(level):
    level_disp = tr_value(level, L)
    if L == "ar":
        return f"{t('risk_word')} {level_disp}"
    return f"{level_disp} {t('risk_word')}"


# ============================================================================
# HELPER / RENDER FUNCTIONS
# ============================================================================
def kpi_card(icon, label, value):
    st.markdown(f"""
    <div class="st-kpi">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def info_card(icon, title, rows_html):
    st.markdown(f"""
    <div class="safetrak-card">
    <h4>{icon} {title}</h4>
    {rows_html}
    </div>
    """, unsafe_allow_html=True)


def risk_gauge(risk_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={"text": t("risk_score")},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": THEME["accent"]},
            "steps": [
                {"range": [0, 35], "color": "#14532d"},
                {"range": [35, 70], "color": "#713f12"},
                {"range": [70, 100], "color": "#7f1d1d"},
            ],
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": THEME["text"]},
                       height=260, margin=dict(t=40, b=10))
    return fig


def history_df():
    """Load prediction history from SQLite (database/predictions.db) and
    shape it into the same flat table the dashboard/history pages expect.
    Persists across Streamlit restarts because it's read straight from disk."""
    rows = _fetch_history_rows()
    records = []
    for r in rows:
        inputs = r.get("inputs", {}) or {}
        records.append({
            "id": r.get("id"),
            "time": r.get("timestamp"),
            "scenario": r.get("scenario"),
            "severity": r.get("severity"),
            "risk_score": r.get("risk_score"),
            "risk_level": r.get("risk_level"),
            "weather": inputs.get("Weather_conditions", "Unspecified"),
            "vehicle": inputs.get("Type_of_vehicle", "Unspecified"),
            "location": inputs.get("Area_accident_occured", "Unspecified"),
            "experience": inputs.get("Driving_experience", "Unspecified"),
        })
    return pd.DataFrame(records)


def push_history(result, label, context=None):
    """Persist one full prediction (inputs, prediction, probabilities, risk
    score/level, severity, decision, recommendations, AI report, and the raw
    pipeline JSON) to SQLite, then invalidate the cached read. `context` is
    carried inside the stored raw JSON only so that a PDF requested later in
    a different language (see Reports -> PDF Language) can regenerate an
    accurate AI narrative -- it does not affect prediction/decision logic."""
    prediction = result["prediction"]
    decision = result["decision"]
    inputs = prediction["inputs_used"]
    raw_payload = {**result, "context": context}
    db.insert_prediction({
        "timestamp": result["generated_at"],
        "language": L,
        "scenario": label,
        "inputs": inputs,
        "prediction": prediction,
        "probabilities": prediction.get("severity_probabilities", {}),
        "risk_score": prediction["risk_score"],
        "risk_level": prediction["risk_level"],
        "severity": prediction["severity"],
        "decision": decision,
        "recommendations": decision.get("recommended_actions", []),
        "ai_report": result.get("report", ""),
        "raw": raw_payload,
    })
    _fetch_history_rows.clear()


def render_result(result, label, context=None):
    """Shared, professional result renderer used by both Manual Prediction
    and Example Scenarios. Reads Prediction / Decision / Agent JSON only --
    none of the underlying logic is touched."""
    prediction, decision = result["prediction"], result["decision"]
    dtma, sraa, caa = result["dtma"], result["sraa"], result["caa"]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(risk_gauge(prediction["risk_score"]), theme=None, use_container_width=True)
        st.markdown(
            f'<span class="badge" style="{badge_class_inline(prediction["risk_level"])}">'
            f'{risk_badge_text(prediction["risk_level"])}</span>',
            unsafe_allow_html=True,
        )
    with col2:
        probs = " &nbsp;·&nbsp; ".join(
            f"{tr_value(k, L)}: {v}%" for k, v in prediction["severity_probabilities"].items()
        )
        info_card("📊", t("prediction"),
                   f"<b>{t('severity')}:</b> {tr_value(prediction['severity'], L)}<br>"
                   f"<b>{t('confidence')}:</b> {probs}")
        actions = ", ".join(tr_value(a, L) for a in decision["recommended_actions"])
        info_card("🧭", t("recommended_actions"),
                   f"<b>{t('priority')}:</b> {tr_value(decision['priority'], L)} &nbsp;|&nbsp; "
                   f"<b>{t('emergency_level')}:</b> {tr_value(decision['emergency_level'], L)}<br>"
                   f"<b>{t('recommended_actions')}:</b> {actions}")

    tab_overview, tab_reco, tab_ai, tab_raw = st.tabs(
        [t("tab_overview"), t("tab_recommendations"), t("tab_ai"), t("tab_raw")]
    )

    inputs = prediction["inputs_used"]
    with tab_overview:
        info_card("📋", t("exec_summary"),
                   f"<b>{t('severity')}:</b> {tr_value(prediction['severity'], L)} &nbsp;|&nbsp; "
                   f"<b>{t('risk_score')}:</b> {prediction['risk_score']} / 100 "
                   f"({tr_value(prediction['risk_level'], L)})")
        c1, c2, c3 = st.columns(3)
        with c1:
            info_card("🌧", t("contributing_factors"),
                       f"<b>{t('weather')}:</b> {display_label(inputs.get('Weather_conditions', '-'), L)}<br>"
                       f"<b>{t('road_surface')}:</b> {display_label(inputs.get('Road_surface_conditions', '-'), L)}<br>"
                       f"<b>{t('light')}:</b> {display_label(inputs.get('Light_conditions', '-'), L)}<br>"
                       f"<b>{t('location_type')}:</b> {display_label(inputs.get('Area_accident_occured', '-'), L)}")
        with c2:
            info_card("🚑", t("emergency_response"),
                       f"<b>{t('dispatch')}:</b> {tr_value(sraa['police_dispatch'], L)}<br>"
                       f"<b>{t('response_time')}:</b> {tr_value(sraa['estimated_response_time'], L)}")
        with c3:
            info_card("📢", t("citizen_advisory"),
                       f"<b>{t('notification')}:</b> {tr_value(caa['citizen_notification'], L)}<br>"
                       f"<b>{t('safety_tip')}:</b> {tr_value(caa['safety_recommendation'], L)}<br>"
                       f"<b>{t('alt_route')}:</b> {tr_value(caa['alternative_route_suggestion'], L)}")

    with tab_reco:
        c1, c2 = st.columns(2)
        with c1:
            info_card("🚦", t("agent_dtma"),
                       f"<b>{t('signal')}:</b> {tr_value(dtma['signal_optimization'], L)}<br>"
                       f"<b>{t('diversion')}:</b> {tr_value(dtma['traffic_diversion'], L)}<br>"
                       f"<b>{t('congestion')}:</b> {tr_value(dtma['congestion_management'], L)}")
        with c2:
            actions = "<br>".join(f"• {tr_value(a, L)}" for a in decision["recommended_actions"])
            info_card("✅", t("recommended_actions"), actions)

    with tab_ai:
        st.markdown(f"#### 📝 {t('ai_explanation')}")
        risk_factors = ", ".join(display_label(inputs.get(k, "-"), L) for k in (
            "Weather_conditions", "Road_surface_conditions", "Light_conditions") if inputs.get(k))
        actions_short = ", ".join(tr_value(a, L) for a in decision["recommended_actions"][:3])
        exec_bullets = [
            f"<b>{t('current_risk')}:</b> {prediction['risk_score']} / 100 ({tr_value(prediction['risk_level'], L)})",
            f"<b>{t('predicted_severity')}:</b> {tr_value(prediction['severity'], L)}",
            f"<b>{t('main_risk_factors')}:</b> {risk_factors or '-'}",
            f"<b>{t('recommended_action')}:</b> {actions_short or '-'}",
            f"<b>{t('emergency_response')}:</b> {tr_value(sraa['police_dispatch'], L)} "
            f"({tr_value(sraa['estimated_response_time'], L)})",
            f"<b>{t('citizen_advice')}:</b> {tr_value(caa['safety_recommendation'], L)}",
        ]
        bullet_html = "".join(
            f"<li style='margin-bottom:0.45rem;'>{b}</li>" for b in exec_bullets
        )
        st.markdown(
            f'<div class="report-box"><ul style="margin:0; padding-{"right" if RTL else "left"}:1.1rem;">'
            f'{bullet_html}</ul></div>',
            unsafe_allow_html=True,
        )
        with st.expander(t("view_full_report")):
            report_html = result["report"].replace(chr(10), "<br>")
            st.markdown(f'<div class="report-box">{report_html}</div>', unsafe_allow_html=True)

    with tab_raw:
        st.json(result)

    push_history(result, label, context)


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    logo_path = str(config.LOGO_PATH)
    st.markdown('<div class="brand-box">', unsafe_allow_html=True)
    if os.path.exists(logo_path):
        _lc1, _lc2, _lc3 = st.columns([1, 2, 1])
        with _lc2:
            st.image(logo_path, width=72)  # single source of the logo; aspect ratio preserved
    else:
        st.markdown("<div style='font-size:2.2rem;'>🚦</div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="brand-title">{t('app_name')}</div>
        <div class="brand-tag">{t('tagline')}</div>
    </div>
    """, unsafe_allow_html=True)

    NAV_KEYS = ["nav_dashboard", "nav_manual", "nav_scenarios", "nav_history", "nav_reports", "nav_settings"]
    nav_labels = [t(k) for k in NAV_KEYS]
    label_to_key = dict(zip(nav_labels, NAV_KEYS))
    selected_label = st.radio(
        t("nav_group_label"), nav_labels,
        index=NAV_KEYS.index(st.session_state.page),
        label_visibility="collapsed",
    )
    st.session_state.page = label_to_key[selected_label]

    st.divider()
    col_a, col_b = st.columns(2)
    LANG_LABELS = {"English": "en", "العربية": "ar"}
    LANG_LABELS_REV = {v: k for k, v in LANG_LABELS.items()}
    THEME_LABELS = {t("theme_dark"): "Dark", t("theme_light"): "Light"}
    THEME_LABELS_REV = {v: k for k, v in THEME_LABELS.items()}
    with col_a:
        lang_label = st.selectbox(t("language"), list(LANG_LABELS.keys()),
                                   index=list(LANG_LABELS.keys()).index(LANG_LABELS_REV[st.session_state.lang]))
        new_lang = LANG_LABELS[lang_label]
    with col_b:
        theme_label = st.selectbox(t("theme"), list(THEME_LABELS.keys()),
                                    index=list(THEME_LABELS.keys()).index(THEME_LABELS_REV[st.session_state.theme]))
        new_theme = THEME_LABELS[theme_label]
    if new_lang != st.session_state.lang or new_theme != st.session_state.theme:
        st.session_state.lang = new_lang
        st.session_state.theme = new_theme
        st.rerun()

page = st.session_state.page

# ============================================================================
# DASHBOARD
# ============================================================================
if page == "nav_dashboard":
    # Logo lives only in the sidebar (see brand-box above) -- no duplicate
    # logo/brand block in the dashboard hero, per single-logo branding rule.
    st.title(t("dashboard_title"))
    st.caption(t("dashboard_sub"))

    df = history_df()
    if df.empty:
        st.info(t("no_data"))
    else:
        today = pd.to_datetime(df["time"]).dt.date.max()
        today_df = df[pd.to_datetime(df["time"]).dt.date == today]
        latest = df.iloc[-1]

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            kpi_card("📅", t("kpi_total"), len(today_df))
        with c2:
            kpi_card("📊", t("kpi_avg"), f"{df['risk_score'].mean():.1f}")
        with c3:
            kpi_card("🔥", t("kpi_high"), f"{df['risk_score'].max():.1f}")
        with c4:
            kpi_card("🕐", t("kpi_latest"), tr_scenario_label(latest["scenario"], L))
        with c5:
            kpi_card("⚠️", t("kpi_level"), tr_value(latest["risk_level"], L))

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"#### {t('dist_title')}")
            ldf = localized_df(df, ["severity"])
            fig = px.pie(ldf, names=col_label("severity"), hole=0.45,
                         color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": THEME["text"]}, height=320)
            st.plotly_chart(fig, theme=None, use_container_width=True)
        with col2:
            st.markdown(f"#### {t('recent_title')}")
            recent = localized_df(df.tail(8), ["time", "scenario", "severity", "risk_score", "risk_level"])
            st.dataframe(recent, width="stretch", hide_index=True)

# ============================================================================
# MANUAL PREDICTION
# ============================================================================
elif page == "nav_manual":
    st.title(t("manual_title"))
    st.caption(t("manual_sub"))

    with st.form("manual_form"):
        cols = st.columns(2)
        values = {}
        for i, (key, label, options) in enumerate(SCENARIO_INPUTS):
            with cols[i % 2]:
                choice = st.selectbox(field_label(label, L), options,
                                       format_func=lambda v: display_label(v, L), key=key)
                values[key] = choice
        context = st.text_input(t("context_label"), "")
        submitted = st.form_submit_button(t("run_btn"))

    if submitted:
        with st.spinner(t("analyzing")):
            result = run_pipeline(values, context or None, L)
        render_result(result, "Manual", context or None)

# ============================================================================
# EXAMPLE SCENARIOS
# ============================================================================
elif page == "nav_scenarios":
    st.title(t("scenarios_title"))
    st.caption(t("scenarios_sub"))

    scenario_keys = list(SCENARIOS.keys())
    scenario_labels = [scenario_display_name(k, L) for k in scenario_keys]
    label_to_scenario_key = dict(zip(scenario_labels, scenario_keys))
    chosen_label = st.selectbox(t("choose_scenario"), scenario_labels)
    scenario_name = label_to_scenario_key[chosen_label]
    scenario = SCENARIOS[scenario_name]

    info_card("📌", scenario_display_name(scenario_name, L),
              scenario_display_context(scenario["context"], L))

    with st.expander(t("view_inputs")):
        for key, label, _ in SCENARIO_INPUTS:
            st.write(f"**{field_label(label, L)}:** {display_label(scenario['inputs'][key], L)}")

    if st.button(t("run_btn"), key="run_scenario"):
        with st.spinner(t("analyzing")):
            result = run_pipeline(scenario["inputs"], scenario["context"], L)
        render_result(result, scenario_name, scenario["context"])

# ============================================================================
# HISTORY ANALYTICS
# ============================================================================
elif page == "nav_history":
    st.title(t("history_title"))
    st.caption(t("history_sub"))

    df = history_df()
    MIN_ROWS = 3
    if df.empty or len(df) < MIN_ROWS:
        st.info(t("no_data") if df.empty else t("not_enough_data"))
    else:
        df["time"] = pd.to_datetime(df["time"])
        df["date"] = df["time"].dt.date

        def chart_layout(fig, height=320, legend=True):
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font={"color": THEME["text"]}, height=height, showlegend=legend,
                               margin=dict(t=40, r=10))
            fig.update_xaxes(title_standoff=15, automargin=True)
            fig.update_yaxes(title_standoff=15, automargin=True)
            return fig

        def avg_risk_by(col, title_key):
            """Bar chart of mean risk_score grouped by a categorical column,
            translated for display. Hidden entirely (no header, no placeholder)
            when there are fewer than 2 distinct groups to compare."""
            grouped = df.groupby(col)["risk_score"].mean().reset_index()
            if len(grouped) < 2:
                return
            st.markdown(f"#### {t(title_key)}")
            grouped[col] = grouped[col].apply(lambda v: tr_value(v, L))
            grouped = grouped.sort_values("risk_score", ascending=False)
            fig = px.bar(grouped, x=col, y="risk_score",
                         labels={col: col_label(col), "risk_score": col_label("risk_score")},
                         color="risk_score", color_continuous_scale="RdYlGn_r")
            st.plotly_chart(chart_layout(fig, height=300, legend=False), theme=None, use_container_width=True)

        # ---- KPIs ----
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("📊", t("kpi_avg"), f"{df['risk_score'].mean():.1f}")
        with c2:
            kpi_card("🔥", t("kpi_high"), f"{df['risk_score'].max():.1f}")
        with c3:
            kpi_card("🧮", t("avg_daily_risk"), f"{df.groupby('date')['risk_score'].mean().mean():.1f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- Risk Score Trend Over Time ----
        st.markdown(f"#### {t('risk_trend')}")
        fig = px.line(df.sort_values("time"), x="time", y="risk_score", markers=True,
                       labels={"time": col_label("time"), "risk_score": col_label("risk_score")})
        st.plotly_chart(chart_layout(fig), theme=None, use_container_width=True)

        row1c1, row1c2 = st.columns(2)
        with row1c1:
            avg_risk_by("weather", "avg_risk_weather")
        with row1c2:
            avg_risk_by("vehicle", "avg_risk_vehicle")

        row2c1, row2c2 = st.columns(2)
        with row2c1:
            avg_risk_by("location", "avg_risk_location")
        with row2c2:
            avg_risk_by("experience", "avg_risk_experience")

        row3c1, row3c2 = st.columns(2)
        with row3c1:
            st.markdown(f"#### {t('severity_pie')}")
            ldf = localized_df(df, ["severity"])
            fig = px.pie(ldf, names=col_label("severity"), hole=0.55,
                         color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(chart_layout(fig, height=300), theme=None, use_container_width=True)
        with row3c2:
            freq = df.groupby("date").size().reset_index(name="count")
            if len(freq) >= 2:
                st.markdown(f"#### {t('prediction_frequency')}")
                fig = px.bar(freq, x="date", y="count",
                             labels={"date": col_label("date"), "count": col_label("count")})
                st.plotly_chart(chart_layout(fig, height=300, legend=False), theme=None, use_container_width=True)

        row4c1, row4c2 = st.columns(2)
        with row4c1:
            top_scenarios = (df.groupby("scenario")["risk_score"].mean()
                              .sort_values(ascending=False).head(5).reset_index())
            if len(top_scenarios) >= 1:
                st.markdown(f"#### {t('top_high_risk')}")
                top_scenarios["scenario"] = top_scenarios["scenario"].apply(lambda v: tr_scenario_label(v, L))
                fig = px.bar(top_scenarios, x="risk_score", y="scenario", orientation="h",
                             labels={"risk_score": col_label("risk_score"), "scenario": col_label("scenario")},
                             color="risk_score", color_continuous_scale="Reds")
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(chart_layout(fig, height=300, legend=False), theme=None, use_container_width=True)
        with row4c2:
            factor_rows = []
            for col in ("weather", "vehicle", "location", "experience"):
                g = df.groupby(col)["risk_score"].mean()
                if not g.empty:
                    top_val = g.idxmax()
                    factor_rows.append({
                        col_label(col): tr_value(top_val, L),
                        col_label("risk_score"): round(g.max(), 1),
                    })
            if factor_rows:
                st.markdown(f"#### {t('highest_risk_factors')}")
                st.dataframe(pd.DataFrame(factor_rows), width="stretch", hide_index=True)

        st.markdown(f"#### {t('recent_predictions')}")
        recent = localized_df(
            df.sort_values("time", ascending=False),
            ["time", "scenario", "severity", "risk_score", "risk_level", "weather", "vehicle", "location"],
        )
        st.dataframe(recent, width="stretch", hide_index=True)

# ============================================================================
# REPORTS  (search / filter / sort / open / preview / delete / PDF / CSV / JSON)
# ============================================================================
elif page == "nav_reports":
    st.title(t("reports_title"))
    st.caption(t("reports_sub"))

    all_rows = _fetch_history_rows()

    if not all_rows:
        st.info(t("no_reports"))
    else:
        # ---- Toolbar: search, export-all, filters, sort ----
        tcol1, tcol2 = st.columns([2, 1])
        with tcol1:
            search_q = st.text_input(t("search_label"), "", placeholder=t("search_placeholder"),
                                      key="reports_search")
        with tcol2:
            st.markdown("<div style='height:1.85rem;'></div>", unsafe_allow_html=True)
            st.download_button(
                t("export_all_csv"), data=reports.export_all_to_csv(),
                file_name="safetrak_predictions.csv", mime="text/csv",
                key="export_all_csv_btn", width="stretch",
            )

        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        scenario_options = sorted({r.get("scenario") or "Manual" for r in all_rows})
        risk_options = sorted({r.get("risk_level") for r in all_rows if r.get("risk_level")})
        severity_options = sorted({r.get("severity") for r in all_rows if r.get("severity")})
        with fcol1:
            f_scenario = st.multiselect(t("filter_scenario"),
                                         scenario_options,
                                         format_func=lambda v: tr_scenario_label(v, L),
                                         key="reports_f_scenario")
        with fcol2:
            f_risk = st.multiselect(t("filter_risk"), risk_options,
                                     format_func=lambda v: tr_value(v, L), key="reports_f_risk")
        with fcol3:
            f_severity = st.multiselect(t("filter_severity"), severity_options,
                                         format_func=lambda v: tr_value(v, L), key="reports_f_severity")
        with fcol4:
            sort_options = {
                t("sort_newest"): ("timestamp", False),
                t("sort_oldest"): ("timestamp", True),
                t("sort_risk_desc"): ("risk_score", False),
                t("sort_risk_asc"): ("risk_score", True),
            }
            sort_choice = st.selectbox(t("sort_label"), list(sort_options.keys()), key="reports_sort")
            sort_field, sort_asc = sort_options[sort_choice]

        # ---- Search / filter / sort (in-memory over the cached SQLite read) ----
        filtered = all_rows
        if search_q:
            q = search_q.lower()
            filtered = [
                r for r in filtered
                if q in (r.get("scenario") or "").lower()
                or q in (r.get("severity") or "").lower()
                or q in (r.get("risk_level") or "").lower()
                or q in (r.get("ai_report") or "").lower()
            ]
        if f_scenario:
            filtered = [r for r in filtered if (r.get("scenario") or "Manual") in f_scenario]
        if f_risk:
            filtered = [r for r in filtered if r.get("risk_level") in f_risk]
        if f_severity:
            filtered = [r for r in filtered if r.get("severity") in f_severity]
        filtered = sorted(filtered, key=lambda r: (r.get(sort_field) or 0), reverse=not sort_asc)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(f"{len(filtered)} {t('reports_count_suffix')}")

        if not filtered:
            st.info(t("no_reports_match"))

        # ---- Report cards ----
        for r in filtered:
            rid = r.get("id")
            inputs = r.get("inputs", {}) or {}
            level = r.get("risk_level") or "Low"
            open_key, preview_key, pdf_key = f"open_{rid}", f"preview_{rid}", f"pdf_bytes_{rid}"

            st.markdown(f"""
            <div class="safetrak-card">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                    <div>
                        <h4 style="margin-bottom:0.2rem;">📄 {tr_scenario_label(r.get('scenario') or 'Manual', L)}
                            <span style="color:{THEME['subtext']}; font-weight:500;">#{rid}</span></h4>
                        <div style="color:{THEME['subtext']}; font-size:0.85rem;">{r.get('timestamp', '-')}</div>
                    </div>
                    <span class="badge" style="{badge_class_inline(level)}">{risk_badge_text(level)}</span>
                </div>
                <div style="margin-top:0.6rem;">
                    <b>{t('severity')}:</b> {tr_value(r.get('severity'), L)} &nbsp;|&nbsp;
                    <b>{t('risk_score')}:</b> {r.get('risk_score')} / 100 &nbsp;|&nbsp;
                    <b>{t('weather')}:</b> {display_label(inputs.get('Weather_conditions', '-'), L)}
                </div>
            </div>
            """, unsafe_allow_html=True)

            bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns(5)
            with bcol1:
                if st.button(t("preview_report"), key=f"btn_preview_{rid}", width="stretch"):
                    st.session_state[preview_key] = not st.session_state.get(preview_key, False)
            with bcol2:
                if st.button(t("open_report"), key=f"btn_open_{rid}", width="stretch"):
                    st.session_state[open_key] = not st.session_state.get(open_key, False)
            with bcol3:
                st.download_button(
                    t("download_json"), data=reports.export_record_to_json(r),
                    file_name=f"safetrak_report_{rid}.json", mime="application/json",
                    key=f"dl_json_{rid}", width="stretch",
                )
            with bcol4:
                # PDF language is chosen independently of the current UI
                # language -- it never changes automatically with it. The
                # picker defaults to the language the prediction itself was
                # made in, but the user is always free to pick the other one.
                pdf_lang_default = "ar" if (r.get("language") or "en") == "ar" else "en"
                pdf_lang_choice = st.radio(
                    t("pdf_language"), list(LANG_LABELS.keys()),
                    index=list(LANG_LABELS.keys()).index(LANG_LABELS_REV[pdf_lang_default]),
                    key=f"pdf_lang_{rid}", horizontal=True, label_visibility="collapsed",
                )
                pdf_lang = LANG_LABELS[pdf_lang_choice]
                # PDF bytes are only generated on click -- never automatically.
                if st.button(t("download_pdf"), key=f"gen_pdf_{rid}", width="stretch"):
                    st.session_state[pdf_key] = reports.generate_pdf_report(r, pdf_lang)
                    st.session_state[f"{pdf_key}_lang"] = pdf_lang
                if pdf_key in st.session_state:
                    dl_lang = st.session_state.get(f"{pdf_key}_lang", pdf_lang_default)
                    st.download_button(
                        t("save_pdf"), data=st.session_state[pdf_key],
                        file_name=f"safetrak_report_{rid}_{dl_lang}.pdf", mime="application/pdf",
                        key=f"dl_pdf_{rid}", width="stretch",
                    )
            with bcol5:
                if st.button(t("delete_report"), key=f"btn_delete_{rid}", width="stretch"):
                    db.delete_prediction(rid)
                    _fetch_history_rows.clear()
                    st.session_state.pop(pdf_key, None)
                    st.session_state.pop(f"{pdf_key}_lang", None)
                    st.success(t("report_deleted"))
                    st.rerun()

            if st.session_state.get(preview_key):
                with st.expander(t("preview_report"), expanded=True):
                    actions = ", ".join(tr_value(a, L) for a in (r.get("recommendations") or []))
                    info_card("📊", t("exec_summary"),
                               f"<b>{t('severity')}:</b> {tr_value(r.get('severity'), L)} &nbsp;|&nbsp; "
                               f"<b>{t('risk_score')}:</b> {r.get('risk_score')} / 100 &nbsp;|&nbsp; "
                               f"<b>{col_label('risk_level')}:</b> {tr_value(r.get('risk_level'), L)}<br>"
                               f"<b>{t('recommended_actions')}:</b> {actions or '-'}")

            if st.session_state.get(open_key):
                with st.expander(t("open_report"), expanded=True):
                    with st.expander(t("view_full_report")):
                        st.markdown(f"##### 📝 {t('ai_explanation')}")
                        report_html = (r.get("ai_report") or "").replace(chr(10), "<br>")
                        st.markdown(f'<div class="report-box">{report_html}</div>', unsafe_allow_html=True)
                    st.markdown(f"##### {t('tab_raw')}")
                    st.json(r.get("raw") or {})

            st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)

# ============================================================================
# SETTINGS
# ============================================================================
else:
    st.title(t("settings_title"))
    st.caption(t("settings_sub"))

    current_lang_name = "English" if L == "en" else "العربية"
    theme_name = t("theme_dark") if st.session_state.theme == "Dark" else t("theme_light")
    total_predictions = db.count_predictions()

    c1, c2, c3 = st.columns(3)
    with c1:
        info_card("🌐", t("settings_lang"), f"<b>{current_lang_name}</b>")
    with c2:
        info_card("🎨", t("settings_theme"), f"<b>{theme_name}</b>")
    with c3:
        info_card("🔢", t("settings_predictions"), f"<b>{total_predictions}</b>")

    c4, c5 = st.columns(2)
    with c4:
        db_note = ("SQLite -- connected") if L == "en" else "SQLite -- متصلة"
        info_card("🗄", t("settings_db"), f"<b>{db_note}</b>")
    with c5:
        info_card("🏷", t("settings_version"), f"<b>{APP_VERSION}</b>")

    st.markdown("<div style='height:0.4rem;'></div>", unsafe_allow_html=True)
    if st.button(t("clear_history")):
        db.delete_all_history()
        _fetch_history_rows.clear()
        st.success(t("history_cleared"))
