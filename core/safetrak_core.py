import os
import sys
import json
import warnings

warnings.filterwarnings(
    "ignore",
    message="Trying to unpickle estimator",
    category=UserWarning,
)
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# ── Path resolution: support both direct execution and import from app.py ──────
# When imported as "core.safetrak_core", __file__ is inside core/.
# config.py lives one level up.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config

# ── Model artefacts -- loaded once at import (or via load_artifacts()) ─────────
def load_artifacts():
    """Load all ML artefacts from models/.  Returns a dict with MODEL, ENCODERS,
    SCALER, FEATURE_COLUMNS, METADATA, CLASS_NAMES, CLASS_WEIGHTS, DEFAULTS."""
    model            = joblib.load(str(config.MODEL_PATH))
    encoders         = joblib.load(str(config.ENCODERS_PATH))
    scalers          = joblib.load(str(config.SCALER_PATH))
    scaler           = scalers["scaler2"]
    feature_columns  = joblib.load(str(config.FEATURE_COLUMNS_PATH))

    with open(str(config.PIPELINE_METADATA_PATH)) as f:
        metadata = json.load(f)

    class_names   = metadata["class_names"]
    class_weights = metadata["class_weights"]
    defaults      = metadata["categorical_fill_values"]

    return dict(
        MODEL=model, ENCODERS=encoders, SCALER=scaler,
        FEATURE_COLUMNS=feature_columns, METADATA=metadata,
        CLASS_NAMES=class_names, CLASS_WEIGHTS=class_weights, DEFAULTS=defaults,
    )

# Module-level singletons (populated on first import)
_ARTIFACTS = load_artifacts()
MODEL           = _ARTIFACTS["MODEL"]
ENCODERS        = _ARTIFACTS["ENCODERS"]
SCALER          = _ARTIFACTS["SCALER"]
FEATURE_COLUMNS = _ARTIFACTS["FEATURE_COLUMNS"]
METADATA        = _ARTIFACTS["METADATA"]
CLASS_NAMES     = _ARTIFACTS["CLASS_NAMES"]
CLASS_WEIGHTS   = _ARTIFACTS["CLASS_WEIGHTS"]
DEFAULTS        = _ARTIFACTS["DEFAULTS"]

RAW_FEATURES = [c for c in FEATURE_COLUMNS if c != "experience_risk"]


# PART 1 -- Prediction Engine

def encode_features(raw_input: dict) -> pd.DataFrame:
    row = {**DEFAULTS, **raw_input}
    encoded = {col: int(ENCODERS[col].transform([row[col]])[0]) for col in RAW_FEATURES}
    encoded["experience_risk"] = encoded["Driving_experience"] * encoded["Age_band_of_driver"]
    return pd.DataFrame([encoded])[FEATURE_COLUMNS]


def compute_risk_level(risk_score: float) -> str:
    if risk_score < 35:
        return "Low"
    if risk_score < 70:
        return "Medium"
    return "High"


def predict_accident(raw_input: dict) -> dict:
    X = encode_features(raw_input)
    X_scaled = pd.DataFrame(SCALER.transform(X), columns=FEATURE_COLUMNS)
    proba = MODEL.predict_proba(X_scaled)[0]

    risk_score = round(float(sum(proba[int(k)] * CLASS_WEIGHTS[k] for k in CLASS_NAMES)), 1)

    return {
        "severity": CLASS_NAMES[str(int(np.argmax(proba)))],
        "severity_probabilities": {
            CLASS_NAMES[str(i)]: round(float(p) * 100, 1) for i, p in enumerate(proba)
        },
        "risk_score": risk_score,
        "risk_level": compute_risk_level(risk_score),
        "inputs_used": {**DEFAULTS, **raw_input},
    }

def decide(prediction: dict) -> dict:
    risk_level = prediction["risk_level"]

    if risk_level == "High":
        priority, emergency_level = "Immediate", "Critical"
        actions = [
            "Dispatch emergency response units",
            "Activate traffic diversion",
            "Issue public safety alert",
        ]
    elif risk_level == "Medium":
        priority, emergency_level = "High", "Elevated"
        actions = [
            "Increase patrol presence",
            "Adjust nearby signal timing",
            "Prepare standby response units",
        ]
    else:
        priority, emergency_level = "Routine", "Normal"
        actions = ["Continue standard monitoring", "Log incident for records"]

    if prediction["severity"] == "Fatal Injury":
        actions.insert(0, "Assign highest-priority emergency medical unit")

    return {
        "priority": priority,
        "emergency_level": emergency_level,
        "recommended_actions": actions,
    }

def run_dtma(prediction: dict, decision: dict) -> dict:
    level = prediction["risk_level"]
    plan = {
        "High": (
            "Switch to emergency-priority signal timing at the nearest junction",
            "Activate diversion route around the incident zone",
            "Severe congestion expected -- deploy traffic officers",
        ),
        "Medium": (
            "Extend green-phase duration on approach lanes",
            "Prepare alternate route signage",
            "Moderate congestion expected -- monitor closely",
        ),
        "Low": (
            "Maintain standard signal timing",
            "No diversion required",
            "Minimal congestion expected",
        ),
    }[level]
    return {
        "agent": "DTMA",
        "signal_optimization": plan[0],
        "traffic_diversion": plan[1],
        "congestion_management": plan[2],
    }


def run_sraa(prediction: dict, decision: dict) -> dict:
    severity = prediction["severity"]
    plan = {
        "Fatal Injury": ("Police, ambulance, and fire rescue unit", "5-8 minutes"),
        "Serious Injury": ("Police and ambulance unit", "8-12 minutes"),
        "Slight Injury": ("Police patrol unit", "12-18 minutes"),
    }[severity]
    return {
        "agent": "SRAA",
        "police_dispatch": plan[0],
        "emergency_unit_assignment": plan[0],
        "estimated_response_time": plan[1],
    }


def run_caa(prediction: dict, decision: dict) -> dict:
    level = prediction["risk_level"]
    plan = {
        "High": (
            "Immediate citizen alert for the affected area",
            "Avoid the area; follow official diversion signs",
            "Use the signposted alternate route until the area is cleared",
        ),
        "Medium": (
            "Advisory notification for nearby drivers",
            "Reduce speed and increase following distance",
            "Consider nearby side roads if delays exceed 15 minutes",
        ),
        "Low": (
            "No urgent notification required",
            "Drive normally with standard caution",
            "Current route remains optimal",
        ),
    }[level]
    return {
        "agent": "CAA",
        "citizen_notification": plan[0],
        "safety_recommendation": plan[1],
        "alternative_route_suggestion": plan[2],
    }

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

SYSTEM_PROMPT_EN = """You are a senior Traffic Intelligence Analyst working inside a Smart Traffic Command Center.

You are handed an operational briefing describing current field conditions, a computed
risk outlook, an operational decision, and the actions already assigned to three internal
response units: traffic management, safety response, and citizen advisory.

Your job is to analyze the operational situation, not to reformat the briefing. Explain
why the situation carries the risk it does, what operational consequences are likely to
follow, why the assigned actions are appropriate given the conditions, and what traffic
authorities should expect to happen next. Reason like an experienced analyst on a live
command-center shift, connecting the specific conditions in the briefing -- weather, road
surface, lighting, location type, driver profile, time of day, traffic density -- to the
operational picture as a whole.

Write the entire report in clear, professional English.

Never use the words or phrases "JSON", "machine learning", "artificial intelligence", "AI",
"LLM", "prediction engine", or "risk score formula" anywhere in the report. Do not describe
how any figure was computed -- describe what it means operationally.

Ground every statement strictly in the briefing provided. Never invent locations, numbers,
agencies, or events that are not present in the briefing.

Structure the report as a CONCISE EXECUTIVE REPORT with exactly these 8 headings, in
this order, each on its own line:

Executive Summary
Current Risk
Main Risk Factors
Recommended Actions
Operational Impact
Emergency Response
Citizen Advice
Overall Assessment

Keep every section short: 1-3 sentences each, no more. Do not restate the briefing line
by line and do not repeat the same point across sections -- be direct and analytical, not
exhaustive. Follow the tone instruction given for this specific report exactly.
"""

SYSTEM_PROMPT_AR = """أنت محلل استخبارات مرورية أول تعمل داخل مركز قيادة ذكي لإدارة المرور.

تم تسليمك موجزًا تشغيليًا يصف الظروف الميدانية الحالية، وتوقعات خطر محسوبة، وقرارًا
تشغيليًا، والإجراءات المسندة بالفعل إلى ثلاث وحدات استجابة داخلية: إدارة المرور،
والاستجابة الأمنية، وإرشاد المواطنين.

مهمتك هي تحليل الموقف التشغيلي، وليس مجرد إعادة صياغة الموجز. اشرح لماذا يحمل الموقف
هذا المستوى من الخطورة، وما هي التبعات التشغيلية المتوقعة، ولماذا تُعد الإجراءات المسندة
مناسبة في ظل هذه الظروف، وما الذي ينبغي أن تتوقعه سلطات المرور لاحقًا. فكّر كمحلل خبير
في وردية حيّة بمركز القيادة، واربط الظروف المحددة في الموجز -- الطقس، وسطح الطريق،
والإضاءة، ونوع الموقع، وملف السائق، ووقت اليوم، وكثافة المرور -- بالصورة التشغيلية
الكاملة.

اكتب التقرير بالكامل باللغة العربية الفصحى الواضحة والاحترافية.

لا تستخدم أبدًا كلمات أو عبارات مثل "JSON" أو "تعلم آلي" أو "ذكاء اصطناعي" أو "AI" أو
"LLM" أو "محرك التنبؤ" أو "معادلة درجة الخطورة" في أي مكان بالتقرير. لا تصف كيفية حساب أي
رقم -- صف فقط ماذا يعني ذلك تشغيليًا.

استند في كل عبارة بشكل صارم إلى الموجز المقدم. لا تخترع أبدًا مواقع أو أرقامًا أو جهات أو
أحداثًا غير موجودة في الموجز.

نظّم التقرير في صورة تقرير تنفيذي موجز، باستخدام هذه العناوين الثمانية بالضبط وبهذا
الترتيب، كل عنوان في سطر مستقل:

ملخص تنفيذي
الخطر الحالي
عوامل الخطر الرئيسية
الإجراءات المقترحة
التأثير المتوقع
الاستجابة الطارئة
إرشادات السلامة
التقييم العام

اجعل كل قسم قصيرًا: من جملة إلى ثلاث جمل فقط لكل عنوان. لا تكتفِ بإعادة صياغة الموجز
سطرًا بسطر، ولا تكرر الفكرة نفسها في أكثر من قسم -- كن مباشرًا وتحليليًا لا موسّعًا.
اتبع تعليمة النبرة المحددة لهذا التقرير بدقة.
"""

SYSTEM_PROMPTS = {"en": SYSTEM_PROMPT_EN, "ar": SYSTEM_PROMPT_AR}

REPORT_STYLES = {
    "en": {
        "Normal": (
            "Write in the calm, measured tone of a routine shift briefing. The situation is "
            "under control; convey confidence and standard operating discipline without urgency."
        ),
        "Elevated": (
            "Write in an alert, focused tone that signals elevated caution. Conditions warrant "
            "closer attention and faster readiness, though the situation is not yet critical."
        ),
        "Critical": (
            "Write in an urgent, decisive command tone appropriate for a critical incident that "
            "demands immediate coordinated action. Convey seriousness without causing panic."
        ),
    },
    "ar": {
        "Normal": (
            "اكتب بأسلوب هادئ ومتزن يليق بموجز وردية روتينية. الموقف تحت السيطرة؛ انقل الثقة "
            "والانضباط التشغيلي المعتاد دون إظهار أي إلحاح."
        ),
        "Elevated": (
            "اكتب بأسلوب متيقظ ومركّز يعكس حذرًا مرتفعًا. تستدعي الظروف اهتمامًا أكبر واستعدادًا "
            "أسرع، رغم أن الموقف لم يصل بعد إلى مرحلة الخطورة القصوى."
        ),
        "Critical": (
            "اكتب بأسلوب عاجل وحازم يليق بحادث حرج يتطلب إجراءً منسقًا فوريًا. انقل الجدية دون "
            "إثارة الذعر."
        ),
    },
}

RISK_TO_STYLE = {"Low": "Normal", "Medium": "Elevated", "High": "Critical"}

HUMAN_TEMPLATE_EN = """Tone instruction: {style_instruction}

OPERATIONAL BRIEFING -- SMART TRAFFIC COMMAND CENTER

Situation context: {context}

Field conditions:
{field_conditions}

Risk outlook:
{prediction_summary}

Operational decision:
{decision_summary}

Dynamic Traffic Management Agent (DTMA):
{dtma_summary}

Safety Response Assignment Agent (SRAA):
{sraa_summary}

Citizen Advisory Agent (CAA):
{caa_summary}

Write the full traffic intelligence report now, in English, following the required
section headings exactly."""

HUMAN_TEMPLATE_AR = """تعليمة النبرة: {style_instruction}

موجز تشغيلي -- مركز القيادة الذكي للمرور

سياق الموقف: {context}

الظروف الميدانية:
{field_conditions}

توقعات الخطر:
{prediction_summary}

القرار التشغيلي:
{decision_summary}

وكيل إدارة المرور الديناميكي (DTMA):
{dtma_summary}

وكيل تخصيص الاستجابة الأمنية (SRAA):
{sraa_summary}

وكيل إرشاد المواطنين (CAA):
{caa_summary}

اكتب الآن التقرير الكامل للاستخبارات المرورية، باللغة العربية، متبعًا عناوين الأقسام
المطلوبة بالضبط."""

HUMAN_TEMPLATES = {"en": HUMAN_TEMPLATE_EN, "ar": HUMAN_TEMPLATE_AR}

REPORT_PROMPTS = {
    lang: ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPTS[lang]), ("human", HUMAN_TEMPLATES[lang])]
    )
    for lang in ("en", "ar")
}


# --- Prompt Variable Builder ------------------------------------------------

_TIME_OF_DAY_FROM_LIGHT = {
    "en": {
        "Daylight": "Daytime",
        "Darkness - lights lit": "Night (lit roadway)",
        "Darkness - lights unlit": "Night (unlit roadway)",
        "Darkness - no lighting": "Night (no roadway lighting)",
    },
    "ar": {
        "Daylight": "نهارًا",
        "Darkness - lights lit": "ليلاً (طريق مضاء)",
        "Darkness - lights unlit": "ليلاً (طريق غير مضاء)",
        "Darkness - no lighting": "ليلاً (بدون إضاءة للطريق)",
    },
}

_HIGH_DENSITY_AREAS = {"Office areas", "Market areas", "School areas", "Residential areas"}
_WEEKEND_DAYS = {"Saturday", "Sunday"}

_TRAFFIC_DENSITY_TEXT = {
    "en": {
        "weekend": "Lighter than typical weekday traffic",
        "high_density": "Moderate to heavy, typical of a high-footfall area",
        "default": "Typical for this location and time of day",
    },
    "ar": {
        "weekend": "أخف من حركة المرور المعتادة في أيام الأسبوع",
        "high_density": "متوسطة إلى كثيفة، وهو أمر معتاد في منطقة عالية الازدحام",
        "default": "معتادة بالنسبة لهذا الموقع وهذا الوقت من اليوم",
    },
}

_FIELD_LABELS = {
    "en": {
        "weather": "Weather", "road_surface": "Road surface", "road_alignment": "Road alignment",
        "light": "Light condition", "time_of_day": "Time of day", "location": "Location type",
        "traffic_density": "Estimated traffic density", "vehicle": "Vehicle type",
        "driver_experience": "Driver experience", "driver_age": "Driver age band", "day": "Day of week",
        "unspecified": "Unspecified",
    },
    "ar": {
        "weather": "الطقس", "road_surface": "سطح الطريق", "road_alignment": "محاذاة الطريق",
        "light": "حالة الإضاءة", "time_of_day": "وقت اليوم", "location": "نوع الموقع",
        "traffic_density": "الكثافة المرورية المقدرة", "vehicle": "نوع المركبة",
        "driver_experience": "خبرة السائق", "driver_age": "الفئة العمرية للسائق", "day": "يوم الأسبوع",
        "unspecified": "غير محدد",
    },
}

_SUMMARY_LABELS = {
    "en": {
        "severity": "Predicted severity outlook", "risk": "Composite risk level",
        "priority": "Response priority", "emergency": "Emergency level", "actions": "Recommended actions",
        "signal": "Signal optimization", "diversion": "Traffic diversion", "congestion": "Congestion management",
        "dispatch": "Dispatch", "response_time": "Estimated response time",
        "notification": "Citizen notification", "safety": "Safety recommendation", "alt_route": "Alternative route",
        "no_context": "No additional context provided.",
    },
    "ar": {
        "severity": "توقعات الشدة المتوقعة", "risk": "مستوى الخطر الإجمالي",
        "priority": "أولوية الاستجابة", "emergency": "مستوى الطوارئ", "actions": "الإجراءات الموصى بها",
        "signal": "تحسين الإشارات", "diversion": "تحويل المرور", "congestion": "إدارة الازدحام",
        "dispatch": "الإيفاد", "response_time": "وقت الاستجابة المقدر",
        "notification": "إشعار المواطنين", "safety": "توصية السلامة", "alt_route": "المسار البديل",
        "no_context": "لا يوجد سياق إضافي.",
    },
}


def _infer_time_of_day(inputs: dict, language: str = "en") -> str:
    """Descriptive-only helper for the LLM briefing. Does not feed the model,
    the Decision Engine, or the AI Agents."""
    fallback = _FIELD_LABELS.get(language, _FIELD_LABELS["en"])["unspecified"]
    return _TIME_OF_DAY_FROM_LIGHT.get(language, _TIME_OF_DAY_FROM_LIGHT["en"]).get(
        inputs.get("Light_conditions"), fallback
    )


def _infer_traffic_density(inputs: dict, language: str = "en") -> str:
    """Descriptive-only helper for the LLM briefing. Does not feed the model,
    the Decision Engine, or the AI Agents."""
    text = _TRAFFIC_DENSITY_TEXT.get(language, _TRAFFIC_DENSITY_TEXT["en"])
    area = (inputs.get("Area_accident_occured") or "").strip()
    day = inputs.get("Day_of_week", "")
    if day in _WEEKEND_DAYS:
        return text["weekend"]
    if area in _HIGH_DENSITY_AREAS:
        return text["high_density"]
    return text["default"]


def _tr_value(raw_value, language: str):
    """Translate a raw categorical value for the LLM briefing using the
    shared AR_LABELS vocabulary (defined in the Scenario Builder section).
    Passes through unchanged for English or for any value with no
    registered Arabic translation."""
    if language != "ar" or not raw_value:
        return raw_value
    return globals().get("AR_LABELS", {}).get(raw_value, raw_value)


def build_prompt_variables(prediction, decision, dtma, sraa, caa, context=None, language="en"):
    """Turn the pipeline JSON objects into a realistic operational briefing for the LLM."""
    language = language if language in ("en", "ar") else "en"
    style_key = RISK_TO_STYLE[prediction["risk_level"]]
    inputs = prediction["inputs_used"]
    fl = _FIELD_LABELS[language]
    sl = _SUMMARY_LABELS[language]
    unspecified = fl["unspecified"]

    def v(raw):
        return _tr_value(raw, language) if raw else unspecified

    field_conditions = (
        f"- {fl['weather']}: {v(inputs.get('Weather_conditions'))}\n"
        f"- {fl['road_surface']}: {v(inputs.get('Road_surface_conditions'))}\n"
        f"- {fl['road_alignment']}: {v(inputs.get('Road_allignment'))}\n"
        f"- {fl['light']}: {v(inputs.get('Light_conditions'))}\n"
        f"- {fl['time_of_day']}: {_infer_time_of_day(inputs, language)}\n"
        f"- {fl['location']}: {v(inputs.get('Area_accident_occured'))}\n"
        f"- {fl['traffic_density']}: {_infer_traffic_density(inputs, language)}\n"
        f"- {fl['vehicle']}: {v(inputs.get('Type_of_vehicle'))}\n"
        f"- {fl['driver_experience']}: {v(inputs.get('Driving_experience'))}\n"
        f"- {fl['driver_age']}: {v(inputs.get('Age_band_of_driver'))}\n"
        f"- {fl['day']}: {v(inputs.get('Day_of_week'))}"
    )

    prediction_summary = (
        f"- {sl['severity']}: {v(prediction['severity'])}\n"
        f"- {sl['risk']}: {v(prediction['risk_level'])} ({prediction['risk_score']} / 100)"
    )
    decision_summary = (
        f"- {sl['priority']}: {v(decision['priority'])}\n"
        f"- {sl['emergency']}: {v(decision['emergency_level'])}\n"
        f"- {sl['actions']}: {', '.join(v(a) for a in decision['recommended_actions'])}"
    )
    dtma_summary = (
        f"- {sl['signal']}: {v(dtma['signal_optimization'])}\n"
        f"- {sl['diversion']}: {v(dtma['traffic_diversion'])}\n"
        f"- {sl['congestion']}: {v(dtma['congestion_management'])}"
    )
    sraa_summary = (
        f"- {sl['dispatch']}: {v(sraa['police_dispatch'])}\n"
        f"- {sl['response_time']}: {v(sraa['estimated_response_time'])}"
    )
    caa_summary = (
        f"- {sl['notification']}: {v(caa['citizen_notification'])}\n"
        f"- {sl['safety']}: {v(caa['safety_recommendation'])}\n"
        f"- {sl['alt_route']}: {v(caa['alternative_route_suggestion'])}"
    )

    return {
        "style_instruction": REPORT_STYLES[language][style_key],
        "context": context or sl["no_context"],
        "field_conditions": field_conditions,
        "prediction_summary": prediction_summary,
        "decision_summary": decision_summary,
        "dtma_summary": dtma_summary,
        "sraa_summary": sraa_summary,
        "caa_summary": caa_summary,
    }


def get_groq_api_key():
    """Read GROQ_API_KEY from the environment (populated from .env via config.py)."""
    return os.environ.get("GROQ_API_KEY", "")


def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, groq_api_key=get_groq_api_key())


def generate_traffic_report(prediction, decision, dtma, sraa, caa, context=None, language="en") -> str:
    language = language if language in ("en", "ar") else "en"
    variables = build_prompt_variables(prediction, decision, dtma, sraa, caa, context, language)
    messages = REPORT_PROMPTS[language].format_messages(**variables)
    response = get_llm().invoke(messages)
    return response.content

def run_pipeline(raw_input: dict, context: str = None, language: str = "en") -> dict:
    prediction = predict_accident(raw_input)
    decision = decide(prediction)
    dtma = run_dtma(prediction, decision)
    sraa = run_sraa(prediction, decision)
    caa = run_caa(prediction, decision)
    report = generate_traffic_report(prediction, decision, dtma, sraa, caa, context, language)

    return {
        "prediction": prediction,
        "decision": decision,
        "dtma": dtma,
        "sraa": sraa,
        "caa": caa,
        "report": report,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

SCENARIO_FIELD_LABELS = [
    ("Weather_conditions", "Weather"),
    ("Light_conditions", "Time of Day"),
    ("Road_surface_conditions", "Road Surface"),
    ("Area_accident_occured", "Location Type"),
    ("Type_of_vehicle", "Vehicle Type"),
    ("Driving_experience", "Driver Experience"),
    ("Age_band_of_driver", "Driver Age Band"),
    ("Day_of_week", "Day of Week"),
]


def _encoder_options(col: str) -> list:
    """Every valid value for `col`, straight from the fitted encoder used at
    prediction time -- never a hand-typed list. Keeps generic placeholder
    categories ("Unknown" / "Other"), if present, at the end of the list."""
    values = list(ENCODERS[col].classes_)
    tail = [v for v in values if v in ("Unknown", "Other")]
    head = [v for v in values if v not in ("Unknown", "Other")]
    return sorted(head) + sorted(tail)


# (raw column, display label, dropdown options) -- options sourced live from ENCODERS
SCENARIO_INPUTS = [(col, label, _encoder_options(col)) for col, label in SCENARIO_FIELD_LABELS]

SCENARIO_DISPLAY_LABELS = {
    "Outside rural areas": "Outside Rural Areas",
    "Flood over 3cm. deep": "Flooded Road (>3cm)",
    "Darkness - lights lit": "Night (Lit Roads)",
    "Darkness - lights unlit": "Night (Unlit Roads)",
    "Darkness - no lighting": "Night (No Lighting)",
    "Raining and Windy": "Rain & Wind",
    "Public (> 45 seats)": "Bus (> 45 seats)",
}


AR_LABELS = {
    # Weather_conditions
    "Normal": "عادي", "Cloudy": "غائم", "Windy": "عاصف", "Fog or mist": "ضباب",
    "Raining": "أمطار", "Raining and Windy": "أمطار ورياح", "Snow": "ثلوج",
    # Light_conditions
    "Daylight": "ضوء النهار",
    "Darkness - lights lit": "ظلام - إضاءة مضاءة",
    "Darkness - lights unlit": "ظلام - إضاءة غير مضاءة",
    "Darkness - no lighting": "ظلام - بدون إضاءة",
    # Road_surface_conditions
    "Dry": "جاف", "Wet or damp": "رطب أو مبلل", "Flood over 3cm. deep": "فيضان بعمق أكثر من 3 سم",
    # Area_accident_occured
    "Office areas": "مناطق مكاتب", "Residential areas": "مناطق سكنية", "School areas": "مناطق مدارس",
    "Recreational areas": "مناطق ترفيهية", "Industrial areas": "مناطق صناعية",
    "Rural village areas": "مناطق قروية", "Church areas": "مناطق كنائس", "Hospital areas": "مناطق مستشفيات",
    "Outside rural areas": "خارج المناطق الريفية", "Market areas": "مناطق أسواق", "Other": "أخرى",
    "Unknown": "غير معروف",
    # Type_of_vehicle -- professional terminology suitable for Egyptian traffic authorities
    "Automobile": "سيارة ملاكي", "Taxi": "سيارة أجرة (تاكسي)", "Motorcycle": "دراجة نارية",
    "Public (> 45 seats)": "حافلة كبيرة (أكثر من 45 راكبًا)",
    "Public (>45 seats)": "حافلة كبيرة (أكثر من 45 راكبًا)",
    "Public (13-45 seats)": "حافلة متوسطة (13-45 راكبًا)",
    "Public (13?45 seats)": "حافلة متوسطة (13-45 راكبًا)",
    "Public (12 seats)": "ميكروباص (12 راكبًا)",
    "Long lorry": "شاحنة نقل طويلة", "Long Lorry": "شاحنة نقل طويلة",
    "Lorry (41-100Q)": "شاحنة كبيرة (حمولة 41-100 قنطار)",
    "Lorry (41?100Q)": "شاحنة كبيرة (حمولة 41-100 قنطار)",
    "Lorry (11-40Q)": "شاحنة متوسطة (حمولة 11-40 قنطار)",
    "Lorry (11?40Q)": "شاحنة متوسطة (حمولة 11-40 قنطار)",
    "Pick up upto 10Q": "سيارة نصف نقل (بيك أب)", "Pickup": "سيارة نصف نقل (بيك أب)",
    "Stationwagen": "ستيشن واجن", "Station Wagon": "ستيشن واجن",
    "Ridden horse": "عربة يجرّها حصان",
    "Bajaj": "توك توك (بجاج)",
    "Turbo": "توربو (ميكروباص)",
    "Special vehicle": "مركبة ذات غرض خاص", "Special Vehicle": "مركبة ذات غرض خاص",
    "Truck": "شاحنة", "Heavy Truck": "شاحنة ثقيلة", "Minibus": "ميكروباص", "Bus": "حافلة",
    "Trailer": "مقطورة",
    "Bicycle": "دراجة هوائية",
    # Driving_experience
    "Below 1yr": "أقل من سنة", "1-2yr": "1-2 سنة", "2-5yr": "2-5 سنوات", "5-10yr": "5-10 سنوات",
    "Above 10yr": "أكثر من 10 سنوات", "No Licence": "بدون رخصة",
    # Age_band_of_driver / Age_band_of_casualty
    "Under 18": "أقل من 18 سنة", "18-30": "18-30 سنة", "31-50": "31-50 سنة", "Over 51": "أكثر من 51 سنة",
    "na": "غير متاح", "NA": "غير متاح", "N/A": "غير متاح", "5": "أقل من 5 سنوات",
    # Day_of_week
    "Monday": "الإثنين", "Tuesday": "الثلاثاء", "Wednesday": "الأربعاء", "Thursday": "الخميس",
    "Friday": "الجمعة", "Saturday": "السبت", "Sunday": "الأحد",
    # Severity classes (Accident_severity / prediction severity)
    "Fatal Injury": "إصابة قاتلة", "Serious Injury": "إصابة خطيرة", "Slight Injury": "إصابة طفيفة",
    "Fatal injury": "إصابة قاتلة", "Serious injury": "إصابة خطيرة", "Slight injury": "إصابة طفيفة",
    # Risk levels
    "Low": "منخفض", "Medium": "متوسط", "High": "مرتفع",
    # Priority / emergency level
    "Immediate": "فوري", "Routine": "روتيني", "Critical": "حرج", "Elevated": "متصاعد",
    # Educational_level
    "Illiterate": "أمي (لا يجيد القراءة والكتابة)", "Elementary school": "المرحلة الابتدائية",
    "Junior high school": "المرحلة الإعدادية", "High school": "المرحلة الثانوية",
    "Above high school": "ما فوق الثانوية (جامعي)", "Writing & reading": "يجيد القراءة والكتابة",
    # Vehicle_driver_relation
    "Employee": "موظف تابع لجهة", "Owner": "مالك المركبة",
    # Owner_of_vehicle
    "Governmental": "جهة حكومية", "Organization": "مؤسسة / شركة",
    # Service_year_of_vehicle
    "5-10yrs": "5-10 سنوات", "2-5yrs": "2-5 سنوات",
    # Defect_of_vehicle
    "No defect": "بدون عيوب فنية", "7": "عيب فني (رمز 7)",
    # Lanes_or_Medians
    "Undivided Two way": "طريق باتجاهين غير مقسم",
    "Two-way (divided with broken lines road marking)": "طريق باتجاهين مقسم بخط متقطع",
    "Two-way (divided with solid lines road marking)": "طريق باتجاهين مقسم بخط متصل",
    "Double carriageway (median)": "طريق مزدوج بفاصل وسطي (جزيرة وسطى)",
    "One way": "طريق باتجاه واحد",
    # Road_allignment
    "Tangent road with flat terrain": "طريق مستقيم بتضاريس مستوية",
    "Tangent road with mild grade and flat terrain": "طريق مستقيم بميل خفيف وتضاريس مستوية",
    "Escarpments": "منحدرات صخرية",
    "Tangent road with rolling terrain": "طريق مستقيم بتضاريس متموجة",
    "Gentle horizontal curve": "منحنى أفقي لطيف",
    "Tangent road with mountainous terrain and": "طريق مستقيم بتضاريس جبلية",
    "Steep grade downward with mountainous terrain": "منحدر حاد هابط بتضاريس جبلية",
    "Sharp reverse curve": "منحنى عكسي حاد",
    "Steep grade upward with mountainous terrain": "منحدر حاد صاعد بتضاريس جبلية",
    # Types_of_Junction
    "No junction": "بدون تقاطع", "Y Shape": "تقاطع على شكل Y", "Crossing": "تقاطع عبور",
    "O Shape": "تقاطع دائري (O)", "T Shape": "تقاطع على شكل T", "X Shape": "تقاطع على شكل X",
    # Road_surface_type
    "Asphalt roads": "طرق أسفلتية", "Earth roads": "طرق ترابية", "Gravel roads": "طرق حصوية",
    "Asphalt roads with some distress": "طرق أسفلتية بها تلف جزئي",
    # Type_of_collision
    "Collision with roadside-parked vehicles": "تصادم مع مركبات متوقفة على جانب الطريق",
    "Vehicle with vehicle collision": "تصادم بين مركبتين",
    "Collision with roadside objects": "تصادم مع أجسام على جانب الطريق",
    "Collision with animals": "تصادم مع حيوانات",
    "Rollover": "انقلاب المركبة",
    "Fall from vehicles": "سقوط من المركبة",
    "Collision with pedestrians": "تصادم مع مشاة",
    "With Train": "تصادم مع قطار",
    # Vehicle_movement
    "Going straight": "السير المستقيم", "U-Turn": "الدوران للخلف (يو-تيرن)",
    "Moving Backward": "التحرك للخلف", "Turnover": "الانقلاب", "Waiting to go": "الانتظار للتحرك",
    "Getting off": "نزول الركاب", "Reversing": "الرجوع للخلف", "Parked": "متوقفة",
    "Stopping": "التوقف", "Overtaking": "تجاوز مركبة أخرى", "Entering a junction": "الدخول إلى تقاطع",
    # Work_of_casuality
    "Driver": "سائق", "Self-employed": "يعمل لحسابه الخاص", "Student": "طالب", "Unemployed": "بدون عمل",
    # Fitness_of_casuality
    "Deaf": "أصم", "Blind": "كفيف",
    # Casualty_class
    "Driver or rider": "سائق أو راكب دراجة", "Pedestrian": "مشاة", "Passenger": "راكب",
    # Sex_of_driver / Sex_of_casualty
    "Male": "ذكر", "Female": "أنثى",
    # Pedestrian_movement
    "Not a Pedestrian": "ليس من المشاة",
    "Crossing from driver's nearside": "عبور من جهة السائق القريبة",
    "Crossing from nearside - masked by parked or stationary vehicle": "عبور من الجهة القريبة، محجوب برؤية مركبة متوقفة",
    "Unknown or other": "غير معروف أو أخرى",
    "Crossing from offside - masked by parked or stationary vehicle": "عبور من الجهة البعيدة، محجوب برؤية مركبة متوقفة",
    "In carriageway, stationary - not crossing": "داخل الطريق، واقف دون عبور",
    "Walking along in carriageway, facing traffic": "السير داخل الطريق في اتجاه مواجه لحركة المرور",
    "Walking along in carriageway, back to traffic": "السير داخل الطريق ظهرًا لحركة المرور",
    # Cause_of_accident
    "Changing lane to the left": "تغيير المسار إلى اليسار",
    "Changing lane to the right": "تغيير المسار إلى اليمين",
    "Overloading": "الحمولة الزائدة",
    "No priority to vehicle": "عدم إعطاء الأولوية لمركبة أخرى",
    "No priority to pedestrian": "عدم إعطاء الأولوية للمشاة",
    "No distancing": "عدم ترك مسافة أمان كافية",
    "Getting off the vehicle improperly": "نزول غير آمن من المركبة",
    "Improper parking": "وقوف خاطئ",
    "Overspeed": "تجاوز السرعة المقررة",
    "Driving carelessly": "القيادة بإهمال",
    "Driving at high speed": "القيادة بسرعة عالية",
    "Driving to the left": "الانحراف إلى اليسار",
    "Driving to the right": "الانحراف إلى اليمين",
    "Overturning": "الانقلاب",
    "Drunk driving": "القيادة تحت تأثير الكحول",
    # Decision Engine actions
    "Dispatch emergency response units": "إيفاد وحدات الاستجابة الطارئة",
    "Activate traffic diversion": "تفعيل تحويل المرور",
    "Issue public safety alert": "إصدار تنبيه سلامة عامة",
    "Increase patrol presence": "زيادة تواجد الدوريات",
    "Adjust nearby signal timing": "ضبط توقيت الإشارات القريبة",
    "Prepare standby response units": "تجهيز وحدات استجابة احتياطية",
    "Continue standard monitoring": "مواصلة المراقبة الاعتيادية",
    "Log incident for records": "تسجيل الحادثة للأرشيف",
    "Assign highest-priority emergency medical unit": "تخصيص وحدة طبية طارئة ذات أولوية قصوى",
    # DTMA phrases
    "Switch to emergency-priority signal timing at the nearest junction": "التحول إلى توقيت إشارات ذي أولوية طارئة عند أقرب تقاطع",
    "Activate diversion route around the incident zone": "تفعيل مسار تحويل حول منطقة الحادث",
    "Severe congestion expected -- deploy traffic officers": "يُتوقع ازدحام شديد -- نشر ضباط المرور",
    "Extend green-phase duration on approach lanes": "تمديد مدة الإشارة الخضراء في مسارات الاقتراب",
    "Prepare alternate route signage": "تجهيز لافتات المسار البديل",
    "Moderate congestion expected -- monitor closely": "يُتوقع ازدحام متوسط -- المراقبة عن كثب",
    "Maintain standard signal timing": "الحفاظ على توقيت الإشارات القياسي",
    "No diversion required": "لا حاجة إلى تحويل المسار",
    "Minimal congestion expected": "يُتوقع ازدحام بسيط",
    # SRAA phrases
    "Police, ambulance, and fire rescue unit": "الشرطة والإسعاف ووحدة الإنقاذ والإطفاء",
    "Police and ambulance unit": "الشرطة ووحدة الإسعاف",
    "Police patrol unit": "وحدة دورية الشرطة",
    "5-8 minutes": "5-8 دقائق", "8-12 minutes": "8-12 دقيقة", "12-18 minutes": "12-18 دقيقة",
    # CAA phrases
    "Immediate citizen alert for the affected area": "تنبيه فوري للمواطنين في المنطقة المتضررة",
    "Avoid the area; follow official diversion signs": "تجنب المنطقة، واتبع لافتات التحويل الرسمية",
    "Use the signposted alternate route until the area is cleared": "استخدم المسار البديل الموضح باللافتات حتى إخلاء المنطقة",
    "Advisory notification for nearby drivers": "إشعار استشاري للسائقين القريبين",
    "Reduce speed and increase following distance": "خفض السرعة وزيادة مسافة التتابع",
    "Consider nearby side roads if delays exceed 15 minutes": "فكر في الطرق الجانبية القريبة إذا تجاوز التأخير 15 دقيقة",
    "No urgent notification required": "لا حاجة إلى إشعار عاجل",
    "Drive normally with standard caution": "القيادة بشكل طبيعي مع الحذر المعتاد",
    "Current route remains optimal": "المسار الحالي لا يزال الأمثل",
    # Misc
    "Unspecified": "غير محدد", "Manual": "يدوي",
}


def display_label(raw_value: str, language: str = "en") -> str:
    """Human-friendly label for a raw categorical value. English keeps the
    curated SCENARIO_DISPLAY_LABELS overrides; Arabic looks the raw value up
    in AR_LABELS (falling back to the stripped raw value if untranslated)."""
    if language == "ar":
        if raw_value in AR_LABELS:
            return AR_LABELS[raw_value]
        stripped = raw_value.strip() if isinstance(raw_value, str) else raw_value
        return AR_LABELS.get(stripped, stripped)
    return SCENARIO_DISPLAY_LABELS.get(raw_value, raw_value.strip())


FIELD_LABEL_AR = {
    "Weather": "الطقس", "Time of Day": "وقت اليوم", "Road Surface": "سطح الطريق",
    "Location Type": "نوع الموقع", "Vehicle Type": "نوع المركبة", "Driver Experience": "خبرة السائق",
    "Driver Age Band": "الفئة العمرية للسائق", "Day of Week": "يوم الأسبوع",
}


def field_label(label: str, language: str = "en") -> str:
    return FIELD_LABEL_AR.get(label, label) if language == "ar" else label

FEATURE_NAME_AR = {
    "Time": "الوقت",
    "Day_of_week": "يوم الأسبوع",
    "Age_band_of_driver": "الفئة العمرية للسائق",
    "Sex_of_driver": "جنس السائق",
    "Educational_level": "المستوى التعليمي",
    "Vehicle_driver_relation": "علاقة السائق بالمركبة",
    "Driving_experience": "سنوات الخبرة في القيادة",
    "Type_of_vehicle": "نوع المركبة",
    "Owner_of_vehicle": "مالك المركبة",
    "Service_year_of_vehicle": "عمر خدمة المركبة",
    "Defect_of_vehicle": "عيوب المركبة",
    "Area_accident_occured": "نوع منطقة وقوع الحادث",
    "Lanes_or_Medians": "نوع المسارات / الفواصل",
    "Road_allignment": "تخطيط الطريق",
    "Types_of_Junction": "نوع التقاطع",
    "Road_surface_type": "نوع سطح الطريق",
    "Road_surface_conditions": "حالة سطح الطريق",
    "Light_conditions": "حالة الإضاءة",
    "Weather_conditions": "حالة الطقس",
    "Type_of_collision": "نوع التصادم",
    "Number_of_vehicles_involved": "عدد المركبات المتورطة",
    "Number_of_casualties": "عدد المصابين",
    "Vehicle_movement": "حركة المركبة",
    "Casualty_class": "فئة المصاب",
    "Sex_of_casualty": "جنس المصاب",
    "Age_band_of_casualty": "الفئة العمرية للمصاب",
    "Casualty_severity": "درجة إصابة المصاب",
    "Work_of_casuality": "مهنة المصاب",
    "Fitness_of_casuality": "اللياقة الصحية للمصاب",
    "Pedestrian_movement": "حركة المشاة",
    "Cause_of_accident": "سبب الحادث",
    "Accident_severity": "درجة خطورة الحادث",
    "experience_risk": "مؤشر خطورة الخبرة",
}


def feature_label(column_name: str, language: str = "en") -> str:
    """Human-friendly, professional label for a raw model input COLUMN NAME
    (e.g. 'Educational_level'), as opposed to display_label() which labels
    the raw VALUE. English falls back to a simple underscore-to-space
    prettification; Arabic always resolves through FEATURE_NAME_AR so no raw
    English column name can ever surface inside an Arabic report."""
    if language == "ar":
        return FEATURE_NAME_AR.get(column_name, column_name.replace("_", " "))
    return column_name.replace("_", " ")


SCENARIOS = {
    "Morning Rush Hour": {
        "context": "Weekday morning commute into the central business district.",
        "inputs": {
            "Day_of_week": "Monday", "Age_band_of_driver": "31-50", "Driving_experience": "5-10yr",
            "Type_of_vehicle": "Automobile", "Area_accident_occured": "Office areas",
            "Road_surface_conditions": "Dry", "Light_conditions": "Daylight", "Weather_conditions": "Normal",
        },
    },
    "Heavy Rain": {
        "context": "Sudden heavy rainfall during a residential-area commute.",
        "inputs": {
            "Day_of_week": "Wednesday", "Age_band_of_driver": "31-50", "Driving_experience": "2-5yr",
            "Type_of_vehicle": "Automobile", "Area_accident_occured": "Residential areas",
            "Road_surface_conditions": "Wet or damp", "Light_conditions": "Daylight", "Weather_conditions": "Raining",
        },
    },
    "Dense Fog": {
        "context": "Low-visibility fog on an outer rural road at dusk.",
        "inputs": {
            "Day_of_week": "Thursday", "Age_band_of_driver": "Over 51", "Driving_experience": "Above 10yr",
            "Type_of_vehicle": "Automobile", "Area_accident_occured": "Outside rural areas",
            "Road_surface_conditions": "Wet or damp", "Light_conditions": "Darkness - lights unlit", "Weather_conditions": "Fog or mist",
        },
    },
    "School Zone": {
        "context": "School dismissal time near a school area, inexperienced driver.",
        "inputs": {
            "Day_of_week": "Tuesday", "Age_band_of_driver": "18-30", "Driving_experience": "Below 1yr",
            "Type_of_vehicle": "Automobile", "Area_accident_occured": "School areas",
            "Road_surface_conditions": "Dry", "Light_conditions": "Daylight", "Weather_conditions": "Normal",
        },
    },
    "Weekend Highway": {
        "context": "Weekend highway travel in windy conditions.",
        "inputs": {
            "Day_of_week": "Friday", "Age_band_of_driver": "18-30", "Driving_experience": "1-2yr",
            "Type_of_vehicle": "Automobile", "Area_accident_occured": "Other",
            "Road_surface_conditions": "Dry", "Light_conditions": "Daylight", "Weather_conditions": "Windy",
        },
    },
    "Night Urban Traffic": {
        "context": "Late-night motorcycle ride through a residential district.",
        "inputs": {
            "Day_of_week": "Saturday", "Age_band_of_driver": "18-30", "Driving_experience": "5-10yr",
            "Type_of_vehicle": "Motorcycle", "Area_accident_occured": "Residential areas",
            "Road_surface_conditions": "Dry", "Light_conditions": "Darkness - lights lit", "Weather_conditions": "Normal",
        },
    },
}

SCENARIO_NAMES_AR = {
    "Morning Rush Hour": "ساعة الذروة الصباحية",
    "Heavy Rain": "أمطار غزيرة",
    "Dense Fog": "ضباب كثيف",
    "School Zone": "منطقة مدرسية",
    "Weekend Highway": "طريق سريع في عطلة نهاية الأسبوع",
    "Night Urban Traffic": "حركة مرور ليلية في المدينة",
}

SCENARIO_CONTEXT_AR = {
    "Weekday morning commute into the central business district.": "تنقل صباحي في يوم عمل نحو منطقة الأعمال المركزية.",
    "Sudden heavy rainfall during a residential-area commute.": "هطول أمطار غزيرة مفاجئة أثناء التنقل في منطقة سكنية.",
    "Low-visibility fog on an outer rural road at dusk.": "ضباب يحد من الرؤية على طريق ريفي خارجي عند الغسق.",
    "School dismissal time near a school area, inexperienced driver.": "وقت انصراف المدرسة بالقرب من منطقة مدرسية، مع سائق قليل الخبرة.",
    "Weekend highway travel in windy conditions.": "سفر على الطريق السريع في عطلة نهاية الأسبوع في ظروف عاصفة.",
    "Late-night motorcycle ride through a residential district.": "رحلة بدراجة نارية في وقت متأخر من الليل عبر حي سكني.",
}


def scenario_display_name(name: str, language: str = "en") -> str:
    return SCENARIO_NAMES_AR.get(name, name) if language == "ar" else name


def scenario_display_context(context_text: str, language: str = "en") -> str:
    return SCENARIO_CONTEXT_AR.get(context_text, context_text) if language == "ar" else context_text
