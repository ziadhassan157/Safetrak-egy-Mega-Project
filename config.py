import os
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)   # does not overwrite existing env vars
except ImportError:
    pass   # python-dotenv not installed -- rely on the OS environment

# ── Root / base directory (this file's parent) ────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ── Sub-directories ───────────────────────────────────────────────────────────
MODELS_DIR    = BASE_DIR / "models"
ASSETS_DIR    = BASE_DIR / "assets"
DATABASE_DIR  = BASE_DIR / "database"
REPORTS_DIR   = BASE_DIR / "reports"
EXPORTS_DIR   = BASE_DIR / "exports"
CORE_DIR      = BASE_DIR / "core"

# ── Artifact paths ────────────────────────────────────────────────────────────
MODEL_PATH            = MODELS_DIR / "model.pkl"
ENCODERS_PATH         = MODELS_DIR / "encoders.pkl"
SCALER_PATH           = MODELS_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH  = MODELS_DIR / "feature_columns.pkl"
LABEL_ENCODER_PATH    = MODELS_DIR / "label_encoder.pkl"
PIPELINE_METADATA_PATH = MODELS_DIR / "pipeline_metadata.json"

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = DATABASE_DIR / "predictions.db"

# ── Assets ────────────────────────────────────────────────────────────────────
LOGO_PATH = ASSETS_DIR / "logo.png"
AR_FONT_REGULAR = ASSETS_DIR / "NotoNaskhArabic-Regular.ttf"
AR_FONT_BOLD    = ASSETS_DIR / "NotoNaskhArabic-Bold.ttf"

# ── API keys (read from environment / .env) ───────────────────────────────────
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "")


def ensure_runtime_dirs():
    """Create database/, reports/, exports/, and assets/ if they do not exist.
    Safe to call multiple times; never raises if they already exist."""
    for d in (DATABASE_DIR, REPORTS_DIR, EXPORTS_DIR, ASSETS_DIR):
        d.mkdir(parents=True, exist_ok=True)
