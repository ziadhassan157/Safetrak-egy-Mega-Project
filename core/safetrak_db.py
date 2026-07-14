import os
import sys
import json
import sqlite3
from contextlib import contextmanager

# ── Path resolution ────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config

# Re-export the directory constants so other modules (safetrak_reports, app)
# that did "import safetrak_db as db" and used "db.ASSETS_DIR" etc. continue
# to work without modification.
BASE_DIR     = str(config.BASE_DIR)
DATABASE_DIR = str(config.DATABASE_DIR)
REPORTS_DIR  = str(config.REPORTS_DIR)
EXPORTS_DIR  = str(config.EXPORTS_DIR)
ASSETS_DIR   = str(config.ASSETS_DIR)
DB_PATH      = str(config.DB_PATH)

_JSON_COLUMNS = (
    "inputs_json", "prediction_json", "probabilities_json",
    "decision_json", "recommendations_json", "raw_json",
)


def ensure_folders():
    """Create database/, reports/, exports/, and assets/ if missing."""
    config.ensure_runtime_dirs()


@contextmanager
def get_connection():
    ensure_folders()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create database/predictions.db and the predictions table if they
    don't already exist. Safe to call repeatedly."""
    ensure_folders()
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp           TEXT NOT NULL,
                language            TEXT,
                scenario            TEXT,
                inputs_json         TEXT,
                prediction_json     TEXT,
                probabilities_json  TEXT,
                risk_score          REAL,
                risk_level          TEXT,
                severity            TEXT,
                decision_json       TEXT,
                recommendations_json TEXT,
                ai_report           TEXT,
                raw_json            TEXT
            )
            """
        )
        # Indexes to keep history/search/filter queries fast as the table grows.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_scenario ON predictions(scenario)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_risk_level ON predictions(risk_level)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_language ON predictions(language)")
    return DB_PATH


def _row_to_dict(row) -> dict:
    """Convert a sqlite3.Row into a plain dict, expanding the JSON columns
    back into nested Python objects."""
    d = dict(row)
    for col in _JSON_COLUMNS:
        raw = d.pop(col, None)
        key = col[:-5]  # strip "_json"
        try:
            d[key] = json.loads(raw) if raw else {}
        except (TypeError, json.JSONDecodeError):
            d[key] = {}
    return d


def insert_prediction(record: dict) -> int:
    """Insert one prediction record and return its new row id.

    Expected keys in `record`:
        timestamp, language, scenario, inputs, prediction, probabilities,
        risk_score, risk_level, severity, decision, recommendations,
        ai_report, raw
    """
    init_db()
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (timestamp, language, scenario, inputs_json, prediction_json,
                 probabilities_json, risk_score, risk_level, severity,
                 decision_json, recommendations_json, ai_report, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("timestamp"),
                record.get("language"),
                record.get("scenario"),
                json.dumps(record.get("inputs", {}), ensure_ascii=False),
                json.dumps(record.get("prediction", {}), ensure_ascii=False),
                json.dumps(record.get("probabilities", {}), ensure_ascii=False),
                record.get("risk_score"),
                record.get("risk_level"),
                record.get("severity"),
                json.dumps(record.get("decision", {}), ensure_ascii=False),
                json.dumps(record.get("recommendations", []), ensure_ascii=False),
                record.get("ai_report", ""),
                json.dumps(record.get("raw", {}), ensure_ascii=False),
            ),
        )
        return cur.lastrowid


def load_history(limit: int = None) -> list:
    """Return all predictions (most recent first) as a list of dicts."""
    init_db()
    query = "SELECT * FROM predictions ORDER BY id DESC"
    if limit:
        query += " LIMIT ?"
        with get_connection() as conn:
            rows = conn.execute(query, (int(limit),)).fetchall()
    else:
        with get_connection() as conn:
            rows = conn.execute(query).fetchall()
    return [_row_to_dict(r) for r in rows]


def delete_prediction(pred_id: int) -> bool:
    """Delete a single prediction by id. Returns True if a row was removed."""
    init_db()
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
    return cur.rowcount > 0


def delete_all_history() -> int:
    """Wipe every prediction. Returns the number of rows removed."""
    init_db()
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM predictions")
    return cur.rowcount


def search_history(keyword: str) -> list:
    """Case-insensitive search across scenario, severity, risk level, and
    the AI-generated report text."""
    init_db()
    like = f"%{keyword}%"
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM predictions
            WHERE scenario LIKE ? OR severity LIKE ? OR risk_level LIKE ?
               OR ai_report LIKE ?
            ORDER BY id DESC
            """,
            (like, like, like, like),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def filter_history(language: str = None, scenario: str = None, risk_level: str = None,
                    severity: str = None, date_from: str = None, date_to: str = None) -> list:
    """Filter predictions by any combination of fields. All args optional;
    omitted fields are not filtered on."""
    init_db()
    clauses, params = [], []
    if language:
        clauses.append("language = ?"); params.append(language)
    if scenario:
        clauses.append("scenario = ?"); params.append(scenario)
    if risk_level:
        clauses.append("risk_level = ?"); params.append(risk_level)
    if severity:
        clauses.append("severity = ?"); params.append(severity)
    if date_from:
        clauses.append("timestamp >= ?"); params.append(date_from)
    if date_to:
        clauses.append("timestamp <= ?"); params.append(date_to)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"SELECT * FROM predictions {where} ORDER BY id DESC"
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def count_predictions() -> int:
    """Total number of stored predictions."""
    init_db()
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()
    return row["c"] if row else 0
