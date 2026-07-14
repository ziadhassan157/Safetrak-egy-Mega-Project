# SAFE TRAK EGY 🚦

## About
The Automated Content Generation System will create a pipeline that uses generative models (such as GPT or GANs) to automatically generate content. The system will be trained on a specific dataset, fine-tuned using advanced techniques such as attention mechanisms, and integrated into an automated pipeline.
---

## Features

| Feature | Details |
|---|---|
| **ML Prediction** | LightGBM model trained on the Egyptian RTA dataset. Outputs severity class, severity probabilities, and a composite risk score (0–100). |
| **Decision Engine** | Rule-based dispatcher that converts a risk level into priority, emergency level, and recommended actions. |
| **AI Agents (3)** | DTMA (traffic management), SRAA (safety response), CAA (citizen advisory) — deterministic rule outputs. |
| **GenAI Report** | LangChain + Groq (`llama-3.3-70b-versatile`) generates a structured executive report in English or Arabic. |
| **SQLite History** | Every prediction persisted in `database/predictions.db` — survives restarts. |
| **PDF Export** | Professional bilingual PDF (ReportLab) with Arabic reshaping + bidi. |
| **CSV / JSON Export** | Bulk CSV and per-record JSON. |
| **Localization** | Full Arabic & English UI with RTL layout. |
| **Dark / Light themes** | CSS-injected Streamlit theming. |
| **Ngrok support** | Optional public tunnel via `launch_ngrok.py`. |

---

## Project Structure

```
SafeTrack-Egy/
├── app.py                  # Streamlit dashboard (entry point)
├── config.py               # Centralised paths + .env loader
├── launch_ngrok.py         # Optional ngrok tunnel launcher
├── requirements.txt
├── .env.example            # Template — copy to .env and fill in keys
├── .gitignore
│
├── core/
│   ├── __init__.py
│   ├── safetrak_core.py    # Prediction engine, decision engine, AI agents, pipeline
│   ├── safetrak_db.py      # SQLite CRUD layer
│   └── safetrak_reports.py # PDF / CSV / JSON reporting
│
├── models/                 # ML artefacts (never committed to git if large)
│   ├── model.pkl
│   ├── encoders.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   ├── label_encoder.pkl
│   └── pipeline_metadata.json
│
├── assets/
│   └── logo.png            # Brand logo (+ optional Arabic font TTFs)
│
├── database/               # Auto-created at runtime
│   └── predictions.db
│
├── reports/                # Auto-created at runtime (generated PDFs land here)
└── exports/                # Auto-created at runtime (CSV exports)
```

---

## Quick Start

### 1. Clone / navigate to the project

```bash
cd SafeTrack-Egy
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

```bash
copy .env.example .env   # Windows
# or
cp .env.example .env     # macOS / Linux
```

Then open `.env` and fill in your Groq API key:

```
GROQ_API_KEY=gsk_...
NGROK_AUTH_TOKEN=         # optional — only needed for launch_ngrok.py
```

Get a free Groq key at https://console.groq.com/keys

### 5. Run the dashboard

```bash
streamlit run app.py
```

The browser opens automatically at `http://localhost:8501`.

---

## Optional: Public URL via ngrok

```bash
python launch_ngrok.py
```

This starts Streamlit and opens an ngrok HTTP tunnel. The public URL is printed to the console. Fill in `NGROK_AUTH_TOKEN` in `.env` to avoid free-tier bandwidth limits.

---

## Arabic Font Support (optional but recommended for PDF)

For proper Arabic PDF rendering, download the Noto Naskh Arabic font and place it in `assets/`:

```
assets/NotoNaskhArabic-Regular.ttf
assets/NotoNaskhArabic-Bold.ttf
```

Download from: https://fonts.google.com/noto/specimen/Noto+Naskh+Arabic

Without the font files, Arabic PDFs still build (falling back to Helvetica) but glyph shaping may be imperfect.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | **Yes** | Groq API key for LLM report generation |
| `NGROK_AUTH_TOKEN` | No | ngrok auth token (for `launch_ngrok.py` only) |

---

## Tech Stack

- **Streamlit** — dashboard framework  
- **LightGBM + scikit-learn** — ML prediction  
- **LangChain + Groq** — GenAI reports  
- **SQLite** — persistent history  
- **ReportLab + arabic-reshaper + python-bidi** — bilingual PDF generation  
- **Plotly** — interactive charts  
- **python-dotenv** — `.env` file loading  
