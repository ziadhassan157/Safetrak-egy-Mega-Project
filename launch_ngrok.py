"""
launch_ngrok.py -- Optional ngrok tunnel launcher for SAFE TRAK EGY.

Usage (after filling in .env):
    python launch_ngrok.py

This script:
  1. Starts the Streamlit app in a subprocess (port 8501)
  2. Opens an ngrok HTTP tunnel to that port
  3. Prints the public URL

The main app works perfectly without this script. Run it only when you
want to expose the local server to the public internet.

Requires:
  - pyngrok (pip install pyngrok)
  - NGROK_AUTH_TOKEN set in .env  (https://dashboard.ngrok.com/get-started/your-authtoken)
"""

import os
import sys
import time
import subprocess

# ── Load .env ──────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed; falling back to OS environment variables.")

NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
PORT = 8501

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set. AI-generated reports will not work.")

try:
    from pyngrok import ngrok
except ImportError:
    print("pyngrok is not installed. Run: pip install pyngrok")
    sys.exit(1)

# ── Authenticate ngrok (optional -- free tier works without a token but
#    with bandwidth limits) ─────────────────────────────────────────────────────
if NGROK_AUTH_TOKEN:
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    print("ngrok authenticated.")
else:
    print("No NGROK_AUTH_TOKEN found; using unauthenticated free tier.")

# ── Kill any existing tunnels / Streamlit processes from a previous run ────────
ngrok.kill()

# ── Start Streamlit ────────────────────────────────────────────────────────────
streamlit_env = os.environ.copy()
streamlit_env["GROQ_API_KEY"] = GROQ_API_KEY

print(f"Starting Streamlit on port {PORT} ...")
streamlit_process = subprocess.Popen(
    [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(PORT),
        "--server.headless", "true",
        "--server.address", "localhost",
    ],
    cwd=_HERE,
    env=streamlit_env,
)

# Give Streamlit time to boot before opening the tunnel
time.sleep(8)

# ── Open ngrok tunnel ──────────────────────────────────────────────────────────
public_url = ngrok.connect(PORT, "http")
print(f"\nSAFE TRAK EGY is live at: {public_url}")
print("Open the URL above in your browser.")
print("Press Ctrl+C to stop the server and close the tunnel.\n")

try:
    streamlit_process.wait()
except KeyboardInterrupt:
    print("\nShutting down ...")
    ngrok.kill()
    streamlit_process.terminate()
    print("Done.")
