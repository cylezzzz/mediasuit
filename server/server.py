from __future__ import annotations

import os, json, time, platform, socket
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

BOOT_TS = time.time()

# ------------------------------------------------------------
# App-Grundgerüst
# ------------------------------------------------------------
app = FastAPI(title="MediaSuite Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lokal ok
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Static Web Mount: G:\mediasuit\web -> /
# ------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.getenv(
    "WEB_DIR",
    os.path.abspath(os.path.join(HERE, "..", "web"))
)

if not os.path.isdir(WEB_DIR):
    # Fallback: aktuelles Verzeichnis, damit / nicht 404 wirft
    WEB_DIR = os.getcwd()

# Root liefert index.html aus WEB_DIR
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

# ------------------------------------------------------------
# Ollama Settings (für LLM-Aufrufe)
# ------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_CHAT = f"{OLLAMA_URL}/api/chat"
OLLAMA_TAGS = f"{OLLAMA_URL}/api/tags"
MODEL       = os.getenv("MODEL", "llama3.1:latest")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))
MAX_TOKENS  = int(os.getenv("MAX_TOKENS", "128"))

async def _ollama_chat(system: str, user: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS}
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(OLLAMA_CHAT, json=payload)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Ollama error: {r.text}")
        data = r.json()
        return (data.get("message", {}) or {}).get("content", "").strip()

# ------------------------------------------------------------
# Unified Endpoint: Analyse + Antwort in einem Call
# ------------------------------------------------------------
class RespondReq(BaseModel):
    analysis_system: str
    analysis_user: str
    reply_system: str
    reply_user: str

@app.post("/api/respond")
async def respond(req: RespondReq):
    # 1) Analyse (erwartet kompaktes JSON vom Modell)
    raw_ana = await _ollama_chat(req.analysis_system, req.analysis_user)
    analysis = None
    try:
        analysis = json.loads(raw_ana)
    except Exception:
        try:
            s = raw_ana.find("{"); e = raw_ana.rfind("}")
            if s != -1 and e != -1 and e > s:
                analysis = json.loads(raw_ana[s:e+1])
        except Exception:
            analysis = None

    # 2) Antwort
    reply_text = await _ollama_chat(req.reply_system, req.reply_user)

    return {
        "ok": True,
        "analysis": analysis,
        "raw_analysis": raw_ana,
        "reply": reply_text,
    }

# ------------------------------------------------------------
# Basis-API für deine UI (keine 404 mehr)
# ------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"ok": True, "status": "running", "uptime_sec": int(time.time() - BOOT_TS)}

@app.get("/api/status")
async def status():
    return {
        "ok": True,
        "host": socket.gethostname(),
        "ip": socket.gethostbyname(socket.gethostname()) if hasattr(socket, "gethostname") else "127.0.0.1",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "uptime_sec": int(time.time() - BOOT_TS),
        "model": MODEL,
        "ollama_url": OLLAMA_URL,
    }

@app.get("/api/ops")
async def ops():
    # Hier könntest du laufende Jobs ausgeben; vorerst leer
    return {"ok": True, "ops": []}

@app.get("/api/models")
async def models():
    # Versuche echte Ollama-Modelle zu holen; fallbacks sicher
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(OLLAMA_TAGS)
            if r.status_code == 200:
                data = r.json()  # {"models":[{"name": "...", ...}]}
                names = [m.get("name") for m in data.get("models", []) if m.get("name")]
                return {"ok": True, "models": names}
    except Exception:
        pass
    # Fallback
    return {"ok": True, "models": [MODEL]}

# ------------------------------------------------------------
# Launcher-Hook
# ------------------------------------------------------------
def build_app():
    """Von start.py importiert."""
    return app

# Direktstart zum Testen (optional)
if __name__ == "__main__":
    import uvicorn
    print(f"Serving WEB from: {WEB_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=3000)
