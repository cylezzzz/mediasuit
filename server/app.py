from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from server.routes.universal_generate import router as universal_router
from pathlib import Path

app = FastAPI()

# Statische Dateien (HTML, Assets)
ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
OUT = ROOT / "outputs"
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
app.mount("/outputs", StaticFiles(directory=OUT), name="outputs")

# API
app.include_router(universal_router, prefix="/api")
