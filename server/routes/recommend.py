
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import shutil, re

from huggingface_hub import snapshot_download

from ..settings import load_settings

router = APIRouter()

class Rec(BaseModel):
    type: str            # "image" | "video"
    name: str
    repo_id: str         # huggingface repo id
    nsfw_capable: bool = False
    backend: str = "diffusers_dir"
    recommended_use: Optional[str] = None
    tags: list[str] = []
    subfolder: Optional[str] = None

RECS: list[Rec] = [
    Rec(type="image", name="Realistic Vision V6.0", repo_id="SG161222/Realistic_Vision_V6.0_B1_noVAE", nsfw_capable=True, backend="diffusers_dir", recommended_use="Ultra-realistische Portraits & Szenen", tags=["photoreal","portrait","general"]),
    Rec(type="image", name="SDXL Base 1.0", repo_id="stabilityai/stable-diffusion-xl-base-1.0", nsfw_capable=True, backend="diffusers_dir", recommended_use="Allgemeine Fotorealistik mit XL", tags=["sdxl","photoreal"]),
    Rec(type="video", name="Stable Video Diffusion (img2vid-xt)", repo_id="stabilityai/stable-video-diffusion-img2vid-xt", nsfw_capable=True, backend="diffusers_dir", recommended_use="Ultrarealistische Kurzclips aus Standbildern", tags=["svd","video","img2vid"]),
]

def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())

@router.get("/recommendations")
def list_recommendations(type: str = Query(..., pattern="^(image|video)$")):
    return [r.dict() for r in RECS if r.type == type]

@router.post("/models/install")
def install_model(repo_id: str, type: str, name: str | None = None, subfolder: str | None = None):
    s = load_settings()
    base = Path(s.paths.image if type == "image" else s.paths.video)
    base.mkdir(parents=True, exist_ok=True)

    dest_name = safe_name(name or repo_id.split("/")[-1])
    dest_dir = base / dest_name

    if dest_dir.exists() and any(dest_dir.iterdir()):
        raise HTTPException(status_code=400, detail="Zielordner existiert bereits und ist nicht leer.")

    try:
        snapshot_download(repo_id=repo_id, local_dir=str(dest_dir), repo_type="model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HuggingFace Download-Fehler: {e}")

    if subfolder:
        sub = dest_dir / subfolder
        if sub.exists() and sub.is_dir():
            for p in sub.iterdir():
                shutil.move(str(p), dest_dir / p.name)
            shutil.rmtree(sub, ignore_errors=True)

    return {"ok": True, "installed_to": str(dest_dir).replace("\\","/")}
