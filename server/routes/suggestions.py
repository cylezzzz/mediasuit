# server/routes/suggestions.py
from __future__ import annotations
from fastapi import APIRouter, Query
from pathlib import Path
import json, random

from ..settings import load_settings

router = APIRouter()

SUG_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "suggestions.json"

def load_suggestions() -> dict:
    if SUG_FILE.exists():
        try:
            return json.loads(SUG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"image": {"sfw": {"categories": [], "modifiers": []}, "nsfw": {"categories": [], "modifiers": []}},
            "video": {"sfw": {"categories": [], "modifiers": []}, "nsfw": {"categories": [], "modifiers": []}}}

@router.get("/suggestions")
def get_suggestions(type: str = Query(..., pattern="^(image|video)$"),
                    nsfw: bool = Query(False)):
    data = load_suggestions().get(type, {})
    key = "nsfw" if nsfw else "sfw"
    payload = data.get(key, {"categories": [], "modifiers": []})
    return payload

@router.get("/prompts/random")
def random_prompt(type: str = Query(..., pattern="^(image|video)$"),
                  nsfw: bool = Query(False),
                  category: str | None = None):
    data = load_suggestions().get(type, {})
    key = "nsfw" if nsfw else "sfw"
    payload = data.get(key, {"categories": [], "modifiers": []})
    pool = []
    for cat in payload.get("categories", []):
        if category and cat.get("id") != category:
            continue
        pool.extend(cat.get("prompts", []))
    if not pool:
        return {"prompt": None}
    base = random.choice(pool)
    mods = payload.get("modifiers", [])
    if mods:
        chosen = ", ".join(random.sample(mods, k=min(3, len(mods))))
        base = f"{base}, {chosen}"
    return {"prompt": base}
