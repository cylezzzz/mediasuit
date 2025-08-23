# server/routes/models.py
from __future__ import annotations
from fastapi import APIRouter, Query
from pathlib import Path

from ..settings import load_settings
from ..models_registry import list_models

router = APIRouter()

@router.get("/models")
def get_models(type: str | None = Query(default=None, pattern="^(image|video|llm)$"),
               nsfw: bool | None = None):
    s = load_settings()
    allm = list_models(Path(s.paths.image), Path(s.paths.video))
    items = []
    for k, lst in allm.items():
        if type and k != type:
            continue
        for m in lst:
            if nsfw is not None and bool(m.get("nsfw_capable", False)) != nsfw:
                continue
            items.append(m)
    return items
