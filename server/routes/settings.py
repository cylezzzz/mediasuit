# server/routes/settings.py
from fastapi import APIRouter, Body
from ..settings import load_settings, save_settings

router = APIRouter()

@router.get("/settings")
async def get_settings():
    s = load_settings()
    return {"ok": True, "settings": s.model_dump()}

@router.put("/settings")
async def update_settings(patch: dict = Body(...)):
    s = save_settings(patch)
    return {"ok": True, "settings": s.model_dump()}
