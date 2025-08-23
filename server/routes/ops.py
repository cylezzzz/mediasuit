
from __future__ import annotations
from fastapi import APIRouter
from ..state import OPS

router = APIRouter()

@router.get("/ops")
def list_ops():
    return {"ok": True, "ops": OPS.list()}
