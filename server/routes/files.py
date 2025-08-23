# server/routes/files.py
from fastapi import APIRouter, Query
from pathlib import Path
from ..settings import load_settings

router = APIRouter()

@router.get("/files")
async def list_files(type: str = Query(..., pattern="^(image|video)$")):
    s = load_settings()
    base = Path(s.paths.images if type == "image" else s.paths.videos)
    base.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(base.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
        if not p.is_file():
            continue
        rel = f"/outputs/{'images' if type=='image' else 'videos'}/{p.name}"
        items.append({
            "name": p.name,
            "url": rel,
            "size": p.stat().st_size,
            "mtime": int(p.stat().st_mtime),
        })
    return {"ok": True, "items": items}
