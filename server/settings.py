# server/settings.py
from pydantic import BaseModel
from pathlib import Path
import json, threading

SETTINGS_PATH = Path("config/settings.json")
_LOCK = threading.Lock()

class Paths(BaseModel):
    images: str = "outputs/images"
    videos: str = "outputs/videos"
    image: str = "models/image"
    video: str = "models/video"

class AppSettings(BaseModel):
    paths: Paths = Paths()
    ui: dict = {"theme":"dark"}
    defaults: dict = {
        "img_resolution": "1024x768",
        "vid_resolution": "1280x720",
        "vid_fps": 24,
        "vid_length_sec": 3
    }
    version: int = 1

_settings_cache: AppSettings | None = None

def load_settings() -> AppSettings:
    global _settings_cache
    if _settings_cache is not None:
        return _settings_cache
    if SETTINGS_PATH.exists():
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        _settings_cache = AppSettings(**data)
    else:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _settings_cache = AppSettings()
        SETTINGS_PATH.write_text(_settings_cache.model_dump_json(indent=2), encoding="utf-8")
    return _settings_cache

def save_settings(patch: dict) -> AppSettings:
    global _settings_cache
    with _LOCK:
        cur = load_settings()
        data = cur.model_dump()
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(data.get(k), dict):
                data[k].update(v)
            else:
                data[k] = v
        data["version"] = int(data.get("version", 1)) + 1
        _settings_cache = AppSettings(**data)
        SETTINGS_PATH.write_text(_settings_cache.model_dump_json(indent=2), encoding="utf-8")
        return _settings_cache
