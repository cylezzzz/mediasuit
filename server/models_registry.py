# server/models_registry.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess
import json
import re
import os

# Dateien/Ordner, die wir als Modelle akzeptieren
ALLOWED = {
    "image": {".safetensors", ".ckpt", ".pt", ".pth", ".onnx"},
    "video": {".safetensors", ".ckpt", ".pt", ".pth", ".onnx"},
}
# diffusers Ordner haben diese Datei
DIFFUSERS_INDEX = "model_index.json"


def nsfw_heuristic(name: str) -> bool:
    """
    Sehr einfache Heuristik, ob ein Modell NSFW-tauglich ist – kann
    per Sidecar überschrieben werden.
    """
    n = (name or "").lower()
    return any(x in n for x in ["nsfw", "erot", "adult", "nude", "boudoir", "lewd", "xxx"])


def load_sidecar(p: Path) -> Dict[str, Any]:
    """
    Lädt optionale Metadaten aus einer Sidecar-Datei:
      <modelname>.model.json
    Liegt neben dem Model-Ordner oder der Model-Datei.
    """
    try:
        base = p if p.is_dir() else p.with_suffix("")
        sc = base.parent / f"{base.name}.model.json"
        if sc.exists():
            return json.loads(sc.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _human_size_bytes(path: Path) -> int:
    try:
        if path.is_dir():
            return sum(pp.stat().st_size for pp in path.rglob("*") if pp.is_file())
        return path.stat().st_size
    except Exception:
        return 0


def scan_kind(kind: str, dir_path: Path) -> List[Dict[str, Any]]:
    """
    Scannt ein Modellverzeichnis (image|video) und erzeugt Metadaten für:
      - diffusers Ordner (mit model_index.json)
      - Single-File-Modelle (.safetensors, .ckpt, .pt, .onnx)
    """
    items: List[Dict[str, Any]] = []
    if not dir_path.exists():
        return items

    for f in dir_path.iterdir():
        # 1) diffusers Ordner
        if f.is_dir():
            mi = f / DIFFUSERS_INDEX
            if mi.exists():
                meta: Dict[str, Any] = {
                    "id": f"{kind}:{f.name}",
                    "name": f.name,
                    "type": kind,
                    "file": None,
                    "path": str(f),
                    "backend": "diffusers_dir",
                    "nsfw_capable": nsfw_heuristic(f.name),
                    "modalities": ["text2img", "img2img"] if kind == "image" else ["text2video", "img2video"],
                    "recommended_use": None,
                    "tags": [],
                    "has_sidecar": False,
                    "size_bytes": _human_size_bytes(f),
                }
                sc = load_sidecar(f)
                if sc:
                    meta["has_sidecar"] = True
                    if "recommended_use" in sc:
                        meta["recommended_use"] = sc["recommended_use"]
                    if "tags" in sc:
                        meta["tags"] = sc["tags"] or []
                    if "nsfw_capable" in sc:
                        meta["nsfw_capable"] = bool(sc["nsfw_capable"])
                    if "modalities" in sc and isinstance(sc["modalities"], list):
                        meta["modalities"] = sc["modalities"]
                items.append(meta)
                continue

        # 2) Single-File Modelle
        if f.is_file() and f.suffix.lower() in ALLOWED.get(kind, set()):
            meta: Dict[str, Any] = {
                "id": f"{kind}:{f.name}",
                "name": f.stem,
                "type": kind,
                "file": f.name,
                "path": str(f),
                "backend": "diffusers_single",
                "nsfw_capable": nsfw_heuristic(f.name),
                "modalities": ["text2img", "img2img"] if kind == "image" else ["text2video", "img2video"],
                "recommended_use": None,
                "tags": [],
                "has_sidecar": False,
                "size_bytes": _human_size_bytes(f),
            }
            sc = load_sidecar(f)
            if sc:
                meta["has_sidecar"] = True
                if "recommended_use" in sc:
                    meta["recommended_use"] = sc["recommended_use"]
                if "tags" in sc:
                    meta["tags"] = sc["tags"] or []
                if "nsfw_capable" in sc:
                    meta["nsfw_capable"] = bool(sc["nsfw_capable"])
                if "modalities" in sc and isinstance(sc["modalities"], list):
                    meta["modalities"] = sc["modalities"]
            items.append(meta)
            continue

    return items


def _ollama_list() -> List[Dict[str, Any]]:
    """
    Liest lokal installierte Ollama-Modelle (wenn Ollama vorhanden ist).
    Gibt eine Liste von LLM-Metadaten zurück.
    """
    out_models: List[Dict[str, Any]] = []
    exe = "ollama"
    try:
        # `ollama list --format json` (neue Versionen) oder `--json` (ältere) – wir probieren beides
        for args in (["list", "--format", "json"], ["list", "--json"]):
            try:
                proc = subprocess.run(
                    [exe, *args],
                    capture_output=True,
                    text=True,
                    check=True,
                    shell=False,
                )
                raw = proc.stdout.strip()
                if not raw:
                    continue
                # Manchmal kommt JSONL – Zeile für Zeile parsen
                lines = [ln for ln in raw.splitlines() if ln.strip()]
                for ln in lines:
                    try:
                        row = json.loads(ln)
                        name = row.get("name") or row.get("model") or "ollama-model"
                        size = row.get("size", 0)
                        modified = row.get("modified_at") or row.get("modified") or None
                        out_models.append({
                            "id": f"llm:{name}",
                            "name": name,
                            "type": "llm",
                            "file": None,
                            "path": None,
                            "backend": "ollama",
                            "nsfw_capable": True,   # LLMs werden nicht gefiltert
                            "modalities": ["chat", "text"],
                            "recommended_use": "LLM für Chat/Steuerungs-Agents (Ollama)",
                            "tags": ["llm", "ollama"],
                            "has_sidecar": False,
                            "size_bytes": int(size) if isinstance(size, (int, float)) else 0,
                            "modified": modified,
                        })
                    except Exception:
                        # evtl. war es doch ein einzelnes JSON-Objekt
                        try:
                            obj = json.loads(raw)
                            # Format: {"models":[...]}
                            models = obj.get("models") or []
                            for m in models:
                                name = m.get("name") or m.get("model") or "ollama-model"
                                size = m.get("size", 0)
                                modified = m.get("modified_at") or m.get("modified") or None
                                out_models.append({
                                    "id": f"llm:{name}",
                                    "name": name,
                                    "type": "llm",
                                    "file": None,
                                    "path": None,
                                    "backend": "ollama",
                                    "nsfw_capable": True,
                                    "modalities": ["chat", "text"],
                                    "recommended_use": "LLM für Chat/Steuerungs-Agents (Ollama)",
                                    "tags": ["llm", "ollama"],
                                    "has_sidecar": False,
                                    "size_bytes": int(size) if isinstance(size, (int, float)) else 0,
                                    "modified": modified,
                                })
                        except Exception:
                            pass
                if out_models:
                    break
            except subprocess.CalledProcessError:
                continue
    except FileNotFoundError:
        # Ollama nicht installiert
        pass
    except Exception:
        pass
    return out_models


def list_models(image_dir: Path, video_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Liefert alle gefundenen Modelle nach Typ gruppiert:
      {
        "image": [...],
        "video": [...],
        "llm":   [...]
      }
    """
    image = scan_kind("image", image_dir)
    video = scan_kind("video", video_dir)
    llm = _ollama_list()
    return {"image": image, "video": video, "llm": llm}
