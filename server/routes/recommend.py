# server/routes/recommend.py - Erweiterte Empfehlungs-API
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
import shutil
import re
import logging
import asyncio
import os

from huggingface_hub import snapshot_download, hf_hub_download
from ..settings import load_settings
from ..model_catalog import (
    get_catalog_for_api,
    get_models_by_type, 
    get_recommended_models,
    get_beginner_friendly_models,
    get_models_by_requirements,
    search_models,
    CATALOG_MODELS
)

logger = logging.getLogger(__name__)

router = APIRouter()

class InstallRequest(BaseModel):
    """Modell-Installation Request"""
    repo_id: str
    type: str
    name: Optional[str] = None
    subfolder: Optional[str] = None
    precision: Optional[str] = None
    force: bool = False

class InstallResponse(BaseModel):
    """Modell-Installation Response"""
    success: bool
    message: str
    installed_to: str
    model_info: Optional[Dict[str, Any]] = None

def safe_name(name: str) -> str:
    """Erstelle sicheren Dateinamen"""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())[:100]

def get_model_destination(type: str, name: str) -> Path:
    """Bestimme Zielverzeichnis für Modell"""
    s = load_settings()
    base = Path(s.paths.image if type == "image" else s.paths.video if type == "video" else s.paths.get("llm", "models/llm"))
    base.mkdir(parents=True, exist_ok=True)
    
    dest_name = safe_name(name)
    return base / dest_name

async def download_with_progress(repo_id: str, local_dir: str, subfolder: str = None):
    """Download mit Progress-Tracking"""
    try:
        # Verwende snapshot_download für komplette Repos
        if subfolder:
            # Einzelne Datei herunterladen
            filename = hf_hub_download(
                repo_id=repo_id,
                filename=subfolder,
                local_dir=local_dir,
                repo_type="model"
            )
            logger.info(f"Downloaded single file: {filename}")
        else:
            # Komplettes Repository
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                repo_type="model",
                resume_download=True
            )
            logger.info(f"Downloaded repository: {repo_id} to {local_dir}")
            
        return True
    except Exception as e:
        logger.error(f"Download failed for {repo_id}: {e}")
        raise

@router.get("/catalog")
async def get_full_catalog():
    """Vollständiger Modell-Katalog"""
    try:
        catalog = get_catalog_for_api()
        return {
            "ok": True,
            "catalog": catalog,
            "stats": {
                "total_models": catalog["total_count"],
                "image_models": len(catalog["image"]),
                "video_models": len(catalog["video"]),
                "llm_models": len(catalog["llm"]),
                "recommended": len(catalog["recommended"]),
                "beginner_friendly": len(catalog["beginner"])
            }
        }
    except Exception as e:
        logger.error(f"Catalog fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Katalog konnte nicht geladen werden: {e}")

@router.get("/recommendations")
def list_recommendations(
    type: str = Query(..., pattern="^(image|video|llm|all)$"),
    max_vram: Optional[float] = Query(None, description="Maximale VRAM in GB"),
    max_size: Optional[float] = Query(None, description="Maximale Modelgröße in GB"),
    beginner_friendly: bool = Query(False, description="Nur anfängerfreundliche Modelle"),
    limit: int = Query(12, le=50, description="Anzahl Ergebnisse")
):
    """Empfohlene Modelle mit erweiterten Filtern"""
    try:
        if type == "all":
            if beginner_friendly:
                models = get_beginner_friendly_models()
            else:
                models = get_recommended_models(limit * 3)  # Mehr laden für Filterung
        else:
            models = get_models_by_type(type)
            
        # Hardware-Filter anwenden
        if max_vram is not None or max_size is not None:
            filtered_models = []
            for model in models:
                if max_vram is not None and model.vram_gb > max_vram:
                    continue
                if max_size is not None and model.size_gb > max_size:
                    continue
                filtered_models.append(model)
            models = filtered_models
        
        # Nach Popularität und Qualität sortieren
        models = sorted(models, 
                       key=lambda m: m.popularity_score * m.quality_rating, 
                       reverse=True)[:limit]
        
        return [
            {
                "id": model.id,
                "name": model.name,
                "type": model.type,
                "repo_id": model.repo_id,
                "architecture": model.architecture,
                "format": model.format,
                "precision": model.precision,
                "nsfw_capable": model.nsfw_capable,
                "size_gb": model.size_gb,
                "vram_gb": model.vram_gb,
                "recommended_use": model.recommended_use,
                "description": model.description,
                "tags": model.tags,
                "modalities": model.modalities,
                "requirements": model.requirements,
                "subfolder": model.subfolder,
                "license": model.license,
                "author": model.author,
                "popularity_score": model.popularity_score,
                "quality_rating": model.quality_rating,
                "verified": model.verified
            }
            for model in models
        ]
        
    except Exception as e:
        logger.error(f"Recommendations failed: {e}")
        raise HTTPException(status_code=500, detail=f"Empfehlungen konnten nicht geladen werden: {e}")

@router.get("/search")
async def search_catalog(
    q: str = Query(..., description="Suchbegriff"),
    type: Optional[str] = Query(None, pattern="^(image|video|llm)$"),
    architecture: Optional[str] = Query(None, description="Architektur-Filter")
):
    """Suche im Modell-Katalog"""
    try:
        results = search_models(q)
        
        # Zusätzliche Filter
        if type:
            results = [m for m in results if m.type == type]
        if architecture:
            results = [m for m in results if m.architecture == architecture]
        
        return {
            "ok": True,
            "query": q,
            "results": [
                {
                    "id": model.id,
                    "name": model.name,
                    "type": model.type,
                    "repo_id": model.repo_id,
                    "architecture": model.architecture,
                    "recommended_use": model.recommended_use,
                    "description": model.description,
                    "tags": model.tags,
                    "size_gb": model.size_gb,
                    "vram_gb": model.vram_gb,
                    "popularity_score": model.popularity_score,
                    "quality_rating": model.quality_rating
                }
                for model in results[:20]  # Limitiere Suchergebnisse
            ],
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Suche fehlgeschlagen: {e}")

@router.post("/models/install")
async def install_model(request: InstallRequest):
    """Installiere Modell aus Katalog oder custom Repository"""
    try:
        # Validiere Request
        if not request.repo_id.strip():
            raise HTTPException(status_code=400, detail="Repository ID ist erforderlich")
        
        if request.type not in ["image", "video", "llm"]:
            raise HTTPException(status_code=400, detail="Typ muss image, video oder llm sein")
        
        # Name bestimmen
        model_name = request.name or request.repo_id.split("/")[-1]
        dest_dir = get_model_destination(request.type, model_name)
        
        # Prüfe ob bereits vorhanden
        if dest_dir.exists() and any(dest_dir.iterdir()) and not request.force:
            raise HTTPException(
                status_code=409, 
                detail=f"Modell bereits vorhanden: {dest_dir}. Verwende force=true zum Überschreiben."
            )
        
        # Erstelle Zielverzeichnis
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Suche Modell im Katalog für zusätzliche Infos
        catalog_model = None
        for model in CATALOG_MODELS:
            if model.repo_id == request.repo_id:
                catalog_model = model
                break
        
        logger.info(f"Starting installation: {request.repo_id} -> {dest_dir}")
        
        # Download ausführen
        try:
            await download_with_progress(
                repo_id=request.repo_id,
                local_dir=str(dest_dir),
                subfolder=request.subfolder
            )
        except Exception as e:
            # Cleanup bei Fehler
            if dest_dir.exists():
                shutil.rmtree(dest_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"Download fehlgeschlagen: {e}")
        
        # Post-Processing wenn Single-File in Subfolder
        if request.subfolder:
            # Verschiebe Datei aus Subfolder zur Root
            subfolder_path = dest_dir / request.subfolder
            if subfolder_path.exists() and subfolder_path.is_file():
                target_path = dest_dir / subfolder_path.name
                if target_path != subfolder_path:
                    shutil.move(str(subfolder_path), str(target_path))
                    # Entferne leere Verzeichnisse
                    try:
                        parent_dir = subfolder_path.parent
                        if parent_dir != dest_dir and not any(parent_dir.iterdir()):
                            shutil.rmtree(parent_dir)
                    except:
                        pass
        
        # Erstelle Sidecar-Datei mit Katalog-Informationen
        if catalog_model:
            sidecar_path = dest_dir / f"{dest_dir.name}.model.json"
            sidecar_data = {
                "id": catalog_model.id,
                "name": catalog_model.name,
                "type": catalog_model.type,
                "repo_id": catalog_model.repo_id,
                "architecture": catalog_model.architecture,
                "format": catalog_model.format,
                "precision": catalog_model.precision,
                "nsfw_capable": catalog_model.nsfw_capable,
                "recommended_use": catalog_model.recommended_use,
                "description": catalog_model.description,
                "tags": catalog_model.tags,
                "modalities": catalog_model.modalities,
                "requirements": catalog_model.requirements,
                "license": catalog_model.license,
                "author": catalog_model.author,
                "installed_by": "LocalMediaSuite",
                "installation_date": "2024-01-01"  # In Produktion: datetime.now().isoformat()
            }
            
            try:
                with open(sidecar_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(sidecar_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Created sidecar file: {sidecar_path}")
            except Exception as e:
                logger.warning(f"Could not create sidecar file: {e}")
        
        # Validiere Installation
        if not dest_dir.exists() or not any(dest_dir.iterdir()):
            raise HTTPException(status_code=500, detail="Installation scheint fehlgeschlagen - Zielverzeichnis leer")
        
        # Erstelle Response
        model_info = {
            "name": model_name,
            "type": request.type,
            "path": str(dest_dir),
            "repo_id": request.repo_id,
            "size_mb": sum(f.stat().st_size for f in dest_dir.rglob("*") if f.is_file()) // (1024 * 1024)
        }
        
        if catalog_model:
            model_info.update({
                "architecture": catalog_model.architecture,
                "recommended_use": catalog_model.recommended_use,
                "tags": catalog_model.tags,
                "verified": catalog_model.verified
            })
        
        logger.info(f"Successfully installed: {request.repo_id} to {dest_dir}")
        
        return InstallResponse(
            success=True,
            message=f"Modell '{model_name}' erfolgreich installiert",
            installed_to=str(dest_dir).replace("\\", "/"),
            model_info=model_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Installation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Installation fehlgeschlagen: {e}")

@router.delete("/models/{model_id}")
async def delete_model(model_id: str, force: bool = Query(False)):
    """Lösche installiertes Modell"""
    try:
        s = load_settings()
        
        # Parse model_id (format: "type:name")
        if ":" not in model_id:
            raise HTTPException(status_code=400, detail="Ungültige Model-ID Format (erwartet: type:name)")
        
        model_type, model_name = model_id.split(":", 1)
        
        if model_type not in ["image", "video", "llm"]:
            raise HTTPException(status_code=400, detail="Ungültiger Modell-Typ")
        
        # Finde Modell-Pfad
        base_dirs = {
            "image": Path(s.paths.image),
            "video": Path(s.paths.video),
            "llm": Path(s.paths.get("llm", "models/llm"))
        }
        
        base_dir = base_dirs[model_type]
        model_path = None
        
        # Suche nach Modell (Ordner oder Datei)
        for item in base_dir.rglob("*"):
            if item.name == model_name or item.stem == model_name:
                model_path = item
                break
        
        if not model_path or not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Modell nicht gefunden: {model_id}")
        
        # Sicherheitsabfrage für wichtige Modelle (außer bei force)
        if not force and model_path.stat().st_size > 1024**3:  # > 1GB
            raise HTTPException(
                status_code=400, 
                detail="Großes Modell - verwende force=true zum Bestätigen der Löschung"
            )
        
        # Lösche Modell
        try:
            if model_path.is_file():
                model_path.unlink()
                logger.info(f"Deleted model file: {model_path}")
            else:
                shutil.rmtree(model_path)
                logger.info(f"Deleted model directory: {model_path}")
                
            # Lösche auch Sidecar-Dateien
            for sidecar_pattern in ["*.model.json", "*.civitai.info"]:
                for sidecar_file in model_path.parent.glob(f"{model_path.stem}.{sidecar_pattern.split('.')[-2]}.*"):
                    try:
                        sidecar_file.unlink()
                        logger.info(f"Deleted sidecar: {sidecar_file}")
                    except:
                        pass
                        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Löschung fehlgeschlagen: {e}")
        
        return {
            "ok": True,
            "message": f"Modell '{model_name}' erfolgreich gelöscht",
            "deleted_path": str(model_path).replace("\\", "/")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Löschung fehlgeschlagen: {e}")

@router.get("/models/{model_id}/info")
async def get_model_info(model_id: str):
    """Detaillierte Informationen über ein Modell"""
    try:
        # Suche im Katalog
        catalog_model = None
        for model in CATALOG_MODELS:
            if model.id == model_id or model.repo_id.endswith(model_id):
                catalog_model = model
                break
        
        if not catalog_model:
            raise HTTPException(status_code=404, detail="Modell nicht im Katalog gefunden")
        
        # Prüfe ob installiert
        s = load_settings()
        base_dirs = {
            "image": Path(s.paths.image),
            "video": Path(s.paths.video), 
            "llm": Path(s.paths.get("llm", "models/llm"))
        }
        
        installed = False
        installed_path = None
        installed_size = 0
        
        base_dir = base_dirs.get(catalog_model.type)
        if base_dir and base_dir.exists():
            model_name = catalog_model.repo_id.split("/")[-1]
            potential_paths = [
                base_dir / model_name,
                base_dir / safe_name(model_name),
                base_dir / catalog_model.name
            ]
            
            for path in potential_paths:
                if path.exists():
                    installed = True
                    installed_path = str(path).replace("\\", "/")
                    installed_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                    break
        
        return {
            "ok": True,
            "model": {
                "id": catalog_model.id,
                "name": catalog_model.name,
                "type": catalog_model.type,
                "repo_id": catalog_model.repo_id,
                "architecture": catalog_model.architecture,
                "format": catalog_model.format,
                "precision": catalog_model.precision,
                "nsfw_capable": catalog_model.nsfw_capable,
                "size_gb": catalog_model.size_gb,
                "vram_gb": catalog_model.vram_gb,
                "recommended_use": catalog_model.recommended_use,
                "description": catalog_model.description,
                "tags": catalog_model.tags,
                "modalities": catalog_model.modalities,
                "requirements": catalog_model.requirements,
                "license": catalog_model.license,
                "author": catalog_model.author,
                "popularity_score": catalog_model.popularity_score,
                "quality_rating": catalog_model.quality_rating,
                "verified": catalog_model.verified,
                "installed": installed,
                "installed_path": installed_path,
                "installed_size_mb": installed_size // (1024 * 1024) if installed_size > 0 else 0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model info failed: {e}")
        raise HTTPException(status_code=500, detail=f"Modell-Info konnte nicht geladen werden: {e}")

@router.get("/architectures")
async def get_supported_architectures():
    """Liste aller unterstützten Modell-Architekturen"""
    architectures = {}
    
    for model in CATALOG_MODELS:
        arch = model.architecture
        if arch not in architectures:
            architectures[arch] = {
                "name": arch,
                "types": set(),
                "count": 0,
                "description": _get_architecture_description(arch),
                "requirements": set()
            }
        
        architectures[arch]["types"].add(model.type)
        architectures[arch]["count"] += 1
        architectures[arch]["requirements"].update(model.requirements)
    
    # Konvertiere Sets zu Listen für JSON-Serialisierung
    for arch_info in architectures.values():
        arch_info["types"] = list(arch_info["types"])
        arch_info["requirements"] = list(arch_info["requirements"])
    
    return {
        "ok": True,
        "architectures": architectures,
        "total": len(architectures)
    }

def _get_architecture_description(arch: str) -> str:
    """Beschreibung für Architekturen"""
    descriptions = {
        "sd15": "Stable Diffusion 1.5 - Der Klassiker für 512x512 Bilder",
        "sd20": "Stable Diffusion 2.0 - Verbesserte Version mit 768x768",
        "sd21": "Stable Diffusion 2.1 - Weiter optimierte Version",
        "sdxl": "Stable Diffusion XL - High-Resolution 1024x1024+ Bilder",
        "sd3": "Stable Diffusion 3 - Neueste Generation mit verbessertem Prompt-Following",
        "svd": "Stable Video Diffusion - Bild-zu-Video-Generation",
        "animatediff": "AnimateDiff - Animation-Erweiterung für SD-Modelle",
        "llama": "Llama - Meta's Large Language Model Familie",
        "mixtral": "Mixtral - Mixture-of-Experts Modell von Mistral AI",
        "qwen": "Qwen - Alibaba's mehrsprachiges LLM",
        "phi": "Phi - Microsoft's kompakte aber leistungsstarke LLMs",
        "controlnet": "ControlNet - Präzise Kontrolle über Bildgeneration"
    }
    return descriptions.get(arch, f"Modell-Architektur: {arch}")

@router.post("/models/batch-install")
async def batch_install_models(
    requests: List[InstallRequest] = Body(..., description="Liste von Modellen zum Installieren")
):
    """Installiere mehrere Modelle gleichzeitig"""
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Maximal 10 Modelle pro Batch erlaubt")
    
    results = []
    
    for i, request in enumerate(requests):
        try:
            result = await install_model(request)
            results.append({
                "index": i,
                "success": True,
                "model": request.repo_id,
                "message": result.message,
                "installed_to": result.installed_to
            })
        except Exception as e:
            results.append({
                "index": i,
                "success": False,
                "model": request.repo_id,
                "error": str(e)
            })
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    return {
        "ok": True,
        "summary": {
            "total": len(requests),
            "successful": successful,
            "failed": failed
        },
        "results": results
    }

@router.get("/compatibility/{repo_id}")
async def check_model_compatibility(repo_id: str):
    """Prüfe Systemkompatibilität für ein Modell"""
    try:
        # Suche Modell im Katalog
        catalog_model = None
        for model in CATALOG_MODELS:
            if model.repo_id == repo_id:
                catalog_model = model
                break
        
        if not catalog_model:
            return {
                "ok": False,
                "error": "Modell nicht im Katalog gefunden",
                "compatible": False
            }
        
        # Simuliere Hardware-Prüfung (in Realität: GPU-Info abfragen)
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if has_cuda else 0
        except:
            has_cuda = False
            gpu_memory_gb = 0
        
        # System-RAM (vereinfacht)
        try:
            import psutil
            system_ram_gb = psutil.virtual_memory().total / (1024**3)
        except:
            system_ram_gb = 8  # Default-Annahme
        
        # Kompatibilitätsprüfung
        compatible = True
        warnings = []
        blockers = []
        
        # VRAM-Prüfung
        if catalog_model.vram_gb > gpu_memory_gb and has_cuda:
            if catalog_model.vram_gb > gpu_memory_gb * 1.5:
                compatible = False
                blockers.append(f"Benötigt {catalog_model.vram_gb}GB VRAM, verfügbar: {gpu_memory_gb:.1f}GB")
            else:
                warnings.append(f"Knapper VRAM: {catalog_model.vram_gb}GB benötigt, {gpu_memory_gb:.1f}GB verfügbar")
        
        # CPU-Fallback für GPU-Modelle
        if not has_cuda and "cuda" in catalog_model.requirements:
            warnings.append("GPU nicht verfügbar - wird auf CPU laufen (langsamer)")
        
        # RAM-Prüfung für LLMs
        if catalog_model.type == "llm" and catalog_model.size_gb * 1.5 > system_ram_gb:
            if catalog_model.size_gb * 2 > system_ram_gb:
                compatible = False
                blockers.append(f"Nicht genug RAM: {catalog_model.size_gb * 1.5:.1f}GB benötigt, {system_ram_gb:.1f}GB verfügbar")
            else:
                warnings.append("Knapper RAM - System könnte langsam werden")
        
        return {
            "ok": True,
            "model": {
                "repo_id": repo_id,
                "name": catalog_model.name,
                "requirements": catalog_model.requirements
            },
            "system": {
                "has_cuda": has_cuda,
                "gpu_memory_gb": gpu_memory_gb,
                "system_ram_gb": system_ram_gb
            },
            "compatible": compatible,
            "warnings": warnings,
            "blockers": blockers,
            "recommendation": _get_compatibility_recommendation(catalog_model, compatible, warnings, blockers)
        }
        
    except Exception as e:
        logger.error(f"Compatibility check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Kompatibilitätsprüfung fehlgeschlagen: {e}")

def _get_compatibility_recommendation(model, compatible: bool, warnings: List[str], blockers: List[str]) -> str:
    """Generiere Kompatibilitäts-Empfehlung"""
    if not compatible:
        return f"❌ Nicht kompatibel. Probleme: {'; '.join(blockers)}"
    elif warnings:
        return f"⚠️ Funktioniert, aber: {'; '.join(warnings)}"
    else:
        return f"✅ Vollständig kompatibel. {model.name} sollte problemlos laufen."