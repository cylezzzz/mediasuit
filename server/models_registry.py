# server/models_registry.py - Erweiterte universelle Modell-Erkennung
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import subprocess
import json
import re
import os
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Standardisierte Modell-Information"""
    id: str
    name: str
    type: str  # image, video, llm
    file: Optional[str]
    path: str
    backend: str
    format: str  # diffusers_dir, safetensors, ckpt, onnx, gguf, etc.
    nsfw_capable: bool
    modalities: List[str]
    recommended_use: Optional[str]
    tags: List[str]
    has_sidecar: bool
    size_bytes: int
    architecture: Optional[str]  # sd15, sdxl, svd, llama, etc.
    precision: Optional[str]     # fp16, fp32, int8, etc.
    requirements: List[str]      # cuda, cpu, specific_version
    metadata: Dict[str, Any]

# Erweiterte Dateiformate pro Typ
MODEL_FORMATS = {
    "image": {
        # Diffusers Ordner
        "diffusers_dir": {"model_index.json", "unet", "text_encoder", "vae"},
        # Single Files
        "safetensors": {".safetensors"},
        "checkpoint": {".ckpt", ".pt", ".pth"},
        "onnx": {".onnx"},
        "tensorrt": {".trt", ".engine"},
        "coreml": {".mlpackage", ".mlmodel"},
        # Quantized
        "gguf": {".gguf"},
        "ggml": {".ggml"},
    },
    "video": {
        "diffusers_dir": {"model_index.json", "unet", "image_encoder"},
        "safetensors": {".safetensors"},
        "checkpoint": {".ckpt", ".pt", ".pth"},
        "onnx": {".onnx"},
    },
    "llm": {
        "ollama": {},  # Handled separately
        "gguf": {".gguf"},
        "safetensors": {".safetensors"},
        "pytorch": {".bin", ".pt", ".pth"},
        "onnx": {".onnx"},
    }
}

# Architektur-Erkennung basierend auf Modellnamen/Pfaden
ARCHITECTURE_PATTERNS = {
    # Stable Diffusion
    "sd15": [
        r"stable.?diffusion.?v?1\.?5", r"sd.?1\.?5", r"v1-5", r"runwayml",
        r"realistic.?vision", r"dreamshaper", r"deliberate"
    ],
    "sd20": [r"stable.?diffusion.?v?2\.?0", r"sd.?2\.?0", r"v2-0", r"768-v-ema"],
    "sd21": [r"stable.?diffusion.?v?2\.?1", r"sd.?2\.?1", r"v2-1"],
    "sdxl": [
        r"stable.?diffusion.?xl", r"sdxl", r"xl.*base", r"xl.*refiner",
        r"juggernaut.*xl", r"real.*xl", r"pony.*xl"
    ],
    "sd3": [r"stable.?diffusion.?3", r"sd.?3", r"sd3.*medium"],
    
    # Video Models
    "svd": [r"stable.?video.?diffusion", r"svd", r"img2vid"],
    "animatediff": [r"animatediff", r"motion.*module"],
    "i2vgen": [r"i2vgen", r"videocrafter"],
    
    # LLM Architectures
    "llama": [r"llama", r"vicuna", r"alpaca", r"wizard.*lm"],
    "mistral": [r"mistral", r"mixtral", r"zephyr"],
    "qwen": [r"qwen", r"qwen2"],
    "phi": [r"phi.*3", r"phi.*2"],
    "gemma": [r"gemma"],
    "yi": [r"yi.*34b", r"yi.*6b"],
}

def detect_architecture(name: str, path: str) -> Optional[str]:
    """Erkenne Modell-Architektur basierend auf Namen und Pfad"""
    text = f"{name} {path}".lower()
    
    for arch, patterns in ARCHITECTURE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return arch
    return None

def detect_precision(path: Path) -> Optional[str]:
    """Erkenne Modell-Precision aus Dateinamen"""
    name = path.name.lower()
    if "fp16" in name or "half" in name:
        return "fp16"
    elif "fp32" in name or "full" in name:
        return "fp32"
    elif "int8" in name or "8bit" in name:
        return "int8"
    elif "int4" in name or "4bit" in name:
        return "int4"
    elif "q4" in name:
        return "q4"
    elif "q8" in name:
        return "q8"
    return None

def nsfw_heuristic_advanced(name: str, path: str, tags: List[str] = None) -> bool:
    """Erweiterte NSFW-Erkennung"""
    text = f"{name} {path} {' '.join(tags or [])}".lower()
    
    nsfw_indicators = [
        "nsfw", "adult", "erotic", "nude", "naked", "xxx", "porn", "hentai",
        "lewd", "ecchi", "r18", "18+", "uncensored", "nudes", "realistic_vision",
        "dreamshaper", "anything", "counterfeit", "absolute_reality"
    ]
    
    return any(indicator in text for indicator in nsfw_indicators)

def detect_model_format(path: Path) -> tuple[str, str]:
    """Erkenne Modellformat und Backend"""
    if path.is_dir():
        # Prüfe auf Diffusers-Verzeichnis
        if (path / "model_index.json").exists():
            return "diffusers_dir", "diffusers"
        # Prüfe auf transformers-Verzeichnis
        elif (path / "config.json").exists():
            return "transformers_dir", "transformers"
        # Prüfe auf ONNX-Verzeichnis
        elif list(path.glob("*.onnx")):
            return "onnx_dir", "onnxruntime"
        else:
            return "unknown_dir", "unknown"
    
    # Single-File Formate
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return "safetensors", "safetensors"
    elif suffix in [".ckpt", ".pt", ".pth"]:
        return "checkpoint", "torch"
    elif suffix == ".onnx":
        return "onnx", "onnxruntime"
    elif suffix == ".gguf":
        return "gguf", "llama_cpp"
    elif suffix == ".ggml":
        return "ggml", "llama_cpp"
    elif suffix in [".trt", ".engine"]:
        return "tensorrt", "tensorrt"
    elif suffix in [".mlpackage", ".mlmodel"]:
        return "coreml", "coreml"
    else:
        return "unknown", "unknown"

def get_model_requirements(architecture: str, format: str, size_gb: float) -> List[str]:
    """Bestimme Systemanforderungen für Modell"""
    requirements = []
    
    # GPU-Anforderungen basierend auf Architektur
    if architecture in ["sdxl", "sd3"]:
        requirements.append("cuda_8gb")  # Mindestens 8GB VRAM
    elif architecture in ["sd15", "sd20", "sd21"]:
        requirements.append("cuda_4gb")  # Mindestens 4GB VRAM
    elif architecture == "svd":
        requirements.append("cuda_12gb")  # Video braucht mehr Speicher
    
    # LLM-spezifische Anforderungen
    if architecture in ["llama", "mistral", "qwen"]:
        if size_gb > 20:
            requirements.append("ram_32gb")
        elif size_gb > 10:
            requirements.append("ram_16gb")
        else:
            requirements.append("ram_8gb")
    
    # Format-spezifische Anforderungen
    if format == "gguf":
        requirements.append("llama_cpp")
    elif format == "onnx":
        requirements.append("onnxruntime")
    elif format == "tensorrt":
        requirements.append("tensorrt")
    elif format == "coreml":
        requirements.append("coreml")
    
    return requirements

def load_sidecar_enhanced(path: Path) -> Dict[str, Any]:
    """Erweiterte Sidecar-Datei Unterstützung"""
    sidecar_data = {}
    
    # Standard .model.json
    base_name = path.stem if path.is_file() else path.name
    sidecar_path = path.parent / f"{base_name}.model.json"
    
    if sidecar_path.exists():
        try:
            sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not parse sidecar {sidecar_path}: {e}")
    
    # CivitAI .civitai.info
    civitai_path = path.parent / f"{base_name}.civitai.info"
    if civitai_path.exists():
        try:
            civitai_data = json.loads(civitai_path.read_text(encoding="utf-8"))
            sidecar_data.update({
                "civitai_id": civitai_data.get("id"),
                "civitai_url": civitai_data.get("modelVersions", [{}])[0].get("downloadUrl"),
                "civitai_tags": [tag.get("name") for tag in civitai_data.get("tags", [])],
                "civitai_nsfw": civitai_data.get("nsfw", False)
            })
        except Exception as e:
            logger.warning(f"Could not parse civitai info {civitai_path}: {e}")
    
    # Hugging Face README.md
    readme_path = path / "README.md" if path.is_dir() else path.parent / "README.md"
    if readme_path.exists():
        try:
            readme_content = readme_path.read_text(encoding="utf-8")
            # Extrahiere Metadaten aus YAML frontmatter
            if readme_content.startswith("---"):
                yaml_end = readme_content.find("---", 3)
                if yaml_end > 0:
                    yaml_content = readme_content[3:yaml_end]
                    # Simple YAML parsing für tags
                    if "tags:" in yaml_content:
                        tags_section = re.search(r"tags:\s*\n((?:\s*-\s*.+\n)*)", yaml_content)
                        if tags_section:
                            tags = [line.strip()[2:] for line in tags_section.group(1).split('\n') if line.strip().startswith('-')]
                            sidecar_data["hf_tags"] = tags
        except Exception as e:
            logger.warning(f"Could not parse README {readme_path}: {e}")
    
    return sidecar_data

def scan_directory_universal(kind: str, dir_path: Path) -> List[ModelInfo]:
    """Universeller Directory-Scanner für alle Modelltypen"""
    models: List[ModelInfo] = []
    
    if not dir_path.exists():
        return models
    
    # Rekursiver Scan mit verbesserter Erkennung
    for item in dir_path.rglob("*"):
        try:
            model_info = analyze_model_path(item, kind)
            if model_info:
                models.append(model_info)
        except Exception as e:
            logger.error(f"Error analyzing {item}: {e}")
    
    # Dedupliziere Modelle (gleicher Pfad)
    seen_paths = set()
    unique_models = []
    for model in models:
        if model.path not in seen_paths:
            seen_paths.add(model.path)
            unique_models.append(model)
    
    return unique_models

def analyze_model_path(path: Path, expected_type: str) -> Optional[ModelInfo]:
    """Analysiere einen Pfad und erstelle ModelInfo wenn gültig"""
    
    # Skip versteckte Dateien/Ordner
    if path.name.startswith('.'):
        return None
    
    # Skip temporäre/cache Dateien
    if any(skip in path.name.lower() for skip in ['.tmp', '.cache', '__pycache__', '.git']):
        return None
    
    format_str, backend = detect_model_format(path)
    
    # Skip unbekannte Formate
    if format_str == "unknown":
        return None
    
    # Validiere gegen erwarteten Typ
    valid_formats = MODEL_FORMATS.get(expected_type, {})
    if format_str not in valid_formats and format_str.replace('_dir', '_dir') not in valid_formats:
        return None
    
    # Grundlegende Eigenschaften
    name = path.stem if path.is_file() else path.name
    size_bytes = get_path_size(path)
    architecture = detect_architecture(name, str(path))
    precision = detect_precision(path)
    
    # Sidecar-Daten laden
    sidecar_data = load_sidecar_enhanced(path)
    has_sidecar = bool(sidecar_data)
    
    # NSFW-Erkennung
    nsfw_capable = sidecar_data.get("nsfw_capable")
    if nsfw_capable is None:
        nsfw_capable = (
            sidecar_data.get("civitai_nsfw", False) or
            nsfw_heuristic_advanced(name, str(path), sidecar_data.get("tags", []))
        )
    
    # Modalitäten basierend auf Typ und Architektur
    modalities = determine_modalities(expected_type, architecture, format_str)
    
    # Tags zusammenführen
    tags = []
    tags.extend(sidecar_data.get("tags", []))
    tags.extend(sidecar_data.get("civitai_tags", []))
    tags.extend(sidecar_data.get("hf_tags", []))
    if architecture:
        tags.append(architecture)
    if precision:
        tags.append(precision)
    
    # Requirements
    requirements = get_model_requirements(architecture or "unknown", format_str, size_bytes / (1024**3))
    
    return ModelInfo(
        id=f"{expected_type}:{name}",
        name=name,
        type=expected_type,
        file=path.name if path.is_file() else None,
        path=str(path),
        backend=backend,
        format=format_str,
        nsfw_capable=nsfw_capable,
        modalities=modalities,
        recommended_use=sidecar_data.get("recommended_use"),
        tags=list(set(tags)),  # Dedupliziere Tags
        has_sidecar=has_sidecar,
        size_bytes=size_bytes,
        architecture=architecture,
        precision=precision,
        requirements=requirements,
        metadata=sidecar_data
    )

def determine_modalities(model_type: str, architecture: str, format_str: str) -> List[str]:
    """Bestimme unterstützte Modalitäten basierend auf Typ und Architektur"""
    if model_type == "image":
        if architecture in ["sd15", "sd20", "sd21", "sdxl", "sd3"]:
            return ["text2img", "img2img"]
        else:
            return ["text2img", "img2img"]  # Default für Bild-Modelle
    
    elif model_type == "video":
        if architecture == "svd":
            return ["img2video"]
        elif architecture == "animatediff":
            return ["text2video", "img2video"]
        else:
            return ["text2video", "img2video"]  # Default für Video-Modelle
    
    elif model_type == "llm":
        modalities = ["text", "chat"]
        if "instruct" in (architecture or "").lower():
            modalities.append("instruct")
        if format_str == "gguf":
            modalities.append("quantized")
        return modalities
    
    return []

def get_path_size(path: Path) -> int:
    """Berechne Gesamtgröße eines Pfads (Datei oder Verzeichnis)"""
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    elif path.is_dir():
        try:
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        except OSError:
            return 0
    return 0

def scan_ollama_models() -> List[ModelInfo]:
    """Erweiterte Ollama-Modell-Erkennung"""
    models = []
    
    try:
        # Versuche mehrere Ollama-Befehle
        for cmd in [["ollama", "list", "--format", "json"], ["ollama", "list", "--json"]]:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                output = result.stdout.strip()
                
                if not output:
                    continue
                
                # Parse JSONL oder JSON
                for line in output.splitlines():
                    if not line.strip():
                        continue
                    
                    try:
                        model_data = json.loads(line)
                        model_name = model_data.get("name", "unknown")
                        size_bytes = model_data.get("size", 0)
                        
                        # Erweiterte Metadaten extrahieren
                        architecture = detect_architecture(model_name, "")
                        
                        # Versuche detaillierte Modell-Info zu holen
                        try:
                            info_result = subprocess.run(
                                ["ollama", "show", model_name, "--format", "json"],
                                capture_output=True, text=True, timeout=5
                            )
                            if info_result.returncode == 0:
                                info_data = json.loads(info_result.stdout)
                                # Weitere Metadaten aus model info
                                template = info_data.get("template", "")
                                system = info_data.get("system", "")
                            else:
                                template = system = ""
                        except:
                            template = system = ""
                        
                        models.append(ModelInfo(
                            id=f"llm:{model_name}",
                            name=model_name,
                            type="llm", 
                            file=None,
                            path=None,
                            backend="ollama",
                            format="ollama",
                            nsfw_capable=True,  # LLMs sind nicht gefiltert
                            modalities=["chat", "text", "instruct"] if "instruct" in model_name.lower() else ["chat", "text"],
                            recommended_use=f"LLM für Chat und Textgenerierung ({architecture or 'unknown architecture'})",
                            tags=["llm", "ollama", architecture] if architecture else ["llm", "ollama"],
                            has_sidecar=False,
                            size_bytes=size_bytes,
                            architecture=architecture,
                            precision=detect_precision(Path(model_name)),
                            requirements=["ollama"],
                            metadata={
                                "modified": model_data.get("modified_at"),
                                "digest": model_data.get("digest"),
                                "template": template,
                                "system": system
                            }
                        ))
                        
                    except json.JSONDecodeError:
                        # Fallback: versuche als einzelnes JSON-Objekt zu parsen
                        try:
                            full_data = json.loads(output)
                            if "models" in full_data:
                                for model in full_data["models"]:
                                    model_name = model.get("name", "unknown")
                                    size_bytes = model.get("size", 0)
                                    architecture = detect_architecture(model_name, "")
                                    
                                    models.append(ModelInfo(
                                        id=f"llm:{model_name}",
                                        name=model_name,
                                        type="llm",
                                        file=None, 
                                        path=None,
                                        backend="ollama",
                                        format="ollama",
                                        nsfw_capable=True,
                                        modalities=["chat", "text"],
                                        recommended_use=f"LLM für Chat ({architecture or 'unknown'})",
                                        tags=["llm", "ollama", architecture] if architecture else ["llm", "ollama"],
                                        has_sidecar=False,
                                        size_bytes=size_bytes,
                                        architecture=architecture,
                                        precision=None,
                                        requirements=["ollama"],
                                        metadata=model
                                    ))
                        except:
                            continue
                
                if models:
                    break  # Erfolgreich geparst, stoppe weitere Versuche
                    
            except subprocess.CalledProcessError:
                continue
                
    except FileNotFoundError:
        logger.info("Ollama nicht gefunden - LLM-Support deaktiviert")
    except Exception as e:
        logger.error(f"Ollama scan failed: {e}")
    
    return models

def list_all_models(image_dir: Path, video_dir: Path, llm_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Hauptfunktion: Liste alle Modelle mit erweiterter Unterstützung"""
    
    logger.info("🔍 Scanne Modelle...")
    
    # Scanne lokale Verzeichnisse
    image_models = scan_directory_universal("image", image_dir)
    video_models = scan_directory_universal("video", video_dir) 
    local_llm_models = scan_directory_universal("llm", llm_dir)
    
    # Scanne Ollama
    ollama_models = scan_ollama_models()
    
    # Kombiniere LLM-Modelle
    all_llm_models = local_llm_models + ollama_models
    
    # Konvertiere zu Dict-Format für API-Kompatibilität
    def model_to_dict(model: ModelInfo) -> Dict[str, Any]:
        return {
            "id": model.id,
            "name": model.name,
            "type": model.type,
            "file": model.file,
            "path": model.path,
            "backend": model.backend,
            "format": model.format,
            "nsfw_capable": model.nsfw_capable,
            "modalities": model.modalities,
            "recommended_use": model.recommended_use,
            "tags": model.tags,
            "has_sidecar": model.has_sidecar,
            "size_bytes": model.size_bytes,
            "architecture": model.architecture,
            "precision": model.precision,
            "requirements": model.requirements,
            "metadata": model.metadata
        }
    
    result = {
        "image": [model_to_dict(m) for m in image_models],
        "video": [model_to_dict(m) for m in video_models], 
        "llm": [model_to_dict(m) for m in all_llm_models]
    }
    
    logger.info(f"✅ Modelle gefunden: {len(image_models)} Bild, {len(video_models)} Video, {len(all_llm_models)} LLM")
    
    return result

# Kompatibilitäts-Wrapper für bestehenden Code
def list_models(image_dir: Path, video_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Kompatibilitäts-Wrapper für alte API"""
    llm_dir = image_dir.parent / "llm"  # Ableitung aus image_dir
    return list_all_models(image_dir, video_dir, llm_dir)