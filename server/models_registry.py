# server/models_registry_fixed.py - Korrigierte Modell-Erkennung
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Vereinfachte aber zuverlässige Modell-Information"""
    id: str
    name: str
    type: str  # image, video, llm
    path: str
    format: str  # diffusers_dir, safetensors, ckpt, gguf, ollama
    backend: str  # diffusers, llama_cpp, ollama
    nsfw_capable: bool
    modalities: List[str]
    architecture: Optional[str] = None
    size_bytes: int = 0
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "path": self.path,
            "format": self.format,
            "backend": self.backend,
            "nsfw_capable": self.nsfw_capable,
            "modalities": self.modalities,
            "architecture": self.architecture,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata or {}
        }

def detect_model_format_and_type(path: Path) -> tuple[str, str, str]:
    """
    Erkenne Format, Typ und Backend eines Modells
    Returns: (format, type, backend)
    """
    
    if path.is_dir():
        # Prüfe auf Diffusers-Verzeichnis
        if (path / "model_index.json").exists():
            # Lese model_index.json um Typ zu bestimmen
            try:
                with open(path / "model_index.json", 'r', encoding='utf-8') as f:
                    model_index = json.load(f)
                    
                # Prüfe auf Video-Pipeline
                if any('video' in str(component).lower() for component in model_index.values()):
                    return "diffusers_dir", "video", "diffusers"
                else:
                    return "diffusers_dir", "image", "diffusers"
                    
            except Exception as e:
                logger.warning(f"Could not read model_index.json in {path}: {e}")
                return "diffusers_dir", "image", "diffusers"
        
        # Prüfe auf Transformers-Verzeichnis (LLM)
        elif (path / "config.json").exists():
            return "transformers_dir", "llm", "transformers"
        
        # Unbekanntes Verzeichnis
        else:
            return "unknown_dir", "unknown", "unknown"
    
    else:
        # Single-File Formate
        suffix = path.suffix.lower()
        
        if suffix == ".safetensors":
            # Bestimme Typ basierend auf Pfad
            if "video" in str(path).lower():
                return "safetensors", "video", "diffusers"
            else:
                return "safetensors", "image", "diffusers"
                
        elif suffix in [".ckpt", ".pt", ".pth"]:
            # PyTorch Checkpoints - meist für Bilder
            return "checkpoint", "image", "diffusers"
            
        elif suffix == ".gguf":
            return "gguf", "llm", "llama_cpp"
            
        else:
            return "unknown", "unknown", "unknown"

def detect_architecture(name: str, path: str) -> Optional[str]:
    """Erkenne Modell-Architektur"""
    text = f"{name} {path}".lower()
    
    # Video Architekturen
    if any(term in text for term in ["stable-video-diffusion", "svd", "img2vid"]):
        return "svd"
    elif "animatediff" in text:
        return "animatediff"
    
    # Image Architekturen
    elif any(term in text for term in ["stable-diffusion-xl", "sdxl", "xl"]):
        return "sdxl"
    elif any(term in text for term in ["stable-diffusion-3", "sd3"]):
        return "sd3"
    elif any(term in text for term in ["stable-diffusion-2", "sd2", "v2-"]):
        return "sd20"
    elif any(term in text for term in ["stable-diffusion", "sd", "v1-5", "runwayml"]):
        return "sd15"
    
    # LLM Architekturen
    elif any(term in text for term in ["llama", "vicuna", "alpaca"]):
        return "llama"
    elif any(term in text for term in ["mistral", "mixtral"]):
        return "mistral"
    elif "qwen" in text:
        return "qwen"
    elif "phi" in text:
        return "phi"
    
    return None

def determine_modalities(model_type: str, architecture: str, format: str) -> List[str]:
    """Bestimme unterstützte Modalitäten"""
    
    if model_type == "image":
        if architecture in ["sd15", "sd20", "sd21", "sdxl", "sd3"]:
            return ["text2img", "img2img"]
        else:
            return ["text2img", "img2img"]  # Default für Bild-Modelle
    
    elif model_type == "video":
        if architecture == "svd":
            return ["img2video"]  # SVD benötigt immer ein Startbild
        elif architecture == "animatediff":
            return ["text2video", "img2video"]
        else:
            return ["img2video"]  # Sicherer Default
    
    elif model_type == "llm":
        return ["text", "chat", "instruct"]
    
    return []

def get_path_size(path: Path) -> int:
    """Berechne Gesamtgröße eines Pfads"""
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

def scan_directory_for_models(directory: Path, expected_type: str) -> List[ModelInfo]:
    """Scanne Verzeichnis nach Modellen"""
    models = []
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return models
    
    logger.info(f"Scanning {directory} for {expected_type} models...")
    
    # Scanne alle Einträge im Verzeichnis
    for item in directory.iterdir():
        try:
            # Skip versteckte Dateien/Ordner
            if item.name.startswith('.'):
                continue
            
            format_type, detected_type, backend = detect_model_format_and_type(item)
            
            # Skip unbekannte Formate
            if format_type == "unknown":
                continue
            
            # Validiere Typ (falls angegeben)
            if expected_type != "any" and detected_type != expected_type:
                continue
            
            # Grundlegende Eigenschaften
            name = item.name
            size_bytes = get_path_size(item)
            architecture = detect_architecture(name, str(item))
            modalities = determine_modalities(detected_type, architecture or "", format_type)
            
            # NSFW-Heuristik (einfach aber funktional)
            nsfw_capable = any(term in name.lower() for term in [
                "realistic", "dreamshaper", "anything", "nsfw", "uncensored"
            ])
            
            # Erstelle ModelInfo
            model = ModelInfo(
                id=f"{detected_type}:{name}",
                name=name,
                type=detected_type,
                path=str(item),
                format=format_type,
                backend=backend,
                nsfw_capable=nsfw_capable,
                modalities=modalities,
                architecture=architecture,
                size_bytes=size_bytes,
                metadata={"discovered": True}
            )
            
            models.append(model)
            logger.info(f"Found {detected_type} model: {name} ({format_type})")
            
        except Exception as e:
            logger.error(f"Error processing {item}: {e}")
            continue
    
    return models

def scan_ollama_models() -> List[ModelInfo]:
    """Scanne Ollama für verfügbare LLM-Modelle"""
    models = []
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True
        )
        
        lines = result.stdout.strip().split('\n')
        
        # Skip header line
        for line in lines[1:]:
            if not line.strip():
                continue
                
            parts = line.split()
            if len(parts) >= 1:
                model_name = parts[0]
                
                # Skip wenn Name ungültig
                if not model_name or model_name == "NAME":
                    continue
                
                architecture = detect_architecture(model_name, "")
                
                model = ModelInfo(
                    id=f"llm:{model_name}",
                    name=model_name,
                    type="llm",
                    path="",  # Ollama hat keinen lokalen Pfad
                    format="ollama",
                    backend="ollama",
                    nsfw_capable=True,  # Ollama-Modelle sind nicht gefiltert
                    modalities=["text", "chat", "instruct"],
                    architecture=architecture,
                    size_bytes=0,  # Ollama verwaltet das intern
                    metadata={"ollama": True}
                )
                
                models.append(model)
                logger.info(f"Found Ollama model: {model_name}")
        
    except subprocess.CalledProcessError as e:
        logger.warning(f"Ollama command failed: {e}")
    except FileNotFoundError:
        logger.info("Ollama not found - LLM support via Ollama disabled")
    except Exception as e:
        logger.error(f"Error scanning Ollama models: {e}")
    
    return models

def list_all_models(image_dir: Path, video_dir: Path, llm_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Hauptfunktion: Liste alle verfügbaren Modelle"""
    
    logger.info("🔍 Starting model discovery...")
    
    # Scanne lokale Verzeichnisse
    image_models = scan_directory_for_models(image_dir, "image")
    video_models = scan_directory_for_models(video_dir, "video") 
    local_llm_models = scan_directory_for_models(llm_dir, "llm")
    
    # Scanne Ollama
    ollama_models = scan_ollama_models()
    
    # Kombiniere LLM-Modelle
    all_llm_models = local_llm_models + ollama_models
    
    # Konvertiere zu Dict-Format
    result = {
        "image": [model.to_dict() for model in image_models],
        "video": [model.to_dict() for model in video_models], 
        "llm": [model.to_dict() for model in all_llm_models]
    }
    
    # Log Zusammenfassung
    logger.info(f"✅ Model discovery complete:")
    logger.info(f"   📸 Image models: {len(image_models)}")
    logger.info(f"   🎬 Video models: {len(video_models)}")
    logger.info(f"   🤖 LLM models: {len(all_llm_models)}")
    
    return result

def list_models(image_dir: Path, video_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Kompatibilitäts-Wrapper für alte API"""
    llm_dir = image_dir.parent / "llm"
    return list_all_models(image_dir, video_dir, llm_dir)

def get_model_by_id(model_id: str, base_dirs: Dict[str, Path]) -> Optional[ModelInfo]:
    """Finde spezifisches Modell nach ID"""
    
    all_models = list_all_models(
        base_dirs.get("image", Path("models/image")),
        base_dirs.get("video", Path("models/video")),
        base_dirs.get("llm", Path("models/llm"))
    )
    
    for model_type, models in all_models.items():
        for model_dict in models:
            if model_dict.get("id") == model_id:
                # Konvertiere Dict zurück zu ModelInfo
                return ModelInfo(
                    id=model_dict["id"],
                    name=model_dict["name"],
                    type=model_dict["type"],
                    path=model_dict["path"],
                    format=model_dict["format"],
                    backend=model_dict["backend"],
                    nsfw_capable=model_dict["nsfw_capable"],
                    modalities=model_dict["modalities"],
                    architecture=model_dict.get("architecture"),
                    size_bytes=model_dict.get("size_bytes", 0),
                    metadata=model_dict.get("metadata", {})
                )
    
    return None

def validate_model_path(model_info: ModelInfo) -> bool:
    """Validiere ob Modell-Pfad existiert und zugänglich ist"""
    
    if model_info.backend == "ollama":
        # Für Ollama prüfen wir via subprocess
        try:
            result = subprocess.run(
                ["ollama", "show", model_info.name],
                capture_output=True,
                timeout=5,
                check=True
            )
            return result.returncode == 0
        except Exception:
            return False
    
    elif model_info.path:
        path = Path(model_info.path)
        return path.exists()
    
    return False

def get_compatible_models_for_mode(mode: str, base_dirs: Dict[str, Path]) -> List[ModelInfo]:
    """Finde alle Modelle die einen bestimmten Modus unterstützen"""
    
    all_models = list_all_models(
        base_dirs.get("image", Path("models/image")),
        base_dirs.get("video", Path("models/video")),
        base_dirs.get("llm", Path("models/llm"))
    )
    
    compatible = []
    
    for model_type, models in all_models.items():
        for model_dict in models:
            modalities = model_dict.get("modalities", [])
            
            # Prüfe Kompatibilität
            if mode in modalities:
                model_info = ModelInfo(
                    id=model_dict["id"],
                    name=model_dict["name"],
                    type=model_dict["type"],
                    path=model_dict["path"],
                    format=model_dict["format"],
                    backend=model_dict["backend"],
                    nsfw_capable=model_dict["nsfw_capable"],
                    modalities=model_dict["modalities"],
                    architecture=model_dict.get("architecture"),
                    size_bytes=model_dict.get("size_bytes", 0),
                    metadata=model_dict.get("metadata", {})
                )
                
                # Zusätzlich Pfad validieren
                if validate_model_path(model_info):
                    compatible.append(model_info)
                else:
                    logger.warning(f"Model path invalid: {model_info.name} at {model_info.path}")
    
    return compatible

def create_mock_models_for_testing() -> Dict[str, List[Dict[str, Any]]]:
    """Erstelle Mock-Modelle für Testing wenn keine echten verfügbar sind"""
    
    mock_models = {
        "image": [
            {
                "id": "image:test-sd15",
                "name": "Test SD 1.5",
                "type": "image",
                "path": "/dev/null",
                "format": "safetensors",
                "backend": "diffusers",
                "nsfw_capable": True,
                "modalities": ["text2img", "img2img"],
                "architecture": "sd15",
                "size_bytes": 4000000000,
                "metadata": {"mock": True}
            },
            {
                "id": "image:test-sdxl",
                "name": "Test SDXL Base",
                "type": "image", 
                "path": "/dev/null",
                "format": "diffusers_dir",
                "backend": "diffusers",
                "nsfw_capable": True,
                "modalities": ["text2img", "img2img"],
                "architecture": "sdxl",
                "size_bytes": 7000000000,
                "metadata": {"mock": True}
            }
        ],
        "video": [
            {
                "id": "video:test-svd",
                "name": "Test SVD",
                "type": "video",
                "path": "/dev/null",
                "format": "diffusers_dir",
                "backend": "diffusers",
                "nsfw_capable": True,
                "modalities": ["img2video"],
                "architecture": "svd",
                "size_bytes": 10000000000,
                "metadata": {"mock": True}
            }
        ],
        "llm": [
            {
                "id": "llm:test-llama",
                "name": "Test Llama 3",
                "type": "llm",
                "path": "",
                "format": "ollama",
                "backend": "ollama",
                "nsfw_capable": True,
                "modalities": ["text", "chat", "instruct"],
                "architecture": "llama",
                "size_bytes": 0,
                "metadata": {"mock": True, "ollama": True}
            }
        ]
    }
    
    logger.warning("🧪 Using mock models for testing - no real models found")
    return mock_models

def auto_discover_models_with_fallback(image_dir: Path, video_dir: Path, llm_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Entdecke Modelle automatisch mit Fallback auf Mock-Modelle"""
    
    try:
        models = list_all_models(image_dir, video_dir, llm_dir)
        
        # Prüfe ob überhaupt Modelle gefunden wurden
        total_models = sum(len(models[key]) for key in models.keys())
        
        if total_models == 0:
            logger.warning("No models found in any directory - using mock models")
            return create_mock_models_for_testing()
        
        return models
        
    except Exception as e:
        logger.error(f"Model discovery failed: {e}")
        logger.warning("Falling back to mock models")
        return create_mock_models_for_testing()

# Hauptfunktion für einfache Nutzung
def discover_models(base_path: str = "models") -> Dict[str, List[Dict[str, Any]]]:
    """Einfache Modell-Entdeckung mit Standard-Pfaden"""
    
    base = Path(base_path)
    
    return auto_discover_models_with_fallback(
        image_dir=base / "image",
        video_dir=base / "video", 
        llm_dir=base / "llm"
    )