# server/routes/universal_generate_fixed.py - Korrigierte universelle Generierung
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
from typing import Optional, Dict, Any, Union
import time, io, uuid, logging, json
from PIL import Image
from PIL.Image import Image as PILImage
import asyncio

# Import aller verfügbaren Backends
from ..settings import load_settings
from ..models_registry import list_all_models
from ..state import OPS

logger = logging.getLogger(__name__)

# Backend-Verfügbarkeit prüfen
BACKENDS = {}

try:
    import torch
    from diffusers import (
        AutoPipelineForText2Image, AutoPipelineForImage2Image,
        StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
        StableVideoDiffusionPipeline, DiffusionPipeline
    )
    BACKENDS['diffusers'] = True
    logger.info("✅ Diffusers backend available")
except ImportError as e:
    BACKENDS['diffusers'] = False
    logger.warning(f"❌ Diffusers not available: {e}")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    BACKENDS['transformers'] = True
    logger.info("✅ Transformers backend available")
except ImportError:
    BACKENDS['transformers'] = False

try:
    from llama_cpp import Llama
    BACKENDS['llama_cpp'] = True
    logger.info("✅ llama.cpp backend available")
except ImportError:
    BACKENDS['llama_cpp'] = False

router = APIRouter()

def get_device():
    """Bestimme bestes verfügbares Device"""
    if BACKENDS.get('diffusers') and torch.cuda.is_available():
        return "cuda"
    elif BACKENDS.get('diffusers') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_dtype():
    """Bestimme optimalen Datentyp"""
    if not BACKENDS.get('diffusers'):
        return None
    device = get_device()
    if device == 'cpu':
        return torch.float32
    else:
        return torch.float16

def find_model_by_id(model_id: str) -> Optional[Dict[str, Any]]:
    """Finde Modell nach ID in allen Kategorien"""
    s = load_settings()
    img_dir = Path(s.paths.image)
    vid_dir = Path(s.paths.video)
    llm_dir = Path(s.paths.get('llm', 'models/llm'))
    
    all_models = list_all_models(img_dir, vid_dir, llm_dir)
    
    for model_type, models in all_models.items():
        for model in models:
            if model.get('id') == model_id or model.get('name') == model_id:
                return model
    return None

def validate_model_compatibility(model_info: Dict[str, Any], mode: str) -> bool:
    """Prüfe ob Modell den gewünschten Modus unterstützt"""
    modalities = model_info.get('modalities', [])
    
    if mode in ['text2img', 'img2img']:
        return any(m in modalities for m in ['text2img', 'img2img'])
    elif mode in ['text2video', 'img2video']:
        return any(m in modalities for m in ['text2video', 'img2video'])
    elif mode == 'text':
        return any(m in modalities for m in ['text', 'chat', 'instruct'])
    
    return False

def load_image_from_upload(upload_file: UploadFile) -> PILImage:
    """Lade und konvertiere Upload-Bild"""
    try:
        content = upload_file.file.read()
        img = Image.open(io.BytesIO(content))
        
        # Konvertiere zu RGB
        if img.mode != 'RGB':
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                rgb_img.paste(img, mask=img.split()[-1] if len(img.split()) > 3 else None)
            else:
                rgb_img.paste(img)
            return rgb_img
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

def load_image_from_url(init_url: str) -> Optional[PILImage]:
    """Lade Bild aus lokaler URL"""
    if not init_url or not init_url.startswith("/outputs/"):
        return None
    
    try:
        # Entferne führenden Slash und baue Pfad
        local_path = Path(".") / init_url.lstrip("/")
        
        if not local_path.exists():
            logger.warning(f"Local image not found: {local_path}")
            return None
            
        img = Image.open(local_path)
        return img.convert("RGB")
    except Exception as e:
        logger.warning(f"Could not load local image {init_url}: {e}")
        return None

class UniversalPipeline:
    """Universelle Pipeline für alle Modelltypen"""
    
    def __init__(self, model_path: Path, model_info: Dict[str, Any]):
        self.model_path = model_path
        self.model_info = model_info
        self.pipeline = None
        self.device = get_device()
        self.dtype = get_dtype()
        
    def load(self) -> bool:
        """Lade Pipeline basierend auf Modellformat"""
        try:
            format_type = self.model_info.get('format', 'unknown')
            architecture = self.model_info.get('architecture', 'unknown')
            backend = self.model_info.get('backend', 'unknown')
            
            logger.info(f"Loading {format_type} model with {backend} backend: {self.model_path}")
            
            if not BACKENDS.get('diffusers'):
                raise ValueError("Diffusers backend not available")
            
            # Diffusers Directory
            if format_type == 'diffusers_dir':
                self.pipeline = self._load_diffusers_directory(architecture)
            
            # Single Files
            elif format_type in ['safetensors', 'checkpoint']:
                self.pipeline = self._load_single_file(architecture)
            
            # Ollama (external)
            elif backend == 'ollama':
                self.pipeline = self._create_ollama_interface()
            
            else:
                raise ValueError(f"Unsupported model format: {format_type}")
            
            return self.pipeline is not None
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            return False
    
    def _load_diffusers_directory(self, architecture: str):
        """Lade Diffusers-Verzeichnis"""
        try:
            # Video Modelle (SVD)
            if architecture == 'svd':
                return StableVideoDiffusionPipeline.from_pretrained(
                    str(self.model_path),
                    torch_dtype=self.dtype,
                    variant="fp16" if self.dtype == torch.float16 else None
                ).to(self.device)
            
            # Standard SD Modelle  
            else:
                return AutoPipelineForText2Image.from_pretrained(
                    str(self.model_path),
                    torch_dtype=self.dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(self.device)
                
        except Exception as e:
            logger.error(f"Error loading diffusers directory: {e}")
            raise
    
    def _load_single_file(self, architecture: str):
        """Lade Single-File Modell"""
        try:
            return StableDiffusionPipeline.from_single_file(
                str(self.model_path),
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
        except Exception as e:
            logger.error(f"Error loading single file: {e}")
            raise
    
    def _create_ollama_interface(self):
        """Erstelle Ollama-Interface"""
        return OllamaInterface(self.model_info.get('name', 'unknown'))
    
    def generate_image(self, prompt: str, **kwargs) -> PILImage:
        """Universelle Bildgeneration"""
        if not self.pipeline:
            raise RuntimeError("Pipeline not loaded")
        
        try:
            # Memory optimization
            if torch.cuda.is_available() and hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
                self.pipeline.enable_attention_slicing()
            
            # Parameter-Mapping
            generation_kwargs = {}
            
            if kwargs.get('width'):
                generation_kwargs['width'] = int(kwargs['width'])
            if kwargs.get('height'):
                generation_kwargs['height'] = int(kwargs['height'])
            if kwargs.get('num_inference_steps'):
                generation_kwargs['num_inference_steps'] = int(kwargs['num_inference_steps'])
            if kwargs.get('guidance_scale'):
                generation_kwargs['guidance_scale'] = float(kwargs['guidance_scale'])
            if kwargs.get('negative_prompt'):
                generation_kwargs['negative_prompt'] = kwargs['negative_prompt']
            
            # Img2Img Parameter
            if kwargs.get('image'):
                generation_kwargs['image'] = kwargs['image']
            if kwargs.get('strength'):
                generation_kwargs['strength'] = float(kwargs['strength'])
            
            # Generierung
            if kwargs.get('image') and hasattr(self.pipeline, 'from_single_file'):
                # Für img2img müssen wir eine andere Pipeline laden
                img2img_pipeline = AutoPipelineForImage2Image.from_pretrained(
                    str(self.model_path),
                    torch_dtype=self.dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(self.device)
                result = img2img_pipeline(prompt=prompt, **generation_kwargs)
            else:
                result = self.pipeline(prompt=prompt, **generation_kwargs)
            
            return result.images[0]
                
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise
    
    def generate_video(self, prompt: str = None, image: PILImage = None, **kwargs):
        """Universelle Videogeneration"""
        if not self.pipeline:
            raise RuntimeError("Pipeline not loaded")
        
        architecture = self.model_info.get('architecture')
        
        if architecture != 'svd':
            raise ValueError(f"Video generation not supported for architecture: {architecture}")
        
        if image is None:
            raise ValueError("SVD requires input image for video generation")
        
        try:
            # Memory optimization
            if torch.cuda.is_available() and hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
            
            num_frames = kwargs.get('num_frames', 14)
            
            result = self.pipeline(
                image=image,
                num_frames=num_frames,
                decode_chunk_size=8,
                motion_bucket_id=kwargs.get('motion_bucket_id', 127),
                noise_aug_strength=kwargs.get('noise_aug_strength', 0.02)
            )
            return result.frames[0]
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            raise
    
    def cleanup(self):
        """Räume Pipeline-Ressourcen auf"""
        if self.pipeline and BACKENDS.get('diffusers'):
            try:
                if hasattr(self.pipeline, 'to') and torch.cuda.is_available():
                    self.pipeline.to('cpu')
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")
        self.pipeline = None

class OllamaInterface:
    """Interface für Ollama-Modelle"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generiere Text via Ollama"""
        try:
            import subprocess
            import json
            
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "temperature": kwargs.get('temperature', 0.8),
                    "top_p": kwargs.get('top_p', 0.95),
                    "num_predict": kwargs.get('max_tokens', 512)
                },
                "stream": False
            }
            
            result = subprocess.run(
                ["ollama", "generate"],
                input=json.dumps(request_data),
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                response_data = json.loads(result.stdout)
                return response_data.get('response', '')
            else:
                raise RuntimeError(f"Ollama error: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

@router.post("/generate/universal")
async def universal_generate(
    prompt: str = Form(...),
    model_id: str = Form(...),
    mode: str = Form("text2img"),
    nsfw: bool = Form(False),
    
    # Image Parameters
    init_image: Optional[UploadFile] = File(None),
    init_url: Optional[str] = Form(None),
    width: int = Form(1024),
    height: int = Form(768),
    num_inference_steps: int = Form(28),
    guidance_scale: float = Form(7.0),
    strength: float = Form(0.65),
    negative_prompt: Optional[str] = Form(None),
    
    # Video Parameters
    num_frames: int = Form(14),
    fps: int = Form(24),
    motion_bucket_id: int = Form(127),
    noise_aug_strength: float = Form(0.02),
    
    # Output Parameters
    output_format: str = Form("png"),
    jpg_quality: int = Form(92)
):
    """Universeller Generator für alle Modelltypen"""
    
    # Validierung
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt darf nicht leer sein")
    
    # Modell finden
    model_info = find_model_by_id(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Modell nicht gefunden: {model_id}")
    
    # Modalitäts-Kompatibilität prüfen
    if not validate_model_compatibility(model_info, mode):
        available_modes = model_info.get('modalities', [])
        raise HTTPException(
            status_code=400, 
            detail=f"Modell {model_info['name']} unterstützt nicht den Modus '{mode}'. Verfügbar: {available_modes}"
        )
    
    # Output-Setup
    s = load_settings()
    model_type = model_info.get('type')
    
    if model_type == 'image' or mode.endswith('2img'):
        out_dir = Path(s.paths.images)
        file_ext = f".{output_format.lower()}"
        url_prefix = "/outputs/images/"
    elif model_type == 'video' or mode.endswith('2video'):
        out_dir = Path(s.paths.videos)
        file_ext = ".mp4"
        url_prefix = "/outputs/videos/"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model type: {model_type}")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Unique filename
    timestamp = int(time.time())
    filename = f"{model_type}_{mode}_{timestamp}_{uuid.uuid4().hex[:8]}{file_ext}"
    output_path = out_dir / filename
    
    # Operation tracking
    op_id = f"universal-{timestamp}"
    OPS.start(op_id, kind=f"{model_type}:{mode}", nsfw=nsfw, params={
        "model": model_info['name'],
        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt
    })
    
    pipeline = None
    
    try:
        # Pipeline laden
        OPS.update(op_id, progress=0.1)
        model_path = Path(model_info['path']) if model_info.get('path') else None
        
        if not model_path or not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Modell-Pfad nicht gefunden: {model_path}")
        
        pipeline = UniversalPipeline(model_path, model_info)
        
        if not pipeline.load():
            raise HTTPException(status_code=500, detail="Pipeline konnte nicht geladen werden")
        
        OPS.update(op_id, progress=0.3)
        
        # Input-Bild laden wenn nötig
        input_image = None
        if mode in ['img2img', 'img2video'] or init_image or init_url:
            try:
                if init_image:
                    input_image = load_image_from_upload(init_image)
                elif init_url:
                    input_image = load_image_from_url(init_url)
                
                if input_image and mode.endswith('2img'):
                    input_image = input_image.resize((width, height), Image.Resampling.LANCZOS)
                    
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Input-Bild konnte nicht geladen werden: {e}")
        
        # Generierung basierend auf Modus
        OPS.update(op_id, progress=0.5)
        
        if mode in ['text2img', 'img2img']:
            # Bildgeneration
            generation_kwargs = {
                'width': width,
                'height': height,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'negative_prompt': negative_prompt,
            }
            
            if mode == 'img2img':
                if input_image is None:
                    raise HTTPException(status_code=400, detail="img2img benötigt ein Eingangsbild")
                generation_kwargs['image'] = input_image
                generation_kwargs['strength'] = strength
            
            result_image = pipeline.generate_image(prompt, **generation_kwargs)
            
            # Speichern
            if output_format.lower() == 'png':
                result_image.save(output_path, 'PNG')
            else:
                result_image.save(output_path, 'JPEG', quality=jpg_quality, optimize=True)
        
        elif mode in ['text2video', 'img2video']:
            # Videogeneration
            if mode == 'text2video' and input_image is None:
                # Erst Bild generieren für text2video
                OPS.update(op_id, progress=0.4)
                temp_image = pipeline.generate_image(prompt, width=width, height=height)
                input_image = temp_image
            
            if input_image is None:
                raise HTTPException(status_code=400, detail=f"{mode} benötigt ein Eingangsbild")
            
            OPS.update(op_id, progress=0.6)
            
            frames = pipeline.generate_video(
                prompt=prompt, 
                image=input_image, 
                num_frames=num_frames,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength
            )
            
            # Als Video speichern
            import imageio
            imageio.mimwrite(
                str(output_path),
                frames,
                fps=fps,
                quality=9,
                codec='libx264',
                output_params=['-pix_fmt', 'yuv420p']
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unbekannter Modus: {mode}")
        
        OPS.update(op_id, progress=0.9)
        
        # Erfolg
        OPS.finish(op_id)
        
        return {
            "ok": True,
            "mode": mode,
            "model": model_info['name'],
            "file": f"{url_prefix}{filename}",
            "metadata": {
                "prompt": prompt,
                "model_id": model_id,
                "architecture": model_info.get('architecture'),
                "format": model_info.get('format'),
                "resolution": f"{width}x{height}" if mode.endswith('2img') else None,
                "frames": num_frames if mode.endswith('2video') else None,
                "duration_sec": num_frames / fps if mode.endswith('2video') else None,
                "generation_time": time.time() - timestamp
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        OPS.finish(op_id, error=str(e))
        logger.error(f"Universal generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation fehlgeschlagen: {e}")
        
    finally:
        # Cleanup
        if pipeline:
            pipeline.cleanup()

@router.get("/models/compatible")
async def get_compatible_models(mode: str):
    """Liste kompatible Modelle für einen Modus"""
    s = load_settings()
    img_dir = Path(s.paths.image)
    vid_dir = Path(s.paths.video)
    llm_dir = Path(s.paths.get('llm', 'models/llm'))
    
    all_models = list_all_models(img_dir, vid_dir, llm_dir)
    compatible = []
    
    for model_type, models in all_models.items():
        for model in models:
            if validate_model_compatibility(model, mode):
                compatible.append(model)
    
    return {
        "ok": True,
        "mode": mode,
        "compatible_models": compatible,
        "count": len(compatible)
    }

@router.get("/generation/status/{op_id}")
async def get_generation_status(op_id: str):
    """Prüfe Status einer laufenden Generierung"""
    ops = OPS.list()
    for op in ops:
        if op.get('id') == op_id:
            return {
                "ok": True,
                "status": op.get('status'),
                "progress": op.get('progress', 0),
                "error": op.get('error'),
                "params": op.get('params', {}),
                "duration": time.time() - op.get('ts_start', time.time())
            }
    
    return {
        "ok": False,
        "error": "Operation not found"
    }