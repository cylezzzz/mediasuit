# server/routes/universal_generate.py - Universeller Generator für alle Modellarten
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import time, io, uuid, logging
from PIL import Image
from PIL.Image import Image as PILImage

# Import aller möglichen Backends
from ..settings import load_settings
from ..models_registry import list_all_models, ModelInfo
from ..state import OPS

logger = logging.getLogger(__name__)

# Versuche alle möglichen ML-Bibliotheken zu importieren
BACKENDS = {}

# Diffusers (Hauptbibliothek)
try:
    import torch
    from diffusers import (
        AutoPipelineForText2Image, AutoPipelineForImage2Image,
        StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
        StableVideoDiffusionPipeline, DiffusionPipeline,
        StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
    )
    BACKENDS['diffusers'] = True
    logger.info("✅ Diffusers backend available")
except ImportError as e:
    BACKENDS['diffusers'] = False
    logger.warning(f"❌ Diffusers not available: {e}")

# ONNX Runtime
try:
    import onnxruntime as ort
    from diffusers import OnnxStableDiffusionPipeline
    BACKENDS['onnxruntime'] = True
    logger.info("✅ ONNX Runtime backend available")
except ImportError:
    BACKENDS['onnxruntime'] = False
    logger.warning("❌ ONNX Runtime not available")

# llama.cpp (für GGUF-Modelle)
try:
    from llama_cpp import Llama
    BACKENDS['llama_cpp'] = True
    logger.info("✅ llama.cpp backend available")
except ImportError:
    BACKENDS['llama_cpp'] = False
    logger.warning("❌ llama.cpp not available")

# Transformers (für LLMs)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    BACKENDS['transformers'] = True
    logger.info("✅ Transformers backend available")
except ImportError:
    BACKENDS['transformers'] = False
    logger.warning("❌ Transformers not available")

# SafeTensors
try:
    from safetensors.torch import load_file as load_safetensors
    BACKENDS['safetensors'] = True
except ImportError:
    BACKENDS['safetensors'] = False

router = APIRouter()

class UniversalPipeline:
    """Universelle Pipeline die alle Modelltypen handhaben kann"""
    
    def __init__(self, model_path: Path, model_info: Dict[str, Any]):
        self.model_path = model_path
        self.model_info = model_info
        self.pipeline = None
        self.device = self._get_device()
        self.dtype = self._get_dtype()
        
    def _get_device(self) -> str:
        """Bestimme bestes verfügbares Device"""
        if BACKENDS.get('diffusers') and torch.cuda.is_available():
            return "cuda"
        elif BACKENDS.get('diffusers') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"
    
    def _get_dtype(self):
        """Bestimme optimalen Datentyp"""
        if not BACKENDS.get('diffusers'):
            return None
        
        precision = self.model_info.get('precision', 'fp16')
        if precision == 'fp32' or self.device == 'cpu':
            return torch.float32
        else:
            return torch.float16
    
    def load(self) -> bool:
        """Lade Pipeline basierend auf Modellformat"""
        try:
            format_type = self.model_info.get('format', 'unknown')
            architecture = self.model_info.get('architecture', 'unknown')
            backend = self.model_info.get('backend', 'unknown')
            
            logger.info(f"Loading {format_type} model with {backend} backend: {self.model_path}")
            
            # Diffusers Directory (Hauptfall)
            if format_type == 'diffusers_dir' and BACKENDS['diffusers']:
                self.pipeline = self._load_diffusers_directory(architecture)
            
            # SafeTensors/Checkpoint Single Files
            elif format_type in ['safetensors', 'checkpoint'] and BACKENDS['diffusers']:
                self.pipeline = self._load_single_file(architecture)
            
            # ONNX Models
            elif format_type == 'onnx' and BACKENDS['onnxruntime']:
                self.pipeline = self._load_onnx_model()
            
            # GGUF Models (LLM)
            elif format_type == 'gguf' and BACKENDS['llama_cpp']:
                self.pipeline = self._load_gguf_model()
            
            # Transformers Models (LLM)
            elif format_type == 'transformers_dir' and BACKENDS['transformers']:
                self.pipeline = self._load_transformers_model()
            
            # Ollama (external)
            elif backend == 'ollama':
                self.pipeline = self._create_ollama_interface()
            
            else:
                raise ValueError(f"Unsupported model format: {format_type} with backend {backend}")
            
            return self.pipeline is not None
            
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            return False
    
    def _load_diffusers_directory(self, architecture: str):
        """Lade Diffusers-Verzeichnis basierend auf Architektur"""
        
        # SDXL Modelle
        if architecture == 'sdxl':
            return StableDiffusionXLPipeline.from_pretrained(
                str(self.model_path),
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
        
        # Standard SD Modelle  
        elif architecture in ['sd15', 'sd20', 'sd21']:
            return StableDiffusionPipeline.from_pretrained(
                str(self.model_path),
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
        
        # Video Modelle (SVD)
        elif architecture == 'svd':
            return StableVideoDiffusionPipeline.from_pretrained(
                str(self.model_path),
                torch_dtype=self.dtype
            ).to(self.device)
        
        # Auto-Pipeline als Fallback
        else:
            return AutoPipelineForText2Image.from_pretrained(
                str(self.model_path),
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
    
    def _load_single_file(self, architecture: str):
        """Lade Single-File Modell (.safetensors/.ckpt)"""
        
        # SDXL Single Files
        if architecture == 'sdxl':
            return StableDiffusionXLPipeline.from_single_file(
                str(self.model_path),
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
        
        # Standard SD Single Files
        else:
            return StableDiffusionPipeline.from_single_file(
                str(self.model_path),
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)
    
    def _load_onnx_model(self):
        """Lade ONNX-Modell"""
        providers = ['CPUExecutionProvider']
        if self.device == 'cuda':
            providers.insert(0, 'CUDAExecutionProvider')
        
        return OnnxStableDiffusionPipeline.from_pretrained(
            str(self.model_path),
            provider=providers
        )
    
    def _load_gguf_model(self):
        """Lade GGUF-Modell mit llama.cpp"""
        # Finde .gguf Datei im Verzeichnis
        gguf_files = list(self.model_path.glob("*.gguf"))
        if not gguf_files:
            raise FileNotFoundError("No .gguf file found in model directory")
        
        gguf_file = gguf_files[0]  # Nimm erste .gguf Datei
        
        return Llama(
            model_path=str(gguf_file),
            n_ctx=4096,
            n_gpu_layers=-1 if self.device == 'cuda' else 0,
            verbose=False
        )
    
    def _load_transformers_model(self):
        """Lade Transformers-Modell"""
        device_map = "auto" if self.device == "cuda" else None
        
        model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=self.dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map=device_map
        )
    
    def _create_ollama_interface(self):
        """Erstelle Ollama-Interface"""
        return OllamaInterface(self.model_info.get('name', 'unknown'))
    
    def generate_image(self, prompt: str, **kwargs) -> PILImage:
        """Universelle Bildgeneration"""
        if not self.pipeline:
            raise RuntimeError("Pipeline not loaded")
        
        model_type = self.model_info.get('type')
        architecture = self.model_info.get('architecture')
        
        # Parameter-Mapping basierend auf Modelltyp
        generation_kwargs = self._map_image_parameters(architecture, kwargs)
        
        try:
            if hasattr(self.pipeline, '__call__'):
                result = self.pipeline(prompt=prompt, **generation_kwargs)
                return result.images[0] if hasattr(result, 'images') else result
            else:
                raise ValueError(f"Pipeline doesn't support image generation: {type(self.pipeline)}")
                
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise
    
    def generate_video(self, prompt: str = None, image: PILImage = None, **kwargs):
        """Universelle Videogeneration"""
        if not self.pipeline:
            raise RuntimeError("Pipeline not loaded")
        
        architecture = self.model_info.get('architecture')
        
        if architecture == 'svd':
            # SVD benötigt Eingangsbild
            if image is None:
                raise ValueError("SVD requires input image for video generation")
            
            result = self.pipeline(
                image=image,
                num_frames=kwargs.get('num_frames', 14),
                decode_chunk_size=8,
                motion_bucket_id=kwargs.get('motion_bucket_id', 127),
                noise_aug_strength=kwargs.get('noise_aug_strength', 0.02)
            )
            return result.frames[0]
        else:
            raise ValueError(f"Video generation not supported for architecture: {architecture}")
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Universelle Textgeneration"""
        if not self.pipeline:
            raise RuntimeError("Pipeline not loaded")
        
        backend = self.model_info.get('backend')
        
        if backend == 'llama_cpp':
            # GGUF mit llama.cpp
            result = self.pipeline(
                prompt,
                max_tokens=kwargs.get('max_tokens', 512),
                temperature=kwargs.get('temperature', 0.8),
                top_p=kwargs.get('top_p', 0.95),
                echo=False
            )
            return result['choices'][0]['text']
        
        elif backend == 'transformers':
            # Transformers Pipeline
            result = self.pipeline(
                prompt,
                max_new_tokens=kwargs.get('max_tokens', 512),
                temperature=kwargs.get('temperature', 0.8),
                top_p=kwargs.get('top_p', 0.95),
                do_sample=True
            )
            return result[0]['generated_text'][len(prompt):]
        
        elif backend == 'ollama':
            # Ollama Interface
            return self.pipeline.generate(prompt, **kwargs)
        
        else:
            raise ValueError(f"Text generation not supported for backend: {backend}")
    
    def _map_image_parameters(self, architecture: str, kwargs: Dict) -> Dict:
        """Mappe Parameter auf modellspezifische Namen"""
        mapped = {}
        
        # Basis-Parameter für alle SD-Modelle
        if kwargs.get('width'):
            mapped['width'] = int(kwargs['width'])
        if kwargs.get('height'):
            mapped['height'] = int(kwargs['height'])
        if kwargs.get('num_inference_steps'):
            mapped['num_inference_steps'] = int(kwargs['num_inference_steps'])
        if kwargs.get('guidance_scale'):
            mapped['guidance_scale'] = float(kwargs['guidance_scale'])
        if kwargs.get('negative_prompt'):
            mapped['negative_prompt'] = kwargs['negative_prompt']
        
        # SDXL-spezifische Parameter
        if architecture == 'sdxl':
            if kwargs.get('original_size'):
                mapped['original_size'] = kwargs['original_size']
            if kwargs.get('crops_coords_top_left'):
                mapped['crops_coords_top_left'] = kwargs['crops_coords_top_left']
            if kwargs.get('target_size'):
                mapped['target_size'] = kwargs['target_size']
        
        # Img2Img spezifische Parameter
        if kwargs.get('image'):
            mapped['image'] = kwargs['image']
        if kwargs.get('strength'):
            mapped['strength'] = float(kwargs['strength'])
        
        return mapped
    
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
            
            # Bereite Ollama-Aufruf vor
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "options": {
                    "temperature": kwargs.get('temperature', 0.8),
                    "top_p": kwargs.get('top_p', 0.95),
                    "max_tokens": kwargs.get('max_tokens', 512)
                },
                "stream": False
            }
            
            # Rufe Ollama auf
            result = subprocess.run(
                ["ollama", "generate"],
                input=json.dumps(request_data),
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                response_data = json.loads(result.stdout)
                return response_data.get('response', '')
            else:
                raise RuntimeError(f"Ollama error: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

# Utility Functions
def find_model_by_id(model_id: str) -> Optional[Dict[str, Any]]:
    """Finde Modell nach ID"""
    s = load_settings()
    img_dir = Path(s.paths.image)
    vid_dir = Path(s.paths.video)
    llm_dir = Path(s.paths.get('llm', 'models/llm'))
    
    all_models = list_all_models(img_dir, vid_dir, llm_dir)
    
    for model_type, models in all_models.items():
        for model in models:
            if model.get('id') == model_id:
                return model
    return None

def get_compatible_modalities(model_info: Dict[str, Any], mode: str) -> bool:
    """Prüfe ob Modell den gewünschten Modus unterstützt"""
    modalities = model_info.get('modalities', [])
    return mode in modalities

# Main Generation Routes
@router.post("/generate/universal")
async def universal_generate(
    prompt: str = Form(...),
    model_id: str = Form(...),
    mode: str = Form("text2img"),  # text2img, img2img, text2video, img2video, text
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
    
    # Text Parameters
    max_tokens: int = Form(512),
    temperature: float = Form(0.8),
    top_p: float = Form(0.95),
    
    # Output Parameters
    output_format: str = Form("png"),  # png, jpg, mp4
    jpg_quality: int = Form(92)
):
    """
    Universeller Generator der alle Modelltypen und -formate unterstützt:
    - Diffusers (SD 1.5, SD 2.x, SDXL, SVD)
    - Single Files (.safetensors, .ckpt)
    - ONNX Modelle
    - GGUF LLMs (llama.cpp)
    - Transformers LLMs
    - Ollama LLMs
    """
    
    # Validierung
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt darf nicht leer sein")
    
    # Modell finden
    model_info = find_model_by_id(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Modell nicht gefunden: {model_id}")
    
    # Modalitäts-Kompatibilität prüfen
    if not get_compatible_modalities(model_info, mode):
        raise HTTPException(
            status_code=400, 
            detail=f"Modell {model_info['name']} unterstützt nicht den Modus '{mode}'. Verfügbar: {model_info.get('modalities', [])}"
        )
    
    # Settings und Ausgabe-Setup
    s = load_settings()
    model_type = model_info.get('type')
    
    if model_type in ['image'] or mode.endswith('2img'):
        out_dir = Path(s.paths.images)
        file_ext = f".{output_format.lower()}"
        url_prefix = "/outputs/images/"
    elif model_type in ['video'] or mode.endswith('2video'):
        out_dir = Path(s.paths.videos) 
        file_ext = ".mp4"
        url_prefix = "/outputs/videos/"
    else:  # text/llm
        out_dir = Path(s.paths.get('text_outputs', 'outputs/text'))
        file_ext = ".txt"
        url_prefix = "/outputs/text/"
    
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
                    # Upload verarbeiten
                    image_data = await init_image.read()
                    input_image = Image.open(io.BytesIO(image_data)).convert('RGB')
                elif init_url and init_url.startswith('/outputs/'):
                    # Lokales Bild laden
                    local_path = Path(f".{init_url}")
                    if local_path.exists():
                        input_image = Image.open(local_path).convert('RGB')
                
                if input_image and mode.endswith('2img'):
                    input_image = input_image.resize((width, height), Image.Resampling.LANCZOS)
                    
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Input-Bild konnte nicht geladen werden: {e}")
        
        # Generierung basierend auf Modus
        OPS.update(op_id, progress=0.5)
        
        generation_kwargs = {
            'width': width,
            'height': height,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'negative_prompt': negative_prompt,
            'strength': strength,
            'image': input_image,
            'num_frames': num_frames,
            'motion_bucket_id': motion_bucket_id,
            'noise_aug_strength': noise_aug_strength,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p
        }
        
        if mode in ['text2img', 'img2img']:
            # Bildgeneration
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
                input_image = pipeline.generate_image(prompt, width=width, height=height)
            
            if input_image is None:
                raise HTTPException(status_code=400, detail=f"{mode} benötigt ein Eingangsbild")
            
            frames = pipeline.generate_video(prompt, image=input_image, **generation_kwargs)
            
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
        
        elif mode == 'text':
            # Textgeneration
            generated_text = pipeline.generate_text(prompt, **generation_kwargs)
            
            # Als Textdatei speichern
            output_path.write_text(generated_text, encoding='utf-8')
            
            # Zusätzlich in Response zurückgeben
            OPS.finish(op_id)
            return {
                "ok": True,
                "mode": mode,
                "model": model_info['name'],
                "file": f"{url_prefix}{filename}",
                "text": generated_text,
                "metadata": {
                    "prompt": prompt,
                    "model_id": model_id,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            }
        
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
                "duration_sec": num_frames / fps if mode.endswith('2video') else None
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

@router.get("/models/supported-modes")
async def get_supported_modes():
    """Liste alle unterstützten Modi und ihre Anforderungen"""
    return {
        "ok": True,
        "modes": {
            "text2img": {
                "name": "Text zu Bild",
                "description": "Generiere Bilder aus Textbeschreibungen",
                "required_model_types": ["image"],
                "required_modalities": ["text2img"],
                "parameters": ["prompt", "width", "height", "steps", "guidance", "negative_prompt"]
            },
            "img2img": {
                "name": "Bild zu Bild", 
                "description": "Verändere/Remixe vorhandene Bilder",
                "required_model_types": ["image"],
                "required_modalities": ["img2img"],
                "parameters": ["prompt", "init_image", "strength", "guidance", "negative_prompt"]
            },
            "text2video": {
                "name": "Text zu Video",
                "description": "Generiere Videos aus Text (erstellt automatisch Startbild)",
                "required_model_types": ["video", "image"],
                "required_modalities": ["text2video", "img2video"],
                "parameters": ["prompt", "num_frames", "fps", "motion_bucket_id"]
            },
            "img2video": {
                "name": "Bild zu Video", 
                "description": "Animiere Standbilder zu Videos",
                "required_model_types": ["video"],
                "required_modalities": ["img2video"],
                "parameters": ["init_image", "num_frames", "fps", "motion_bucket_id", "noise_aug_strength"]
            },
            "text": {
                "name": "Textgeneration",
                "description": "Generiere und vervollständige Texte",
                "required_model_types": ["llm"],
                "required_modalities": ["text", "chat"],
                "parameters": ["prompt", "max_tokens", "temperature", "top_p"]
            }
        },
        "backends": BACKENDS,
        "recommendations": {
            "best_image": "Verwende SDXL-Modelle für beste Bildqualität",
            "best_video": "SVD (Stable Video Diffusion) für professionelle Videos", 
            "best_text": "Llama 3 oder Mixtral für intelligente Gespräche",
            "hardware": "GPU mit 8GB+ VRAM für beste Performance"
        }
    }

@router.get("/models/compatibility-check")
async def check_system_compatibility():
    """Prüfe Systemkompatibilität für verschiedene Modelltypen"""
    
    compatibility = {
        "system": {},
        "backends": BACKENDS,
        "recommendations": []
    }
    
    # System-Info sammeln
    try:
        import psutil
        compatibility["system"]["ram_gb"] = psutil.virtual_memory().total / (1024**3)
        compatibility["system"]["cpu_count"] = psutil.cpu_count()
    except:
        compatibility["system"]["ram_gb"] = 0
        compatibility["system"]["cpu_count"] = 0
    
    # GPU-Info
    if BACKENDS.get('diffusers'):
        try:
            import torch
            compatibility["system"]["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                compatibility["system"]["gpu_name"] = torch.cuda.get_device_name(0)
                compatibility["system"]["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            else:
                compatibility["system"]["gpu_memory_gb"] = 0
        except:
            compatibility["system"]["cuda_available"] = False
            compatibility["system"]["gpu_memory_gb"] = 0
    
    # Empfehlungen basierend auf Hardware
    gpu_mem = compatibility["system"].get("gpu_memory_gb", 0)
    ram_gb = compatibility["system"].get("ram_gb", 0)
    
    if gpu_mem >= 12:
        compatibility["recommendations"].extend([
            "✅ Kann alle Modelltypen (SD15, SDXL, SVD) ausführen",
            "🚀 Empfohlen: SDXL + SVD für beste Qualität"
        ])
    elif gpu_mem >= 8:
        compatibility["recommendations"].extend([
            "✅ Kann SDXL und die meisten Modelle ausführen", 
            "⚠️ SVD könnte knapp werden - Model CPU Offloading aktivieren"
        ])
    elif gpu_mem >= 4:
        compatibility["recommendations"].extend([
            "✅ Kann SD 1.5 Modelle gut ausführen",
            "⚠️ SDXL mit CPU Offloading möglich",
            "❌ SVD zu ressourcenhungrig"
        ])
    else:
        compatibility["recommendations"].extend([
            "❌ GPU zu schwach für moderne Modelle",
            "💡 Verwende ONNX/Quantisierte Modelle oder CPU-Mode"
        ])
    
    if ram_gb >= 32:
        compatibility["recommendations"].append("✅ Kann große LLMs (Mixtral 8x7B) laden")
    elif ram_gb >= 16:
        compatibility["recommendations"].append("✅ Kann mittlere LLMs (Llama 3 8B) laden")
    else:
        compatibility["recommendations"].append("⚠️ Nur kleine LLMs oder quantisierte Versionen empfohlen")
    
    return {
        "ok": True,
        **compatibility
    }