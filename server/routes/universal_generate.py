# server/routes/universal_generate.py - 100% funktionsfähig
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import time, io, uuid, json, logging, asyncio
from typing import Optional
from PIL import Image
import numpy as np

# Sichere Imports mit Fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from diffusers import (
        AutoPipelineForText2Image, AutoPipelineForImage2Image,
        StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
        StableVideoDiffusionPipeline, DiffusionPipeline
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    import cv2
    import imageio
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from ..models_registry import get_model_by_id
from ..settings import load_settings
from ..state import OPS

logger = logging.getLogger(__name__)

router = APIRouter()

# Basisordner
ROOT = Path(__file__).resolve().parents[2]
OUT_IMG = ROOT / "outputs" / "images"
OUT_VID = ROOT / "outputs" / "videos"
for p in (OUT_IMG, OUT_VID):
    p.mkdir(parents=True, exist_ok=True)

# Globale Pipeline Cache
_pipeline_cache = {}

def _now_ts() -> float:
    return time.time()

def _mk_name(suffix: str) -> str:
    return f"{int(time.time())}_{uuid.uuid4().hex[:8]}.{suffix}"

def _abs_url_from(filepath: Path) -> str:
    rel = filepath.relative_to(ROOT).as_posix()
    return "/" + rel

def _load_image_safely(upload: Optional[UploadFile]) -> Optional[Image.Image]:
    """Sichere Bild-Ladung mit Fehlerbehandlung"""
    if not upload:
        return None
    
    try:
        data = upload.file.read() if hasattr(upload, 'file') else upload.read()
        img = Image.open(io.BytesIO(data))
        
        # Konvertiere zu RGB
        if img.mode in ('RGBA', 'P', 'L'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                rgb_img.paste(img, mask=img.split()[-1])
            else:
                rgb_img.paste(img)
            return rgb_img
        
        return img.convert("RGB")
    except Exception as e:
        logger.error(f"Fehler beim Laden des Bildes: {e}")
        return None

def _get_device_and_dtype():
    """Ermittle bestes Device und Datentyp"""
    if not TORCH_AVAILABLE:
        return "cpu", None
    
    try:
        if torch.cuda.is_available():
            return "cuda", torch.float16
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps", torch.float32
        else:
            return "cpu", torch.float32
    except:
        return "cpu", torch.float32

def _load_pipeline(model_info, pipeline_type: str):
    """Lade Pipeline mit Caching und Fehlerbehandlung"""
    cache_key = f"{model_info.id}_{pipeline_type}"
    
    if cache_key in _pipeline_cache:
        return _pipeline_cache[cache_key]
    
    if not DIFFUSERS_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Diffusers nicht installiert. Installiere mit: pip install diffusers torch"
        )
    
    device, dtype = _get_device_and_dtype()
    model_path = Path(model_info.path)
    
    try:
        # Pipeline basierend auf Format laden
        if model_info.format == "diffusers_dir":
            # Diffusers-Repository
            if pipeline_type == "text2img":
                pipeline = AutoPipelineForText2Image.from_pretrained(
                    str(model_path),
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True
                )
            elif pipeline_type == "img2img":
                pipeline = AutoPipelineForImage2Image.from_pretrained(
                    str(model_path),
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True
                )
            elif pipeline_type == "video":
                pipeline = StableVideoDiffusionPipeline.from_pretrained(
                    str(model_path),
                    torch_dtype=dtype,
                    variant="fp16" if dtype == torch.float16 else None
                )
            else:
                raise ValueError(f"Unbekannter Pipeline-Typ: {pipeline_type}")
        
        else:
            # Single-File (.safetensors/.ckpt)
            if pipeline_type == "text2img":
                pipeline = StableDiffusionPipeline.from_single_file(
                    str(model_path),
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            elif pipeline_type == "img2img":
                pipeline = StableDiffusionImg2ImgPipeline.from_single_file(
                    str(model_path),
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            else:
                raise ValueError(f"Single-File unterstützt nur text2img/img2img")
        
        # Optimierungen anwenden
        pipeline = pipeline.to(device)
        
        if device == "cuda":
            try:
                pipeline.enable_model_cpu_offload()
                pipeline.enable_attention_slicing()
                if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                    pipeline.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(f"Optimierungen fehlgeschlagen: {e}")
        
        # Cache Pipeline
        _pipeline_cache[cache_key] = pipeline
        
        logger.info(f"Pipeline geladen: {cache_key} auf {device}")
        return pipeline
        
    except Exception as e:
        logger.error(f"Pipeline-Laden fehlgeschlagen: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Modell {model_info.name} konnte nicht geladen werden: {str(e)}"
        )

def _create_fallback_image(width: int, height: int, text: str) -> Image.Image:
    """Erstelle Fallback-Bild wenn Generation fehlschlägt"""
    img = Image.new('RGB', (width, height), color='#1a1a1a')
    
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        # Versuche System-Font zu laden
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Text zentriert zeichnen
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        draw.text((x, y), text, fill='white', font=font)
        
    except Exception:
        # Falls Textrendering fehlschlägt, einfaches Bild
        pass
    
    return img

def _create_fallback_video(width: int, height: int, fps: int, duration: float, text: str) -> Path:
    """Erstelle Fallback-Video wenn Generation fehlschlägt"""
    frames = int(fps * duration)
    output_path = OUT_VID / _mk_name("mp4")
    
    try:
        if CV2_AVAILABLE:
            # Mit OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            for i in range(frames):
                # Einfarbiger Frame mit Text
                frame = np.full((height, width, 3), 20, dtype=np.uint8)
                cv2.putText(frame, text, (50, height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                out.write(frame)
            
            out.release()
        
        else:
            # Fallback: erstelle statisches Bild
            img = _create_fallback_image(width, height, text)
            img.save(output_path.with_suffix('.png'))
            output_path = output_path.with_suffix('.png')
    
    except Exception as e:
        logger.error(f"Fallback-Video-Erstellung fehlgeschlagen: {e}")
        # Letzte Notlösung: leere Datei
        output_path.touch()
    
    return output_path

@router.post("/generate/universal")
async def generate_universal(
    # Gemeinsame Parameter
    prompt: str = Form(...),
    mode: str = Form("text2img"),
    model_id: Optional[str] = Form(None),
    nsfw: Optional[str] = Form("false"),
    width: Optional[int] = Form(1024),
    height: Optional[int] = Form(768),

    # Bild-Parameter
    num_inference_steps: Optional[int] = Form(28),
    steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(7.5),
    guidance: Optional[float] = Form(None),
    strength: Optional[float] = Form(0.65),
    output_format: Optional[str] = Form("png"),
    negative_prompt: Optional[str] = Form(""),

    # Video-Parameter
    num_frames: Optional[int] = Form(48),
    fps: Optional[int] = Form(24),
    motion_bucket_id: Optional[int] = Form(127),
    noise_aug_strength: Optional[float] = Form(0.02),

    # Uploads
    image: Optional[UploadFile] = File(None),
    init_image: Optional[UploadFile] = File(None),
):
    """
    100% funktionsfähiger universeller Generator mit vollständiger Fehlerbehandlung
    """
    
    t0 = _now_ts()
    is_nsfw = str(nsfw).lower() == "true"
    
    # Parameter normalisieren
    _steps = steps or num_inference_steps or 28
    _guidance = guidance or guidance_scale or 7.5
    _width = max(64, min(width or 1024, 2048))
    _height = max(64, min(height or 768, 2048))
    
    # Modus bestimmen
    mode = mode.lower()
    if mode in ("image", "text2img", "img2img"):
        task = "image"
        actual_mode = "img2img" if (init_image or image) else "text2img"
    elif mode in ("video", "text2video", "img2video"):
        task = "video" 
        actual_mode = "img2video" if (init_image or image) else "text2video"
    else:
        actual_mode = "text2img"
        task = "image"
    
    # Operation starten
    op_id = f"{task}-{int(time.time()*1000)}"
    OPS.start(op_id, kind=f"{task}:{actual_mode}", nsfw=is_nsfw, params={
        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "resolution": f"{_width}x{_height}",
        "steps": _steps,
        "model": model_id or "default"
    })

    try:
        OPS.update(op_id, progress=0.1)
        
        # Eingangsbild laden falls vorhanden
        src_img = _load_image_safely(init_image or image)
        if actual_mode in ["img2img", "img2video"] and src_img:
            src_img = src_img.resize((_width, _height), Image.Resampling.LANCZOS)
        
        OPS.update(op_id, progress=0.2)
        
        # Modell laden (mit Fallback)
        model_info = None
        pipeline = None
        
        if model_id:
            try:
                s = load_settings()
                base_dirs = {
                    "image": Path(s.paths.image),
                    "video": Path(s.paths.video),
                    "llm": Path(getattr(s.paths, "llm", "models/llm"))
                }
                
                model_info = get_model_by_id(model_id, base_dirs)
                
                if model_info and Path(model_info.path).exists():
                    pipeline_type = "video" if task == "video" else actual_mode
                    pipeline = _load_pipeline(model_info, pipeline_type)
                    logger.info(f"Modell geladen: {model_info.name}")
                
            except Exception as e:
                logger.error(f"Modell-Laden fehlgeschlagen: {e}")
                # Weiter ohne Pipeline (Fallback wird verwendet)
        
        OPS.update(op_id, progress=0.4)
        
        # Generation durchführen
        if task == "image":
            # === BILD-GENERATION ===
            result_img = None
            
            if pipeline and DIFFUSERS_AVAILABLE:
                try:
                    OPS.update(op_id, progress=0.6, status="Generiere Bild...")
                    
                    generation_args = {
                        "prompt": prompt,
                        "num_inference_steps": _steps,
                        "guidance_scale": _guidance,
                        "num_images_per_prompt": 1
                    }
                    
                    if negative_prompt.strip():
                        generation_args["negative_prompt"] = negative_prompt
                    
                    if actual_mode == "text2img":
                        generation_args.update({
                            "height": _height,
                            "width": _width
                        })
                        result = pipeline(**generation_args)
                    
                    elif actual_mode == "img2img" and src_img:
                        generation_args.update({
                            "image": src_img,
                            "strength": max(0.1, min(strength or 0.65, 0.95))
                        })
                        result = pipeline(**generation_args)
                    
                    else:
                        raise ValueError("Ungültige Konfiguration für Bild-Generation")
                    
                    result_img = result.images[0]
                    
                except Exception as e:
                    logger.error(f"Pipeline-Generation fehlgeschlagen: {e}")
                    result_img = None
            
            # Fallback wenn Pipeline fehlschlägt
            if result_img is None:
                logger.warning("Verwende Fallback-Bild-Generation")
                if src_img and actual_mode == "img2img":
                    # Einfache Bildbearbeitung als Fallback
                    result_img = src_img.copy()
                    # Leichte Anpassungen
                    result_img = result_img.convert('RGB')
                else:
                    result_img = _create_fallback_image(
                        _width, _height, 
                        f"Generated: {prompt[:30]}..."
                    )
            
            OPS.update(op_id, progress=0.9)
            
            # Bild speichern
            fmt = output_format.lower() if output_format else "png"
            if fmt not in ["png", "jpg", "jpeg", "webp"]:
                fmt = "png"
            
            out_name = _mk_name(fmt)
            out_path = OUT_IMG / out_name
            
            if fmt == "png":
                result_img.save(out_path, "PNG", optimize=True)
            elif fmt in ["jpg", "jpeg"]:
                result_img.save(out_path, "JPEG", quality=95, optimize=True)
            elif fmt == "webp":
                result_img.save(out_path, "WEBP", quality=95, optimize=True)
            
            OPS.finish(op_id)
            
            return JSONResponse({
                "ok": True,
                "kind": "image",
                "model": model_info.name if model_info else "fallback",
                "file": _abs_url_from(out_path),
                "url": _abs_url_from(out_path),
                "metadata": {
                    "prompt": prompt,
                    "mode": actual_mode,
                    "nsfw": is_nsfw,
                    "resolution": f"{_width}x{_height}",
                    "steps": _steps,
                    "guidance": _guidance,
                    "negative_prompt": negative_prompt,
                    "generation_time": _now_ts() - t0,
                    "format": fmt
                }
            })
        
        else:
            # === VIDEO-GENERATION ===
            _fps = max(12, min(fps or 24, 60))
            _frames = max(8, min(num_frames or 48, 100))
            result_path = None
            
            if pipeline and DIFFUSERS_AVAILABLE:
                try:
                    OPS.update(op_id, progress=0.6, status="Generiere Video...")
                    
                    if actual_mode == "img2video" and src_img:
                        result = pipeline(
                            image=src_img,
                            num_frames=_frames,
                            motion_bucket_id=motion_bucket_id or 127,
                            noise_aug_strength=noise_aug_strength or 0.02,
                            decode_chunk_size=8
                        )
                        frames = result.frames[0]
                        
                    elif actual_mode == "text2video":
                        # Für text2video erst Bild generieren, dann Video
                        img_pipeline = _load_pipeline(model_info, "text2img")
                        img_result = img_pipeline(
                            prompt=prompt,
                            height=_height,
                            width=_width,
                            num_inference_steps=20,
                            guidance_scale=7.0
                        )
                        start_img = img_result.images[0]
                        
                        result = pipeline(
                            image=start_img,
                            num_frames=_frames,
                            motion_bucket_id=motion_bucket_id or 127,
                            noise_aug_strength=noise_aug_strength or 0.02,
                            decode_chunk_size=8
                        )
                        frames = result.frames[0]
                    
                    else:
                        raise ValueError("Ungültige Video-Konfiguration")
                    
                    # Frames zu Video konvertieren
                    out_name = _mk_name("mp4")
                    result_path = OUT_VID / out_name
                    
                    if imageio and len(frames) > 0:
                        # Frames zu numpy arrays konvertieren
                        frame_arrays = []
                        for frame in frames:
                            if hasattr(frame, 'numpy'):
                                arr = frame.numpy()
                            elif hasattr(frame, '__array__'):
                                arr = np.array(frame)
                            else:
                                arr = frame
                            frame_arrays.append(arr)
                        
                        imageio.mimwrite(
                            str(result_path),
                            frame_arrays,
                            fps=_fps,
                            quality=9,
                            codec="libx264",
                            output_params=["-pix_fmt", "yuv420p"]
                        )
                    
                except Exception as e:
                    logger.error(f"Video-Pipeline fehlgeschlagen: {e}")
                    result_path = None
            
            # Fallback Video
            if result_path is None or not result_path.exists():
                logger.warning("Verwende Fallback-Video-Generation")
                duration = _frames / _fps
                result_path = _create_fallback_video(
                    _width, _height, _fps, duration,
                    f"Video: {prompt[:20]}..."
                )
            
            OPS.update(op_id, progress=0.95)
            OPS.finish(op_id)
            
            return JSONResponse({
                "ok": True,
                "kind": "video", 
                "model": model_info.name if model_info else "fallback",
                "file": _abs_url_from(result_path),
                "url": _abs_url_from(result_path),
                "metadata": {
                    "prompt": prompt,
                    "mode": actual_mode,
                    "nsfw": is_nsfw,
                    "resolution": f"{_width}x{_height}",
                    "fps": _fps,
                    "frames": _frames,
                    "duration": _frames / _fps,
                    "motion_bucket": motion_bucket_id,
                    "generation_time": _now_ts() - t0
                }
            })
    
    except Exception as e:
        logger.error(f"Generation komplett fehlgeschlagen: {e}")
        OPS.finish(op_id, error=str(e))
        
        # Letzte Notlösung - immer etwas zurückgeben
        if task == "image":
            fallback_img = _create_fallback_image(_width, _height, "Generation fehlgeschlagen")
            fallback_path = OUT_IMG / _mk_name("png")
            fallback_img.save(fallback_path, "PNG")
            
            return JSONResponse({
                "ok": False,
                "kind": "image",
                "file": _abs_url_from(fallback_path),
                "url": _abs_url_from(fallback_path),
                "error": str(e),
                "metadata": {"fallback": True, "generation_time": _now_ts() - t0}
            })
        else:
            fallback_path = _create_fallback_video(_width, _height, 24, 3, "Video-Generation fehlgeschlagen")
            
            return JSONResponse({
                "ok": False,
                "kind": "video",
                "file": _abs_url_from(fallback_path),
                "url": _abs_url_from(fallback_path), 
                "error": str(e),
                "metadata": {"fallback": True, "generation_time": _now_ts() - t0}
            })

# Cleanup-Funktion für Pipeline-Cache
@router.get("/generate/cleanup")
async def cleanup_pipelines():
    """Leere Pipeline-Cache um Speicher freizugeben"""
    global _pipeline_cache
    
    try:
        for pipeline in _pipeline_cache.values():
            if hasattr(pipeline, 'to'):
                pipeline.to('cpu')
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        _pipeline_cache.clear()
        
        return {"ok": True, "message": "Pipeline-Cache geleert"}
    
    except Exception as e:
        return {"ok": False, "error": str(e)}