# server/routes/generate.py - Optimiert und vollständig
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
from typing import Optional, Tuple, Union
import time, io, uuid, os, logging

from PIL import Image
from PIL.Image import Image as PILImage

# eigene Module
from ..settings import load_settings
from ..models_registry import list_models
from ..state import OPS

# Torch / Diffusers - mit Fallback falls nicht installiert
try:
    import torch
    from diffusers import (
        AutoPipelineForText2Image,
        AutoPipelineForImage2Image,
        StableDiffusionPipeline,
        StableDiffusionImg2ImgPipeline,
        StableVideoDiffusionPipeline,
        DiffusionPipeline
    )
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Diffusers not available: {e}")
    DIFFUSERS_AVAILABLE = False

router = APIRouter()

# ---------------------------
# Hilfsfunktionen
# ---------------------------
def _device() -> str:
    """Bestimme verfügbares Gerät für Inferenz"""
    if not DIFFUSERS_AVAILABLE:
        return "cpu"
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except:
        return "cpu"

def _dtype():
    """Bestimme optimalen Datentyp"""
    if not DIFFUSERS_AVAILABLE:
        return None
    try:
        return torch.float16 if torch.cuda.is_available() else torch.float32
    except:
        return None

def _parse_res_xy(s: str, fallback: Tuple[int, int]) -> Tuple[int, int]:
    """Parse Auflösungsstring wie '1024x768' zu (1024, 768)"""
    try:
        parts = s.lower().replace('x', '*').replace('×', '*').split('*')
        if len(parts) >= 2:
            w, h = int(parts[0].strip()), int(parts[1].strip())
            return max(64, min(w, 4096)), max(64, min(h, 4096))
    except:
        pass
    return fallback

def _project_root() -> Path:
    """Ermittle Projekt-Root-Verzeichnis"""
    return Path(__file__).resolve().parent.parent.parent

def _load_image_from_upload(upload: UploadFile) -> PILImage:
    """Lade Bild aus Upload-File"""
    try:
        if hasattr(upload, 'file'):
            data = upload.file.read()
        else:
            data = upload.read()
        
        img = Image.open(io.BytesIO(data))
        
        # Konvertiere zu RGB falls nötig
        if img.mode in ('RGBA', 'P', 'L'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                rgb_img.paste(img, mask=img.split()[-1])
            else:
                rgb_img.paste(img)
            return rgb_img
        
        return img.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ungültiges Bildformat: {e}")

def _load_image_from_local_url(init_url: Optional[str]) -> Optional[PILImage]:
    """Lade Bild aus lokaler URL (z.B. /outputs/images/...)"""
    if not init_url or not init_url.startswith("/outputs/"):
        return None
    
    try:
        root = _project_root()
        local_path = root / init_url.lstrip("/")
        
        if not local_path.exists():
            return None
            
        img = Image.open(local_path)
        return img.convert("RGB")
    except Exception as e:
        logging.warning(f"Could not load local image {init_url}: {e}")
        return None

def _resolve_model(kind: str, model_id: Optional[str]) -> tuple[Optional[Path], Optional[str]]:
    """
    Finde verfügbares Modell für 'image' oder 'video'.
    Returns: (pfad, backend) oder (None, None)
    """
    try:
        s = load_settings()
        img_dir = Path(s.paths.image)
        vid_dir = Path(s.paths.video)
        
        models = list_models(img_dir, vid_dir).get(kind, [])
        if not models:
            return None, None
            
        target = None
        if model_id:
            # Suche spezifisches Modell
            for m in models:
                if m.get("id") == model_id or m.get("name") == model_id:
                    target = m
                    break
                    
        if target is None:
            # Nimm erstes verfügbares
            target = models[0]
            
        model_path = target.get("path")
        if model_path:
            path_obj = Path(model_path)
            if path_obj.exists():
                return path_obj, target.get("backend", "diffusers_single")
                
        return None, None
        
    except Exception as e:
        logging.error(f"Error resolving model: {e}")
        return None, None

def _ensure_output_dir(s, output_type: str) -> Path:
    """Stelle sicher dass Output-Verzeichnis existiert"""
    if output_type == "image":
        out_dir = Path(getattr(s, 'outputs_images', s.paths.images))
    else:
        out_dir = Path(getattr(s, 'outputs_videos', s.paths.videos))
        
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def _cleanup_pipeline(pipe):
    """Cleanup nach Pipeline-Nutzung"""
    if pipe is not None and hasattr(pipe, 'to'):
        try:
            # Versuche GPU-Speicher freizugeben
            if torch.cuda.is_available():
                pipe.to('cpu')
                torch.cuda.empty_cache()
        except:
            pass

# ---------------------------
# Bild-Generierung (Stable Diffusion)
# ---------------------------
@router.post("/generate/image")
async def generate_image(
    prompt: str = Form(...),
    nsfw: bool = Form(False),
    mode: str = Form("text2img"),  # "text2img" | "img2img"
    model_id: Optional[str] = Form(None),
    init_image: Optional[UploadFile] = File(None),
    init_url: Optional[str] = Form(None),
    img_resolution: str = Form("1024x768"),
    img_format: str = Form("png"),
    jpg_quality: int = Form(92),
    num_inference_steps: int = Form(28),
    guidance_scale: float = Form(7.0),
    strength: float = Form(0.65),  # für img2img
):
    """
    Vollständige Bild-Generierung über Diffusers:
    - Text2Img über AutoPipeline/StableDiffusion
    - Img2Img (Upload oder Remix via init_url)
    """
    
    # Prüfe Diffusers-Verfügbarkeit
    if not DIFFUSERS_AVAILABLE:
        raise HTTPException(
            status_code=501, 
            detail="Diffusers nicht installiert. Bitte installiere: pip install diffusers torch"
        )
    
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt darf nicht leer sein")
    
    s = load_settings()
    out_dir = _ensure_output_dir(s, "image")
    
    # Parse Parameter
    W, H = _parse_res_xy(img_resolution, (1024, 768))
    ext = ".png" if img_format.lower() == "png" else ".jpg"
    fname = f"img_{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"
    out_path = out_dir / fname
    
    # Validiere Parameter
    num_inference_steps = max(1, min(int(num_inference_steps), 100))
    guidance_scale = max(1.0, min(float(guidance_scale), 20.0))
    strength = max(0.01, min(float(strength), 0.99))
    jpg_quality = max(10, min(int(jpg_quality), 100))

    # Operation starten
    op_id = f"img-{int(time.time()*1000)}"
    OPS.start(op_id, kind=f"image:{mode}", nsfw=nsfw, params={
        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "resolution": f"{W}x{H}",
        "steps": num_inference_steps,
        "guidance": guidance_scale,
        "strength": strength if mode == "img2img" else None
    })

    pipe = None
    
    try:
        # Modell auflösen
        mpath, backend = _resolve_model("image", model_id)
        if mpath is None or backend is None:
            OPS.finish(op_id, error="Kein Bildmodell gefunden.")
            raise HTTPException(
                status_code=400, 
                detail="Kein Bildmodell gefunden. Installiere Modelle über den Katalog oder kopiere sie in models/image/"
            )

        OPS.update(op_id, progress=0.1)

        # Init-Bild laden falls nötig
        base_img: Optional[PILImage] = None
        if mode == "img2img" or init_image is not None or (init_url and init_url.startswith("/outputs/")):
            try:
                if init_image is not None:
                    base_img = _load_image_from_upload(init_image)
                elif init_url:
                    base_img = _load_image_from_local_url(init_url)
                    
                if base_img is not None:
                    # Resize zu Zielauflösung
                    base_img = base_img.resize((W, H), Image.Resampling.LANCZOS)
            except Exception as e:
                OPS.finish(op_id, error=f"Init-Bild konnte nicht geladen werden: {e}")
                raise HTTPException(status_code=400, detail=f"Init-Bild Fehler: {e}")

        if mode == "img2img" and base_img is None:
            OPS.finish(op_id, error="img2img benötigt ein Eingangsbild.")
            raise HTTPException(status_code=400, detail="img2img benötigt ein Eingangsbild (Upload oder Remix).")

        OPS.update(op_id, progress=0.2)

        # Pipeline laden und ausführen
        torch_dtype = _dtype()
        device = _device()
        
        try:
            if backend == "diffusers_dir":
                # Diffusers-Ordner mit model_index.json
                if mode == "text2img":
                    pipe = AutoPipelineForText2Image.from_pretrained(
                        str(mpath), 
                        torch_dtype=torch_dtype,
                        safety_checker=None,  # Deaktiviere Safety Checker für lokale Nutzung
                        requires_safety_checker=False
                    )
                else:
                    pipe = AutoPipelineForImage2Image.from_pretrained(
                        str(mpath), 
                        torch_dtype=torch_dtype,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
            else:
                # Single-File (.safetensors/.ckpt)
                if mode == "text2img":
                    pipe = StableDiffusionPipeline.from_single_file(
                        str(mpath), 
                        torch_dtype=torch_dtype,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
                else:
                    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
                        str(mpath), 
                        torch_dtype=torch_dtype,
                        safety_checker=None,
                        requires_safety_checker=False
                    )
            
            pipe.to(device)
            
            # Memory optimization
            if torch.cuda.is_available():
                pipe.enable_model_cpu_offload()
                pipe.enable_attention_slicing()

            OPS.update(op_id, progress=0.4)

            # Generierung
            if mode == "text2img":
                result = pipe(
                    prompt=prompt,
                    height=H,
                    width=W,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=1
                )
            else:
                result = pipe(
                    prompt=prompt,
                    image=base_img,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )

            img = result.images[0]
            OPS.update(op_id, progress=0.9)

        except Exception as e:
            OPS.finish(op_id, error=f"Pipeline-Fehler: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Bild-Generation fehlgeschlagen: {str(e)}")

        # Speichern
        try:
            if ext == ".png":
                img.save(out_path, "PNG")
            else:
                img.save(out_path, "JPEG", quality=jpg_quality, optimize=True)
                
            OPS.finish(op_id)
            
            return {
                "ok": True, 
                "file": f"/outputs/images/{fname}",
                "metadata": {
                    "prompt": prompt,
                    "model": model_id or "default",
                    "resolution": f"{W}x{H}",
                    "steps": num_inference_steps,
                    "guidance": guidance_scale
                }
            }
            
        except Exception as e:
            OPS.finish(op_id, error=f"Speichern fehlgeschlagen: {e}")
            raise HTTPException(status_code=500, detail=f"Bild konnte nicht gespeichert werden: {e}")

    finally:
        _cleanup_pipeline(pipe)

# ---------------------------
# Video-Generierung (SVD)
# ---------------------------
@router.post("/generate/video")
async def generate_video(
    prompt: str = Form(...),
    nsfw: bool = Form(False),
    mode: str = Form("text2video"),  # "text2video" | "img2video"
    model_id: Optional[str] = Form(None),
    init_image: Optional[UploadFile] = File(None),
    init_url: Optional[str] = Form(None),
    vid_resolution: str = Form("1024x576"),
    vid_fps: int = Form(24),
    vid_length_sec: int = Form(3),
    num_frames_override: int = Form(0),
):
    """
    Video-Generierung über Stable Video Diffusion:
    - img2video: nimmt Upload/Remix-Bild
    - text2video: erzeugt zuerst Startbild via Bildmodell, dann SVD
    """
    
    if not DIFFUSERS_AVAILABLE:
        raise HTTPException(
            status_code=501, 
            detail="Diffusers nicht installiert. Benötigt für Video-Generation."
        )
    
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt darf nicht leer sein")

    s = load_settings()
    out_dir = _ensure_output_dir(s, "video")

    # Parse Parameter
    VW, VH = _parse_res_xy(vid_resolution, (1024, 576))
    vid_fps = max(12, min(int(vid_fps), 60))
    vid_length_sec = max(1, min(int(vid_length_sec), 30))
    
    fname = f"vid_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
    out_path = out_dir / fname

    # Operation starten
    op_id = f"vid-{int(time.time()*1000)}"
    OPS.start(op_id, kind=f"video:{mode}", nsfw=nsfw, params={
        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "resolution": f"{VW}x{VH}",
        "fps": vid_fps,
        "length": vid_length_sec
    })

    img_pipe = None
    vid_pipe = None

    try:
        # Videomodell auflösen
        mpath, backend = _resolve_model("video", model_id)
        if mpath is None or backend is None:
            OPS.finish(op_id, error="Kein Videomodell gefunden.")
            raise HTTPException(
                status_code=400, 
                detail="Kein Videomodell (SVD) gefunden. Installiere über Katalog oder kopiere in models/video/"
            )

        OPS.update(op_id, progress=0.1)

        # Startframe bestimmen
        base_img: Optional[PILImage] = None
        
        try:
            # Versuche existierendes Bild zu laden
            if mode == "img2video" or init_image is not None or (init_url and init_url.startswith("/outputs/")):
                if init_image is not None:
                    base_img = _load_image_from_upload(init_image)
                elif init_url:
                    base_img = _load_image_from_local_url(init_url)

            # text2video → Frame via Bildmodell generieren
            if base_img is None and mode == "text2video":
                im_path, im_backend = _resolve_model("image", None)
                if im_path is None:
                    OPS.finish(op_id, error="Kein Bildmodell für text2video gefunden.")
                    raise HTTPException(
                        status_code=400, 
                        detail="text2video benötigt ein Bildmodell zur Startframe-Generierung"
                    )
                
                # Generiere Startframe
                torch_dtype = _dtype()
                device = _device()
                
                try:
                    if im_backend == "diffusers_dir":
                        img_pipe = AutoPipelineForText2Image.from_pretrained(
                            str(im_path), 
                            torch_dtype=torch_dtype,
                            safety_checker=None,
                            requires_safety_checker=False
                        )
                    else:
                        img_pipe = StableDiffusionPipeline.from_single_file(
                            str(im_path), 
                            torch_dtype=torch_dtype,
                            safety_checker=None,
                            requires_safety_checker=False
                        )
                    
                    img_pipe.to(device)
                    
                    result = img_pipe(
                        prompt=prompt, 
                        height=VH, 
                        width=VW, 
                        num_inference_steps=25, 
                        guidance_scale=7.0
                    )
                    base_img = result.images[0]
                    
                    # Cleanup Image Pipeline
                    _cleanup_pipeline(img_pipe)
                    img_pipe = None
                    
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Startframe-Generation fehlgeschlagen: {e}")

            if base_img is None:
                OPS.finish(op_id, error="Kein Eingangsbild verfügbar.")
                raise HTTPException(status_code=400, detail="Kein Eingangsbild für Video gefunden")
                
            # Resize zu Video-Auflösung
            base_img = base_img.resize((VW, VH), Image.Resampling.LANCZOS)
            
        except Exception as e:
            OPS.finish(op_id, error=f"Frame-Beschaffung: {e}")
            raise HTTPException(status_code=500, detail=f"Startframe-Verarbeitung fehlgeschlagen: {e}")

        OPS.update(op_id, progress=0.3)

        # SVD Video-Pipeline
        try:
            torch_dtype = _dtype()
            device = _device()

            vid_pipe = StableVideoDiffusionPipeline.from_pretrained(
                str(mpath),
                torch_dtype=torch_dtype,
                variant="fp16" if torch_dtype == torch.float16 else None
            )
            vid_pipe.to(device)
            
            # Memory optimization
            if torch.cuda.is_available():
                vid_pipe.enable_model_cpu_offload()

            # Frames bestimmen
            if int(num_frames_override) > 0:
                num_frames = max(8, min(int(num_frames_override), 65))
            else:
                num_frames = max(8, min(int(vid_length_sec) * int(vid_fps), 65))

            OPS.update(op_id, progress=0.5)

            # Video generieren
            result = vid_pipe(
                image=base_img,
                decode_chunk_size=8,
                num_frames=num_frames,
                motion_bucket_id=127,  # Standardwert für SVD
                noise_aug_strength=0.02
            )
            
            frames = result.frames[0]  # Liste von PIL Images oder numpy arrays
            OPS.update(op_id, progress=0.8)

        except Exception as e:
            OPS.finish(op_id, error=f"Video-Pipeline: {e}")
            raise HTTPException(status_code=500, detail=f"Video-Generation fehlgeschlagen: {e}")

        # Als MP4 speichern
        try:
            import imageio
            
            # Konvertiere Frames zu numpy falls nötig
            frame_arrays = []
            for frame in frames:
                if hasattr(frame, 'numpy'):  # torch tensor
                    arr = frame.numpy()
                elif hasattr(frame, '__array__'):  # PIL Image
                    arr = np.array(frame)
                else:
                    arr = frame
                frame_arrays.append(arr)

            imageio.mimwrite(
                str(out_path), 
                frame_arrays, 
                fps=int(vid_fps), 
                quality=9, 
                codec="libx264",
                output_params=["-pix_fmt", "yuv420p"]  # Für bessere Kompatibilität
            )

            OPS.finish(op_id)
            
            return {
                "ok": True, 
                "file": f"/outputs/videos/{fname}",
                "metadata": {
                    "prompt": prompt,
                    "model": model_id or "default",
                    "resolution": f"{VW}x{VH}",
                    "fps": vid_fps,
                    "frames": len(frame_arrays),
                    "duration": len(frame_arrays) / vid_fps
                }
            }

        except Exception as e:
            OPS.finish(op_id, error=f"Video-Export: {e}")
            raise HTTPException(status_code=500, detail=f"Video konnte nicht gespeichert werden: {e}")

    finally:
        _cleanup_pipeline(img_pipe)
        _cleanup_pipeline(vid_pipe)