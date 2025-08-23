# server/routes/generate.py
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
from typing import Optional, Tuple
import time, io, uuid

from PIL import Image
from PIL.Image import Image as PILImage

# eigene Module
from ..settings import load_settings
from ..models_registry import list_models
from ..state import OPS

# Torch / Diffusers
import torch
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableVideoDiffusionPipeline,
)

router = APIRouter()


# ---------------------------
# Hilfsfunktionen
# ---------------------------
def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _parse_res_xy(s: str, fallback: Tuple[int, int]) -> Tuple[int, int]:
    try:
        w, h = [int(x) for x in s.lower().split("x")]
        return max(64, w), max(64, h)
    except Exception:
        return fallback


def _project_root() -> Path:
    # server/routes/ -> server/ -> repo-root
    return Path(__file__).resolve().parent.parent.parent


def _load_image_from_upload(upload: UploadFile) -> PILImage:
    data = upload.file.read() if hasattr(upload, "file") else upload.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def _load_image_from_local_url(init_url: Optional[str]) -> Optional[PILImage]:
    if not init_url or not init_url.startswith("/outputs/"):
        return None
    root = _project_root()
    local = root / init_url.lstrip("/")
    if local.exists():
        return Image.open(local).convert("RGB")
    return None


def _resolve_model(kind: str, model_id: Optional[str]) -> tuple[Optional[Path], Optional[str]]:
    """
    Liefert (pfad, backend) für 'image' oder 'video'.
    Backend: "diffusers_dir" (Ordner mit model_index.json) oder "diffusers_single" (safetensors/ckpt)
    """
    s = load_settings()
    # Settings-Pfade – Settings sollten strings liefern
    img_dir = Path(s.paths.image)
    vid_dir = Path(s.paths.video)
    pool = list_models(img_dir, vid_dir).get(kind, [])
    if not pool:
        return None, None
    target = None
    if model_id:
        for m in pool:
            if m.get("id") == model_id:
                target = m
                break
    if target is None:
        target = pool[0]
    return (Path(target.get("path")) if target.get("path") else None, target.get("backend", None))


# ---------------------------
# Bild-Generierung (SD)
# ---------------------------
@router.post("/generate/image")
async def generate_image(
    prompt: str = Form(...),
    nsfw: bool = Form(False),
    mode: str = Form("text2img"),  # "text2img" | "img2img"
    model_id: Optional[str] = Form(None),
    init_image: UploadFile | None = File(None),
    init_url: Optional[str] = Form(None),
    img_resolution: str = Form("1024x768"),
    img_format: str = Form("png"),
    jpg_quality: int = Form(92),
    num_inference_steps: int = Form(28),
    guidance_scale: float = Form(7.0),
    strength: float = Form(0.65),  # für img2img
):
    """
    Volle Bild-Generierung über diffusers:
    - Text2Img über AutoPipeline/StableDiffusion
    - Img2Img (Upload oder Remix via init_url)
    - Auflösung, Format, Qualität, Steps, Guidance, Strength steuerbar
    """
    s = load_settings()
    out_dir = Path(s.outputs_images) if hasattr(s, "outputs_images") else Path(s.paths.images)
    out_dir.mkdir(parents=True, exist_ok=True)

    W, H = _parse_res_xy(img_resolution, (1024, 768))
    ext = ".png" if img_format.lower() == "png" else ".jpg"
    fname = f"img_{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"
    out_path = out_dir / fname

    # OPS start
    op_id = f"img-{int(time.time()*1000)}"
    OPS.start(op_id, kind=f"image:{mode}", nsfw=nsfw, params={
        "resolution": img_resolution, "steps": num_inference_steps, "guidance": guidance_scale, "strength": strength
    })

    # Modell auflösen
    mpath, backend = _resolve_model("image", model_id)
    if mpath is None or backend is None:
        OPS.finish(op_id, error="Kein Bildmodell gefunden.")
        raise HTTPException(status_code=400, detail="Kein Bildmodell gefunden. Lege ein Model in models/image ab (Diffusers-Ordner oder .safetensors/.ckpt).")

    # Init-Bild laden, wenn nötig
    base_img: Optional[PILImage] = None
    if mode == "img2img" or init_image is not None or (init_url and init_url.startswith("/outputs/")):
        try:
            if init_image is not None:
                base_img = _load_image_from_upload(init_image)
            else:
                base_img = _load_image_from_local_url(init_url)
            if base_img is not None:
                base_img = base_img.resize((W, H))
        except Exception as e:
            OPS.finish(op_id, error=f"Init-Bild konnte nicht geladen werden: {e}")
            raise HTTPException(status_code=400, detail=f"Init-Bild konnte nicht geladen werden: {e}")

    # Pipeline laden und ausführen
    try:
        torch_dtype = _dtype()
        dev = _device()

        if backend == "diffusers_dir":
            if mode == "text2img":
                pipe = AutoPipelineForText2Image.from_pretrained(str(mpath), torch_dtype=torch_dtype)
                pipe.to(dev)
                img = pipe(
                    prompt=prompt, height=H, width=W,
                    num_inference_steps=int(num_inference_steps),
                    guidance_scale=float(guidance_scale)
                ).images[0]
            else:
                if base_img is None:
                    OPS.finish(op_id, error="img2img benötigt ein Eingangsbild.")
                    raise HTTPException(status_code=400, detail="img2img benötigt ein Eingangsbild.")
                pipe = AutoPipelineForImage2Image.from_pretrained(str(mpath), torch_dtype=torch_dtype)
                pipe.to(dev)
                img = pipe(
                    prompt=prompt, image=base_img, strength=float(strength),
                    guidance_scale=float(guidance_scale),
                    num_inference_steps=int(num_inference_steps)
                ).images[0]
        else:
            # Single-File (safetensors/ckpt)
            if mode == "text2img":
                pipe = StableDiffusionPipeline.from_single_file(str(mpath), torch_dtype=torch_dtype)
                pipe.to(dev)
                img = pipe(
                    prompt=prompt, height=H, width=W,
                    num_inference_steps=int(num_inference_steps),
                    guidance_scale=float(guidance_scale)
                ).images[0]
            else:
                if base_img is None:
                    OPS.finish(op_id, error="img2img benötigt ein Eingangsbild.")
                    raise HTTPException(status_code=400, detail="img2img benötigt ein Eingangsbild.")
                pipe = StableDiffusionImg2ImgPipeline.from_single_file(str(mpath), torch_dtype=torch_dtype)
                pipe.to(dev)
                img = pipe(
                    prompt=prompt, image=base_img, strength=float(strength),
                    guidance_scale=float(guidance_scale),
                    num_inference_steps=int(num_inference_steps)
                ).images[0]

    except Exception as e:
        OPS.finish(op_id, error=f"Bild-Pipeline Fehler: {e}")
        raise HTTPException(status_code=500, detail=f"Bild-Pipeline Fehler: {e}")

    # Speichern
    try:
        if ext == ".png":
            img.save(out_path, "PNG")
        else:
            img.save(out_path, "JPEG", quality=int(jpg_quality))
        OPS.finish(op_id)
        return {"ok": True, "file": f"/outputs/images/{fname}"}
    except Exception as e:
        OPS.finish(op_id, error=f"Speichern fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=f"Bild konnte nicht gespeichert werden: {e}")


# ---------------------------
# Video-Generierung (SVD)
# ---------------------------
@router.post("/generate/video")
async def generate_video(
    prompt: str = Form(...),
    nsfw: bool = Form(False),
    mode: str = Form("text2video"),  # "text2video" | "img2video"
    model_id: Optional[str] = Form(None),
    init_image: UploadFile | None = File(None),
    init_url: Optional[str] = Form(None),
    vid_resolution: str = Form("1280x720"),
    vid_fps: int = Form(24),
    vid_length_sec: int = Form(3),
    num_frames_override: int = Form(0),  # optional: direkt Framelänge setzen
):
    """
    Volle Video-Generierung über Stable Video Diffusion:
    - img2video: nimmt Upload/Remix-Bild
    - text2video: erzeugt zuerst ein Startbild via Bildmodell (Text2Img), dann SVD
    - Auflösung, FPS, Länge steuerbar
    """
    s = load_settings()
    out_dir = Path(s.outputs_videos) if hasattr(s, "outputs_videos") else Path(s.paths.videos)
    out_dir.mkdir(parents=True, exist_ok=True)

    VW, VH = _parse_res_xy(vid_resolution, (1280, 720))
    fname = f"vid_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
    out_path = out_dir / fname

    # OPS
    op_id = f"vid-{int(time.time()*1000)}"
    OPS.start(op_id, kind=f"video:{mode}", nsfw=nsfw, params={
        "resolution": vid_resolution, "fps": vid_fps, "length": vid_length_sec
    })

    # Videomodell auflösen
    mpath, backend = _resolve_model("video", model_id)
    if mpath is None or backend is None:
        OPS.finish(op_id, error="Kein Videomodell (SVD) gefunden.")
        raise HTTPException(status_code=400, detail="Kein Videomodell gefunden. Lege ein SVD Diffusers-Model in models/video ab.")

    # Startframe bestimmen
    base_img: Optional[PILImage] = None
    try:
        if mode == "img2video" or init_image is not None or (init_url and init_url.startswith("/outputs/")):
            if init_image is not None:
                base_img = _load_image_from_upload(init_image)
            else:
                base_img = _load_image_from_local_url(init_url)

        # text2video → Frame via Bildmodell synthesieren
        if base_img is None and mode == "text2video":
            im_path, im_backend = _resolve_model("image", None)
            if im_path is None or im_backend is None:
                OPS.finish(op_id, error="Kein Bildmodell für text2video vorhanden.")
                raise HTTPException(status_code=400, detail="Kein Bildmodell für text2video gefunden.")
            torch_dtype = _dtype()
            dev = _device()
            if im_backend == "diffusers_dir":
                pipe = AutoPipelineForText2Image.from_pretrained(str(im_path), torch_dtype=torch_dtype)
                pipe.to(dev)
                base_img = pipe(prompt=prompt, height=VH, width=VW, num_inference_steps=28, guidance_scale=7.0).images[0]
            else:
                pipe = StableDiffusionPipeline.from_single_file(str(im_path), torch_dtype=torch_dtype)
                pipe.to(dev)
                base_img = pipe(prompt=prompt, height=VH, width=VW, num_inference_steps=28, guidance_scale=7.0).images[0]

        if base_img is None:
            OPS.finish(op_id, error="Kein Eingangsbild für Video vorhanden.")
            raise HTTPException(status_code=400, detail="Kein Eingangsbild für Video vorhanden.")
        base_img = base_img.resize((VW, VH))
    except Exception as e:
        OPS.finish(op_id, error=f"Frame-Beschaffung fehlgeschlagen: {e}")
        raise HTTPException(status_code=500, detail=f"Frame-Beschaffung fehlgeschlagen: {e}")

    # SVD ausführen
    try:
        torch_dtype = _dtype()
        dev = _device()

        pipe = StableVideoDiffusionPipeline.from_pretrained(
            str(mpath),
            torch_dtype=torch_dtype,
            variant="fp16" if torch_dtype == torch.float16 else None
        )
        pipe.to(dev)

        # Frames bestimmen
        nf = int(num_frames_override) if int(num_frames_override) > 0 else int(vid_length_sec) * int(vid_fps)
        nf = max(8, min(nf, 65))  # SVD ist typischerweise auf <=65 Frames sinnvoll
        OPS.update(op_id, progress=0.5)

        result = pipe(base_img, decode_chunk_size=8, num_frames=nf)
        frames = result.frames[0]  # Liste von numpy-Frames
        OPS.update(op_id, progress=0.8)

        # Speichern als MP4
        import imageio
        imageio.mimwrite(out_path, frames, fps=int(vid_fps), quality=9, codec="libx264")

        OPS.finish(op_id)
        return {"ok": True, "file": f"/outputs/videos/{fname}"}

    except Exception as e:
        OPS.finish(op_id, error=f"Video-Pipeline Fehler: {e}")
        raise HTTPException(status_code=500, detail=f"Video-Pipeline Fehler: {e}")
