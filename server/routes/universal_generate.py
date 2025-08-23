# G:\mediasuit\server\routes\universal_generate.py
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import time, io, uuid, json
from typing import Optional
from PIL import Image

router = APIRouter()

# Basisordner
ROOT = Path(__file__).resolve().parents[2]  # ...\mediasuit
OUT_IMG = ROOT / "outputs" / "images"
OUT_VID = ROOT / "outputs" / "videos"
for p in (OUT_IMG, OUT_VID):
    p.mkdir(parents=True, exist_ok=True)

def _now_ts() -> float:
    return time.time()

def _mk_name(suffix: str) -> str:
    return f"{int(time.time())}_{uuid.uuid4().hex[:8]}.{suffix}"

def _abs_url_from(filepath: Path) -> str:
    # Deine Static-Route sollte /outputs direkt serven (z. B. StaticFiles).
    # Dann kann die UI die Datei unter /outputs/... abrufen:
    rel = filepath.relative_to(ROOT).as_posix()
    return "/" + rel  # z. B. /outputs/images/....png

@router.post("/generate/universal")
async def generate_universal(
    # gemeinsame Felder
    prompt: str = Form(...),
    mode: str = Form("text2img"),  # text2img|img2img|text2video|img2video|image|video
    model_id: Optional[str] = Form(None),
    nsfw: Optional[str] = Form("false"),
    width: Optional[int] = Form(768),
    height: Optional[int] = Form(768),

    # Bild-Parameter (Bezeichnungen passen zu image.html / image_nsfw.html)
    num_inference_steps: Optional[int] = Form(None),
    steps: Optional[int] = Form(None),                # Fallback von manchen Seiten
    guidance_scale: Optional[float] = Form(None),
    guidance: Optional[float] = Form(None),           # Fallback
    strength: Optional[float] = Form(None),
    output_format: Optional[str] = Form("png"),
    negative_prompt: Optional[str] = Form(None),

    # Video-Parameter (Bezeichnungen passen zu index.html/video‑Sektion)
    num_frames: Optional[int] = Form(None),
    fps: Optional[int] = Form(None),
    motion_bucket_id: Optional[int] = Form(None),

    # Uploads (img2img / img2video)
    image: Optional[UploadFile] = File(None),
    init_image: Optional[UploadFile] = File(None),
):
    """
    Universeller Generator-Endpoint, kompatibel zu deinen HTML-Seiten.
    Aktuell: Dummy-Ausgabe (Platzhalter), damit die UI ohne Fehler läuft.
    Später: Backend (Diffusers/SVD/…) hier einhängen.
    """
    t0 = _now_ts()
    is_nsfw = (str(nsfw).lower() == "true")

    # Robustheit: vereinheitliche Steps/Guidance
    _steps = steps or num_inference_steps or 25
    _guidance = guidance or guidance_scale or 7.5

    # Modus normalisieren (einige Seiten senden 'image'/'video')
    mode = mode.lower()
    if mode in ("image", "text2img", "img2img"):
        task = "image"
    elif mode in ("video", "text2video", "img2video"):
        task = "video"
    else:
        raise HTTPException(400, detail=f"Unbekannter Modus: {mode}")

    # Eingangsbild laden (falls vorhanden)
    src_upload = init_image or image
    src_img: Image.Image | None = None
    if src_upload is not None:
        if not src_upload.content_type.startswith("image/"):
            raise HTTPException(400, detail="Upload muss ein Bild sein")
        data = await src_upload.read()
        try:
            src_img = Image.open(io.BytesIO(data)).convert("RGBA")
        except Exception as e:
            raise HTTPException(400, detail=f"Upload konnte nicht gelesen werden: {e}")

    # ---- DUMMY PIPELINE (Ergebnis als Platzhalter erzeugen) ----
    # Bild: wir erzeugen ein einfaches PNG in Zielauflösung.
    if task == "image":
        W = int(width or 768)
        H = int(height or 768)
        fmt = (output_format or "png").lower()
        if fmt not in ("png", "jpg", "jpeg", "webp"):
            fmt = "png"

        if src_img is None:
            # Text2Img-Dummy: einfarbiges Canvas mit Meta-Balken
            img = Image.new("RGB", (W, H), (14, 20, 30))
        else:
            # Img2Img-Dummy: Quelle auf Zielgröße skalieren
            img = src_img.convert("RGB").resize((W, H))

        out_name = _mk_name("jpg" if fmt == "jpg" else fmt)
        out_path = OUT_IMG / out_name
        img.save(out_path, quality=95)  # für PNG ignoriert PIL "quality"

        meta = {
            "mode": mode,
            "nsfw": is_nsfw,
            "resolution": f"{W}x{H}",
            "steps": _steps,
            "guidance": _guidance,
            "negative_prompt": negative_prompt or "",
            "duration_sec": None,
            "fps": None,
            "motion_bucket": None,
            "generation_time": _now_ts() - t0,
        }
        return JSONResponse({
            "ok": True,
            "kind": "image",
            "model": model_id or "",
            "file": _abs_url_from(out_path),
            "url": _abs_url_from(out_path),   # einige deiner Seiten erwarten "url"
            "metadata": meta,
        })

    # Video: wir liefern eine winzige MP4-Attrappe (hier als PNG‑Serie + Notiz).
    # Für echte Videos später eine SVD/Latent-Video-Pipeline einhängen.
    else:
        frames = int(num_frames or 48)
        _fps = int(fps or 24)
        W = int(width or 576)
        H = int(height or 576)

        # Dummy: wir speichern ein einzelnes PNG als Platzhalter und nennen es .mp4,
        # damit die UI einen abspielbaren Pfad bekommt. (Besser: kleine echte MP4 via ffmpeg.)
        placeholder = Image.new("RGB", (W, H), (16, 10, 20))
        out_name = _mk_name("mp4")
        out_png = OUT_VID / (out_name.replace(".mp4", ".png"))
        placeholder.save(out_png)

        meta = {
            "mode": mode,
            "nsfw": is_nsfw,
            "resolution": f"{W}x{H}",
            "frames": frames,
            "fps": _fps,
            "motion_bucket": motion_bucket_id,
            "generation_time": _now_ts() - t0,
        }

        # UI erwartet meist "file" oder "url"; wir geben beides zurück.
        # Solange kein echtes MP4 erzeugt wird, zeigen wir auf das PNG.
        url = _abs_url_from(out_png)
        return JSONResponse({
            "ok": True,
            "kind": "video",
            "model": model_id or "",
            "file": url,
            "url": url,
            "metadata": meta,
        })
