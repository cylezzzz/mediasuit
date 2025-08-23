# server/app.py - Vollständig funktionierende FastAPI App
from fastapi import FastAPI, HTTPException, Query, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import json
import time
import uuid
import random
import logging
from typing import Optional, List, Dict, Any
import psutil
import platform

# Sichere Imports
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(
    title="LocalMediaSuite API",
    description="Vollständige lokale KI-Medien-Generierung", 
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pfade konfigurieren
ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"

# Verzeichnisse erstellen
for directory in [OUTPUTS_DIR, OUTPUTS_DIR / "images", OUTPUTS_DIR / "videos", MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Static Files
if OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

# Global State für Operations
OPERATIONS = {}

# === API ROUTES ===

@app.get("/api/health")
async def health_check():
    """System-Gesundheitscheck"""
    return {
        "ok": True,
        "status": "healthy",
        "version": "2.0.0",
        "uptime": time.time(),
        "pil_available": PIL_AVAILABLE
    }

@app.get("/api/status")
async def get_status():
    """System-Status mit Hardware-Info"""
    try:
        vm = psutil.virtual_memory()
        return {
            "ok": True,
            "system": {
                "os": platform.platform(),
                "python": platform.python_version(),
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "ram_percent": vm.percent,
                "total_ram_gb": round(vm.total / (1024**3), 1),
                "available_ram_gb": round(vm.available / (1024**3), 1)
            },
            "server": {
                "uptime": time.time(),
                "pil_available": PIL_AVAILABLE
            }
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "system": {
                "os": platform.platform(),
                "python": platform.python_version(),
                "cpu_percent": 0,
                "ram_percent": 0,
                "total_ram_gb": 0
            }
        }

@app.get("/api/ops")
async def list_operations():
    """Liste laufende Operationen"""
    current_time = time.time()
    active_ops = []
    
    for op_id, op_data in OPERATIONS.items():
        if current_time - op_data.get("start_time", 0) < 300:  # 5 Minuten
            active_ops.append({
                "id": op_id,
                "kind": op_data.get("kind", "unknown"),
                "status": op_data.get("status", "running"),
                "progress": op_data.get("progress", 0),
                "start_time": op_data.get("start_time", current_time),
                "duration": current_time - op_data.get("start_time", current_time)
            })
    
    return {"ok": True, "ops": active_ops}

@app.get("/api/models")
async def get_models(
    type: Optional[str] = Query(None, pattern="^(image|video|llm)$"),
    nsfw: Optional[bool] = Query(None)
):
    """Liste verfügbare Modelle"""
    
    # Mock-Modelle für Fallback
    mock_models = [
        {
            "id": "fallback-sd15",
            "name": "Fallback SD 1.5",
            "type": "image",
            "format": "safetensors",
            "backend": "diffusers",
            "nsfw_capable": True,
            "modalities": ["text2img", "img2img"],
            "architecture": "sd15",
            "path": "/dev/null",
            "size_bytes": 4000000000,
            "description": "Fallback-Modell für Bild-Generierung"
        },
        {
            "id": "fallback-svd",
            "name": "Fallback SVD",
            "type": "video", 
            "format": "diffusers_dir",
            "backend": "diffusers",
            "nsfw_capable": True,
            "modalities": ["img2video"],
            "architecture": "svd",
            "path": "/dev/null",
            "size_bytes": 9000000000,
            "description": "Fallback-Modell für Video-Generierung"
        },
        {
            "id": "fallback-llama",
            "name": "Fallback LLM",
            "type": "llm",
            "format": "ollama",
            "backend": "ollama",
            "nsfw_capable": True,
            "modalities": ["text", "chat"],
            "architecture": "llama",
            "path": "",
            "size_bytes": 0,
            "description": "Fallback für Text-Generierung"
        }
    ]
    
    # Scanne echte Modelle falls vorhanden
    real_models = []
    
    try:
        # Prüfe model-Verzeichnisse
        for model_type in ["image", "video", "llm"]:
            model_dir = MODELS_DIR / model_type
            if model_dir.exists():
                for item in model_dir.iterdir():
                    if item.is_dir() or item.suffix in ['.safetensors', '.ckpt', '.gguf']:
                        real_models.append({
                            "id": f"{model_type}:{item.name}",
                            "name": item.name,
                            "type": model_type,
                            "format": "diffusers_dir" if item.is_dir() else item.suffix[1:],
                            "backend": "diffusers" if model_type != "llm" else "transformers",
                            "nsfw_capable": True,
                            "modalities": ["text2img", "img2img"] if model_type == "image" else 
                                         (["img2video"] if model_type == "video" else ["text", "chat"]),
                            "architecture": "sd15" if model_type == "image" else ("svd" if model_type == "video" else "llama"),
                            "path": str(item),
                            "size_bytes": sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) if item.is_dir() else item.stat().st_size,
                            "description": f"Lokales {model_type}-Modell"
                        })
    except Exception as e:
        logger.warning(f"Modell-Scan fehlgeschlagen: {e}")
    
    # Kombiniere echte und Mock-Modelle
    all_models = real_models + mock_models
    
    # Filter anwenden
    if type:
        all_models = [m for m in all_models if m["type"] == type]
    
    if nsfw is not None:
        all_models = [m for m in all_models if m["nsfw_capable"] == nsfw]
    
    return all_models

@app.get("/api/models/compatible")
async def get_compatible_models(mode: str = Query(...)):
    """Hole kompatible Modelle für einen Modus"""
    
    all_models = await get_models()
    compatible = []
    
    for model in all_models:
        modalities = model.get("modalities", [])
        
        # Prüfe Kompatibilität
        if mode in modalities:
            compatible.append(model)
        elif mode == "text2img" and "text2img" in modalities:
            compatible.append(model)
        elif mode == "img2img" and "img2img" in modalities:
            compatible.append(model)
        elif mode == "text2video" and model["type"] == "video":
            compatible.append(model)
        elif mode == "img2video" and "img2video" in modalities:
            compatible.append(model)
    
    return {
        "ok": True,
        "mode": mode,
        "compatible_models": compatible
    }

@app.get("/api/files")
async def list_files(type: str = Query(..., pattern="^(image|video)$")):
    """Liste generierte Dateien"""
    
    file_dir = OUTPUTS_DIR / (f"{type}s")  # images oder videos
    items = []
    
    if file_dir.exists():
        for file_path in sorted(file_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if file_path.is_file():
                rel_path = f"/outputs/{type}s/{file_path.name}"
                items.append({
                    "name": file_path.name,
                    "url": rel_path,
                    "size": file_path.stat().st_size,
                    "mtime": int(file_path.stat().st_mtime)
                })
    
    return {"ok": True, "items": items}

@app.post("/api/respond")
async def flirt_respond(request: dict):
    """Enhanced Flirt AI Response System"""
    
    try:
        analysis_user = request.get("analysis_user", "")
        reply_user = request.get("reply_user", "")
        
        # Erweiterte Analyse-Simulation
        user_length = len(analysis_user)
        word_count = len(analysis_user.split())
        emoji_count = analysis_user.count("😊") + analysis_user.count("😍") + analysis_user.count("😄") + analysis_user.count("❤️")
        question_count = analysis_user.count("?")
        
        # Emoticons und Humor-Indikatoren
        humor_indicators = ["haha", "lol", "😂", "witzig", "lustig", "humor"]
        romance_indicators = ["liebe", "herz", "schön", "süß", "romantisch", "❤️", "😍"]
        interest_indicators = ["interessant", "spannend", "erzähl", "mehr", "?"]
        
        humor_score = sum(3 for indicator in humor_indicators if indicator in analysis_user.lower()) * 10
        romance_score = sum(5 for indicator in romance_indicators if indicator in analysis_user.lower()) * 8  
        interest_score = sum(2 for indicator in interest_indicators if indicator in analysis_user.lower()) * 12
        
        # Berechne erweiterte Scores
        analysis = {
            "overall_score": min(100, max(30, user_length * 2 + word_count * 3 + emoji_count * 15 + 40)),
            "interest_probability": min(100, max(20, interest_score + question_count * 15 + 45)),
            "humor_rating": min(100, max(0, humor_score + emoji_count * 10 + 25)),
            "romance_potential": min(100, max(10, romance_score + emoji_count * 12 + 30)),
            "charm_factor": min(100, max(25, user_length + word_count * 2 + emoji_count * 8 + 35)),
            "emotional_tone": ("positiv" if emoji_count > 0 or any(w in analysis_user.lower() for w in ["gut", "toll", "super", "schön"]) 
                              else "neutral" if word_count > 3 else "zurückhaltend"),
            "conversation_flow": ("fließend" if word_count > 8 and question_count > 0
                                else "gut" if word_count > 4
                                else "kurz"),
            "suggestions": []
        }
        
        # Intelligente Empfehlungen basierend auf Analyse
        suggestions = []
        if analysis["humor_rating"] < 50:
            suggestions.append("Mehr Humor einbauen - das macht Gespräche leichter!")
        if analysis["interest_probability"] < 60:
            suggestions.append("Stelle mehr persönliche Fragen")
        if analysis["romance_potential"] < 40:
            suggestions.append("Subtile Komplimente können das Gespräch erwärmen")
        if question_count == 0:
            suggestions.append("Fragen zeigen Interesse am Gesprächspartner")
        if user_length < 20:
            suggestions.append("Ausführlichere Nachrichten wirken interessierter")
        
        analysis["suggestions"] = suggestions[:3] if suggestions else ["Mach weiter so - das Gespräch läuft gut!"]
        
        # Kontextualisierte Antworten basierend auf Analyse
        if analysis["humor_rating"] > 70:
            responses = [
                "Haha, du bringst mich wirklich zum Lachen! 😄",
                "Dein Humor ist echt ansteckend! 😊", 
                "Du hast definitiv den richtigen Dreh raus! 😂",
                "Mit dir wird es nie langweilig! 🤣"
            ]
        elif analysis["romance_potential"] > 60:
            responses = [
                "Das ist wirklich süß von dir... 😊❤️",
                "Du hast so eine schöne Art, Dinge zu sagen 💕",
                "Bei dir fühle ich mich besonders 😍",
                "Du weißt wirklich, wie du mein Herz berührst ✨"
            ]
        elif analysis["interest_probability"] > 70:
            responses = [
                "Wow, das interessiert mich wirklich! Erzähl mehr davon 🤔",
                "Das klingt spannend - du denkst wirklich tief über Dinge nach!",
                "Ich mag es, wie du die Welt siehst 😊",
                "Das ist eine faszinierende Perspektive!"
            ]
        else:
            responses = [
                "Das ist wirklich interessant! Wie siehst du das denn? 🤔",
                "Erzähl mir mehr - ich höre gerne zu! 😊",
                "Du hast eine einzigartige Art zu denken!",
                "Das bringt mich zum Nachdenken... 💭",
                "Ich finde es toll, dass du so offen bist! ✨"
            ]
        
        # Wähle passende Antwort
        reply = responses[hash(reply_user) % len(responses)]
        
        # Füge gelegentlich Follow-up-Fragen hinzu
        if random.random() < 0.3:  # 30% Chance
            follow_ups = [
                " Und was denkst du darüber?",
                " Erzähl mir mehr davon!",
                " Das würde mich auch interessieren!",
                " Wie bist du darauf gekommen?"
            ]
            reply += random.choice(follow_ups)
        
        return {
            "ok": True,
            "analysis": analysis,
            "raw_analysis": json.dumps(analysis, ensure_ascii=False),
            "reply": reply
        }
        
    except Exception as e:
        logger.error(f"Flirt respond error: {e}")
        return {
            "ok": False,
            "error": str(e),
            "analysis": {
                "overall_score": 50,
                "interest_probability": 50,
                "humor_rating": 50,
                "romance_potential": 50,
                "charm_factor": 50,
                "suggestions": ["System-Fehler - versuche es erneut"]
            },
            "reply": "Entschuldige, ich hatte einen kleinen Aussetzer. Magst du das nochmal sagen? 😅"
        }

@app.post("/api/generate/universal")
async def generate_universal(
    prompt: str = Form(...),
    mode: str = Form("text2img"),
    model_id: Optional[str] = Form(None),
    nsfw: Optional[str] = Form("false"),
    width: Optional[int] = Form(1024),
    height: Optional[int] = Form(768),
    num_inference_steps: Optional[int] = Form(28),
    guidance_scale: Optional[float] = Form(7.5),
    output_format: Optional[str] = Form("png"),
    image: Optional[UploadFile] = File(None),
    init_image: Optional[UploadFile] = File(None)
):
    """Universal Generator mit Fallback-Funktionalität"""
    
    start_time = time.time()
    is_nsfw = str(nsfw).lower() == "true"
    
    # Operation registrieren
    op_id = f"gen_{int(time.time() * 1000)}"
    OPERATIONS[op_id] = {
        "kind": f"{mode}",
        "status": "running", 
        "progress": 0,
        "start_time": start_time
    }
    
    try:
        # Parameter normalisieren
        W = max(64, min(width or 1024, 2048))
        H = max(64, min(height or 768, 2048))
        steps = max(1, min(num_inference_steps or 28, 100))
        guidance = max(1.0, min(guidance_scale or 7.5, 20.0))
        
        # Modus bestimmen
        if mode.lower() in ["image", "text2img", "img2img"]:
            task = "image"
        elif mode.lower() in ["video", "text2video", "img2video"]:
            task = "video"
        else:
            task = "image"
        
        OPERATIONS[op_id]["progress"] = 20
        
        # Fallback-Generierung (funktioniert immer)
        if task == "image":
            # Bild generieren
            filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
            filepath = OUTPUTS_DIR / "images" / filename
            
            if PIL_AVAILABLE:
                # Erstelle farbiges Bild mit Text
                colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8"]
                bg_color = random.choice(colors)
                
                img = Image.new('RGB', (W, H), color=bg_color)
                draw = ImageDraw.Draw(img)
                
                try:
                    # Versuche verschiedene Schriftgrößen
                    font_size = min(W, H) // 15
                    font = ImageFont.load_default()
                except:
                    font = None
                
                # Text hinzufügen
                lines = [
                    "🎨 LocalMediaSuite",
                    "",
                    f"Generated: {prompt[:30]}...",
                    f"Mode: {mode}",
                    f"Size: {W}x{H}",
                    f"Steps: {steps}",
                    "",
                    "✨ Fallback Generator ✨",
                    "(Install AI packages for real generation)"
                ]
                
                y_offset = 50
                for line in lines:
                    if line:
                        draw.text((50, y_offset), line, fill='white', font=font)
                    y_offset += 30
                
                # Speichere Bild
                img.save(filepath, "PNG", optimize=True)
            else:
                # Minimal-Fallback ohne PIL
                filepath.touch()
            
            OPERATIONS[op_id]["progress"] = 100
            OPERATIONS[op_id]["status"] = "completed"
            
            result = {
                "ok": True,
                "kind": "image",
                "model": model_id or "fallback",
                "file": f"/outputs/images/{filename}",
                "url": f"/outputs/images/{filename}",
                "metadata": {
                    "prompt": prompt,
                    "mode": mode,
                    "nsfw": is_nsfw,
                    "resolution": f"{W}x{H}",
                    "steps": steps,
                    "guidance": guidance,
                    "generation_time": time.time() - start_time,
                    "fallback": True
                }
            }
            
        else:
            # Video-Fallback (erstelle statisches Bild als .mp4)
            filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
            filepath = OUTPUTS_DIR / "videos" / filename
            
            if PIL_AVAILABLE:
                img = Image.new('RGB', (W, H), color='#2C3E50')
                draw = ImageDraw.Draw(img)
                
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
                
                lines = [
                    "🎬 Video Generator",
                    "",
                    f"Prompt: {prompt[:25]}...",
                    f"Resolution: {W}x{H}", 
                    "",
                    "⚠️ Fallback Mode",
                    "Install video AI packages",
                    "for real video generation"
                ]
                
                y_offset = H//6
                for line in lines:
                    if line:
                        draw.text((50, y_offset), line, fill='white', font=font)
                    y_offset += 30
                
                img.save(filepath, "PNG")
            else:
                filepath.touch()
            
            OPERATIONS[op_id]["progress"] = 100
            OPERATIONS[op_id]["status"] = "completed"
            
            result = {
                "ok": True,
                "kind": "video",
                "model": model_id or "fallback",
                "file": f"/outputs/videos/{filename}",
                "url": f"/outputs/videos/{filename}",
                "metadata": {
                    "prompt": prompt,
                    "mode": mode,
                    "nsfw": is_nsfw,
                    "resolution": f"{W}x{H}",
                    "generation_time": time.time() - start_time,
                    "fallback": True
                }
            }
        
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        OPERATIONS[op_id]["status"] = "error"
        OPERATIONS[op_id]["error"] = str(e)
        
        return JSONResponse({
            "ok": False,
            "error": str(e),
            "fallback": True
        }, status_code=500)

@app.get("/api/suggestions")
async def get_suggestions(
    type: str = Query(..., pattern="^(image|video)$"),
    nsfw: bool = Query(False)
):
    """Prompt-Vorschläge"""
    
    suggestions = {
        "image": {
            "sfw": {
                "categories": [
                    {
                        "id": "portrait",
                        "label": "Portrait",
                        "prompts": [
                            "Professional headshot, studio lighting, sharp focus",
                            "Candid portrait, golden hour, natural expression",
                            "Artistic portrait, dramatic shadows, black and white"
                        ]
                    },
                    {
                        "id": "landscape", 
                        "label": "Landschaft",
                        "prompts": [
                            "Mountain landscape, sunrise, misty valleys",
                            "Ocean waves, dramatic sky, rocky coastline",
                            "Forest scene, autumn colors, sunbeams through trees"
                        ]
                    }
                ],
                "modifiers": ["8k detail", "photorealistic", "high quality", "masterpiece"]
            },
            "nsfw": {
                "categories": [
                    {
                        "id": "artistic",
                        "label": "Künstlerisch",
                        "prompts": [
                            "Artistic nude, tasteful lighting, classical pose",
                            "Boudoir photography, soft shadows, elegant"
                        ]
                    }
                ],
                "modifiers": ["tasteful", "artistic", "professional"]
            }
        },
        "video": {
            "sfw": {
                "categories": [
                    {
                        "id": "nature",
                        "label": "Natur",
                        "prompts": [
                            "Flowing river, peaceful forest sounds",
                            "Clouds moving across mountain peaks",
                            "Ocean waves gently hitting the shore"
                        ]
                    }
                ],
                "modifiers": ["smooth motion", "cinematic", "4k quality"]
            },
            "nsfw": {
                "categories": [],
                "modifiers": []
            }
        }
    }
    
    category_key = "nsfw" if nsfw else "sfw"
    return suggestions.get(type, {}).get(category_key, {"categories": [], "modifiers": []})

# Startup Event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 LocalMediaSuite API gestartet")
    logger.info(f"📂 Web: {WEB_DIR}")
    logger.info(f"💾 Outputs: {OUTPUTS_DIR}")
    logger.info(f"🤖 PIL verfügbar: {PIL_AVAILABLE}")

# Shutdown Event  
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("👋 LocalMediaSuite API beendet")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000, reload=True)