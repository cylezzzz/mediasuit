#!/usr/bin/env python3
"""
LocalMediaSuite Enhanced Startup Script
100% Funktionalität mit automatischen Fallbacks
"""
import sys
import os
import subprocess
import logging
import time
import socket
from pathlib import Path
from typing import Optional

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Prüfe Python-Version"""
    if sys.version_info < (3, 9):
        logger.error("❌ Python 3.9+ erforderlich")
        print(f"Aktuelle Version: {sys.version}")
        print("Bitte installiere Python 3.9 oder höher")
        return False
    logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} OK")
    return True

def check_and_install_dependencies():
    """Prüfe und installiere Abhängigkeiten"""
    logger.info("🔍 Prüfe Abhängigkeiten...")
    
    # Kritische Abhängigkeiten
    critical_packages = [
        ("fastapi", "FastAPI Web Framework"),
        ("uvicorn", "ASGI Server"),
        ("pydantic", "Data Validation"),
        ("PIL", "Pillow Image Processing")
    ]
    
    missing_critical = []
    
    for package, description in critical_packages:
        try:
            if package == "PIL":
                import PIL
            else:
                __import__(package)
            logger.info(f"✅ {description} verfügbar")
        except ImportError:
            missing_critical.append(package)
            logger.warning(f"⚠️ {description} fehlt")
    
    # Installiere fehlende kritische Pakete
    if missing_critical:
        logger.info("📦 Installiere fehlende kritische Abhängigkeiten...")
        install_packages = {
            "fastapi": "fastapi",
            "uvicorn": "uvicorn[standard]", 
            "pydantic": "pydantic",
            "PIL": "pillow"
        }
        
        to_install = [install_packages.get(pkg, pkg) for pkg in missing_critical]
        to_install.extend(["python-multipart", "numpy", "requests"])
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + to_install, timeout=300)
            logger.info("✅ Kritische Abhängigkeiten installiert")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"❌ Installation fehlgeschlagen: {e}")
            print("\n💡 Manuelle Installation:")
            print(f"pip install {' '.join(to_install)}")
            return False, False
    
    # Prüfe AI-Bibliotheken (optional)
    ai_available = True
    ai_packages = [("torch", "PyTorch"), ("diffusers", "Diffusers")]
    
    for package, description in ai_packages:
        try:
            __import__(package)
            logger.info(f"✅ {description} verfügbar")
        except ImportError:
            logger.warning(f"⚠️ {description} nicht verfügbar")
            ai_available = False
    
    if not ai_available:
        logger.warning("⚠️ AI-Bibliotheken fehlen - Fallback-Modus aktiv")
        print("💡 Für AI-Features installiere: pip install torch diffusers transformers")
    
    return True, ai_available

def setup_directories():
    """Erstelle notwendige Verzeichnisse"""
    logger.info("📁 Erstelle Verzeichnisstruktur...")
    
    directories = [
        "outputs/images",
        "outputs/videos", 
        "models/image",
        "models/video",
        "models/llm",
        "config",
        "server",
        "web"
    ]
    
    created = 0
    for dir_path in directories:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created += 1
            
        # .gitkeep für leere Model-Verzeichnisse
        if "models" in dir_path:
            gitkeep = path / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()
    
    logger.info(f"✅ {created} Verzeichnisse erstellt, {len(directories)-created} bereits vorhanden")

def find_free_port(start_port: int = 3000) -> int:
    """Finde freien Port"""
    for port in range(start_port, start_port + 20):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", port))
                return port
        except OSError:
            continue
    return start_port

def get_local_ip() -> str:
    """Ermittle lokale IP"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

def create_minimal_server():
    """Erstelle minimalen Fallback-Server wenn Server-Module fehlen"""
    
    # Erstelle server/__init__.py
    server_init = Path("server/__init__.py")
    if not server_init.exists():
        server_init.write_text("")
    
    # Erstelle server/app.py falls nicht vorhanden
    server_app = Path("server/app.py")
    if not server_app.exists():
        app_content = '''from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import time
import uuid

app = FastAPI(title="LocalMediaSuite", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
ROOT = Path(__file__).parent.parent
WEB_DIR = ROOT / "web"
OUTPUTS_DIR = ROOT / "outputs"

if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
if OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

@app.get("/api/health")
async def health():
    return {"ok": True, "status": "running", "mode": "minimal"}

@app.post("/api/respond")
async def flirt_respond(request: dict):
    """Flirt AI Mock Response"""
    user_msg = request.get("reply_user", "")
    
    responses = [
        "Das ist interessant! Erzähl mir mehr 😊",
        "Du hast einen tollen Sinn für Humor! 😄", 
        "Das gefällt mir an dir - du denkst anders! 🤔",
        "Wow, das hätte ich nicht erwartet! 😍",
        "Du überraschst mich immer wieder! ✨"
    ]
    
    analysis = {
        "overall_score": min(100, max(30, len(user_msg) * 2 + 50)),
        "interest_probability": min(100, max(20, len(user_msg) + 40)),
        "humor_rating": 60 + (user_msg.count("😂") + user_msg.count("haha")) * 10,
        "romance_potential": 55,
        "charm_factor": min(100, max(40, len(user_msg) + 30)),
        "suggestions": ["Mehr persönliche Fragen", "Humorvoll bleiben"]
    }
    
    return {
        "ok": True,
        "analysis": analysis,
        "reply": responses[len(user_msg) % len(responses)]
    }

@app.get("/api/models")
async def get_models():
    return [{"id": "fallback", "name": "Fallback Model", "type": "image"}]

@app.post("/api/generate/universal")
async def generate_universal(request: dict = None):
    """Fallback Generator"""
    # Erstelle Dummy-Dateien
    import random
    from PIL import Image, ImageDraw, ImageFont
    
    filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
    filepath = OUTPUTS_DIR / "images" / filename
    filepath.parent.mkdir(exist_ok=True)
    
    # Dummy-Bild
    img = Image.new('RGB', (1024, 768), color=f'#{random.randint(100, 200):02x}{random.randint(100, 200):02x}{random.randint(100, 200):02x}')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.load_default()
        draw.text((50, 50), "LocalMediaSuite\\nFallback Generator\\nInstalliere AI-Pakete\\nfür echte Generierung", 
                 fill='white', font=font)
    except:
        pass
    
    img.save(filepath)
    
    return {
        "ok": True,
        "file": f"/outputs/images/{filename}",
        "url": f"/outputs/images/{filename}",
        "metadata": {"mode": "fallback", "generation_time": 0.1}
    }
'''
        server_app.write_text(app_content)
        logger.info("✅ Minimaler Server erstellt")

def create_basic_web_files():
    """Erstelle grundlegende Web-Dateien falls nicht vorhanden"""
    
    web_dir = Path("web")
    web_dir.mkdir(exist_ok=True)
    
    # index.html falls nicht vorhanden
    index_file = web_dir / "index.html"
    if not index_file.exists():
        index_content = '''<!DOCTYPE html>
<html>
<head>
    <title>LocalMediaSuite</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial; margin: 40px; background: #1a1a1a; color: white; }
        .card { background: #2a2a2a; padding: 20px; border-radius: 10px; margin: 10px; }
        a { color: #4CAF50; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>🚀 LocalMediaSuite</h1>
    <div class="card">
        <h2>System läuft!</h2>
        <p>✅ Server erfolgreich gestartet</p>
        <p>📡 API verfügbar unter <a href="/api/docs">/api/docs</a></p>
        <p>🔧 Installiere vollständige Features mit den Artefakt-Dateien</p>
    </div>
</body>
</html>'''
        index_file.write_text(index_content, encoding='utf-8')
        logger.info("✅ Basis index.html erstellt")

def start_server(port: int, reload: bool = False, ai_available: bool = True):
    """Starte FastAPI Server"""
    try:
        import uvicorn
        
        # Bereite Server vor
        setup_directories()
        create_minimal_server()
        create_basic_web_files()
        
        # Server-Konfiguration
        config = {
            "host": "0.0.0.0", 
            "port": port,
            "reload": reload,
            "log_level": "info",
            "access_log": True
        }
        
        # App importieren
        try:
            from server.app import app
            logger.info("✅ Server-App importiert")
        except ImportError as e:
            logger.warning(f"⚠️ Server-Import fehlgeschlagen: {e}")
            logger.info("🔄 Verwende Fallback-Konfiguration...")
            
            # Fallback: Direkte App-Erstellung
            from fastapi import FastAPI
            from fastapi.staticfiles import StaticFiles
            
            app = FastAPI(title="LocalMediaSuite Minimal")
            
            if Path("web").exists():
                app.mount("/", StaticFiles(directory="web", html=True), name="web")
            
            @app.get("/api/health")
            async def health():
                return {"ok": True, "mode": "minimal", "ai_available": ai_available}
        
        # Startup-Information
        local_ip = get_local_ip()
        
        print("\n" + "="*70)
        print("🚀 LocalMediaSuite Enhanced gestartet!")
        print("="*70)
        print(f"📡 Server läuft auf Port {port}")
        print(f"🤖 AI-Features: {'✅ Verfügbar' if ai_available else '⚠️ Fallback-Modus'}")
        print(f"📊 Reload-Modus: {'✅ Aktiv' if reload else '❌ Deaktiviert'}")
        print("\n🌐 Zugriff über:")
        print(f"   • Lokal:        http://127.0.0.1:{port}")
        print(f"   • Netzwerk:     http://{local_ip}:{port}")
        print(f"   • API Docs:     http://{local_ip}:{port}/api/docs")
        print("\n🎯 Verfügbare Features:")
        if ai_available:
            print("   • Vollständige AI-Generierung (Bilder & Videos)")
            print("   • Erweiterte Flirt-KI mit Analyse")
            print("   • Modell-Katalog mit Installation")
        else:
            print("   • Fallback-Generierung (Platzhalter)")
            print("   • Basis-Flirt-System")
            print("   • Grundlegende Funktionen")
        print("\n💡 Vollständige Features:")
        print("   Kopiere die Artefakt-Dateien in dein Projekt!")
        print("\n" + "="*70)
        print("🛑 Beenden: Strg+C")
        print("="*70 + "\n")
        
        # Server starten
        uvicorn.run(app, **config)
        
    except KeyboardInterrupt:
        logger.info("\n👋 Server durch Benutzer beendet")
    except Exception as e:
        logger.error(f"❌ Server-Start fehlgeschlagen: {e}")
        logger.info("💡 Versuche manuelle Installation:")
        logger.info("pip install fastapi uvicorn[standard] pillow")
        sys.exit(1)

def show_help():
    """Zeige Hilfe-Information"""
    print("""
LocalMediaSuite Enhanced Startup

VERWENDUNG:
    python start_enhanced.py [OPTIONEN]

OPTIONEN:
    --dev, --reload     Entwicklungsmodus mit Hot-Reload
    --headless          Starte ohne Desktop-UI (nur Server)
    --port PORT         Verwende spezifischen Port (Standard: 3000)
    --help, -h          Zeige diese Hilfe

BEISPIELE:
    python start_enhanced.py                    # Normal starten
    python start_enhanced.py --dev              # Entwicklungsmodus  
    python start_enhanced.py --port 8080        # Port 8080
    python start_enhanced.py --headless         # Nur Server

ERSTE SCHRITTE:
    1. Dieses Script startet einen funktionsfähigen Basis-Server
    2. Kopiere die Artefakt-Dateien für vollständige Features
    3. Installiere AI-Pakete: pip install torch diffusers transformers
    4. Öffne http://localhost:3000 im Browser

PROBLEMBEHEBUNG:
    • Python 3.9+ erforderlich
    • Bei Fehlern: pip install fastapi uvicorn pillow
    • Für AI: pip install torch diffusers transformers
""")

def main():
    """Hauptfunktion"""
    # Hilfe anzeigen
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        return
    
    print("🔧 LocalMediaSuite Enhanced wird gestartet...")
    
    # Python-Version prüfen
    if not check_python_version():
        sys.exit(1)
    
    # Abhängigkeiten prüfen und installieren
    deps_ok, ai_available = check_and_install_dependencies()
    if not deps_ok:
        print("\n❌ Kritische Abhängigkeiten fehlen!")
        print("💡 Versuche: pip install fastapi uvicorn[standard] pillow python-multipart")
        sys.exit(1)
    
    # Kommandozeilen-Parameter
    reload = "--dev" in sys.argv or "--reload" in sys.argv
    headless = "--headless" in sys.argv
    
    # Port bestimmen
    port = 3000
    if "--port" in sys.argv:
        try:
            port_idx = sys.argv.index("--port") + 1
            if port_idx < len(sys.argv):
                port = int(sys.argv[port_idx])
            else:
                raise ValueError("Port-Wert fehlt")
        except (ValueError, IndexError):
            logger.warning("⚠️ Ungültiger Port-Parameter, verwende 3000")
            port = 3000
    
    # Freien Port finden
    original_port = port
    port = find_free_port(port)
    if port != original_port:
        logger.info(f"🔄 Port {original_port} belegt, verwende {port}")
    
    # Modus-Information
    if reload:
        logger.info("🔄 Development-Modus: Hot-Reload aktiv")
    if headless:
        logger.info("🖥️ Headless-Modus: Nur Server, keine UI")
    
    # Server starten
    try:
        start_server(port, reload, ai_available)
    except KeyboardInterrupt:
        logger.info("👋 Programm beendet")
    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()