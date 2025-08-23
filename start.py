# start.py – LocalMediaSuite Enhanced Startup
from __future__ import annotations
import threading
import time
import sys
import os
import socket
import subprocess
import logging
from pathlib import Path
from server.server import build_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

HOST = "0.0.0.0"
PORT = 3000
RELOAD = "--reload" in sys.argv or "--dev" in sys.argv

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    # Check core dependencies
    try:
        import fastapi
        import uvicorn
        import pydantic
    except ImportError as e:
        missing_deps.append(f"FastAPI stack: {e}")
    
    # Check optional dependencies
    optional_missing = []
    try:
        import torch
    except ImportError:
        optional_missing.append("torch (für GPU-Beschleunigung)")
    
    try:
        import diffusers
    except ImportError:
        optional_missing.append("diffusers (für KI-Generierung)")
    
    if missing_deps:
        print("❌ Kritische Abhängigkeiten fehlen:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n💡 Installiere sie mit: pip install -r requirements.txt")
        return False
    
    if optional_missing:
        print("⚠️ Optionale Abhängigkeiten fehlen:")
        for dep in optional_missing:
            print(f"   - {dep}")
        print("   KI-Generierung wird möglicherweise nicht funktionieren.\n")
    
    return True

def setup_directories():
    """Create necessary directories if they don't exist"""
    dirs_to_create = [
        "outputs/images",
        "outputs/videos", 
        "models/image",
        "models/video",
        "models/llm",
        "config"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    # Create .gitkeep files for empty directories
    for dir_path in dirs_to_create:
        gitkeep_path = Path(dir_path) / ".gitkeep"
        if not gitkeep_path.exists():
            gitkeep_path.touch()
    
    logger.info("📁 Verzeichnisstruktur erstellt/überprüft")

def get_local_ip():
    """Get the local IP address"""
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

def print_startup_info(port: int):
    """Print startup information with all available URLs"""
    local_ip = get_local_ip()
    
    print("\n" + "="*60)
    print("🚀 LocalMediaSuite gestartet!")
    print("="*60)
    print(f"📡 Server läuft auf Port {port}")
    print("\n🌐 Erreichbar unter:")
    print(f"   • Lokal:      http://127.0.0.1:{port}")
    print(f"   • Netzwerk:   http://{local_ip}:{port}")
    print(f"   • Localhost:  http://localhost:{port}")
    
    if local_ip != "127.0.0.1":
        print(f"\n📱 Für mobile Geräte im Netzwerk: http://{local_ip}:{port}")
    
    print("\n🎯 Verfügbare Seiten:")
    print("   • /             - Startseite")
    print("   • /image.html   - Bild-Generator (SFW)")
    print("   • /image_nsfw.html - Bild-Generator (NSFW)")
    print("   • /video.html   - Video-Generator (SFW)")
    print("   • /video_nsfw.html - Video-Generator (NSFW)")
    print("   • /gallery.html - Galerie")
    print("   • /catalog.html - Modell-Katalog")
    print("   • /settings.html - Einstellungen")
    
    print("\n🔧 API-Endpunkte:")
    print(f"   • http://{local_ip}:{port}/api/status")
    print(f"   • http://{local_ip}:{port}/api/models")
    print(f"   • http://{local_ip}:{port}/docs (FastAPI Dokumentation)")
    
    print("\n" + "="*60)
    print("💡 Tipp: Öffne die Desktop-UI für Live-Status und Systeminfos")
    print("🛑 Beenden: Strg+C oder Desktop-UI schließen")
    print("="*60 + "\n")

def run_server(host: str, port: int, reload: bool = False):
    """Start the FastAPI server"""
    try:
        import uvicorn
        from server.server import build_app
        
        app = build_app()
        
        # Configure uvicorn with better settings
        config = uvicorn.Config(
            app=app,
            host=host, 
            port=port,
            reload=reload,
            log_level="info",
            access_log=True,
            server_header=False,
            date_header=False
        )
        
        server = uvicorn.Server(config)
        server.run()
        
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)

def start_with_ui(host: str, port: int, reload: bool = False):
    """Start server with desktop UI"""
    
    # Setup
    setup_directories()
    print_startup_info(port)
    
    # Start server in background thread
    server_thread = threading.Thread(
        target=run_server, 
        args=(host, port, reload), 
        daemon=True,
        name="FastAPIServer"
    )
    server_thread.start()
    
    # Give server time to start
    time.sleep(2)
    
    # Check if server started successfully
    try:
        import urllib.request
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5)
        logger.info("✅ Server erfolgreich gestartet")
    except Exception as e:
        logger.error(f"❌ Server nicht erreichbar: {e}")
        print("⚠️  Server möglicherweise nicht gestartet. Prüfe die Logs.")
    
    # ---- Desktop UI (Tkinter) ----
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
        import urllib.request
        import json
    except ImportError:
        logger.error("Tkinter nicht verfügbar - starte ohne Desktop-UI")
        print("🖥️  Desktop-UI nicht verfügbar (Tkinter fehlt)")
        print("💻 Server läuft trotzdem - öffne Browser manuell")
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("👋 Server beendet")
            return
    
    def fetch_json(url, timeout=3):
        """Fetch JSON data from URL with error handling"""
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            logger.debug(f"API call failed: {url} - {e}")
            return None
    
    # Create main window
    root = tk.Tk()
    root.title("LocalMediaSuite – Server Control")
    root.geometry("900x600")
    root.configure(bg='#1a1a1a')
    
    # Configure styling
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TLabel', background='#1a1a1a', foreground='white')
    style.configure('TFrame', background='#1a1a1a')
    style.configure('TLabelFrame', background='#1a1a1a', foreground='white')
    
    # Header Frame
    header_frame = ttk.Frame(root, padding=10)
    header_frame.pack(fill="x")
    
    title_label = ttk.Label(
        header_frame, 
        text="LocalMediaSuite Server", 
        font=('Arial', 16, 'bold')
    )
    title_label.pack(side="left")
    
    url_label = ttk.Label(
        header_frame, 
        text=f"🌐 http://{get_local_ip()}:{port}",
        font=('Arial', 10),
        foreground='#4CAF50'
    )
    url_label.pack(side="right")
    
    # System Status Frame
    status_frame = ttk.LabelFrame(root, text="📊 Systemstatus", padding=10)
    status_frame.pack(fill="x", padx=10, pady=5)
    
    status_labels = {}
    status_info = [
        ('os', 'Betriebssystem: -'),
        ('python', 'Python: -'),
        ('cpu', 'CPU: -%'),
        ('ram', 'RAM: -% (- GB)'),
        ('server', 'Server: Starte...')
    ]
    
    for key, text in status_info:
        label = ttk.Label(status_frame, text=text)
        label.pack(anchor="w", pady=2)
        status_labels[key] = label
    
    # Operations Frame
    ops_frame = ttk.LabelFrame(root, text="⚡ Laufende Vorgänge", padding=10)
    ops_frame.pack(fill="both", expand=True, padx=10, pady=5)
    
    # Operations Treeview
    columns = ('operation', 'status', 'progress', 'duration')
    ops_tree = ttk.Treeview(ops_frame, columns=columns, show="headings", height=8)
    
    ops_tree.heading('operation', text='Vorgang')
    ops_tree.heading('status', text='Status')
    ops_tree.heading('progress', text='Fortschritt')
    ops_tree.heading('duration', text='Dauer')
    
    ops_tree.column('operation', width=200)
    ops_tree.column('status', width=100)
    ops_tree.column('progress', width=100)
    ops_tree.column('duration', width=100)
    
    # Scrollbar for treeview
    ops_scrollbar = ttk.Scrollbar(ops_frame, orient="vertical", command=ops_tree.yview)
    ops_tree.configure(yscrollcommand=ops_scrollbar.set)
    
    ops_tree.pack(side="left", fill="both", expand=True)
    ops_scrollbar.pack(side="right", fill="y")
    
    # Control Buttons Frame
    controls_frame = ttk.Frame(root, padding=10)
    controls_frame.pack(fill="x")
    
    def open_browser():
        """Open browser with main URL"""
        try:
            import webbrowser
            webbrowser.open(f"http://{get_local_ip()}:{port}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Browser konnte nicht geöffnet werden: {e}")
    
    def refresh_data():
        """Manually refresh all data"""
        update_status()
    
    # Buttons
    ttk.Button(controls_frame, text="🌐 Browser öffnen", command=open_browser).pack(side="left", padx=5)
    ttk.Button(controls_frame, text="🔄 Aktualisieren", command=refresh_data).pack(side="left", padx=5)
    
    # Status indicator
    status_indicator = ttk.Label(controls_frame, text="🔴 Starte...", foreground='orange')
    status_indicator.pack(side="right", padx=5)
    
    def update_status():
        """Update all status information"""
        # System status
        system_data = fetch_json(f"http://127.0.0.1:{port}/api/status")
        if system_data and system_data.get("ok"):
            sys_info = system_data.get("system", {})
            status_labels['os'].config(text=f"Betriebssystem: {sys_info.get('os', 'Unbekannt')}")
            status_labels['python'].config(text=f"Python: {sys_info.get('python', 'Unbekannt')}")
            status_labels['cpu'].config(text=f"CPU: {sys_info.get('cpu_percent', 0):.1f}%")
            status_labels['ram'].config(text=f"RAM: {sys_info.get('ram_percent', 0):.1f}% ({sys_info.get('total_ram_gb', 0):.1f} GB)")
            status_labels['server'].config(text="Server: ✅ Online")
            status_indicator.config(text="🟢 Online", foreground='green')
        else:
            status_labels['server'].config(text="Server: ❌ Offline")
            status_indicator.config(text="🔴 Offline", foreground='red')
        
        # Operations status
        ops_data = fetch_json(f"http://127.0.0.1:{port}/api/ops")
        ops_tree.delete(*ops_tree.get_children())
        
        if ops_data and ops_data.get("ok"):
            ops_list = ops_data.get("ops", [])
            current_time = time.time()
            
            for op in ops_list[-10:]:  # Show last 10 operations
                start_time = op.get("ts_start", current_time)
                duration = int(current_time - start_time)
                progress = f"{op.get('progress', 0):.1f}%"
                
                # Format operation name
                op_name = op.get('kind', 'Unknown')
                if op.get('params', {}).get('prompt'):
                    prompt_preview = op['params']['prompt'][:30] + "..." if len(op['params']['prompt']) > 30 else op['params']['prompt']
                    op_name += f": {prompt_preview}"
                
                # Status with emoji
                status = op.get('status', 'unknown')
                status_display = {
                    'running': '🔄 Läuft',
                    'done': '✅ Fertig', 
                    'error': '❌ Fehler'
                }.get(status, status)
                
                ops_tree.insert("", "end", values=(
                    op_name,
                    status_display,
                    progress,
                    f"{duration}s"
                ))
    
    # Update loop
    def update_loop():
        """Periodic update loop"""
        update_status()
        root.after(2000, update_loop)  # Update every 2 seconds
    
    # Start update loop
    root.after(1000, update_loop)  # Start after 1 second
    
    def on_closing():
        """Handle window closing"""
        if messagebox.askokcancel("Beenden", "LocalMediaSuite Server beenden?"):
            logger.info("👋 Desktop-UI geschlossen - Server wird beendet")
            root.destroy()
            os._exit(0)  # Force exit to stop server thread
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start UI
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("👋 Unterbrochen durch Benutzer")
    except Exception as e:
        logger.error(f"UI Error: {e}")

def start_headless(host: str, port: int, reload: bool = False):
    """Start server without UI (headless mode)"""
    setup_directories()
    print_startup_info(port)
    
    try:
        run_server(host, port, reload)
    except KeyboardInterrupt:
        logger.info("👋 Server beendet durch Benutzer")
    except Exception as e:
        logger.error(f"Server Error: {e}")

def main():
    """Main entry point with argument parsing"""
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print("""
LocalMediaSuite Startup Script

Usage:
  python start.py [options]

Options:
  --headless     Starte ohne Desktop-UI
  --reload       Aktiviere Hot-Reload für Entwicklung
  --dev          Alias für --reload
  --port PORT    Verwende anderen Port (default: 3000)
  --host HOST    Bind zu anderer Adresse (default: 0.0.0.0)
  --help, -h     Zeige diese Hilfe

Beispiele:
  python start.py                    # Normal mit Desktop-UI
  python start.py --headless         # Ohne UI
  python start.py --dev              # Entwicklungsmodus
  python start.py --port 8080        # Anderer Port
  python start.py --headless --port 8080  # Kombiniert
""")
            return
    
    # Parse port
    port = PORT
    if '--port' in sys.argv:
        try:
            port_idx = sys.argv.index('--port') + 1
            port = int(sys.argv[port_idx])
        except (IndexError, ValueError):
            print("❌ Ungültiger Port angegeben")
            sys.exit(1)
    
    # Parse host
    host = HOST
    if '--host' in sys.argv:
        try:
            host_idx = sys.argv.index('--host') + 1
            host = sys.argv[host_idx]
        except IndexError:
            print("❌ Ungültiger Host angegeben")
            sys.exit(1)
    
    # Parse other flags
    headless = '--headless' in sys.argv
    reload = '--reload' in sys.argv or '--dev' in sys.argv
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Start appropriate mode
    if headless:
        logger.info("🖥️  Starte im Headless-Modus (ohne Desktop-UI)")
        start_headless(host, port, reload)
    else:
        logger.info("🖥️  Starte mit Desktop-UI")
        start_with_ui(host, port, reload)

if __name__ == "__main__":
    main()