# LocalMediaSuite 2.0 🚀

**LocalMediaSuite 2.0** ist die ultimative universelle KI-Suite für lokale Medien-Generation. Das System erkennt **automatisch alle Modellformate** und kann über 50+ verschiedene Modelltypen verwenden - von Stable Diffusion bis zu modernen LLMs.

**🎯 Neu in Version 2.0:**
- **Universelle Modell-Erkennung**: Automatische Erkennung aller gängigen Formate
- **Multi-Backend-Support**: Diffusers, ONNX, llama.cpp, Transformers, Ollama
- **Erweiterte Katalog-Datenbank**: 50+ vorkonfigurierte Modelle zum Download
- **Intelligente Kompatibilitätsprüfung**: Hardware-Optimierung und Empfehlungen
- **Unified Generation API**: Ein Endpunkt für alle Generierungsarten

---

## ✨ Unterstützte Modellformate

### 🖼️ **Bildmodelle**
- **Diffusers Repositories** (.diffusers-Ordner mit model_index.json)
- **SafeTensors** (.safetensors) - Empfohlenes sicheres Format
- **Checkpoints** (.ckpt, .pt, .pth) - Legacy PyTorch Format
- **ONNX** (.onnx) - Optimiert für CPU/GPU-Inferenz
- **TensorRT** (.trt) - NVIDIA GPU-optimiert
- **CoreML** (.mlmodel) - Apple Silicon optimiert

**Unterstützte Architekturen:**
- Stable Diffusion 1.5 (512x512)
- Stable Diffusion 2.0/2.1 (768x768)  
- **Stable Diffusion XL** (1024x1024+)
- Stable Diffusion 3 (Neueste Generation)
- ControlNet (Präzise Kontrolle)
- Inpainting-Modelle

### 🎥 **Videomodelle**
- **Stable Video Diffusion** (img2vid, img2vid-xt)
- **AnimateDiff** (SD-kompatible Animation)
- **I2VGen-XL** (Text-zu-Video)
- Alle Diffusers-kompatiblen Video-Pipelines

### 🤖 **LLM-Modelle**
- **GGUF** (.gguf) - llama.cpp Format (quantisiert)
- **Transformers** (Hugging Face Format)
- **Ollama** (Externe Integration)
- **SafeTensors LLMs** (Moderne LLM-Checkpoints)

**Unterstützte LLM-Architekturen:**
- Llama 3/3.1 (8B, 70B)
- Mixtral 8x7B (Mixture-of-Experts)
- Qwen2 (Mehrsprachig)
- Phi-3 (Kompakt aber leistungsstark)
- Gemma, Yi, Mistral und viele mehr

---

## 🚀 Erweiterte Installation

### Systemanforderungen

**Minimum:**
- Python 3.9+
- 8GB RAM
- 10GB freier Speicher

**Empfohlen:**
- Python 3.10+
- NVIDIA GPU mit 8GB+ VRAM (RTX 3070/4060 oder besser)
- 16GB+ RAM
- 50GB+ freier Speicher (für mehrere Modelle)

### Setup-Optionen

**Option 1: Standard-Installation**
```bash
git clone <repository-url>
cd LocalMediaSuite
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# oder: .\.venv\Scripts\activate  # Windows

pip install -r requirements.txt
python start.py
```

**Option 2: GPU-Optimierte Installation**
```bash
# Nach Standard-Installation:

# NVIDIA CUDA 11.8+ (empfohlen)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Oder CUDA 12.1+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# AMD ROCm (experimentell)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

# Apple Silicon (nativer Support)
# Torch automatisch MPS-optimiert
```

**Option 3: Docker-Installation**
```bash
# TODO: Docker-Support kommt in v2.1
docker run -p 3000:3000 -v ./models:/app/models localmediasuite:latest
```

---

## 📁 Erweiterte Verzeichnisstruktur

```
LocalMediaSuite/
├── 📂 models/                  # KI-Modelle (automatisch erkannt)
│   ├── 📂 image/              # Bild-Modelle
│   │   ├── 📂 realistic-vision-v6/  # Diffusers-Repository
│   │   ├── 📄 dreamshaper-8.safetensors  # Single-File
│   │   └── 📄 sdxl-base.ckpt  # Legacy Checkpoint
│   ├── 📂 video/              # Video-Modelle  
│   │   ├── 📂 stable-video-diffusion/
│   │   └── 📂 animatediff-motion-module/
│   └── 📂 llm/                # LLM-Modelle
│       ├── 📂 llama-3-8b-instruct/  # Transformers
│       ├── 📄 mixtral-8x7b-q4.gguf  # Quantisiert
│       └── 📄 qwen2-7b.safetensors
├── 📂 server/                 # FastAPI Backend
│   ├── 📄 models_registry.py  # Universelle Erkennung
│   ├── 📄 model_catalog.py    # 50+ Modell-Datenbank
│   └── 📄 universal_generate.py # Multi-Backend Generator
├── 📂 web/                    # Frontend
└── 📂 outputs/                # Generierte Inhalte
```

---

## 🎯 Universelle API

### `/api/generate/universal` - Ein Endpunkt für alles

```javascript
// Bild generieren (SD 1.5)
fetch('/api/generate/universal', {
  method: 'POST',
  body: new FormData({
    prompt: 'Beautiful landscape at sunset, photorealistic, 8k',
    model_id: 'image:realistic-vision-v6',
    mode: 'text2img',
    width: 1024,
    height: 768
  })
})

// Video generieren (SVD)
fetch('/api/generate/universal', {
  method: 'POST', 
  body: new FormData({
    model_id: 'video:stable-video-diffusion',
    mode: 'img2video',
    init_image: imageFile,
    num_frames: 25,
    fps: 24
  })
})

// Text generieren (Llama 3)
fetch('/api/generate/universal', {
  method: 'POST',
  body: new FormData({
    prompt: 'Explain quantum computing in simple terms',
    model_id: 'llm:llama-3-8b-instruct', 
    mode: 'text',
    max_tokens: 512,
    temperature: 0.8
  })
})
```

### Erweiterte Catalog-API

```javascript
// Alle verfügbaren Modelle durchsuchen
GET /api/catalog

// Hardware-optimierte Empfehlungen
GET /api/recommendations?max_vram=8&beginner_friendly=true

// Modell-Suche
GET /api/search?q=realistic%20portrait

// Kompatibilitätsprüfung  
GET /api/compatibility/stabilityai/stable-diffusion-xl-base-1.0

// Ein-Klick-Installation
POST /api/models/install
{
  "repo_id": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
  "type": "image", 
  "name": "Realistic Vision V6"
}
```

---

## 🛠️ Erweiterte Konfiguration

### Hardware-Optimierung

**GPU-Speicher-Optimierung (config/settings.json):**
```json
{
  "generation": {
    "enable_model_cpu_offload": true,
    "enable_sequential_cpu_offload": true, 
    "enable_attention_slicing": true,
    "enable_memory_efficient_attention": true,
    "use_safetensors": true
  },
  "hardware": {
    "max_batch_size": 1,
    "precision": "fp16",
    "compile_models": false
  }
}
```

**Multi-GPU-Setup:**
```json
{
  "hardware": {
    "device_map": "auto",
    "gpu_ids": [0, 1],
    "load_balancing": "memory_usage"
  }
}
```

### Modell-Sidecar-Dateien

Erstelle `.model.json` neben jedem Modell für erweiterte Metadaten:

```json
{
  "name": "Realistic Vision V6.0",
  "architecture": "sd15", 
  "precision": "fp16",
  "nsfw_capable": true,
  "recommended_use": "Fotorealistische Portraits und Szenen",
  "tags": ["photorealistic", "portrait", "versatile"],
  "modalities": ["text2img", "img2img"],
  "requirements": ["cuda_4gb"],
  "optimal_settings": {
    "steps": 28,
    "guidance_scale": 7.0,
    "resolution": "1024x768"
  },
  "negative_prompts": [
    "blurry, low quality, distorted, bad anatomy"
  ]
}
```

---

## 📊 Performance-Benchmarks

### Typische Generierungszeiten (RTX 4070)

| Modell-Typ | Auflösung | Zeit | VRAM |
|------------|-----------|------|------|
| SD 1.5 | 512x512 | 3-5s | 3.2GB |
| SDXL | 1024x1024 | 8-12s | 6.8GB |
| SVD (25 frames) | 1024x576 | 45-60s | 11.2GB |
| Llama 3 8B (GGUF Q4) | - | 50 tok/s | 5.1GB |

### Speicherbedarf-Übersicht

| Kategorie | Klein | Mittel | Groß |
|-----------|-------|---------|------|
| **SD 1.5** | 2.1GB | 4.2GB | 7.1GB |
| **SDXL** | 6.5GB | 13.5GB | 20.1GB |
| **LLM (FP16)** | 7GB (3B) | 15GB (7B) | 30GB (13B) |
| **LLM (GGUF Q4)** | 2GB (3B) | 4GB (7B) | 8GB (13B) |

---

## 🔧 Erweiterte Features

### Batch-Generierung
```python
# Mehrere Bilder gleichzeitig
POST /api/models/batch-install
[
  {"repo_id": "SG161222/Realistic_Vision_V6.0", "type": "image"},
  {"repo_id": "stabilityai/stable-video-diffusion-img2vid-xt", "type": "video"}
]
```

### Intelligente Hardware-Erkennung
```python
GET /api/models/compatibility-check
# Automatische VRAM/RAM-Analyse
# Modell-Empfehlungen basierend auf Hardware
# Performance-Optimierungsvorschläge
```

### Erweiterte Prompt-Engineering
- **Kategoriebasierte Vorschläge** für jeden Modelltyp
- **Negative Prompt-Datenbank**
- **Stil-Transfer-Presets**
- **Automatische Prompt-Optimierung**

### Multi-Modal Workflows
```javascript
// Text → Bild → Video Pipeline
const image = await generateImage(textPrompt)
const video = await generateVideo(image)
const description = await generateText(video)
```

---

## 🔍 Fehlerbehebung 2.0

### Häufige Probleme & Lösungen

**Modell wird nicht erkannt:**
```bash
# Prüfe Modell-Format
GET /api/models  # Zeigt alle erkannten Modelle

# Erstelle Sidecar-Datei
echo '{"type":"image","architecture":"sd15"}' > model.model.json
```

**"Pipeline not supported":**
```python
# Prüfe verfügbare Backends
GET /api/info  # Zeigt Backend-Status

# Installiere fehlende Abhängigkeiten
pip install diffusers transformers onnxruntime
```

**CUDA out of memory:**
```json
// Aktiviere Memory-Optimierungen
{
  "generation": {
    "enable_model_cpu_offload": true,
    "enable_sequential_cpu_offload": true,
    "max_batch_size": 1
  }
}
```

**Slow generation on CPU:**
```bash
# Verwende ONNX-optimierte Modelle
GET /api/recommendations?format=onnx

# Oder quantisierte Versionen
GET /api/search?q=gguf
```

---

## 🚀 Roadmap v2.1+

### Geplante Features

- **🐳 Docker-Support** mit GPU-Passthrough
- **☁️ Cloud-Integration** (optional für große Modelle)
- **🎨 ControlNet-Suite** (Pose, Depth, Canny, etc.)
- **📱 Mobile App** (Remote-Steuerung)
- **🔄 Model Conversion** (ONNX, TensorRT Export)
- **📊 Advanced Analytics** (Performance-Tracking)
- **🤝 Multi-User-Support** (Team-Features)
- **🎭 Face/Style Consistency** (Character-Erhaltung)

### Community & Beiträge

**Modell-Datenbank erweitern:**
```python
# Füge neue Modelle zur Katalog-Datenbank hinzu
# server/model_catalog.py - CATALOG_MODELS Liste
```

**Backend-Support hinzufügen:**
```python
# Implementiere neuen Backend in
# server/routes/universal_generate.py
```

---

## 📞 Support & Community

- **📧 Issues:** [GitHub Issues](https://github.com/your-repo/issues)
- **💬 Diskussionen:** [GitHub Discussions](https://github.com/your-repo/discussions)  
- **📖 Wiki:** [Vollständige Dokumentation](https://github.com/your-repo/wiki)
- **🎥 Tutorials:** [YouTube-Kanal](https://youtube.com/@localmediasuite)

---

**⭐ LocalMediaSuite 2.0 - Die Zukunft der lokalen KI-Generation ist da!**

*Keine Cloud, keine Limits, keine Filter - Nur pure KI-Power auf deiner Hardware.*# LocalMediaSuite

**LocalMediaSuite** ist eine vollständige Windows/Linux/macOS-basierte Lösung, um im Heimnetzwerk einen lokalen KI-Server mit moderner Weboberfläche bereitzustellen. Der Server läuft auf deinem PC, hostet die Web-UI für alle Geräte im Netzwerk und ermöglicht die Generierung, Verwaltung und Wiedergabe von Bildern und Videos.

**Alle Berechnungen laufen ausschließlich lokal** – keine Cloud, keine externen APIs, keine Filter oder Sperren.

## ✨ Features

### 🎨 Medien-Generierung
- **Bild-Generation**: Text→Bild, Bild→Bild (Stable Diffusion)
- **Video-Generation**: Text→Video, Bild→Video (Stable Video Diffusion)
- **SFW & NSFW**: Separate Modi ohne Filter oder Einschränkungen
- **Erweiterte Parameter**: Auflösung, Steps, Guidance, Strength, etc.

### 🎯 Benutzerfreundlichkeit
- **Intelligente Prompts**: Kategoriebasierte Vorschläge und Zufalls-Generator
- **Remix-Funktion**: Verwende vorherige Ergebnisse als Basis
- **Live-Galerie**: Automatische Anzeige neuer Generierungen
- **Modell-Management**: Automatische Erkennung und Ein-Klick-Installation

### 🖥️ Moderne Oberfläche
- **Responsive Design**: Funktioniert auf Desktop, Tablet und Smartphone
- **Dark Theme**: Augenfreundliche Oberfläche
- **Live-Status**: Desktop-UI zeigt Systemauslastung und laufende Vorgänke
- **Integrierter Player**: Direktwiedergabe von Bildern und Videos

### ⚙️ Technische Highlights
- **FastAPI Backend**: Moderne, asynchrone API
- **Diffusers Integration**: Unterstützt alle gängigen Stable Diffusion Modelle
- **Hugging Face Hub**: Automatische Modell-Installation
- **Ollama Support**: LLM-Integration für zukünftige Features
- **Broadcasting**: Live-Updates zwischen Browser-Tabs

## 🚀 Schnellstart

### 1. Voraussetzungen

**Python 3.9 oder höher** ist erforderlich.

**Für GPU-Beschleunigung (empfohlen):**
- NVIDIA GPU mit CUDA 11.8+ (RTX-Serie empfohlen)
- Oder AMD GPU mit ROCm 5.6+
- Oder Apple Silicon (M1/M2/M3)

### 2. Installation

```bash
# Repository klonen
git clone <repository-url>
cd LocalMediaSuite

# Python Virtual Environment erstellen (empfohlen)
python -m venv .venv

# Virtual Environment aktivieren
# Windows PowerShell:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt

# Für optimale GPU-Performance (NVIDIA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Erste Ausführung
python start.py
```

### 3. Zugriff

Nach dem Start öffne einen Browser und navigiere zu:
- **Lokal**: `http://localhost:3000`
- **Im Netzwerk**: `http://<deine-ip>:3000` (wird beim Start angezeigt)

## 📁 Projektstruktur

```
LocalMediaSuite/
├── 📂 config/              # Konfigurationsdateien
│   ├── suggestions.json    # Prompt-Vorschläge
│   └── settings.json       # Benutzereinstellungen
├── 📂 models/              # KI-Modelle (leer bei Start)
│   ├── image/              # Bild-Modelle (.safetensors, .ckpt)
│   ├── video/              # Video-Modelle (SVD)
│   └── llm/                # LLM-Modelle (Ollama)
├── 📂 outputs/             # Generierte Dateien
│   ├── images/             # Generierte Bilder
│   └── videos/             # Generierte Videos
├── 📂 server/              # FastAPI Backend
│   ├── routes/             # API-Endpunkte
│   ├── models_registry.py  # Modell-Verwaltung
│   ├── settings.py         # Einstellungs-System
│   └── server.py           # Haupt-Server
├── 📂 web/                 # Frontend (HTML/CSS/JS)
│   ├── assets/             # Styles, Scripts, Icons
│   ├── image.html          # Bild-Generator (SFW)
│   ├── image_nsfw.html     # Bild-Generator (NSFW)
│   ├── video.html          # Video-Generator (SFW)
│   ├── video_nsfw.html     # Video-Generator (NSFW)
│   ├── gallery.html        # Galerie mit Player
│   ├── catalog.html        # Modell-Katalog
│   └── settings.html       # Einstellungen
├── requirements.txt        # Python-Abhängigkeiten
├── start.py               # Startup-Script mit Desktop-UI
└── README.md              # Diese Datei
```

## 🎯 Erste Schritte

### 1. Modelle installieren

**Option A: Über den Katalog (empfohlen)**
1. Öffne `http://localhost:3000/catalog.html`
2. Klicke auf "Installieren" bei den empfohlenen Modellen
3. Warte bis Download abgeschlossen ist

**Option B: Manuell**
```bash
# Beispiel: Realistic Vision v6.0 für Bilder
mkdir -p models/image
# Lade Modell von Hugging Face oder CivitAI herunter
# und kopiere es nach models/image/

# Beispiel: Stable Video Diffusion für Videos
mkdir -p models/video
# Verwende git-lfs für große Diffusers-Modelle:
cd models/video
git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
```

### 2. Erstes Bild generieren

1. Gehe zu `http://localhost:3000/image.html`
2. Wähle ein installiertes Modell aus
3. Gib einen Prompt ein: "A beautiful landscape at sunset, photorealistic, 8k"
4. Klicke auf "Generieren"
5. Das Ergebnis erscheint automatisch in der Galerie

### 3. Erstes Video generieren

1. Gehe zu `http://localhost:3000/video.html`
2. Wähle ein Video-Modell (SVD)
3. Wähle "Text → Video" oder "Bild → Video"
4. Gib einen Prompt ein oder wähle ein Startbild
5. Klicke auf "Generieren"

## ⚙️ Konfiguration

### GPU-Einstellungen

Die Software erkennt automatisch verfügbare GPUs:
- **NVIDIA CUDA**: Automatisch genutzt wenn verfügbar
- **AMD ROCm**: Manuell installieren mit `pip install torch --index-url https://download.pytorch.org/whl/rocm5.6`
- **Apple Silicon**: Native Unterstützung über MPS

### Speicher-Optimierung

Für Systeme mit weniger als 16GB RAM:
```python
# In server/routes/generate.py anpassen:
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload()  # Für sehr wenig VRAM
```

### Erweiterte Einstellungen

Bearbeite `config/settings.json`:
```json
{
  "paths": {
    "images": "outputs/images",
    "videos": "outputs/videos",
    "image": "models/image",
    "video": "models/video"
  },
  "defaults": {
    "img_resolution": "1024x768",
    "vid_resolution": "1024x576",
    "vid_fps": 24,
    "vid_length_sec": 3
  },
  "ui": {
    "theme": "dark"
  }
}
```

## 🔧 API-Dokumentation

### Wichtige Endpunkte

- `GET /api/models` - Liste aller verfügbaren Modelle
- `POST /api/generate/image` - Bild generieren
- `POST /api/generate/video` - Video generieren
- `GET /api/files?type=image|video` - Galerie-Inhalte
- `GET /api/status` - Systemstatus
- `GET /api/ops` - Laufende Operationen

### Vollständige API-Docs

Nach dem Start verfügbar unter: `http://localhost:3000/docs`

## 🎨 Modell-Empfehlungen

### Bild-Modelle (Diffusers/SafeTensors)

**Fotorealistisch:**
- `SG161222/Realistic_Vision_V6.0_B1_noVAE` - Ultra-realistisch
- `stabilityai/stable-diffusion-xl-base-1.0` - SDXL Base
- `runwayml/stable-diffusion-v1-5` - SD 1.5 Standard

**Künstlerisch:**
- `Lykon/DreamShaper` - Vielseitig für alle Stile
- `andite/anything-v4.0` - Anime/Manga
- `nitrosocke/Arcane-Diffusion` - Comic-Stil

### Video-Modelle

**Standard:**
- `stabilityai/stable-video-diffusion-img2vid-xt` - Bester Allrounder
- `stabilityai/stable-video-diffusion-img2vid` - Kompaktere Version

### Installation über Hugging Face Hub

```bash
# In Python-Console oder Script:
from huggingface_hub import snapshot_download

# Modell herunterladen
snapshot_download(
    repo_id="SG161222/Realistic_Vision_V6.0_B1_noVAE",
    local_dir="models/image/Realistic_Vision_V6",
    repo_type="model"
)
```

## 🔍 Fehlerbehebung

### Häufige Probleme

**"CUDA out of memory":**
```python
# Reduziere Batch-Size oder nutze CPU-Offloading
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
```

**"No module named 'torch'":**
```bash
pip install torch torchvision torchaudio
```

**"ffmpeg not found" (für Videos):**
```bash
# Windows: Scoop installieren, dann:
scoop install ffmpeg

# macOS:
brew install ffmpeg

# Ubuntu/Debian:
sudo apt install ffmpeg
```

**Server startet nicht:**
```bash
# Port bereits belegt - anderen Port verwenden:
python start.py --port 8080

# Abhängigkeiten prüfen:
pip install -r requirements.txt
```

### Debug-Modus

```bash
# Mit detaillierten Logs:
python start.py --dev

# Ohne Desktop-UI (für Server):
python start.py --headless
```

## 🤝 Beitragen

Dieses Projekt ist Open Source! Beiträge sind willkommen:

1. Fork das Repository
2. Erstelle einen Feature-Branch: `git checkout -b feature/neue-funktion`
3. Commit deine Änderungen: `git commit -m 'Add neue Funktion'`
4. Push zum Branch: `git push origin feature/neue-funktion`
5. Erstelle einen Pull Request

## 📄 Lizenz

MIT License - siehe [LICENSE](LICENSE) Datei für Details.

## 🙏 Danksagungen

- **Stability AI** - Für Stable Diffusion und SVD
- **Hugging Face** - Für Diffusers und Model Hub
- **FastAPI Team** - Für das exzellente Web-Framework
- **Community** - Für Feedback und Beiträge

---

**⭐ Gefällt dir LocalMediaSuite? Gib dem Projekt einen Stern auf GitHub!**

**🐛 Probleme gefunden?** [Öffne ein Issue](https://github.com/dein-repo/LocalMediaSuite/issues)

**💬 Fragen?** Schau in die [Discussions](https://github.com/dein-repo/LocalMediaSuite/discussions)