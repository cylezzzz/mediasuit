# LocalMediaSuite

Windows-basiertes Server-Programm, das im Heimnetzwerk eine Web-UI bereitstellt.
Funktionen:
- Bildgenerierung (Text→Bild, Bild→Bild) in SFW & NSFW
- Videogenerierung (Text→Video, Bild→Video) in SFW & NSFW
- LLM-Auflistung über **Ollama** (keine Dateien in `models/llm/`)
- Galerie mit integriertem Medienplayer
- Katalog-Seite (Modelle/Agenten)
- Settings-Seite

## Schnellstart

```bash
# Optional: Python venv
python -m venv .venv
.\.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt

# Start
python start.py
```
Dann im Browser öffnen: `http://<deine-ip>:3000` (z. B. `http://192.168.0.10:3000`).

> Der Browser wird **nicht** automatisch geöffnet.

## Modelle

- `models/image/`  → Bildmodelle (.safetensors/.ckpt/.pt/.onnx)
- `models/video/`  → Videomodelle (.safetensors/.ckpt/.pt/.onnx)
- `models/llm/`    → **leer**; LLMs werden aus `ollama list` gelesen.

## Hinweis
Die Generatoren sind als **Demo-Implementierung** enthalten (Platzhalter):
- Image: erzeugt ein PNG mit Textoverlay.
- Video: erzeugt einen kurzen MP4-Clip mit Text.

Du kannst später echte Pipelines anbinden (Diffusers, comfyui, Stable Video Diffusion usw.).
# LocalMediaSuite

**LocalMediaSuite** ist eine vollständige Windows-basierte Lösung, um im Heimnetzwerk einen lokalen KI-Server mit Weboberfläche bereitzustellen.  
Der Server läuft auf deinem PC, hostet die Web-UI für alle Geräte im Netzwerk und ermöglicht die Generierung, Verwaltung und Wiedergabe von Medien.  
Alle Berechnungen laufen **ausschließlich lokal** – keine Cloud, keine externen APIs.  

---

## 🎯 Ziel

- **Eigener Heimserver** für Bild- und Videogenerierung (inkl. NSFW/Adult ohne Filter oder Sperren).  
- **Agenten & Modelle** verwalten, automatisch erkennen und passend zu den Funktionen einblenden.  
- **Web-UI** mit allen Seiten (Home, Image, Video, Gallery, Catalog, Settings) – aufrufbar über `http://<deine-ip>:3000/`.  
- **Desktop-UI** zur Serversteuerung und Live-Statusanzeige.  
- **Integrierter Mediaplayer** für Bilder und Videos auf allen relevanten Seiten.  

---

## 🖥️ Architektur

1. **Server (FastAPI + Uvicorn)**
   - Läuft lokal auf Port 3000
   - Liefert alle HTML-Seiten aus dem Ordner `/web/`
   - Liefert generierte Ergebnisse aus `/outputs/`
   - API-Routen für:
     - `/api/generate/...` → Bild/Video generieren
     - `/api/settings` → Einstellungen lesen/schreiben
     - `/api/models` → installierte Modelle & Agenten
     - `/api/files` → Galerie-Inhalte abrufen
     - `/api/status` → Systemstatus (CPU, RAM, Version)
     - `/api/ops` → Laufende Operationen / Fortschritt

2. **Desktop-UI**
   - Startet den Server
   - Zeigt erreichbare IPs und Ports
   - Live-Status: CPU, RAM, Auslastung, laufende Generierungen
   - Holt sich Änderungen von der Web-UI automatisch über `/api/settings`

3. **Web-UI (HTML, Tailwind, JS)**
   - **index.html** → Übersicht aller Seiten
   - **image.html / image_nsfw.html** → Text→Bild, Bild→Bild
   - **video.html / video_nsfw.html** → Text→Video, Bild→Video
   - **gallery.html** → zeigt alle Outputs, automatisch aktualisiert
   - **catalog.html** → Katalog für Modelle & Agenten (Installieren, Infos, Download)
   - **settings.html** → alle Standard-Einstellungen (Qualität, Länge, Auflösung, Theme etc.)
   - **assets/** → Styles, App-JS, Logo (ico/png/svg)

---

## ⚙️ Funktionen

### Medien-Generierung
- **Bild**:
  - Text → Bild
  - Bild → Bild (Remix mit Upload oder Galerie-Auswahl)
- **Video**:
  - Text → Video (Startframe automatisch)
  - Bild → Video (Startframe aus Upload oder Galerie)

### NSFW/Adult
- Keine Filter oder Sperren  
- NSFW-Seiten (`image_nsfw.html`, `video_nsfw.html`) haben identische Funktionen wie SFW, nur für andere Modelle/Kategorien

### Prompts & Kategorien
- Dynamisches Vorschlags-System:
  - Automatische Kategorien & Prompt-Vorschläge
  - Zufalls-Prompts
  - Remix-Funktion: Vorschläge werden passend zum gewählten Modus generiert
- Eigene Texteingabe bleibt jederzeit möglich

### Standard-Einstellungen
- **Bild**: Auflösung, Format (PNG/JPG), Qualität, Steps, Guidance, Strength (bei Remix)  
- **Video**: Auflösung, FPS, Länge, Frames  
- Alle Werte über **Settings-Seite** dauerhaft speicherbar

### Outputs & Galerie
- Ergebnisse werden in `/outputs/images` oder `/outputs/videos` gespeichert  
- **Gallery-Seite** zeigt neue Inhalte automatisch an (Polling + Broadcast)  
- Integrierter Player (Bild/Video startet direkt)

### Agenten & Modelle
- Gespeichert in `/models/image`, `/models/video`, `/models/llm`  
- Automatische Erkennung: nur passende Modelle werden pro Seite angezeigt  
- **Catalog-Seite**:
  - zeigt verfügbare lokale und mögliche neue Modelle
  - Download & Installation in die richtigen Ordner
  - Jedes Modell mit Beschreibung („geeignet für Portrait, NSFW, Landschaft…“)

---

## 📡 API Überblick

- `GET /api/status` → Systeminfos  
- `GET /api/ops` → Laufende Tasks mit Fortschritt  
- `GET /api/settings` → Aktuelle Settings  
- `PUT /api/settings` → Änderungen speichern  
- `GET /api/models` → alle erkannten Modelle/Agenten  
- `GET /api/files?type=image|video` → Galerie-Files  
- `POST /api/generate/image` → Bild generieren  
- `POST /api/generate/video` → Video generieren  

Alle Antworten im JSON-Format, Fehler standardisiert als:
```json
{
  "ok": false,
  "error": {
    "code": 400,
    "message": "Fehlermeldung",
    "path": "/api/generate/image"
  }
}
