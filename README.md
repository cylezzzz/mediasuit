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
