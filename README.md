# 📦 LocalMediaSuite – Dokumentation

Dieses Projekt bietet eine lokale Suite zur Medien-Generierung (Bilder, Videos – SFW/NSFW), Verwaltung, Katalogisierung und Konfiguration.  
Alle Seiten sind modular aufgebaut und nutzen die zentrale API (`/api/generate/universal`, `/api/models`, `/api/files`, …).

---

## 🌐 Übersicht der HTML-Seiten (14)

### 1. `index.html`:contentReference[oaicite:0]{index=0}
- Zentrale Startseite („Alles in einer Seite“).
- Tabs für **Image, Video, Galerie, Katalog, Settings, Vorgänge**.
- Systemstatus (OS, CPU, RAM, Server online).
- Einheitliches Dashboard mit Live-Daten und Generierung direkt aus der Seite.

### 2. `image.html`:contentReference[oaicite:1]{index=1}
- **Bildgenerator (SFW)**.
- Modi: **Text→Bild** und **Bild→Bild**.
- Modell-Auswahl mit Vorschau.
- Parameter: Auflösung, Steps, Guidance, Seed, Format.
- Fortschrittsanzeige und Download.

### 3. `image_nsfw.html`:contentReference[oaicite:2]{index=2}
- **NSFW/Adult Bildgenerator (18+)**.
- Gleiche Funktionen wie `image.html`, aber **ohne Filter**.
- Klare **🔞 Warnungen** und NSFW-Modellauswahl.
- Output mit „Adult Content“-Badge.

### 4. `video.html`:contentReference[oaicite:3]{index=3}
- **Videogenerator (SFW)**.
- Modi: **Text→Video** und **Bild→Video**.
- Parameter: Auflösung, FPS, Dauer, Motion, Noise.
- Fortschrittsanzeige mit Frames/ETA.
- Preview-Panel mit generiertem Video.

### 5. `video_nsfw.html`:contentReference[oaicite:4]{index=4}
- **Videogenerator (NSFW / unrestricted)**.
- Keine Inhaltsbeschränkungen, inkl. **18+ Badge**.
- Erweiterte Parameter: Seed, Qualitätsstufen (Draft, Standard, Hoch, Ultra).
- Stil-Chips (z. B. Cineastisch, Sinnlich, Künstlerisch).
- Detaillierte Warnung & volle kreative Freiheit.

### 6. `advanced_image.html`:contentReference[oaicite:5]{index=5}
- **Erweiterter Bild/Video Generator**.
- Modi: Text→Bild, Bild→Bild, Text→Video, Bild→Video.
- Auswahl zwischen **SFW/NSFW** und Genre-Presets (Fantasy, Sci-Fi, Horror, Romantik).
- Flexible Eingabe mit Vorschau.
- Erweiterte Parameter (Resolution, Steps, CFG, Seed).

### 7. `gallery.html`:contentReference[oaicite:6]{index=6}
- **Galerie für Bilder und Videos**.
- Filter: nur Bilder, nur Videos, alle.
- Sortieroptionen (neueste, älteste, Name, Größe).
- Statistiken (Gesamt, Speicherplatz).
- Vollbild-Ansicht mit Navigation, Download und Löschen.

### 8. `catalog.html`:contentReference[oaicite:7]{index=7}
- **Modell-Katalog**.
- Installiere neue **Bild- und Video-Modelle**.
- Filter nach Typ (Image, Video, LLM).
- NSFW-Filter, Sortierungen.
- Ein-Klick-Empfehlungen, manuelle Installation möglich.
- Verwaltung: Details anzeigen, Modelle löschen.

### 9. `settings.html`:contentReference[oaicite:8]{index=8}
- **Einstellungen für Default-Werte**.
- Bild-Auflösung, Video-Auflösung, FPS, Videolänge.
- Theme-Umschaltung (Dark/Light).
- Speicherung über `/api/settings`.

### 10. `gallery.html` (zweiter Einstieg)  
- Gleiche Funktionalität wie Seite 7, nur andere Navigationseinbettung.

### 11. `catalog.html` (zweiter Einstieg)  
- Gleiche Funktionalität wie Seite 8.

### 12. `settings.html` (zweiter Einstieg)  
- Gleiche Funktionalität wie Seite 9.

### 13. `README.md`
- Dieses Dokument (Projektbeschreibung, Funktionsübersicht).

### 14. `server/routes/...`
- Serverseitige API-Routen für Universal-Generierung, Status, Modelle und Vorgänge.

---

## 🚀 Hauptfeatures
- **Bild- und Video-Generierung (SFW & NSFW)**.
- Einheitliche API (`/api/generate/universal`).
- **Modelle katalogisieren, installieren und verwalten**.
- **Galerie mit Player** für generierte Inhalte.
- **Settings** für Standardwerte & Theme.
- **Dashboard** mit Systemstatus und Vorgängen.

---

## ⚠️ Hinweise
- NSFW/Unrestricted-Seiten (`image_nsfw.html`, `video_nsfw.html`) sind ausschließlich für **18+ Nutzer** bestimmt.
- Nutzer sind für die Einhaltung lokaler Gesetze selbst verantwortlich.
