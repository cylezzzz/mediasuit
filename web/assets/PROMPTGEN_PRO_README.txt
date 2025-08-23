
Datei: web/assets/promptgen.js
Einbau: Am Ende jeder Seite, die ein Prompt-Feld besitzt, genau eine Zeile einfügen:
  <script src="/assets/promptgen.js"></script>

Der Prompt-Generator montiert sich automatisch über das bestehende Prompt-Feld,
ohne bestehendes UI zu verändern. Presets/History werden pro Seite (image/video/...)
in localStorage gespeichert. Shift+Klick auf Chips = Lock/Unlock (Zufall ignoriert).
