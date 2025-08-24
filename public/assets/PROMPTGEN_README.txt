
PromptGen Integration (non-destructive)
--------------------------------------
Füge am Ende JEDER Seite, die ein Prompt-Feld besitzt, genau eine Zeile ein:

  <script src="/assets/promptgen.js"></script>

Der Prompt-Generator erkennt die Seite automatisch (image, video, *_nsfw) anhand der URL.
Er mountet sich direkt ÜBER dem vorhandenen Prompt-Feld (#prompt oder name=prompt) und verändert sonst nichts.
Optional: <textarea id="negative_prompt" ...> wird für "Negative" automatisch befüllt.
