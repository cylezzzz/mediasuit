// web/assets/promptgen.js
(async function() {
  const page = document.body.getAttribute("data-page") || "image";
  let suggestions = {};
  let active = { category: null, styles: [], modifiers: [], negatives: [] };

  async function loadSuggestions() {
    try {
      const res = await fetch(`/api/suggestions?page=${page}`);
      const json = await res.json();
      suggestions = json.data || {};
      renderCategories();
    } catch (e) {
      console.error("PromptGen load error", e);
    }
  }

  function renderCategories() {
    const wrap = document.getElementById("prompt-suggestions");
    if (!wrap) return;
    wrap.innerHTML = "";

    // Kategorien
    (suggestions.categories || []).forEach(cat => {
      const btn = document.createElement("button");
      btn.textContent = cat;
      btn.className = "chip";
      btn.onclick = () => {
        active.category = cat;
        active.styles = [];
        active.modifiers = [];
        renderStyles(cat);
        updatePrompt();
      };
      wrap.appendChild(btn);
    });
  }

  function renderStyles(cat) {
    const wrap = document.getElementById("prompt-suggestions");
    wrap.innerHTML = "";

    const back = document.createElement("button");
    back.textContent = "← Kategorien";
    back.className = "chip-back";
    back.onclick = () => { renderCategories(); };
    wrap.appendChild(back);

    // Styles zu Kategorie
    (suggestions.styles || []).forEach(style => {
      const btn = document.createElement("button");
      btn.textContent = style;
      btn.className = active.styles.includes(style) ? "chip active" : "chip";
      btn.onclick = () => {
        if (active.styles.includes(style)) {
          active.styles = active.styles.filter(s => s !== style);
        } else {
          active.styles.push(style);
        }
        renderStyles(cat);
        renderModifiers();
        updatePrompt();
      };
      wrap.appendChild(btn);
    });

    renderModifiers();
  }

  function renderModifiers() {
    const modWrap = document.createElement("div");
    modWrap.className = "modifiers";
    (suggestions.modifiers || []).forEach(mod => {
      const btn = document.createElement("button");
      btn.textContent = mod;
      btn.className = active.modifiers.includes(mod) ? "chip active" : "chip";
      btn.onclick = () => {
        if (active.modifiers.includes(mod)) {
          active.modifiers = active.modifiers.filter(m => m !== mod);
        } else {
          active.modifiers.push(mod);
        }
        updatePrompt();
        renderModifiers();
      };
      modWrap.appendChild(btn);
    });
    document.getElementById("prompt-suggestions").appendChild(modWrap);
  }

  function updatePrompt() {
    const field = document.querySelector("#prompt, textarea[name=prompt]");
    if (!field) return;
    let parts = [];
    if (active.category) parts.push(active.category);
    if (active.styles.length) parts.push(active.styles.join(", "));
    if (active.modifiers.length) parts.push(active.modifiers.join(", "));
    field.value = parts.join(", ");
  }

  // Initial
  await loadSuggestions();
})();
