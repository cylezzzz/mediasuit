// File: assets/js/nav.js
(() => {
  "use strict";

  // ---------- Helpers ----------
  const qs  = (s, r = document) => r.querySelector(s);
  const qsa = (s, r = document) => Array.from(r.querySelectorAll(s));

  // Ermittelt den "aktuellen" Dateinamen ohne Query/Hash, z.B. "image.html"
  function currentPage() {
    try {
      const url = new URL(location.href);
      const path = url.pathname.replace(/\\/g, "/");
      const name = path.split("/").filter(Boolean).pop() || "index.html";
      return name.toLowerCase();
    } catch {
      const path = location.pathname.replace(/\\/g, "/");
      const name = path.split("/").filter(Boolean).pop() || "index.html";
      return name.toLowerCase();
    }
  }

  // Normalisiert href eines Links zu einem Dateinamen
  function normalizeHref(href) {
    if (!href) return "";
    // Externe Links ignorieren
    if (/^https?:\/\//i.test(href) || href.startsWith("mailto:") || href.startsWith("#")) return "";
    // Query/Hash entfernen
    const clean = href.split("#")[0].split("?")[0];
    // Nur Dateiname
    const name = clean.split("/").filter(Boolean).pop() || clean;
    return name.toLowerCase();
  }

  // Robust localStorage-get/set (Safari Private Mode etc.)
  const store = (() => {
    try {
      const testKey = "__stitch_test__";
      localStorage.setItem(testKey, "1");
      localStorage.removeItem(testKey);
      return {
        get: (k, d = "") => localStorage.getItem(k) ?? d,
        set: (k, v) => localStorage.setItem(k, v),
      };
    } catch {
      const mem = new Map();
      return {
        get: (k, d = "") => (mem.has(k) ? mem.get(k) : d),
        set: (k, v) => mem.set(k, v),
      };
    }
  })();

  // ---------- Active Link Marking ----------
  function markActiveLinks() {
    const here = currentPage();
    // Alle bekannten "active"-Klassen zuerst entfernen (idempotent)
    qsa("nav a[href]").forEach(a => {
      a.classList.remove("bg-blue-600", "text-white", "rounded", "px-2", "font-semibold", "underline");
    });

    qsa("nav a[href]").forEach(link => {
      const href = normalizeHref(link.getAttribute("href"));
      if (!href) return;

      // Match: exakt gleiche Datei ODER location endet auf "/href"
      if (href === here) {
        link.classList.add("bg-blue-600", "text-white", "rounded", "px-2");
      }
    });
  }

  // ---------- Persist: Prompt + Kategorie ----------
  // Kompatibel zu deiner bestehenden ID-Konvention:
  //   #promptInput (textarea/input)
  //   #categorySelect (select)
  // Zusätzlich optional: beliebige Felder mit data-persist="key"
  function restorePersisted() {
    const promptInput   = qs("#promptInput");
    const categorySelect= qs("#categorySelect");

    if (promptInput)    promptInput.value    = store.get("lastPrompt", "");
    if (categorySelect) categorySelect.value = store.get("lastCategory", "");

    qsa("[data-persist]").forEach(el => {
      const key = el.getAttribute("data-persist");
      const val = store.get(key, "");
      if (val !== "") {
        if ("value" in el) el.value = val;
        else el.textContent = val;
      }
    });
  }

  function wirePersistence() {
    const promptInput   = qs("#promptInput");
    const categorySelect= qs("#categorySelect");

    // kleiner Debounce, damit nicht bei jedem Tastendruck geschrieben wird
    const debouncers = new Map();
    const debounced = (key, fn, delay = 150) => {
      clearTimeout(debouncers.get(key));
      const id = setTimeout(fn, delay);
      debouncers.set(key, id);
    };

    if (promptInput) {
      promptInput.addEventListener("input", () => {
        debounced("prompt", () => store.set("lastPrompt", promptInput.value));
      });
    }
    if (categorySelect) {
      categorySelect.addEventListener("change", () => {
        store.set("lastCategory", categorySelect.value);
      });
    }

    qsa("[data-persist]").forEach(el => {
      const key = el.getAttribute("data-persist");
      const ev  = el.tagName === "SELECT" ? "change" : "input";
      el.addEventListener(ev, () => {
        const val = "value" in el ? el.value : el.textContent;
        debounced("persist:"+key, () => store.set(key, val));
      });
    });
  }

  // ---------- Optional: Global Header Injection ----------
  // Wenn auf der Seite ein Platzhalter <div id="global-header"></div> existiert,
  // kann hier ein einheitlicher Header eingefügt werden (falls nicht serverseitig gerendert).
  function maybeInjectHeader() {
    const mount = qs("#global-header");
    if (!mount) return;

    const PAGES = [
      {href:"index.html",            label:"Home"},
      {href:"overview.html",         label:"Overview"},
      {href:"settings.html",         label:"Settings"},
      {href:"studio.html",           label:"Studio"},
      {href:"image.html",            label:"Image"},
      {href:"image_nsfw.html",       label:"Image NSFW"},
      {href:"video.html",            label:"Video"},
      {href:"video_nsfw.html",       label:"Video NSFW"},
      {href:"advanced_image.html",   label:"Advanced Image"},
      {href:"advanced_video.html",   label:"Advanced Video"},
      {href:"unified_generator.html",label:"Unified Generator"},
      {href:"gallery.html",          label:"Galerie"},
      {href:"catalog.html",          label:"Katalog"},
      {href:"character_lab.html",    label:"Charaktere"},
      {href:"create.html",           label:"Erstellen"},
      {href:"flirt.html",            label:"Flirt KI"},
    ];

    const here = currentPage();
    const linkHtml = PAGES.map(p=>{
      const isActive = p.href.toLowerCase() === here;
      const cls = isActive ? "font-semibold underline" : "text-blue-600 hover:underline";
      return `<a href="${p.href}" class="${cls}">${p.label}</a>`;
    }).join("");

    mount.outerHTML = `
<header class="w-full bg-white border-b px-6 py-4 flex flex-wrap gap-4 justify-between items-center">
  <a href="index.html" class="text-xl font-bold">Stitch Dashboard</a>
  <nav class="flex flex-wrap gap-x-4 gap-y-2 text-sm">
    ${linkHtml}
  </nav>
</header>`;
  }

  // ---------- Init ----------
  function init() {
    try {
      maybeInjectHeader();
      markActiveLinks();
      restorePersisted();
      wirePersistence();
    } catch (e) {
      console.error("Nav Init Error:", e);
    }
  }

  // Sofort ausführen, aber sicherstellen, dass DOM steht
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }

  // Exporte für Debug/Tests
  window.StitchNav = {
    _currentPage: currentPage,
    refreshActive: markActiveLinks,
  };
})();
