// File: assets/js/nav.js
(() => {
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
    {href:"flirt.html",            label:"Flirt KI"}
  ];

  const here = (location.pathname.split('/').pop() || "index.html").toLowerCase();

  const linkHtml = PAGES.map(p => {
    const active = here === p.href.toLowerCase();
    const cls = active ? "font-semibold underline" : "text-blue-600 hover:underline";
    return `<a href="${p.href}" class="${cls}">${p.label}</a>`;
  }).join("");

  const html = `
<header class="w-full bg-white border-b px-6 py-4 flex flex-wrap gap-4 justify-between items-center">
  <a href="index.html" class="text-xl font-bold">Stitch Dashboard</a>
  <nav class="flex flex-wrap gap-x-4 gap-y-2 text-sm">
    ${linkHtml}
  </nav>
</header>`;

  const mount = document.getElementById("global-header");
  if (mount) mount.outerHTML = html;
})();
