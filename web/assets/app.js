// web/assets/app.js
// Toast + Fetch Wrapper + Form Helpers
(function(){
  const host = document.createElement('div');
  host.id = 'lms-toast-host';
  host.style.position = 'fixed';
  host.style.right = '16px';
  host.style.bottom = '16px';
  host.style.zIndex = '9999';
  document.body.appendChild(host);

  function showToast(msg, type='info', timeout=3800){
    const el = document.createElement('div');
    el.style.marginTop = '8px';
    el.style.padding = '10px 14px';
    el.style.borderRadius = '8px';
    el.style.fontSize = '14px';
    el.style.maxWidth = '420px';
    el.style.boxShadow = '0 6px 20px rgba(0,0,0,.35)';
    el.style.background = type==='error' ? '#8B0000' : (type==='success' ? '#1b5e20' : '#333');
    el.style.color = '#fff';
    el.textContent = msg;
    host.appendChild(el);
    setTimeout(()=>{ el.style.opacity='0'; el.style.transition='opacity .3s'; setTimeout(()=>el.remove(), 300); }, timeout);
  }

  window.LMS = window.LMS || {};
  window.LMS.toast = showToast;
})();

async function fetchJSON(url, opts={}){
  const r = await fetch(url, opts);
  let data = null;
  try { data = await r.json(); } catch {}
  if (!r.ok || (data && data.ok===false)) {
    const msg = (data && data.error && data.error.message) || `HTTP ${r.status}`;
    LMS.toast(msg, 'error');
    const err = new Error(msg);
    err.response = r;
    err.payload = data;
    throw err;
  }
  return data;
}
window.LMS = window.LMS || {};
window.LMS.fetchJSON = fetchJSON;

function attachModelToForm(form, pageKey){
  form.addEventListener('submit', async (e)=>{
    e.preventDefault();
    const fd = new FormData(form);
    const sel = (window.LMS && LMS.getSelectedModel) ? LMS.getSelectedModel(pageKey) : null;
    if (sel) fd.append('model_id', sel);

    try {
      const endpoint = form.getAttribute('action') || form.dataset.endpoint;
      const res = await fetchJSON(endpoint, { method: 'POST', body: fd });
      LMS.toast('Generierung gestartet', 'success');
      if (res.file && window.LMS && LMS.onGenerated) LMS.onGenerated(res.file);
      try {
        const bc = new BroadcastChannel('lms-events');
        bc.postMessage({ type: 'generated', file: res.file });
        bc.close();
      } catch {}
    } catch (err) {
      const errBox = form.querySelector('.form-error') || (()=> {
        const d = document.createElement('div');
        d.className = 'form-error';
        d.style.marginTop = '8px';
        d.style.color = '#ff9ea8';
        form.appendChild(d);
        return d;
      })();
      const msg = (err && err.payload && err.payload.error && err.payload.error.message) || err.message || 'Fehler';
      errBox.textContent = msg;
    }
  });
}
window.LMS = window.LMS || {};
window.LMS.attachModelToForm = attachModelToForm;
