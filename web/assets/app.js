// web/assets/app.js - Vollständige LMS-Funktionen
(function(){
  // Toast-System
  const host = document.createElement('div');
  host.id = 'lms-toast-host';
  host.style.cssText = 'position:fixed;right:16px;bottom:16px;z-index:9999;pointer-events:none;';
  document.body.appendChild(host);

  function showToast(msg, type='info', timeout=3800){
    const el = document.createElement('div');
    el.style.cssText = `
      margin-top:8px;padding:12px 16px;border-radius:8px;font-size:14px;max-width:420px;
      box-shadow:0 8px 32px rgba(0,0,0,.4);backdrop-filter:blur(8px);
      background:${type==='error' ? '#dc2626' : (type==='success' ? '#16a34a' : '#1f2937')};
      color:#fff;opacity:0;transform:translateX(100%);transition:all .3s ease;
      pointer-events:auto;
    `;
    el.textContent = msg;
    host.appendChild(el);
    
    // Animate in
    requestAnimationFrame(() => {
      el.style.opacity = '1';
      el.style.transform = 'translateX(0)';
    });
    
    // Animate out
    setTimeout(() => {
      el.style.opacity = '0';
      el.style.transform = 'translateX(100%)';
      setTimeout(() => el.remove(), 300);
    }, timeout);
  }

  // Globale LMS-Funktionen
  window.LMS = window.LMS || {};
  window.LMS.toast = showToast;
})();

// Enhanced Fetch mit besserer Fehlerbehandlung
async function fetchJSON(url, opts = {}) {
  try {
    const r = await fetch(url, opts);
    let data = null;
    
    const contentType = r.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      data = await r.json();
    }
    
    if (!r.ok || (data && data.ok === false)) {
      const msg = (data && data.error && data.error.message) || `HTTP ${r.status}: ${r.statusText}`;
      LMS.toast(msg, 'error');
      const err = new Error(msg);
      err.response = r;
      err.payload = data;
      throw err;
    }
    
    return data;
  } catch (err) {
    if (!err.response) {
      LMS.toast('Netzwerkfehler - Server nicht erreichbar', 'error');
    }
    throw err;
  }
}

// Model Selection System
const MODEL_STORAGE_KEY = 'lms_selected_models';

function getSelectedModel(pageKey) {
  try {
    const stored = localStorage.getItem(MODEL_STORAGE_KEY);
    const data = stored ? JSON.parse(stored) : {};
    return data[pageKey] || null;
  } catch {
    return null;
  }
}

function setSelectedModel(pageKey, modelId) {
  try {
    const stored = localStorage.getItem(MODEL_STORAGE_KEY);
    const data = stored ? JSON.parse(stored) : {};
    data[pageKey] = modelId;
    localStorage.setItem(MODEL_STORAGE_KEY, JSON.stringify(data));
  } catch (e) {
    console.warn('Could not save model selection:', e);
  }
}

// Models API
async function loadModels(type, nsfw = false, purpose = null) {
  try {
    let url = `/api/models?type=${type}`;
    if (nsfw !== null) url += `&nsfw=${nsfw}`;
    
    const models = await fetchJSON(url);
    if (!Array.isArray(models)) return [];
    
    // Filter by purpose/modality if specified
    if (purpose) {
      return models.filter(m => {
        const modalities = m.modalities || [];
        if (purpose === 'text2img' || purpose === 'img2img') {
          return modalities.includes('text2img') || modalities.includes('img2img');
        }
        if (purpose === 'text2video' || purpose === 'img2video') {
          return modalities.includes('text2video') || modalities.includes('img2video');
        }
        return true;
      });
    }
    
    return models;
  } catch (e) {
    console.error('Failed to load models:', e);
    return [];
  }
}

// Model Tabs Renderer
function renderModelTabs({ container, models, onSelect, activeId, title = "Verfügbare Modelle" }) {
  if (!container) return;
  
  container.innerHTML = '';
  
  if (models.length === 0) {
    container.innerHTML = `
      <div class="p-4 text-center bg-yellow-900/20 border border-yellow-700 rounded-lg">
        <div class="text-yellow-400 font-medium mb-2">⚠️ Keine Modelle gefunden</div>
        <div class="text-sm text-gray-400">
          Bitte installiere Modelle über den <a href="/catalog.html" class="text-blue-400 underline">Katalog</a> 
          oder kopiere sie manuell in die entsprechenden Ordner.
        </div>
      </div>
    `;
    return;
  }
  
  const header = document.createElement('div');
  header.className = 'flex items-center justify-between mb-3';
  header.innerHTML = `
    <div class="font-semibold text-lg">${title}</div>
    <div class="text-sm text-gray-400">${models.length} gefunden</div>
  `;
  
  const tabsContainer = document.createElement('div');
  tabsContainer.className = 'flex flex-wrap gap-2';
  
  models.forEach((model, index) => {
    const isActive = model.id === activeId;
    const tab = document.createElement('button');
    tab.type = 'button';
    tab.className = `px-4 py-2 rounded-lg text-sm transition-all duration-200 ${
      isActive 
        ? 'bg-blue-600 text-white shadow-lg' 
        : 'bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white'
    }`;
    
    const sizeStr = model.size_bytes ? formatFileSize(model.size_bytes) : '';
    const nsfwBadge = model.nsfw_capable ? ' 🔞' : '';
    
    tab.innerHTML = `
      <div class="font-medium">${model.name}${nsfwBadge}</div>
      ${sizeStr ? `<div class="text-xs opacity-75">${sizeStr}</div>` : ''}
      ${model.recommended_use ? `<div class="text-xs opacity-75 max-w-48 truncate">${model.recommended_use}</div>` : ''}
    `;
    
    tab.addEventListener('click', () => {
      // Update UI
      tabsContainer.querySelectorAll('button').forEach(b => {
        b.className = b.className.replace(/bg-blue-600|text-white|shadow-lg/g, '').replace(/bg-gray-700/, 'bg-gray-800');
        if (!b.className.includes('bg-gray-800')) b.className += ' bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white';
      });
      tab.className = 'px-4 py-2 rounded-lg text-sm transition-all duration-200 bg-blue-600 text-white shadow-lg';
      
      // Notify callback
      if (onSelect) onSelect(model.id);
    });
    
    tabsContainer.appendChild(tab);
  });
  
  container.appendChild(header);
  container.appendChild(tabsContainer);
}

// File size formatter
function formatFileSize(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// Form Handler with Model Selection
function attachModelToForm(form, pageKey) {
  if (!form) return;
  
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn ? submitBtn.textContent : '';
    
    // Clear previous errors
    const errorBox = form.querySelector('.form-error');
    if (errorBox) errorBox.remove();
    
    try {
      // Disable submit button
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Generiere...';
      }
      
      const fd = new FormData(form);
      
      // Add selected model
      const selectedModel = getSelectedModel(pageKey);
      if (selectedModel) {
        fd.append('model_id', selectedModel);
      }
      
      const endpoint = form.getAttribute('action') || form.dataset.endpoint;
      if (!endpoint) throw new Error('Kein API-Endpunkt definiert');
      
      const res = await fetchJSON(endpoint, { 
        method: 'POST', 
        body: fd 
      });
      
      LMS.toast('Generierung erfolgreich gestartet!', 'success');
      
      // Handle result
      if (res.file && window.LMS && LMS.onGenerated) {
        LMS.onGenerated(res.file);
      }
      
      // Broadcast for gallery updates
      try {
        const bc = new BroadcastChannel('lms-events');
        bc.postMessage({ type: 'generated', file: res.file });
        bc.close();
      } catch {}
      
    } catch (err) {
      console.error('Generation failed:', err);
      
      // Show error in form
      const errDiv = document.createElement('div');
      errDiv.className = 'form-error mt-3 p-3 bg-red-900/20 border border-red-700 rounded-lg text-red-400 text-sm';
      const msg = (err?.payload?.error?.message) || err.message || 'Unbekannter Fehler';
      errDiv.textContent = `❌ ${msg}`;
      form.appendChild(errDiv);
      
    } finally {
      // Re-enable submit button
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
      }
    }
  });
}

// Install Model Function
async function installModel({ repo_id, type, name, subfolder }) {
  const response = await fetchJSON('/api/models/install', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ repo_id, type, name, subfolder })
  });
  return response;
}

// API Helper
async function apiCall(url, options = {}) {
  return fetchJSON(url, options);
}

// Assign to global LMS object
Object.assign(window.LMS, {
  fetchJSON,
  loadModels,
  getSelectedModel,
  setSelectedModel,
  renderModelTabs,
  attachModelToForm,
  installModel,
  api: apiCall,
  formatFileSize
});

// Enhanced CSS Classes (injected dynamically)
const style = document.createElement('style');
style.textContent = `
  .btn {
    @apply px-4 py-2 rounded-lg font-medium transition-all duration-200 border;
  }
  .btn.primary {
    @apply bg-blue-600 text-white border-blue-600 hover:bg-blue-700 shadow-lg hover:shadow-xl;
  }
  .btn.ghost {
    @apply bg-transparent text-gray-300 border-gray-600 hover:bg-gray-700 hover:text-white;
  }
  .btn:not(.primary):not(.ghost) {
    @apply bg-gray-800 text-gray-300 border-gray-700 hover:bg-gray-700 hover:text-white;
  }
  .chip {
    @apply px-3 py-1 text-sm rounded-full bg-gray-700 text-gray-300 hover:bg-blue-600 hover:text-white transition-colors cursor-pointer;
  }
  .input {
    @apply bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-gray-100 placeholder-gray-400 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-colors;
  }
  .label {
    @apply block text-sm font-medium text-gray-300 mb-1;
  }
  .card {
    @apply bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl p-4 shadow-lg;
  }
  .card:hover {
    @apply border-gray-600;
  }
`;
document.head.appendChild(style);

console.log('🚀 LMS App loaded successfully');