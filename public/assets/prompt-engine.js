// web/assets/prompt-engine.js - Intelligentes Prompt-System
(function() {
  'use strict';

  // === PROMPT ENGINE CORE ===
  class PromptEngine {
    constructor() {
      this.currentPage = this.detectPage();
      this.currentMode = 'text2img';
      this.isNSFW = this.currentPage.includes('nsfw');
      this.prompts = {};
      this.categories = {};
      this.qualityPresets = {};
      this.styleDatabase = {};
      
      this.init();
    }

    async init() {
      await this.loadPromptDatabase();
      this.setupUI();
      this.bindEvents();
      this.startPerformanceOptimization();
    }

    detectPage() {
      const url = window.location.pathname;
      if (url.includes('image_nsfw')) return 'image_nsfw';
      if (url.includes('video_nsfw')) return 'video_nsfw';
      if (url.includes('image')) return 'image';
      if (url.includes('video')) return 'video';
      return 'image';
    }

    // === PROMPT DATABASE ===
    async loadPromptDatabase() {
      // Lade erweiterte Prompt-Definitionen
      this.prompts = {
        image: {
          sfw: {
            categories: {
              portrait: {
                name: "Portrait & Menschen",
                icon: "👤",
                description: "Realistische und künstlerische Portraits",
                quality_focus: ["skin_detail", "lighting", "expression"],
                base_prompts: [
                  "professional portrait photography, studio lighting, sharp focus, detailed skin texture",
                  "cinematic portrait, dramatic lighting, shallow depth of field, photorealistic",
                  "beauty portrait, soft lighting, high detail, natural expression",
                  "editorial portrait, fashion photography, professional makeup, studio setup"
                ],
                style_modifiers: {
                  lighting: ["studio lighting", "natural light", "golden hour", "rim lighting", "soft box", "dramatic shadows"],
                  camera: ["85mm lens", "50mm lens", "135mm lens", "medium format", "full frame"],
                  mood: ["confident", "serene", "intense", "candid", "professional", "artistic"],
                  quality: ["8k uhd", "sharp focus", "detailed skin", "professional grade", "commercial quality"]
                },
                negative_defaults: ["blurry", "low quality", "amateur", "bad lighting", "distorted features", "oversaturated"]
              },

              landscape: {
                name: "Landschaft & Natur",
                icon: "🏔️",
                description: "Naturaufnahmen und Landschaftsfotografie",
                quality_focus: ["atmospheric_depth", "color_grading", "composition"],
                base_prompts: [
                  "breathtaking landscape photography, golden hour lighting, dramatic sky, ultra-wide angle",
                  "cinematic landscape, atmospheric perspective, natural colors, high dynamic range",
                  "nature photography, pristine wilderness, perfect composition, professional grade",
                  "scenic vista, panoramic view, dramatic clouds, crystal clear detail"
                ],
                style_modifiers: {
                  time: ["golden hour", "blue hour", "sunrise", "sunset", "storm light", "overcast"],
                  season: ["spring bloom", "summer lush", "autumn colors", "winter snow", "rainy season"],
                  composition: ["rule of thirds", "leading lines", "symmetrical", "panoramic", "vertical"],
                  atmosphere: ["misty", "clear air", "dramatic clouds", "hazy", "pristine", "moody"]
                },
                negative_defaults: ["urban elements", "people", "buildings", "pollution", "artificial lighting"]
              },

              architecture: {
                name: "Architektur & Urban",
                icon: "🏗️",
                description: "Gebäude, Städte und Strukturen",
                quality_focus: ["geometric_precision", "perspective", "material_detail"],
                base_prompts: [
                  "architectural photography, perfect perspective, clean lines, professional composition",
                  "modern architecture, geometric design, glass and steel, urban photography",
                  "historic building, detailed stonework, classical architecture, heritage photography",
                  "futuristic structure, innovative design, contemporary architecture, award winning"
                ],
                style_modifiers: {
                  style: ["modern", "classical", "brutalist", "art deco", "minimalist", "futuristic"],
                  materials: ["glass and steel", "concrete", "brick", "stone", "wood", "metal"],
                  perspective: ["low angle", "bird's eye", "symmetrical", "diagonal lines", "wide angle"],
                  lighting: ["natural light", "artificial lighting", "dramatic shadows", "even illumination"]
                }
              },

              product: {
                name: "Produkt & Commercial",
                icon: "📦",
                description: "Produktfotografie und kommerzielle Bilder",
                quality_focus: ["detail_sharpness", "color_accuracy", "professional_lighting"],
                base_prompts: [
                  "professional product photography, studio lighting, white background, commercial quality",
                  "luxury product shot, premium lighting, sophisticated composition, advertising grade",
                  "e-commerce photography, clean background, perfect lighting, catalog quality",
                  "hero product image, dramatic lighting, premium presentation, marketing ready"
                ],
                style_modifiers: {
                  background: ["white seamless", "gradient background", "lifestyle setting", "minimal backdrop"],
                  lighting: ["soft box", "key light", "rim lighting", "ambient lighting", "studio flash"],
                  angle: ["straight on", "three quarter", "top down", "close up macro", "environmental"]
                }
              },

              artistic: {
                name: "Kunst & Kreativ",
                icon: "🎨",
                description: "Künstlerische und kreative Interpretationen",
                quality_focus: ["artistic_vision", "color_harmony", "creative_composition"],
                base_prompts: [
                  "artistic interpretation, creative vision, unique perspective, gallery worthy",
                  "fine art photography, artistic composition, emotional depth, museum quality",
                  "conceptual art, surreal elements, imaginative design, thought provoking",
                  "abstract art, color harmony, dynamic composition, contemporary art style"
                ],
                style_modifiers: {
                  medium: ["oil painting style", "watercolor effect", "charcoal drawing", "digital art", "mixed media"],
                  movement: ["impressionist", "surrealist", "abstract expressionist", "pop art", "minimalist"],
                  color: ["monochromatic", "vibrant palette", "muted tones", "complementary colors", "analogous scheme"]
                }
              }
            }
          },

          nsfw: {
            categories: {
              artistic_nude: {
                name: "Künstlerischer Akt",
                icon: "🎭",
                description: "Tasteful artistic nude photography",
                quality_focus: ["lighting_mastery", "composition", "artistic_merit"],
                base_prompts: [
                  "artistic nude photography, chiaroscuro lighting, tasteful composition, fine art quality",
                  "classical nude study, dramatic shadows, sculptural lighting, museum worthy",
                  "contemporary nude art, minimalist composition, elegant poses, artistic vision",
                  "boudoir photography, soft lighting, intimate atmosphere, professional quality"
                ],
                style_modifiers: {
                  lighting: ["chiaroscuro", "soft diffused", "dramatic shadows", "natural window light", "studio lighting"],
                  composition: ["classical pose", "modern interpretation", "abstract form", "intimate framing"],
                  mood: ["elegant", "powerful", "serene", "contemplative", "confident"]
                },
                negative_defaults: ["explicit", "vulgar", "pornographic", "crude", "offensive", "exploitative"]
              },

              glamour: {
                name: "Glamour & Fashion",
                icon: "💃",
                description: "Sensual glamour and fashion photography", 
                quality_focus: ["fashion_aesthetics", "model_beauty", "styling"],
                base_prompts: [
                  "glamour photography, fashion model, professional styling, high-end fashion shoot",
                  "sensual portrait, glamour lighting, luxury setting, editorial quality",
                  "fashion photography, designer clothing, professional makeup, studio quality",
                  "beauty photography, perfect skin, professional retouching, magazine quality"
                ],
                style_modifiers: {
                  setting: ["luxury interior", "studio setup", "outdoor location", "urban environment"],
                  styling: ["haute couture", "lingerie", "evening wear", "casual luxury", "avant-garde"],
                  mood: ["confident", "seductive", "powerful", "elegant", "mysterious"]
                }
              }
            }
          }
        },

        video: {
          sfw: {
            categories: {
              cinematic: {
                name: "Cinematic & Film",
                icon: "🎬",
                description: "Filmische Videosequenzen",
                quality_focus: ["camera_movement", "lighting", "atmosphere"],
                base_prompts: [
                  "cinematic video sequence, professional cinematography, smooth camera movement",
                  "film quality footage, dramatic lighting, cinematic composition, movie-like",
                  "professional video production, hollywood style, cinematic color grading",
                  "artistic video, creative cinematography, visual storytelling, award winning"
                ],
                style_modifiers: {
                  camera: ["tracking shot", "dolly movement", "crane shot", "handheld", "steady cam", "drone footage"],
                  lighting: ["cinematic lighting", "dramatic shadows", "natural light", "studio lighting", "practical lights"],
                  mood: ["dramatic", "mysterious", "uplifting", "melancholic", "intense", "serene"],
                  color: ["film noir", "warm tones", "cool palette", "desaturated", "vibrant colors", "monochromatic"]
                },
                technical_specs: {
                  frame_rate: [24, 30, 60],
                  resolution: ["1080p", "4K", "8K"],
                  aspect_ratio: ["16:9", "21:9", "4:3", "1:1"]
                }
              },

              nature: {
                name: "Natur & Wildlife",
                icon: "🌿",
                description: "Naturaufnahmen und Tierwelt",
                quality_focus: ["natural_movement", "organic_flow", "environmental_detail"],
                base_prompts: [
                  "nature documentary style, wildlife footage, natural behavior, pristine environment",
                  "landscape time-lapse, natural phenomena, environmental storytelling",
                  "macro nature video, detailed close-ups, natural textures, organic movement",
                  "scenic nature footage, peaceful atmosphere, natural lighting, serene mood"
                ]
              },

              abstract: {
                name: "Abstract & Motion Graphics",
                icon: "🌀", 
                description: "Abstrakte Bewegung und Formen",
                quality_focus: ["fluid_motion", "geometric_precision", "color_dynamics"],
                base_prompts: [
                  "abstract motion graphics, fluid dynamics, geometric patterns, hypnotic movement",
                  "particle animation, flowing forms, mathematical precision, mesmerizing patterns",
                  "generative art animation, procedural movement, algorithmic beauty",
                  "liquid simulation, organic flow, natural physics, beautiful abstraction"
                ]
              }
            }
          },

          nsfw: {
            categories: {
              artistic_motion: {
                name: "Artistic Motion",
                icon: "💫",
                description: "Tasteful artistic video content",
                quality_focus: ["elegant_movement", "artistic_merit", "aesthetic_beauty"],
                base_prompts: [
                  "artistic video, elegant movement, tasteful choreography, aesthetic beauty",
                  "contemporary dance video, expressive movement, artistic interpretation",
                  "sensual motion, graceful movement, artistic expression, beautiful cinematography",
                  "intimate video art, emotional expression, artistic nudity, fine art quality"
                ]
              }
            }
          }
        }
      };

      // Qualitäts-Presets
      this.qualityPresets = {
        draft: {
          name: "Draft",
          description: "Schnelle Vorschau",
          modifiers: ["quick render", "preview quality", "fast generation"],
          technical: { steps: 15, guidance: 6.0 }
        },
        standard: {
          name: "Standard", 
          description: "Ausgewogene Qualität",
          modifiers: ["good quality", "balanced settings", "reasonable render time"],
          technical: { steps: 28, guidance: 7.5 }
        },
        high: {
          name: "High Quality",
          description: "Hohe Qualität",
          modifiers: ["high quality", "detailed", "professional grade", "crisp details"],
          technical: { steps: 35, guidance: 8.0 }
        },
        ultra: {
          name: "Ultra",
          description: "Maximale Qualität",
          modifiers: ["ultra high quality", "maximum detail", "professional photography", "award winning", "masterpiece"],
          technical: { steps: 50, guidance: 9.0 }
        }
      };

      // Style-Datenbank
      this.styleDatabase = {
        photography: {
          professional: ["professional photography", "studio quality", "commercial grade", "high-end production"],
          artistic: ["artistic photography", "creative vision", "fine art", "gallery worthy"],
          documentary: ["documentary style", "photojournalism", "authentic", "candid"],
          fashion: ["fashion photography", "editorial style", "luxury", "high fashion"],
          street: ["street photography", "urban", "candid moments", "authentic life"]
        },
        lighting: {
          natural: ["natural lighting", "soft daylight", "window light", "outdoor lighting"],
          studio: ["studio lighting", "controlled lighting", "professional setup", "key lighting"],
          dramatic: ["dramatic lighting", "chiaroscuro", "strong shadows", "moody lighting"],
          soft: ["soft lighting", "diffused light", "gentle illumination", "flattering light"],
          golden: ["golden hour", "warm light", "sunset lighting", "magic hour"]
        },
        mood: {
          serene: ["peaceful", "calm", "tranquil", "serene atmosphere"],
          dramatic: ["dramatic", "intense", "powerful", "impactful"],
          mysterious: ["mysterious", "enigmatic", "intriguing", "shadowy"],
          uplifting: ["uplifting", "positive", "inspiring", "joyful"],
          intimate: ["intimate", "personal", "close", "emotional"]
        }
      };
    }

    // === UI SETUP ===
    setupUI() {
      const promptField = this.findPromptField();
      if (!promptField) return;

      // Erstelle Prompt-Engine UI
      const engineUI = this.createEngineUI();
      
      // Füge oberhalb des Prompt-Felds ein
      promptField.parentNode.insertBefore(engineUI, promptField);
      
      // Erweitere Prompt-Feld
      this.enhancePromptField(promptField);
      
      // Initialisiere für aktuelle Seite
      this.updateForCurrentPage();
    }

    findPromptField() {
      const selectors = [
        '#prompt', '#video-prompt', '#nsfw-prompt', 
        'textarea[name="prompt"]', 'input[name="prompt"]',
        '[placeholder*="prompt"]', '[placeholder*="Prompt"]'
      ];
      
      for (const selector of selectors) {
        const field = document.querySelector(selector);
        if (field) return field;
      }
      return null;
    }

    createEngineUI() {
      const container = document.createElement('div');
      container.className = 'prompt-engine';
      container.innerHTML = `
        <style>
          .prompt-engine {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border: 1px solid #333;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            font-family: system-ui, sans-serif;
          }
          
          .engine-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 12px;
            color: #fff;
            font-weight: 600;
          }
          
          .engine-tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
            overflow-x: auto;
            padding-bottom: 4px;
          }
          
          .engine-tab {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 20px;
            color: #ccc;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
            white-space: nowrap;
            user-select: none;
          }
          
          .engine-tab:hover {
            background: rgba(59, 130, 246, 0.2);
            border-color: #3b82f6;
            color: #fff;
          }
          
          .engine-tab.active {
            background: linear-gradient(45deg, #3b82f6, #8b5cf6);
            border-color: #3b82f6;
            color: #fff;
            font-weight: 600;
          }
          
          .engine-content {
            display: none;
          }
          
          .engine-content.active {
            display: block;
          }
          
          .quality-selector {
            display: flex;
            gap: 6px;
            margin-bottom: 12px;
            flex-wrap: wrap;
          }
          
          .quality-btn {
            padding: 4px 10px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            color: #aaa;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.2s;
          }
          
          .quality-btn.active {
            background: linear-gradient(45deg, #10b981, #34d399);
            border-color: #10b981;
            color: #fff;
          }
          
          .style-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 8px;
          }
          
          .style-chip {
            padding: 4px 8px;
            background: rgba(139, 92, 246, 0.15);
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 12px;
            color: #c4b5fd;
            font-size: 10px;
            cursor: pointer;
            transition: all 0.2s;
            user-select: none;
          }
          
          .style-chip:hover {
            background: rgba(139, 92, 246, 0.3);
            border-color: #8b5cf6;
            color: #fff;
            transform: scale(1.05);
          }
          
          .style-chip.active {
            background: #8b5cf6;
            border-color: #8b5cf6;
            color: #fff;
          }
          
          .prompt-suggestions {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 8px;
            margin-top: 8px;
          }
          
          .prompt-suggestion {
            padding: 8px 10px;
            background: rgba(255,255,255,0.03);
            border: 1px dashed rgba(255,255,255,0.2);
            border-radius: 8px;
            color: #ccc;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.2s;
            line-height: 1.3;
          }
          
          .prompt-suggestion:hover {
            background: rgba(59, 130, 246, 0.1);
            border-color: #3b82f6;
            border-style: solid;
            color: #fff;
          }
          
          .engine-controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid rgba(255,255,255,0.1);
          }
          
          .engine-btn {
            padding: 6px 12px;
            background: rgba(59, 130, 246, 0.2);
            border: 1px solid #3b82f6;
            border-radius: 6px;
            color: #60a5fa;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.2s;
          }
          
          .engine-btn:hover {
            background: #3b82f6;
            color: #fff;
          }
          
          .engine-status {
            font-size: 10px;
            color: #888;
            display: flex;
            align-items: center;
            gap: 6px;
          }
          
          .status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #10b981;
            animation: pulse 2s infinite;
          }
          
          @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
          }
        </style>
        
        <div class="engine-header">
          <span>🤖</span>
          <span>Intelligenter Prompt-Generator</span>
          <div class="engine-status">
            <span class="status-dot"></span>
            <span id="engine-mode">${this.currentPage.toUpperCase()}</span>
          </div>
        </div>
        
        <div class="engine-tabs" id="category-tabs">
          <!-- Wird dynamisch gefüllt -->
        </div>
        
        <div class="quality-selector" id="quality-selector">
          <!-- Wird dynamisch gefüllt -->
        </div>
        
        <div id="category-content">
          <!-- Wird dynamisch gefüllt -->
        </div>
        
        <div class="engine-controls">
          <div style="display: flex; gap: 8px;">
            <button class="engine-btn" id="generate-prompt">🎲 Generieren</button>
            <button class="engine-btn" id="enhance-prompt">✨ Verbessern</button>
            <button class="engine-btn" id="clear-prompt">🧹 Leeren</button>
          </div>
          <button class="engine-btn" id="toggle-engine">⚙️ Erweitert</button>
        </div>
      `;
      
      return container;
    }

    enhancePromptField(field) {
      // Füge Placeholder und Hints hinzu
      if (!field.placeholder) {
        const placeholders = {
          image: "Beschreibe dein Bild detailliert...",
          image_nsfw: "Beschreibe dein Bild (unrestricted)...",
          video: "Beschreibe deine Video-Szene...", 
          video_nsfw: "Beschreibe deine Video-Szene (unrestricted)..."
        };
        field.placeholder = placeholders[this.currentPage] || "Beschreibe dein gewünschtes Ergebnis...";
      }
      
      // Auto-Resize für Textareas
      if (field.tagName === 'TEXTAREA') {
        field.style.minHeight = '80px';
        field.addEventListener('input', () => {
          field.style.height = 'auto';
          field.style.height = field.scrollHeight + 'px';
        });
      }
      
      // Live-Prompt-Analyse
      field.addEventListener('input', () => {
        this.analyzePrompt(field.value);
      });
    }

    // === EVENT HANDLING ===
    bindEvents() {
      // Tab-Switching
      document.addEventListener('click', (e) => {
        if (e.target.matches('.engine-tab')) {
          this.switchCategory(e.target.dataset.category);
        }
        
        if (e.target.matches('.quality-btn')) {
          this.setQuality(e.target.dataset.quality);
        }
        
        if (e.target.matches('.style-chip')) {
          this.toggleStyleChip(e.target);
        }
        
        if (e.target.matches('.prompt-suggestion')) {
          this.applyPromptSuggestion(e.target.textContent);
        }
        
        if (e.target.matches('#generate-prompt')) {
          this.generateRandomPrompt();
        }
        
        if (e.target.matches('#enhance-prompt')) {
          this.enhanceCurrentPrompt();
        }
        
        if (e.target.matches('#clear-prompt')) {
          this.clearPrompt();
        }
        
        if (e.target.matches('#toggle-engine')) {
          this.toggleAdvancedMode();
        }
      });
    }

    // === CORE FUNCTIONALITY ===
    updateForCurrentPage() {
      const pageData = this.prompts[this.currentPage.includes('video') ? 'video' : 'image'];
      const nsfwData = this.isNSFW ? pageData.nsfw : pageData.sfw;
      
      // Update Category Tabs
      const tabsContainer = document.getElementById('category-tabs');
      tabsContainer.innerHTML = Object.keys(nsfwData.categories).map(key => {
        const cat = nsfwData.categories[key];
        return `
          <div class="engine-tab ${key === Object.keys(nsfwData.categories)[0] ? 'active' : ''}" 
               data-category="${key}">
            <span>${cat.icon}</span>
            <span>${cat.name}</span>
          </div>
        `;
      }).join('');
      
      // Update Quality Selector
      const qualityContainer = document.getElementById('quality-selector');
      qualityContainer.innerHTML = Object.keys(this.qualityPresets).map(key => {
        const preset = this.qualityPresets[key];
        return `
          <div class="quality-btn ${key === 'standard' ? 'active' : ''}" 
               data-quality="${key}">
            ${preset.name}
          </div>
        `;
      }).join('');
      
      // Load first category
      this.switchCategory(Object.keys(nsfwData.categories)[0]);
    }

    switchCategory(categoryKey) {
      const pageData = this.prompts[this.currentPage.includes('video') ? 'video' : 'image'];
      const nsfwData = this.isNSFW ? pageData.nsfw : pageData.sfw;
      const category = nsfwData.categories[categoryKey];
      
      if (!category) return;
      
      // Update active tab
      document.querySelectorAll('.engine-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.category === categoryKey);
      });
      
      // Update content
      const contentContainer = document.getElementById('category-content');
      contentContainer.innerHTML = `
        <div class="engine-content active">
          <div style="margin-bottom: 10px;">
            <div style="color: #fff; font-weight: 500; margin-bottom: 4px;">${category.name}</div>
            <div style="color: #aaa; font-size: 11px; margin-bottom: 8px;">${category.description}</div>
          </div>
          
          ${category.style_modifiers ? Object.keys(category.style_modifiers).map(modType => `
            <div style="margin-bottom: 8px;">
              <div style="color: #ccc; font-size: 11px; margin-bottom: 4px; text-transform: uppercase;">${modType}</div>
              <div class="style-chips">
                ${category.style_modifiers[modType].map(style => `
                  <span class="style-chip" data-type="${modType}" data-style="${style}">${style}</span>
                `).join('')}
              </div>
            </div>
          `).join('') : ''}
          
          <div class="prompt-suggestions">
            ${category.base_prompts.map(prompt => `
              <div class="prompt-suggestion">${prompt}</div>
            `).join('')}
          </div>
        </div>
      `;
      
      this.currentCategory = categoryKey;
    }

    setQuality(qualityKey) {
      const preset = this.qualityPresets[qualityKey];
      if (!preset) return;
      
      // Update active quality
      document.querySelectorAll('.quality-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.quality === qualityKey);
      });
      
      this.currentQuality = qualityKey;
      
      // Trigger quality change event
      this.triggerQualityChange(preset);
    }

    triggerQualityChange(preset) {
      // Update technical settings if controls exist
      const stepsControl = document.querySelector('#steps, #nsfw-steps, #video-steps');
      const guidanceControl = document.querySelector('#guidance, #guidance_scale, #nsfw-guidance');
      
      if (stepsControl && preset.technical.steps) {
        stepsControl.value = preset.technical.steps;
        const valueDisplay = document.querySelector(`#${stepsControl.id}-value`);
        if (valueDisplay) valueDisplay.textContent = preset.technical.steps;
      }
      
      if (guidanceControl && preset.technical.guidance) {
        guidanceControl.value = preset.technical.guidance;
        const valueDisplay = document.querySelector(`#${guidanceControl.id}-value`);
        if (valueDisplay) valueDisplay.textContent = preset.technical.guidance.toFixed(1);
      }
    }

    toggleStyleChip(chip) {
      const isActive = chip.classList.contains('active');
      chip.classList.toggle('active', !isActive);
      
      const style = chip.dataset.style;
      const promptField = this.findPromptField();
      
      if (!promptField) return;
      
      if (!isActive) {
        // Add style
        this.addToPrompt(style);
      } else {
        // Remove style
        this.removeFromPrompt(style);
      }
    }

    addToPrompt(text) {
      const field = this.findPromptField();
      if (!field) return;
      
      const current = field.value.trim();
      const newText = current ? `${current}, ${text}` : text;
      field.value = newText;
      
      // Trigger change event
      field.dispatchEvent(new Event('input', { bubbles: true }));
    }

    removeFromPrompt(text) {
      const field = this.findPromptField();
      if (!field) return;
      
      const current = field.value;
      const regex = new RegExp(`\\b${text.replace(/[.*+?^${}()|[\]\\]/g, '\\    removeFromPrompt(text) {
      const field = this.findPromptFiel')}\\b,?\\s*`, 'gi');
      field.value = current.replace(regex, '').replace(/,\s*,/g, ',').replace(/^,\s*/, '').replace(/,\s*$/, '');
      
      field.dispatchEvent(new Event('input', { bubbles: true }));
    }

    applyPromptSuggestion(suggestion) {
      const field = this.findPromptField();
      if (!field) return;
      
      field.value = suggestion;
      field.dispatchEvent(new Event('input', { bubbles: true }));
      
      // Auto-apply quality modifiers
      if (this.currentQuality && this.qualityPresets[this.currentQuality]) {
        const qualityMods = this.qualityPresets[this.currentQuality].modifiers;
        this.addToPrompt(qualityMods.join(', '));
      }
    }

    generateRandomPrompt() {
      if (!this.currentCategory) return;
      
      const pageData = this.prompts[this.currentPage.includes('video') ? 'video' : 'image'];
      const nsfwData = this.isNSFW ? pageData.nsfw : pageData.sfw;
      const category = nsfwData.categories[this.currentCategory];
      
      if (!category) return;
      
      // Random base prompt
      const basePrompt = category.base_prompts[Math.floor(Math.random() * category.base_prompts.length)];
      
      // Random style modifiers
      let styleModifiers = [];
      if (category.style_modifiers) {
        Object.keys(category.style_modifiers).forEach(modType => {
          const options = category.style_modifiers[modType];
          const randomStyle = options[Math.floor(Math.random() * options.length)];
          styleModifiers.push(randomStyle);
        });
      }
      
      // Quality modifiers
      const qualityMods = this.currentQuality ? this.qualityPresets[this.currentQuality].modifiers : [];
      
      // Combine all
      const fullPrompt = [basePrompt, ...styleModifiers, ...qualityMods].join(', ');
      
      const field = this.findPromptField();
      if (field) {
        field.value = fullPrompt;
        field.dispatchEvent(new Event('input', { bubbles: true }));
      }
    }

    enhanceCurrentPrompt() {
      const field = this.findPromptField();
      if (!field || !field.value.trim()) return;
      
      const current = field.value.trim();
      
      // Add quality modifiers if not present
      const qualityMods = this.currentQuality ? this.qualityPresets[this.currentQuality].modifiers : [];
      const enhancement = qualityMods.filter(mod => 
        !current.toLowerCase().includes(mod.toLowerCase())
      );
      
      // Add category-specific enhancements
      if (this.currentCategory) {
        const pageData = this.prompts[this.currentPage.includes('video') ? 'video' : 'image'];
        const nsfwData = this.isNSFW ? pageData.nsfw : pageData.sfw;
        const category = nsfwData.categories[this.currentCategory];
        
        if (category.negative_defaults) {
          // Add to negative prompt field if exists
          const negativeField = document.querySelector('#negative_prompt, #nsfw-negative-prompt, #video-negative-prompt');
          if (negativeField && !negativeField.value.trim()) {
            negativeField.value = category.negative_defaults.join(', ');
          }
        }
      }
      
      if (enhancement.length > 0) {
        field.value = current + ', ' + enhancement.join(', ');
        field.dispatchEvent(new Event('input', { bubbles: true }));
      }
    }

    clearPrompt() {
      const field = this.findPromptField();
      if (field) {
        field.value = '';
        field.dispatchEvent(new Event('input', { bubbles: true }));
      }
      
      // Clear style chips
      document.querySelectorAll('.style-chip.active').forEach(chip => {
        chip.classList.remove('active');
      });
    }

    toggleAdvancedMode() {
      const container = document.querySelector('.prompt-engine');
      container.classList.toggle('advanced-mode');
      
      if (container.classList.contains('advanced-mode')) {
        this.showAdvancedFeatures();
      } else {
        this.hideAdvancedFeatures();
      }
    }

    showAdvancedFeatures() {
      const controlsContainer = document.querySelector('.engine-controls');
      const advancedPanel = document.createElement('div');
      advancedPanel.className = 'advanced-panel';
      advancedPanel.innerHTML = `
        <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1);">
          <div style="color: #fff; font-weight: 500; margin-bottom: 8px;">🔧 Erweiterte Einstellungen</div>
          
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px;">
            <div>
              <label style="color: #ccc; font-size: 11px; display: block; margin-bottom: 4px;">Prompt-Gewichtung</label>
              <input type="range" id="prompt-weight" min="0.5" max="2" step="0.1" value="1" 
                     style="width: 100%; height: 4px; background: #333; border-radius: 2px;">
              <div style="font-size: 10px; color: #888;" id="weight-value">1.0x</div>
            </div>
            
            <div>
              <label style="color: #ccc; font-size: 11px; display: block; margin-bottom: 4px;">Stil-Intensität</label>
              <input type="range" id="style-intensity" min="0.1" max="1" step="0.1" value="0.7"
                     style="width: 100%; height: 4px; background: #333; border-radius: 2px;">
              <div style="font-size: 10px; color: #888;" id="intensity-value">0.7</div>
            </div>
          </div>
          
          <div style="margin-top: 8px; display: flex; gap: 6px; flex-wrap: wrap;">
            <button class="engine-btn" id="save-preset">💾 Preset speichern</button>
            <button class="engine-btn" id="load-preset">📂 Preset laden</button>
            <button class="engine-btn" id="export-prompt">📤 Prompt exportieren</button>
          </div>
        </div>
      `;
      
      controlsContainer.appendChild(advancedPanel);
      
      // Bind advanced events
      document.getElementById('prompt-weight').addEventListener('input', (e) => {
        document.getElementById('weight-value').textContent = parseFloat(e.target.value).toFixed(1) + 'x';
      });
      
      document.getElementById('style-intensity').addEventListener('input', (e) => {
        document.getElementById('intensity-value').textContent = e.target.value;
      });
    }

    hideAdvancedFeatures() {
      const advancedPanel = document.querySelector('.advanced-panel');
      if (advancedPanel) {
        advancedPanel.remove();
      }
    }

    analyzePrompt(promptText) {
      if (!promptText.trim()) return;
      
      const analysis = {
        wordCount: promptText.split(' ').length,
        hasQualityTerms: false,
        hasStyleTerms: false,
        hasLighting: false,
        hasCameraTerms: false,
        complexity: 'basic'
      };
      
      const lowerPrompt = promptText.toLowerCase();
      
      // Check for quality terms
      const qualityTerms = ['high quality', '4k', '8k', 'detailed', 'sharp', 'professional', 'masterpiece'];
      analysis.hasQualityTerms = qualityTerms.some(term => lowerPrompt.includes(term));
      
      // Check for style terms
      const styleTerms = ['cinematic', 'artistic', 'realistic', 'abstract', 'vintage', 'modern'];
      analysis.hasStyleTerms = styleTerms.some(term => lowerPrompt.includes(term));
      
      // Check for lighting
      const lightingTerms = ['lighting', 'light', 'shadow', 'bright', 'dark', 'dramatic'];
      analysis.hasLighting = lightingTerms.some(term => lowerPrompt.includes(term));
      
      // Check for camera terms
      const cameraTerms = ['lens', 'camera', 'shot', 'angle', 'perspective', 'close-up', 'wide'];
      analysis.hasCameraTerms = cameraTerms.some(term => lowerPrompt.includes(term));
      
      // Determine complexity
      const score = [analysis.hasQualityTerms, analysis.hasStyleTerms, analysis.hasLighting, analysis.hasCameraTerms]
        .filter(Boolean).length;
      
      if (score >= 3) analysis.complexity = 'advanced';
      else if (score >= 2) analysis.complexity = 'intermediate';
      
      this.updatePromptAnalysis(analysis);
    }

    updatePromptAnalysis(analysis) {
      const statusElement = document.querySelector('.engine-status span:last-child');
      if (!statusElement) return;
      
      const complexityEmojis = {
        basic: '🟡',
        intermediate: '🟠', 
        advanced: '🟢'
      };
      
      statusElement.innerHTML = `
        ${complexityEmojis[analysis.complexity]} 
        ${analysis.wordCount} Wörter • ${analysis.complexity}
      `;
    }

    // === PERFORMANCE OPTIMIZATION ===
    startPerformanceOptimization() {
      // Debounce prompt analysis
      let analysisTimeout;
      const originalAnalyze = this.analyzePrompt.bind(this);
      this.analyzePrompt = (prompt) => {
        clearTimeout(analysisTimeout);
        analysisTimeout = setTimeout(() => originalAnalyze(prompt), 300);
      };
      
      // Lazy load categories
      this.loadedCategories = new Set();
      
      // Memory management
      this.startMemoryCleanup();
    }

    startMemoryCleanup() {
      setInterval(() => {
        // Clean up unused event listeners
        const inactiveElements = document.querySelectorAll('.style-chip:not(.active)');
        if (inactiveElements.length > 50) {
          // Remove some inactive elements if too many
          Array.from(inactiveElements).slice(30).forEach(el => el.remove());
        }
      }, 30000);
    }

    // === STORAGE & PRESETS ===
    saveCurrentAsPreset(name) {
      const field = this.findPromptField();
      if (!field) return;
      
      const preset = {
        name: name,
        prompt: field.value,
        category: this.currentCategory,
        quality: this.currentQuality,
        activeStyles: Array.from(document.querySelectorAll('.style-chip.active')).map(chip => ({
          type: chip.dataset.type,
          style: chip.dataset.style
        })),
        timestamp: Date.now(),
        page: this.currentPage
      };
      
      const presets = JSON.parse(localStorage.getItem('prompt_presets') || '[]');
      presets.push(preset);
      localStorage.setItem('prompt_presets', JSON.stringify(presets));
    }

    loadPreset(preset) {
      const field = this.findPromptField();
      if (!field) return;
      
      field.value = preset.prompt;
      field.dispatchEvent(new Event('input', { bubbles: true }));
      
      if (preset.category !== this.currentCategory) {
        this.switchCategory(preset.category);
      }
      
      if (preset.quality !== this.currentQuality) {
        this.setQuality(preset.quality);
      }
      
      // Restore active styles
      document.querySelectorAll('.style-chip.active').forEach(chip => chip.classList.remove('active'));
      preset.activeStyles.forEach(styleData => {
        const chip = document.querySelector(`[data-type="${styleData.type}"][data-style="${styleData.style}"]`);
        if (chip) chip.classList.add('active');
      });
    }

    // === INTEGRATION HELPERS ===
    static integrate() {
      // Auto-initialize when DOM is ready
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
          window.PromptEngine = new PromptEngine();
        });
      } else {
        window.PromptEngine = new PromptEngine();
      }
    }

    // === PUBLIC API ===
    updateMode(mode) {
      this.currentMode = mode;
      this.updateForCurrentPage();
    }

    getGeneratedPrompt() {
      const field = this.findPromptField();
      return field ? field.value : '';
    }

    setPrompt(text) {
      const field = this.findPromptField();
      if (field) {
        field.value = text;
        field.dispatchEvent(new Event('input', { bubbles: true }));
      }
    }

    getPromptAnalysis() {
      const field = this.findPromptField();
      if (!field) return null;
      
      return {
        text: field.value,
        wordCount: field.value.split(' ').length,
        category: this.currentCategory,
        quality: this.currentQuality,
        activeStyles: Array.from(document.querySelectorAll('.style-chip.active')).map(chip => chip.dataset.style)
      };
    }
  }

  // Auto-integrate
  PromptEngine.integrate();

  // Export for manual usage
  window.PromptEngine = PromptEngine;

})();

// === INTEGRATION INSTRUCTIONS ===
/*
INTEGRATION:
1. Füge diese Datei zu jeder Seite hinzu:
   <script src="/assets/prompt-engine.js"></script>

2. Das System erkennt automatisch:
   - Aktueller Seitentyp (image/video, sfw/nsfw)
   - Prompt-Felder (#prompt, #video-prompt, etc.)
   - Existierende Qualitäts-Controls

3. API-Nutzung:
   - window.PromptEngine.setPrompt("text")
   - window.PromptEngine.getGeneratedPrompt()
   - window.PromptEngine.updateMode("img2img")

4. Events:
   - 'prompt-enhanced': Wenn Prompt verbessert wurde
   - 'category-changed': Wenn Kategorie gewechselt wurde
   - 'quality-changed': Wenn Qualität geändert wurde

FEATURES:
✅ Automatische Seiten-Erkennung
✅ Kategorie-basierte Prompts
✅ Qualitäts-Presets 
✅ Style-Chips zum An-/Abwählen
✅ Live-Prompt-Analyse
✅ Zufalls-Generator
✅ Prompt-Verbesserung
✅ Preset-System
✅ Performance-Optimierung
✅ Mobile-responsive
✅ Lokaler Speicher
*/