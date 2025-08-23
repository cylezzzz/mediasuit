# server/model_catalog.py - Erweiterte Modell-Katalog-Datenbank
from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json

@dataclass
class CatalogModel:
    """Katalog-Modell-Definition"""
    id: str
    name: str
    type: str  # image, video, llm
    repo_id: str  # HuggingFace Repository ID
    architecture: str  # sd15, sdxl, svd, llama, etc.
    format: str  # diffusers_dir, safetensors, gguf, etc.
    precision: str  # fp16, fp32, q4, q8, etc.
    nsfw_capable: bool
    size_gb: float
    vram_gb: float  # Geschätzte VRAM-Anforderung
    recommended_use: str
    description: str
    tags: List[str]
    modalities: List[str]
    requirements: List[str]
    download_url: Optional[str] = None
    subfolder: Optional[str] = None
    license: str = "unknown"
    author: str = "unknown"
    created_date: str = "2024-01-01"
    popularity_score: int = 0  # 0-100
    quality_rating: float = 0.0  # 0.0-5.0
    verified: bool = False

# Erweiterte Modell-Katalog-Datenbank
CATALOG_MODELS: List[CatalogModel] = [
    
    # === BILD-MODELLE (STABLE DIFFUSION) ===
    
    # SD 1.5 Basis
    CatalogModel(
        id="sd15_base",
        name="Stable Diffusion v1.5",
        type="image",
        repo_id="runwayml/stable-diffusion-v1-5",
        architecture="sd15",
        format="diffusers_dir", 
        precision="fp16",
        nsfw_capable=True,
        size_gb=4.2,
        vram_gb=4.0,
        recommended_use="Universeller Allrounder für alle Bildstile",
        description="Das originale SD 1.5 Modell - stabil, gut dokumentiert und weit unterstützt. Perfekt als Basis für weitere Experimente.",
        tags=["stable-diffusion", "base-model", "versatile", "classic"],
        modalities=["text2img", "img2img"],
        requirements=["diffusers", "torch"],
        license="CreativeML Open RAIL-M",
        author="RunwayML",
        popularity_score=95,
        quality_rating=4.0,
        verified=True
    ),
    
    # Realistische Modelle
    CatalogModel(
        id="realistic_vision_v6",
        name="Realistic Vision V6.0 B1",
        type="image",
        repo_id="SG161222/Realistic_Vision_V6.0_B1_noVAE",
        architecture="sd15",
        format="diffusers_dir",
        precision="fp16", 
        nsfw_capable=True,
        size_gb=3.97,
        vram_gb=4.0,
        recommended_use="Ultra-realistische Portraits und Fotografien",
        description="Eines der besten Modelle für fotorealistische Bilder. Exzellent für Portraits, Architektur und alltägliche Szenen.",
        tags=["photorealistic", "portrait", "photography", "realistic", "high-quality"],
        modalities=["text2img", "img2img"],
        requirements=["diffusers", "torch"],
        license="CreativeML Open RAIL-M",
        author="SG161222",
        popularity_score=90,
        quality_rating=4.8,
        verified=True
    ),
    
    CatalogModel(
        id="dreamshaper_8",
        name="DreamShaper 8",
        type="image", 
        repo_id="Lykon/DreamShaper",
        architecture="sd15",
        format="safetensors",
        precision="fp16",
        nsfw_capable=True,
        size_gb=2.1,
        vram_gb=4.0,
        recommended_use="Vielseitig für Realismus und künstlerische Stile",
        description="Perfekte Balance zwischen Realismus und kreativer Flexibilität. Funktioniert gut mit verschiedenen Prompt-Stilen.",
        tags=["versatile", "artistic", "balanced", "creative"],
        modalities=["text2img", "img2img"],
        requirements=["diffusers", "torch"],
        license="CreativeML Open RAIL-M",
        author="Lykon",
        popularity_score=88,
        quality_rating=4.5,
        verified=True
    ),
    
    # SDXL Modelle
    CatalogModel(
        id="sdxl_base",
        name="Stable Diffusion XL Base 1.0",
        type="image",
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        architecture="sdxl",
        format="diffusers_dir",
        precision="fp16",
        nsfw_capable=True,
        size_gb=6.9,
        vram_gb=8.0,
        recommended_use="Hochauflösende Bilder mit exzellenten Details",
        description="SDXL Base für 1024x1024 Bilder mit verbesserter Detailqualität und Prompt-Verständnis.",
        tags=["sdxl", "high-resolution", "detailed", "base-model"],
        modalities=["text2img", "img2img"],
        requirements=["diffusers", "torch", "cuda_8gb"],
        license="OpenRAIL++",
        author="Stability AI",
        popularity_score=92,
        quality_rating=4.6,
        verified=True
    ),
    
    CatalogModel(
        id="juggernaut_xl_v9",
        name="Juggernaut XL v9",
        type="image",
        repo_id="RunDiffusion/Juggernaut-XL-v9",
        architecture="sdxl",
        format="safetensors",
        precision="fp16",
        nsfw_capable=True,
        size_gb=6.5,
        vram_gb=8.0,
        recommended_use="Cinematic und fotorealistische SDXL-Bilder",
        description="Premium SDXL-Modell mit kinematischer Qualität. Hervorragend für Portraits und dramatische Szenen.",
        tags=["sdxl", "cinematic", "premium", "photorealistic"],
        modalities=["text2img", "img2img"],
        requirements=["diffusers", "torch", "cuda_8gb"],
        license="CreativeML Open RAIL++-M",
        author="RunDiffusion",
        popularity_score=85,
        quality_rating=4.7,
        verified=True
    ),
    
    CatalogModel(
        id="realvis_xl_v4",
        name="RealVisXL V4.0",
        type="image",
        repo_id="SG161222/RealVisXL_V4.0",
        architecture="sdxl",
        format="safetensors",
        precision="fp16",
        nsfw_capable=True,
        size_gb=6.5,
        vram_gb=8.0,
        recommended_use="Ultra-realistische SDXL Fotografien",
        description="SDXL-Version des beliebten Realistic Vision. Beste Wahl für fotorealistische Hochauflösungsbilder.",
        tags=["sdxl", "photorealistic", "realistic", "portrait"],
        modalities=["text2img", "img2img"],
        requirements=["diffusers", "torch", "cuda_8gb"],
        license="CreativeML Open RAIL++-M",
        author="SG161222",
        popularity_score=87,
        quality_rating=4.8,
        verified=True
    ),
    
    # Anime/Kunst Modelle
    CatalogModel(
        id="anything_v5",
        name="Anything V5 (Ink)",
        type="image",
        repo_id="stablediffusionapi/anything-v5",
        architecture="sd15",
        format="safetensors",
        precision="fp16",
        nsfw_capable=True,
        size_gb=2.1,
        vram_gb=4.0,
        recommended_use="Anime, Manga und künstlerische Illustration",
        description="Spezialisiert auf Anime-Stil mit sauberen Linien und lebendigen Farben. Sehr beliebt in der Community.",
        tags=["anime", "manga", "artistic", "illustration", "colorful"],
        modalities=["text2img", "img2img"],
        requirements=["diffusers", "torch"],
        license="CreativeML Open RAIL-M",
        author="Linaqruf",
        popularity_score=89,
        quality_rating=4.4,
        verified=True
    ),
    
    CatalogModel(
        id="counterfeit_v3",
        name="Counterfeit V3.0",
        type="image",
        repo_id="gsdf/Counterfeit-V3.0",
        architecture="sd15",
        format="safetensors",
        precision="fp16",
        nsfw_capable=True,
        size_gb=2.1,
        vram_gb=4.0,
        recommended_use="Hochwertige Anime-Charaktere und Szenen",
        description="Eines der besten Anime-Modelle mit konsistenter Qualität und detailreichen Charakteren.",
        tags=["anime", "character", "detailed", "consistent"],
        modalities=["text2img", "img2img"],
        requirements=["diffusers", "torch"],
        license="CreativeML Open RAIL-M",
        author="rqdwdw",
        popularity_score=86,
        quality_rating=4.6,
        verified=True
    ),
    
    # Spezielle Stile
    CatalogModel(
        id="deliberate_v2",
        name="Deliberate V2",
        type="image",
        repo_id="XpucT/Deliberate",
        architecture="sd15",
        format="safetensors",
        precision="fp16",
        nsfw_capable=True,
        size_gb=4.27,
        vram_gb=4.0,
        recommended_use="Künstlerische und stilisierte Bilder",
        description="Hervorragend für künstlerische Interpretationen und stilisierte Portraits mit organischen Details.",
        tags=["artistic", "stylized", "organic", "creative"],
        modalities=["text2img", "img2img"],
        requirements=["diffusers", "torch"],
        license="CreativeML Open RAIL-M",
        author="XpucT",
        popularity_score=83,
        quality_rating=4.3,
        verified=True
    ),
    
    # === VIDEO-MODELLE ===
    
    CatalogModel(
        id="svd_img2vid_xt",
        name="Stable Video Diffusion (img2vid-xt)",
        type="video",
        repo_id="stabilityai/stable-video-diffusion-img2vid-xt",
        architecture="svd",
        format="diffusers_dir",
        precision="fp16",
        nsfw_capable=True,
        size_gb=9.56,
        vram_gb=12.0,
        recommended_use="Hochqualitative Videos aus Standbildern",
        description="State-of-the-art Modell für Bild-zu-Video-Generation. Erzeugt 4-Sekunden-Clips mit 1024x576 Auflösung.",
        tags=["svd", "img2video", "high-quality", "cinematic"],
        modalities=["img2video"],
        requirements=["diffusers", "torch", "cuda_12gb"],
        license="Stability AI Non-Commercial",
        author="Stability AI",
        popularity_score=95,
        quality_rating=4.8,
        verified=True
    ),
    
    CatalogModel(
        id="svd_img2vid",
        name="Stable Video Diffusion (img2vid)",
        type="video", 
        repo_id="stabilityai/stable-video-diffusion-img2vid",
        architecture="svd",
        format="diffusers_dir",
        precision="fp16",
        nsfw_capable=True,
        size_gb=9.56,
        vram_gb=10.0,
        recommended_use="Standard Video-Generation aus Bildern",
        description="Standard SVD-Modell für 576x1024 Videos. Etwas weniger VRAM-intensiv als die XT-Version.",
        tags=["svd", "img2video", "standard"],
        modalities=["img2video"],
        requirements=["diffusers", "torch", "cuda_10gb"],
        license="Stability AI Non-Commercial",
        author="Stability AI",
        popularity_score=90,
        quality_rating=4.5,
        verified=True
    ),
    
    CatalogModel(
        id="animatediff_motion_module",
        name="AnimateDiff Motion Module V3",
        type="video",
        repo_id="guoyww/animatediff",
        architecture="animatediff",
        format="safetensors",
        precision="fp16",
        nsfw_capable=True,
        size_gb=1.82,
        vram_gb=8.0,
        recommended_use="Animation von SD-Bildern mit Bewegung",
        description="Erweitert SD-Modelle um Animationsfähigkeiten. Kompatibel mit den meisten SD 1.5 Modellen.",
        tags=["animation", "motion", "sd15-compatible"],
        modalities=["text2video", "img2video"],
        requirements=["animatediff", "diffusers", "torch"],
        license="Apache 2.0",
        author="Yuwei Guo",
        popularity_score=78,
        quality_rating=4.2,
        verified=True
    ),
    
    # === LLM-MODELLE ===
    
    CatalogModel(
        id="llama3_8b_instruct",
        name="Llama 3 8B Instruct",
        type="llm",
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        architecture="llama",
        format="safetensors", 
        precision="fp16",
        nsfw_capable=True,
        size_gb=16.0,
        vram_gb=16.0,
        recommended_use="Hochintelligente Konversation und Aufgaben",
        description="Metas neuestes LLM mit hervorragender Reasoning-Fähigkeit. Perfekt für komplexe Aufgaben und Dialoge.",
        tags=["llama", "instruct", "conversation", "reasoning"],
        modalities=["chat", "instruct", "text"],
        requirements=["transformers", "torch"],
        license="Custom (Llama 3)",
        author="Meta",
        popularity_score=96,
        quality_rating=4.9,
        verified=True
    ),
    
    CatalogModel(
        id="mixtral_8x7b_instruct",
        name="Mixtral 8x7B Instruct",
        type="llm", 
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        architecture="mixtral",
        format="safetensors",
        precision="fp16",
        nsfw_capable=True,
        size_gb=90.0,
        vram_gb=48.0,
        recommended_use="Experten-Level Reasoning und Code-Generation",
        description="Mixture-of-Experts Modell mit 47B Parametern aber nur 13B aktiv. Hervorragend für komplexe Aufgaben.",
        tags=["mixtral", "moe", "expert", "coding", "reasoning"],
        modalities=["chat", "instruct", "code"],
        requirements=["transformers", "torch", "cuda_48gb"],
        license="Apache 2.0",
        author="Mistral AI",
        popularity_score=91,
        quality_rating=4.7,
        verified=True
    ),
    
    # Quantisierte LLM-Modelle (GGUF für llama.cpp)
    CatalogModel(
        id="llama3_8b_q4_k_m",
        name="Llama 3 8B Q4_K_M (GGUF)",
        type="llm",
        repo_id="bartowski/Meta-Llama-3-8B-Instruct-GGUF",
        architecture="llama",
        format="gguf",
        precision="q4",
        nsfw_capable=True,
        size_gb=4.6,
        vram_gb=6.0,
        recommended_use="Effiziente lokale LLM-Nutzung",
        description="Quantisierte Version von Llama 3 8B. Läuft effizient auf Consumer-Hardware mit nur 6GB VRAM.",
        tags=["llama", "quantized", "efficient", "gguf"],
        modalities=["chat", "instruct", "text"],
        requirements=["llama_cpp", "gguf"],
        subfolder="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        license="Custom (Llama 3)",
        author="bartowski",
        popularity_score=88,
        quality_rating=4.4,
        verified=True
    ),
    
    CatalogModel(
        id="qwen2_7b_instruct",
        name="Qwen2 7B Instruct",
        type="llm",
        repo_id="Qwen/Qwen2-7B-Instruct",
        architecture="qwen",
        format="safetensors",
        precision="fp16",
        nsfw_capable=True,
        size_gb=15.0,
        vram_gb=14.0,
        recommended_use="Mehrsprachiges LLM mit starker Reasoning",
        description="Alibabas neuestes LLM mit exzellenter Performance in mehreren Sprachen, besonders Chinesisch und Englisch.",
        tags=["qwen", "multilingual", "reasoning", "efficient"],
        modalities=["chat", "instruct", "multilingual"],
        requirements=["transformers", "torch"],
        license="Tongyi Qianwen",
        author="Alibaba",
        popularity_score=84,
        quality_rating=4.5,
        verified=True
    ),
    
    CatalogModel(
        id="phi3_mini_instruct",
        name="Phi-3 Mini 4K Instruct",
        type="llm",
        repo_id="microsoft/Phi-3-mini-4k-instruct",
        architecture="phi",
        format="safetensors",
        precision="fp16",
        nsfw_capable=True,
        size_gb=7.64,
        vram_gb=8.0,
        recommended_use="Kompakte aber leistungsstarke Konversation",
        description="Microsofts kompaktes aber überraschend fähiges 3.8B LLM. Perfekt für ressourcenbegrenzte Umgebungen.",
        tags=["phi", "compact", "efficient", "microsoft"],
        modalities=["chat", "instruct", "compact"],
        requirements=["transformers", "torch"],
        license="MIT",
        author="Microsoft",
        popularity_score=79,
        quality_rating=4.1,
        verified=True
    ),
    
    # === SPEZIAL-MODELLE ===
    
    CatalogModel(
        id="sd_2_inpainting",
        name="Stable Diffusion 2.0 Inpainting",
        type="image",
        repo_id="stabilityai/stable-diffusion-2-inpainting",
        architecture="sd20",
        format="diffusers_dir",
        precision="fp16",
        nsfw_capable=False,
        size_gb=5.21,
        vram_gb=6.0,
        recommended_use="Präzises Inpainting und Bildbearbeitung",
        description="Spezialisiertes SD 2.0 Modell für Inpainting-Aufgaben. Füllt maskierte Bereiche nahtlos aus.",
        tags=["inpainting", "editing", "specialized", "sd20"],
        modalities=["inpainting", "img2img"],
        requirements=["diffusers", "torch"],
        license="CreativeML Open RAIL++-M",
        author="Stability AI",
        popularity_score=75,
        quality_rating=4.2,
        verified=True
    ),
    
    # ControlNet-kompatible Modelle
    CatalogModel(
        id="openpose_controlnet",
        name="ControlNet OpenPose",
        type="image",
        repo_id="lllyasviel/control_v11p_sd15_openpose",
        architecture="controlnet",
        format="safetensors",
        precision="fp16",
        nsfw_capable=True,
        size_gb=1.45,
        vram_gb=4.0,
        recommended_use="Pose-kontrollierte Bildgeneration",
        description="ControlNet für präzise Pose-Kontrolle. Verwendet OpenPose-Skelett als Eingabe für exakte Körperhaltungen.",
        tags=["controlnet", "pose", "precision", "control"],
        modalities=["controlnet", "pose2img"],
        requirements=["controlnet", "diffusers", "torch"],
        license="Apache 2.0",
        author="Lvmin Zhang",
        popularity_score=82,
        quality_rating=4.4,
        verified=True
    ),
]

# Hilfsfunktionen für Katalog-Verwaltung
def get_models_by_type(model_type: str) -> List[CatalogModel]:
    """Filtere Modelle nach Typ"""
    return [model for model in CATALOG_MODELS if model.type == model_type]

def get_models_by_architecture(architecture: str) -> List[CatalogModel]:
    """Filtere Modelle nach Architektur"""
    return [model for model in CATALOG_MODELS if model.architecture == architecture]

def get_models_by_requirements(max_vram_gb: float = None, max_size_gb: float = None) -> List[CatalogModel]:
    """Filtere Modelle nach Hardware-Anforderungen"""
    filtered = CATALOG_MODELS
    
    if max_vram_gb is not None:
        filtered = [m for m in filtered if m.vram_gb <= max_vram_gb]
    
    if max_size_gb is not None:
        filtered = [m for m in filtered if m.size_gb <= max_size_gb]
    
    return filtered

def get_recommended_models(limit: int = 10) -> List[CatalogModel]:
    """Top-Modelle nach Beliebtheit und Qualität"""
    return sorted(CATALOG_MODELS, key=lambda m: m.popularity_score * m.quality_rating, reverse=True)[:limit]

def get_beginner_friendly_models() -> List[CatalogModel]:
    """Modelle die für Anfänger geeignet sind (weniger VRAM, einfacher)"""
    return [
        model for model in CATALOG_MODELS
        if model.vram_gb <= 8.0 and model.size_gb <= 10.0 and model.verified
    ]

def get_models_as_dict() -> List[Dict[str, Any]]:
    """Konvertiere alle Modelle zu Dictionary-Format für API"""
    return [asdict(model) for model in CATALOG_MODELS]

def search_models(query: str) -> List[CatalogModel]:
    """Suche Modelle nach Name, Tags oder Beschreibung"""
    query = query.lower()
    results = []
    
    for model in CATALOG_MODELS:
        # Suche in Name, Beschreibung, Tags
        searchable_text = f"{model.name} {model.description} {' '.join(model.tags)}".lower()
        
        if query in searchable_text:
            results.append(model)
    
    return results

# Export für API-Verwendung
def get_catalog_for_api() -> Dict[str, List[Dict[str, Any]]]:
    """Strukturiere Katalog für API-Verwendung"""
    return {
        "image": [asdict(m) for m in get_models_by_type("image")],
        "video": [asdict(m) for m in get_models_by_type("video")],
        "llm": [asdict(m) for m in get_models_by_type("llm")],
        "recommended": [asdict(m) for m in get_recommended_models(12)],
        "beginner": [asdict(m) for m in get_beginner_friendly_models()],
        "total_count": len(CATALOG_MODELS)
    }