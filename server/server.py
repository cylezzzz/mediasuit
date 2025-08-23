# server/server.py - Vollständig erweitert um alle neuen Features
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles

# Alle Router importieren
from .routes.models import router as models_router
from .routes.generate import router as generate_router
from .routes.settings import router as settings_router
from .routes.files import router as files_router
from .routes.status import router as status_router
from .routes.ops import router as ops_router
from .routes.suggestions import router as suggestions_router
from .routes.recommend import router as recommend_router
from .routes.universal_generate import router as universal_router  # NEU

def build_app():
    app = FastAPI(
        title="LocalMediaSuite", 
        version="2.0.0",
        description="Universal AI Media Generation Suite with Multi-Backend Support"
    )

    # === API-Router ===
    app.include_router(models_router,      prefix="/api", tags=["Models"])
    app.include_router(generate_router,    prefix="/api", tags=["Generation - Legacy"])
    app.include_router(universal_router,   prefix="/api", tags=["Generation - Universal"])  # NEU
    app.include_router(settings_router,    prefix="/api", tags=["Settings"])
    app.include_router(files_router,       prefix="/api", tags=["Files"])
    app.include_router(status_router,      prefix="/api", tags=["Status"])
    app.include_router(ops_router,         prefix="/api", tags=["Operations"])
    app.include_router(suggestions_router, prefix="/api", tags=["Suggestions"])
    app.include_router(recommend_router,   prefix="/api", tags=["Recommendations & Catalog"])

    # === Static: Outputs + Web-UI ===
    # /outputs -> outputs/ (damit Gallery URLs funktionieren)
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
    # Root-Web: / -> web/ (liefert index.html etc.)
    app.mount("/", StaticFiles(directory="web", html=True), name="web")

    # === Einheitliche Fehler-JSON ===
    @app.exception_handler(StarletteHTTPException)
    async def http_ex_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"ok": False, "error": {
                "code": exc.status_code,
                "message": str(exc.detail),
                "path": str(request.url.path),
            }},
        )

    @app.exception_handler(Exception)
    async def unhandled_ex(request: Request, exc: Exception):
        import traceback
        print(f"Unhandled exception: {exc}")
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": {
                "code": 500,
                "message": "Interner Serverfehler - Details in Logs",
                "path": str(request.url.path),
                "details": str(exc) if app.debug else "Debug-Modus deaktiviert",
            }},
        )

    # === Health Check ===
    @app.get("/api/health")
    async def health_check():
        return {"ok": True, "status": "healthy", "service": "LocalMediaSuite", "version": "2.0.0"}

    # === System Info Endpoint ===
    @app.get("/api/info")
    async def system_info():
        """Erweiterte Systeminfos mit Backend-Status"""
        import platform
        
        # Backend Status
        backends = {}
        try:
            import torch
            backends["torch"] = {
                "available": True,
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        except ImportError:
            backends["torch"] = {"available": False}
        
        try:
            import diffusers
            backends["diffusers"] = {
                "available": True,
                "version": diffusers.__version__
            }
        except ImportError:
            backends["diffusers"] = {"available": False}
        
        try:
            import transformers
            backends["transformers"] = {
                "available": True,
                "version": transformers.__version__
            }
        except ImportError:
            backends["transformers"] = {"available": False}
        
        try:
            from llama_cpp import Llama
            backends["llama_cpp"] = {"available": True}
        except ImportError:
            backends["llama_cpp"] = {"available": False}
        
        try:
            import onnxruntime
            backends["onnxruntime"] = {
                "available": True,
                "version": onnxruntime.__version__,
                "providers": onnxruntime.get_available_providers()
            }
        except ImportError:
            backends["onnxruntime"] = {"available": False}

        return {
            "ok": True,
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "architecture": platform.machine()
            },
            "backends": backends,
            "features": {
                "universal_generation": True,
                "multi_backend_support": True,
                "auto_model_detection": True,
                "catalog_installation": True,
                "nsfw_support": True
            }
        }

    return app