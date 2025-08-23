# server/server.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles

# deine existierenden Router (falls Pfade anders: anpassen)
from .routes.models import router as models_router
from .routes.generate import router as generate_router
from .routes.settings import router as settings_router
from .routes.files import router as files_router
from .routes.status import router as status_router   # NEU
from .routes.ops import router as ops_router         # NEU

def build_app():
    app = FastAPI()

    # === API-Router ===
    app.include_router(models_router,  prefix="/api")
    app.include_router(generate_router, prefix="/api")
    app.include_router(settings_router, prefix="/api")
    app.include_router(files_router,    prefix="/api")
    app.include_router(status_router,   prefix="/api")  # /api/status
    app.include_router(ops_router,      prefix="/api")  # /api/ops

    # === Static: Outputs + Web-UI ===
    # /outputs -> outputs/  (damit Gallery URLs funktionieren)
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
    # Root-Web: / -> web/   (liefert index.html etc.)
    app.mount("/", StaticFiles(directory="web", html=True), name="web")

    # === Einheitliche Fehler-JSON ===
    @app.exception_handler(StarletteHTTPException)
    async def http_ex_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"ok": False, "error": {
                "code": exc.status_code,
                "message": str(exc.detail),
                "path": str(request.url),
            }},
        )

    @app.exception_handler(Exception)
    async def unhandled_ex(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": {
                "code": 500,
                "message": "Unerwarteter Fehler – bitte erneut versuchen.",
                "path": str(request.url),
                "details": str(exc),
            }},
        )

    return app
