from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
PUBLIC_DIR = BASE_DIR / "public"
WEB_DIR = BASE_DIR / "web"

# Static folders
app.mount("/public", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
app.mount("/web", StaticFiles(directory=WEB_DIR, html=True), name="web")

# Root route – zeigt public/index.html beim Aufruf von "/"
@app.get("/")
async def root():
    return FileResponse(PUBLIC_DIR / "index.html")

# API route – für Stichwort-Generierung o.ä.
@app.post("/api/respond")
async def respond(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    category = data.get("category", "")
    
    # Dummy logic – hier kannst du KI-Model oder API anbinden
    return JSONResponse({
        "response": f"Prompt erhalten: '{prompt}' in Kategorie: '{category}'"
    })
