# server/routes/status.py
from fastapi import APIRouter
import platform, psutil

router = APIRouter()

@router.get("/status")
async def get_status():
    vm = psutil.virtual_memory()
    return {
        "ok": True,
        "system": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "cpu_percent": psutil.cpu_percent(interval=0.0),
            "ram_percent": vm.percent,
            "total_ram_gb": round(vm.total / (1024**3), 1),
        }
    }
