
from __future__ import annotations
import threading, time
from typing import List, Dict, Any

class OpsStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._ops: Dict[str, Dict[str, Any]] = {}

    def start(self, op_id: str, *, kind: str, nsfw: bool, params: dict):
        with self._lock:
            self._ops[op_id] = {
                "id": op_id,
                "kind": kind,
                "nsfw": bool(nsfw),
                "params": params,
                "ts_start": time.time(),
                "ts_end": None,
                "status": "running",
                "progress": 0.0,
                "error": None
            }

    def update(self, op_id: str, *, progress: float | None = None, status: str | None = None):
        with self._lock:
            if op_id in self._ops:
                if progress is not None:
                    self._ops[op_id]["progress"] = float(progress)
                if status is not None:
                    self._ops[op_id]["status"] = status

    def finish(self, op_id: str, *, error: str | None = None):
        with self._lock:
            if op_id in self._ops:
                self._ops[op_id]["ts_end"] = time.time()
                self._ops[op_id]["status"] = "error" if error else "done"
                self._ops[op_id]["error"] = error

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._ops.values())

OPS = OpsStore()
