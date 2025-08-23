# start.py – LocalMediaSuite mit verpflichtender Desktop-UI
from __future__ import annotations
import threading
import time
import sys
import os

HOST = "0.0.0.0"
PORT = 3000
RELOAD = False  # bei Bedarf True setzen (Entwicklung)

def print_bind_addresses(port: int):
    # Robuste Ermittlung erreichbarer IPv4-URLs (ohne Zusatzlibs)
    urls = set()
    try:
        import socket
        # bevorzugt: UDP "connect" trick
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                urls.add(f"http://{ip}:{port}/")
        finally:
            s.close()
        # weitere Interfaces grob abfragen
        host = socket.gethostname()
        for info in socket.getaddrinfo(host, None):
            if info[0] == socket.AF_INET:
                ip = info[4][0]
                if ip and not ip.startswith("127."):
                    urls.add(f"http://{ip}:{port}/")
    except Exception:
        pass
    urls.add(f"http://127.0.0.1:{port}/")
    urls.add(f"http://localhost:{port}/")

    print("\n=== LocalMediaSuite erreichbar unter: ===")
    for u in sorted(urls):
        print(f"  -> {u}")
    print("========================================\n")


def run_server(host: str, port: int, reload: bool = False):
    # Startet Uvicorn synchron (blockierend) – wir rufen das in einem Thread auf.
    import uvicorn
    from uvicorn import Config, Server
    from server.server import build_app

    cfg = Config(app=build_app, host=host, port=port, reload=reload, log_level="info")
    server = Server(cfg)
    server.run()


def start_with_ui(host: str, port: int, reload: bool = False):
    """
    Startet den FastAPI-Server im Hintergrundthread und zeigt eine Desktop-UI (Tkinter),
    die dauerhaft Systemstatus & laufende Vorgänge pollt und darstellt.
    """
    print_bind_addresses(port)

    # Server als Daemon-Thread starten
    t = threading.Thread(target=run_server, args=(host, port, reload), daemon=True)
    t.start()

    # ---- Desktop-UI (Tkinter) ----
    import tkinter as tk
    from tkinter import ttk
    import urllib.request, json

    def fetch_json(url, timeout=2):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            return None

    root = tk.Tk()
    root.title("LocalMediaSuite – Server UI")
    root.geometry("760x480")

    # Header
    header = ttk.Frame(root, padding=8)
    header.pack(fill="x")
    ttk.Label(header, text="LocalMediaSuite", font=("Segoe UI", 14, "bold")).pack(side="left")
    ttk.Label(header, text=f" http://localhost:{port}", foreground="#4a8").pack(side="left", padx=8)

    # Systemstatus
    stat = ttk.LabelFrame(root, text="Systemstatus", padding=8)
    stat.pack(fill="x", padx=8, pady=4)
    lbl_os = ttk.Label(stat, text="OS: -"); lbl_os.pack(anchor="w")
    lbl_py = ttk.Label(stat, text="Python: -"); lbl_py.pack(anchor="w")
    lbl_cpu = ttk.Label(stat, text="CPU: -"); lbl_cpu.pack(anchor="w")
    lbl_ram = ttk.Label(stat, text="RAM: -"); lbl_ram.pack(anchor="w")

    # Aktuelle Vorgänge
    opsf = ttk.LabelFrame(root, text="Aktuelle Vorgänge", padding=8)
    opsf.pack(fill="both", expand=True, padx=8, pady=4)
    cols = ("id","kind","status","progress","since")
    tree = ttk.Treeview(opsf, columns=cols, show="headings", height=12)
    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, width=120 if c!="id" else 200, anchor="w")
    tree.pack(fill="both", expand=True)

    # Polling-Loop (1 Hz)
    def tick():
        st = fetch_json(f"http://localhost:{port}/api/status")
        if st and st.get("ok"):
            sysinfo = st.get("system", {})
            lbl_os.config(text=f"OS: {sysinfo.get('os','-')}")
            lbl_py.config(text=f"Python: {sysinfo.get('python','-')}")
            lbl_cpu.config(text=f"CPU: {sysinfo.get('cpu_percent','-')}%")
            lbl_ram.config(text=f"RAM: {sysinfo.get('ram_percent','-')}% ({sysinfo.get('total_ram_gb','-')} GB)")

        ops = fetch_json(f"http://localhost:{port}/api/ops")
        if ops and ops.get("ok"):
            tree.delete(*tree.get_children())
            now = time.time()
            for op in ops["ops"]:
                since = int(now - op.get("ts_start", now))
                tree.insert("", "end", values=(
                    op.get("id"),
                    op.get("kind"),
                    op.get("status"),
                    f"{op.get('progress',0):.2f}",
                    f"{since}s",
                ))

        root.after(1000, tick)

    root.after(800, tick)

    def on_close():
        # UI zu -> Prozess beendet; der Server-Thread ist daemon und folgt dem Prozessende.
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    # Immer UI starten – Status ist zentral.
    start_with_ui(HOST, PORT, RELOAD)
