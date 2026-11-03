"""Tiny local web app: drop a photo, tune it with live preview, print.

Standard library only (http.server). Listens on 127.0.0.1, so only this Mac can use it.
"""
from __future__ import annotations

import io
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import commands, imaging, jobs, printers

STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 60 * 1024 * 1024


class State:
    def __init__(self, printer_spec: str):
        self.printer_spec = printer_spec
        self.image = None              # grayscale working image
        self.image_name = ""
        self.lock = threading.Lock()   # one image in memory at a time
        self.print_lock = threading.Lock()  # one print job at a time


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ThermalPhoto/1.0"

        def log_message(self, fmt, *args):  # keep the terminal quiet except for errors
            if args and str(args[1]).startswith(("4", "5")):
                super().log_message(fmt, *args)

        # ----------------------------------------------------------- helpers
        def _send(self, code: int, body: bytes, ctype: str, headers: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _error(self, msg: str, code: int = 400, **extra):
            self._json({"error": msg, **extra}, code)

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_UPLOAD:
                raise ValueError("File too large (max 60 MB).")
            return self.rfile.read(n)

        def _settings(self) -> tuple[imaging.Settings, dict]:
            body = json.loads(self._body() or b"{}")
            return imaging.Settings.from_dict(body.get("settings")), body

        def _render(self, s: imaging.Settings, stage: str = "dither"):
            with state.lock:
                gray = state.image
            if gray is None:
                raise LookupError("Choose a photo first.")
            adjusted = imaging.adjust(gray, s)
            return adjusted if stage == "adjust" else imaging.dither(adjusted, s)

        # ----------------------------------------------------------- routes
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/info":
                return self._json({
                    "printer": state.printer_spec, "dithers": list(imaging.DITHERS),
                    "defaults": imaging.Settings().to_dict(),
                    "image": None if state.image is None else {
                        "name": state.image_name, "width": state.image.width, "height": state.image.height},
                })
            if self.path == "/api/status":
                return self._status()
            self._error("Not found", 404)

        def do_POST(self):
            try:
                if self.path.startswith("/api/image"):
                    data = self._body()
                    if not data:
                        return self._error("Empty upload.")
                    gray = imaging.load_image(data)
                    with state.lock:
                        state.image = gray
                        state.image_name = self.headers.get("X-Filename", "photo")
                    return self._json({"width": gray.width, "height": gray.height})

                if self.path == "/api/preview":
                    s, body = self._settings()
                    img = self._render(s, body.get("stage", "dither"))
                    buf = io.BytesIO()
                    img.save(buf, "PNG", optimize=False)
                    w_mm, h_mm = imaging.printed_size_mm(img)
                    return self._send(200, buf.getvalue(), "image/png", {
                        "X-Dots": f"{img.width}x{img.height}", "X-MM": f"{w_mm:.0f}x{h_mm:.0f}"})

                if self.path == "/api/print":
                    s, body = self._settings()
                    bitmap = self._render(s)
                    return self._run_job(lambda p: jobs.print_bitmap(p, bitmap, cut=body.get("cut", True)))
                if self.path == "/api/test":
                    s, _ = self._settings()
                    return self._run_job(lambda p: jobs.print_test_page(p, s.width))
                if self.path == "/api/cut":
                    return self._run_job(lambda p: jobs.feed_and_cut(p))
                if self.path == "/api/printer":
                    body = json.loads(self._body() or b"{}")
                    state.printer_spec = (body.get("spec") or "auto").strip()
                    return self._status()
                self._error("Not found", 404)
            except LookupError as e:
                self._error(str(e), 409)
            except (ValueError, OSError) as e:
                self._error(str(e))

        def _status(self):
            try:
                p = printers.open_printer(state.printer_spec)
                st = p.status()
                return self._json({"printer": p.name, "status": st,
                                   "problem": commands.blocking_problem(st),
                                   "warning": commands.status_warning(st)})
            except printers.PrinterError as e:
                return self._json({"printer": state.printer_spec, "error": str(e)})
            except Exception as e:  # e.g. a raw USB error while probing
                return self._json({"printer": state.printer_spec, "error": f"{type(e).__name__}: {e}"})

        def _run_job(self, fn):
            if not state.print_lock.acquire(blocking=False):
                return self._error("Already printing - wait for the current job to finish.", 409)
            try:
                job = fn(printers.open_printer(state.printer_spec))
                return self._json({"ok": True, "job": job})
            except printers.PrinterError as e:
                return self._error(str(e), 503, partial=e.partial)
            except Exception as e:  # never leave the browser hanging
                return self._error(f"{type(e).__name__}: {e}", 500, partial=True)
            finally:
                state.print_lock.release()

    return Handler


def serve(printer_spec: str = "auto", port: int = 8350, open_browser: bool = True,
          host: str = "127.0.0.1") -> None:
    state = State(printer_spec)
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"Thermal photo app running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
