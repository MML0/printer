"""Print jobs: photo printing with pre-flight checks, test page, job log."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from . import commands, imaging
from . import printers
from .printers import Printer, PrinterError

LOG_PATH = Path(__file__).resolve().parent.parent / "print-jobs.log"
DPI = 180
MAX_ROWS = 4000  # ~56 cm of paper; protects against printing a whole roll by mistake


def _log(entry: dict) -> None:
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def preflight(printer: Printer) -> dict | None:
    """Ask the printer for its status; raise if it is definitely not able to print."""
    st = printer.status()
    problem = commands.blocking_problem(st)
    if problem:
        raise PrinterError(problem)
    return st


def send_job(printer: Printer, data: bytes | list[tuple[bytes, float]], kind: str, extra: dict | None = None,
             check: bool = True) -> dict:
    """Send one job. check=False skips the status check (for when status reading misbehaves)."""
    job = {"id": uuid.uuid4().hex[:8], "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "kind": kind, "printer": printer.name, "bytes": printers.job_size(data), **(extra or {})}
    try:
        if check:
            job["status_before"] = preflight(printer)
            job["warning"] = commands.status_warning(job["status_before"])
        if not isinstance(data, (bytes, bytearray)) and not printer.paced:
            data = b"".join(seg for seg, _ in data)
        printer.send(data)
    except PrinterError as e:
        job["result"] = "unknown" if e.partial else "failed"
        job["error"] = str(e)
        _log(job)
        raise
    job["result"] = "sent"
    job["printer"] = printer.name  # USB fills in the exact device after opening it
    _log(job)
    return job


def print_bitmap(printer: Printer, bitmap: Image.Image, *, cut: bool = True,
                 band: int = commands.DEFAULT_BAND, max_rows: int = MAX_ROWS, check: bool = True,
                 settings: imaging.Settings | None = None) -> dict:
    """Print a 1-bit image; `settings` supplies the chunk/pause options (defaults if None)."""
    if bitmap.height > max_rows:
        raise PrinterError(f"Image is {bitmap.height} dots tall ({bitmap.height / DPI * 25.4:.0f} mm); "
                           f"the limit is {max_rows}. Use rotate=90 or a smaller width.")
    s = settings or imaging.Settings()
    extra = {"width": bitmap.width, "height": bitmap.height,
             "length_mm": round(bitmap.height / DPI * 25.4, 1)}
    if s.chunked:
        data = commands.image_segments(bitmap, cut=cut, band=band, budget=s.chunk,
                                       pause=s.pause_ms / 1000, look_back=s.hide_lines)
        extra["chunks"] = {"count": len(data), "chunk": s.chunk, "pause_ms": s.pause_ms,
                           "hide_lines": s.hide_lines, "pauses_s": round(sum(w for _, w in data), 1)}
    else:
        data = commands.image_job(bitmap, cut=cut, band=band)
        extra["chunks"] = None
    return send_job(printer, data, "photo", extra, check=check)


def print_photo(printer: Printer, src, settings: imaging.Settings, **kw) -> dict:
    gray = src if isinstance(src, Image.Image) else imaging.load_image(src)
    return print_bitmap(printer, imaging.render(gray, settings), settings=settings, **kw)


# --------------------------------------------------------------------------- test page

def _ruler(width: int) -> Image.Image:
    """Full-width bar with ticks every 8/64 dots: shows if the edges get clipped."""
    a = np.full((40, width), 255, dtype=np.uint8)
    a[0:6, :] = 0
    for x in range(0, width, 8):
        a[6:14 if x % 64 else 30, x] = 0
    a[6:40, 0:2] = 0
    a[6:40, width - 2:width] = 0
    return Image.fromarray(a).convert("1", dither=Image.NONE)


def _ramp(width: int, height: int = 56) -> Image.Image:
    row = np.linspace(0, 255, width)
    return Image.fromarray(np.tile(row, (height, 1)).astype(np.uint8), "L")


def test_page(width: int = 512) -> bytes:
    c = commands
    out = bytearray(c.INIT + c.ALIGN_CENTER + c.SIZE_DOUBLE + c.BOLD_ON)
    out += c.text("PHOTO PRINTER TEST\n") + c.SIZE_NORMAL + c.BOLD_OFF
    out += c.text(time.strftime("%Y-%m-%d %H:%M:%S") + f"\n{width} dots wide @ {DPI} dpi\n\n")
    out += c.text("Edge ruler - both thick end bars must be visible:\n")
    out += c.raster(_ruler(width)) + c.text("\n")
    ramp = _ramp(width)
    for name in imaging.DITHERS:
        s = imaging.Settings(width=width, dither=name, auto_contrast=False)
        out += c.text(f"{name}\n") + c.raster(imaging.dither(ramp, s)) + c.text("\n")
    out += c.text("If you can read this, text + images work.\n")
    out += c.feed_lines(4) + c.PARTIAL_CUT
    return bytes(out)


def print_test_page(printer: Printer, width: int = 512, check: bool = True) -> dict:
    return send_job(printer, test_page(width), "test-page", check=check)


def feed_and_cut(printer: Printer, cut: bool = True, check: bool = True) -> dict:
    data = commands.INIT + commands.feed_lines(4) + (commands.PARTIAL_CUT if cut else b"")
    return send_job(printer, data, "cut" if cut else "feed", check=check)
