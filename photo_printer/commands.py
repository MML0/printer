"""ESC/POS byte builders for the BIXOLON SRP-350III (standard ESC/POS subset).

Only well-established commands are used:
    ESC @        initialise printer
    ESC a n      alignment
    ESC E n      bold
    GS ! n       character size
    ESC d n      print and feed n lines
    GS v 0       raster bit image (the photo)
    GS V 1       partial cut
    DLE EOT n    real-time status (n = 1..4)
"""
from __future__ import annotations

import numpy as np
from PIL import Image

ESC, GS, DLE = b"\x1b", b"\x1d", b"\x10"

INIT = ESC + b"@"
ALIGN_LEFT, ALIGN_CENTER, ALIGN_RIGHT = ESC + b"a\x00", ESC + b"a\x01", ESC + b"a\x02"
BOLD_ON, BOLD_OFF = ESC + b"E\x01", ESC + b"E\x00"
SIZE_NORMAL, SIZE_DOUBLE = GS + b"!\x00", GS + b"!\x11"
PARTIAL_CUT = GS + b"V\x01"

# A single GS v 0 command can describe a huge image, but sending it in bands
# keeps every command small and well inside what the printer is built for.
DEFAULT_BAND = 256


def feed_lines(n: int) -> bytes:
    return ESC + b"d" + bytes([max(0, min(255, n))])


def text(s: str) -> bytes:
    """Plain ASCII text (non-ASCII characters are replaced with '?')."""
    return s.encode("ascii", "replace")


def pack_rows(img: Image.Image) -> tuple[int, int, bytes]:
    """1-bit image -> (bytes_per_row, rows, data); bit=1 means 'burn a dot'."""
    if img.mode != "1":
        img = img.convert("1")
    black = ~np.asarray(img, dtype=bool)          # mode "1": True = white
    packed = np.packbits(black, axis=1)           # MSB = leftmost dot, pads with 0 (white)
    return packed.shape[1], packed.shape[0], packed.tobytes()


def raster(img: Image.Image, band: int = DEFAULT_BAND) -> bytes:
    """GS v 0 m xL xH yL yH d1...dk, split into bands of `band` rows."""
    width_bytes, rows, data = pack_rows(img)
    band = max(1, min(band, 2047))
    out = bytearray()
    for y in range(0, rows, band):
        n = min(band, rows - y)
        out += GS + b"v0\x00"
        out += bytes([width_bytes & 0xFF, width_bytes >> 8, n & 0xFF, n >> 8])
        out += data[y * width_bytes:(y + n) * width_bytes]
    return bytes(out)


def image_job(img: Image.Image, *, cut: bool = True, feed: int = 4,
              band: int = DEFAULT_BAND) -> bytes:
    """A complete, self-contained print job for one image."""
    job = bytearray(INIT + ALIGN_CENTER)
    job += raster(img, band)
    job += feed_lines(feed)
    if cut:
        job += PARTIAL_CUT
    return bytes(job)


# --------------------------------------------------------------------------- status

STATUS_QUERIES = {n: DLE + b"\x04" + bytes([n]) for n in (1, 2, 3, 4)}


def valid_status_byte(b: int) -> bool:
    # ESC/POS status bytes always have bit0=0, bit1=1, bit4=1, bit7=0.
    return (b & 0x93) == 0x12


def parse_status(raw: dict[int, int | None]) -> dict:
    """Decode DLE EOT 1..4 answers. Missing/invalid answers -> fields stay None."""
    def get(n):
        b = raw.get(n)
        return b if b is not None and valid_status_byte(b) else None

    s1, s2, s3, s4 = get(1), get(2), get(3), get(4)
    st = {
        "responded": any(v is not None for v in (s1, s2, s3, s4)),
        "online": None if s1 is None else not (s1 & 0x08),
        "cover_open": None if s2 is None else bool(s2 & 0x04),
        "paper_end": None,
        "paper_near_end": None if s4 is None else bool(s4 & 0x0C),
        "error": None if s2 is None else bool(s2 & 0x40),
        "cutter_error": None if s3 is None else bool(s3 & 0x08),
        "unrecoverable_error": None if s3 is None else bool(s3 & 0x20),
        "raw": {n: (None if v is None else f"0x{v:02x}") for n, v in raw.items()},
    }
    if s4 is not None or s2 is not None:
        st["paper_end"] = bool((s4 or 0) & 0x60) or bool((s2 or 0) & 0x20)
    return st


def blocking_problem(st: dict | None) -> str | None:
    """Human-readable reason not to print, or None if nothing is definitely wrong."""
    if not st or not st.get("responded"):
        return None
    if st.get("cover_open"):
        return "The printer cover is open. Close it and try again."
    if st.get("paper_end"):
        return "The printer is out of paper. Load a new roll and try again."
    if st.get("cutter_error"):
        return "Auto-cutter error. Turn the printer off, clear any jam, turn it on again."
    if st.get("unrecoverable_error"):
        return "The printer reports an unrecoverable error. Power-cycle it."
    return None


def status_warning(st: dict | None) -> str | None:
    """Something worth showing, but not a reason to refuse the job."""
    if not st or not st.get("responded"):
        return None
    if st.get("online") is False:
        return "Printer reports 'offline' (check the lid, paper and error light)."
    if st.get("paper_near_end"):
        return "Paper is running low."
    return None


def parse_port_status(b: int | None) -> dict | None:
    """USB printer-class GET_PORT_STATUS byte (IEEE 1284 style)."""
    if b is None:
        return None
    return {"raw": f"0x{b:02x}", "paper_empty": bool(b & 0x20),
            "selected": bool(b & 0x10), "no_error": bool(b & 0x08)}


def describe(st: dict | None) -> str:
    """One-line human summary of a parsed status."""
    if st is None:
        return "this connection can't read status"
    raw = " ".join(v or "--" for v in st.get("raw", {}).values())
    if not st.get("responded"):
        return f"no status reply (DLE EOT 1-4: {raw})"
    yn = lambda v, t, f: "?" if v is None else (t if v else f)  # noqa: E731
    return (f"online: {yn(st['online'], 'yes', 'NO')} | cover: {yn(st['cover_open'], 'OPEN', 'closed')} | "
            f"paper: {yn(st['paper_end'], 'OUT', 'ok')}"
            f"{' (low)' if st.get('paper_near_end') else ''} | "
            f"error: {yn(st['error'], 'YES', 'no')}  [DLE EOT 1-4: {raw}]")
