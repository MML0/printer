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
# keeps every command small. 48 rows = 3 KB, below the 56-row bands of the test
# page that printed fine on the SRP-350III (a 256-row photo stopped after ~1 cm).
DEFAULT_BAND = 48

# The SRP-350III's power supply can't keep up with long dark stretches: a 50%-black
# strip reset the printer (it dropped off USB for ~4 s) after ~1 cm, a 15% strip
# printed fine, and the 50% strip survived when sent in pieces with pauses. The
# printer's speed/density/heating commands don't lower the load, so pauses it is.
# A long pause lets the head cool, and the next rows come out a shade lighter (a
# visible band in dark areas). Side-by-side prints were smoothest with tiny pieces:
# a segment ends once it holds BLACK_BUDGET rows' worth of black dots (~2 rows at
# 50%), followed by PAUSE_PER_BLACK_ROW per row's worth. Light areas don't wait.
# Pause length: a 70% strip in these pieces survived 0.025, 0.012 and 0.006 s per
# row's worth but reset with no pause at all; 0.015 keeps a 2x margin.
# All of these can be changed per print from the web app (a printer with a stronger
# supply can print without chunks at all).
BLACK_BUDGET = 1.0
PAUSE_PER_BLACK_ROW = 0.015
MIN_SEGMENT_ROWS = 2


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


def segment_rows(black: np.ndarray, budget: float = BLACK_BUDGET, min_rows: int = MIN_SEGMENT_ROWS,
                 look_back: int = 0) -> list[tuple[int, int]]:
    """Split rows (per-row share of black dots) into (start, end) runs that each hold
    at most ~`budget` rows' worth of black. With `look_back`, a run may end up to that
    many rows early, after the lightest row there, where the pause line shows least."""
    runs, start, acc = [], 0, 0.0
    for r, b in enumerate(black):
        acc += b
        if acc >= budget and r + 1 - start >= min_rows:
            lo = max(start + min_rows - 1, r - look_back)
            window = black[lo:r + 1]                    # latest row that is about the lightest
            end = lo + int(np.flatnonzero(window <= window.min() + 0.03)[-1]) + 1
            runs.append((start, end))
            start, acc = end, float(black[end:r + 1].sum())
    if start < len(black) or not runs:
        runs.append((start, len(black)))
    return runs


def image_segments(img: Image.Image, *, cut: bool = True, feed: int = 4, band: int = DEFAULT_BAND,
                   budget: float = BLACK_BUDGET, min_rows: int = MIN_SEGMENT_ROWS,
                   pause: float = PAUSE_PER_BLACK_ROW, look_back: int = 0) -> list[tuple[bytes, float]]:
    """The print job for one image as (bytes, seconds to wait afterwards) segments.
    Joined together the bytes are a complete, self-contained job."""
    if img.mode != "1":
        img = img.convert("1")
    black = (~np.asarray(img, dtype=bool)).mean(axis=1)
    out = []
    for y0, y1 in segment_rows(black, budget, min_rows, look_back):
        seg = img.crop((0, y0, img.width, y1))
        out.append([raster(seg, band) if y1 > y0 else b"", pause * float(black[y0:y1].sum())])
    out[-1][1] = 0.0
    out[0][0] = INIT + ALIGN_CENTER + out[0][0]
    out[-1][0] += feed_lines(feed) + (PARTIAL_CUT if cut else b"")
    return [(data, wait) for data, wait in out]


def image_job(img: Image.Image, *, cut: bool = True, feed: int = 4,
              band: int = DEFAULT_BAND) -> bytes:
    """A complete, self-contained print job for one image, sent without pauses."""
    return INIT + ALIGN_CENTER + raster(img, band) + feed_lines(feed) + (PARTIAL_CUT if cut else b"")


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
