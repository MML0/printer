"""Photo -> 1-bit bitmap for a thermal printer.

Pipeline (the preview and the printer use exactly the same code):

    load -> grayscale -> rotate -> resize to printer width -> auto-contrast
    -> brightness -> contrast -> gamma -> sharpness -> invert -> dither
"""
from __future__ import annotations

import io
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict, dataclass, fields

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

DITHERS = ("floyd_steinberg", "atkinson", "stucki", "bayer", "threshold")

# Longest side kept in memory after loading. Plenty for a 512-dot-wide print,
# and keeps live preview fast even for 48 MP phone photos.
WORKING_MAX = 2048


@dataclass
class Settings:
    width: int = 512            # dots; SRP-350III prints 512 dots on 80 mm paper
    brightness: float = 1.0     # 1.0 = unchanged
    contrast: float = 1.0       # 1.0 = unchanged
    gamma: float = 1.0          # >1 lightens midtones, <1 darkens them
    sharpness: float = 1.0      # 1.0 = unchanged, 2.0 = sharper
    auto_contrast: bool = True  # stretch the histogram first (good for photos)
    dither: str = "floyd_steinberg"
    threshold: int = 128        # only used by dither="threshold"
    invert: bool = False
    rotate: int = 0             # 0 / 90 / 180 / 270 (90 = landscape printed big)

    _LIMITS = {
        "width": (8, 1024),
        "brightness": (0.0, 3.0),
        "contrast": (0.0, 3.0),
        "gamma": (0.2, 5.0),
        "sharpness": (0.0, 5.0),
        "threshold": (0, 255),
    }

    @classmethod
    def from_dict(cls, d: dict | None) -> "Settings":
        """Build settings from untrusted input (JSON, CLI); clamps every value."""
        s = cls()
        for f in fields(cls):
            if f.name.startswith("_") or d is None or d.get(f.name) is None:
                continue
            v = d[f.name]
            default = getattr(s, f.name)
            try:
                if isinstance(default, bool):
                    v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
                elif isinstance(default, int):
                    v = int(round(float(v)))
                elif isinstance(default, float):
                    v = float(v)
                else:
                    v = str(v)
            except (TypeError, ValueError):
                continue
            setattr(s, f.name, v)
        return s.clamped()

    def clamped(self) -> "Settings":
        for name, (lo, hi) in self._LIMITS.items():
            setattr(self, name, type(getattr(self, name))(min(hi, max(lo, getattr(self, name)))))
        if self.dither not in DITHERS:
            self.dither = "floyd_steinberg"
        self.rotate = int(self.rotate) % 360
        if self.rotate not in (0, 90, 180, 270):
            self.rotate = 0
        return self

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}


# --------------------------------------------------------------------------- load

def load_image(src) -> Image.Image:
    """Open a photo (path, bytes or file object) as a grayscale working image.

    Handles EXIF rotation, transparency, and - on macOS - formats Pillow can't
    read itself (e.g. iPhone HEIC) by converting them with the built-in `sips`.
    """
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    try:
        im = Image.open(src)
        im.load()
    except Exception as first_error:
        im = _open_with_sips(src)
        if im is None:
            raise ValueError(f"Can't read this image ({first_error}). Try a JPEG or PNG.") from None

    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("L")
    im.thumbnail((WORKING_MAX, WORKING_MAX), Image.LANCZOS)
    return im


def _open_with_sips(src) -> Image.Image | None:
    if platform.system() != "Darwin":
        return None
    with tempfile.TemporaryDirectory() as tmp:
        if isinstance(src, (str, os.PathLike)):
            in_path = str(src)
        else:
            src.seek(0)
            in_path = os.path.join(tmp, "input")
            with open(in_path, "wb") as fh:
                fh.write(src.read())
        out_path = os.path.join(tmp, "out.png")
        r = subprocess.run(["sips", "-s", "format", "png", in_path, "--out", out_path],
                           capture_output=True)
        if r.returncode != 0 or not os.path.exists(out_path):
            return None
        im = Image.open(out_path)
        im.load()
        return im


# --------------------------------------------------------------------------- adjust

def adjust(gray: Image.Image, s: Settings) -> Image.Image:
    """Everything before dithering: returns an 8-bit grayscale image at print width."""
    im = gray if gray.mode == "L" else gray.convert("L")
    if s.rotate:
        im = im.rotate(-s.rotate, expand=True)  # PIL rotates counter-clockwise
    w = s.width
    h = max(1, round(im.height * w / im.width))
    im = im.resize((w, h), Image.LANCZOS)

    if s.auto_contrast:
        im = ImageOps.autocontrast(im, cutoff=1)
    if s.brightness != 1.0:
        im = ImageEnhance.Brightness(im).enhance(s.brightness)
    if s.contrast != 1.0:
        im = ImageEnhance.Contrast(im).enhance(s.contrast)
    if s.gamma != 1.0:
        inv = 1.0 / s.gamma
        im = im.point([round(255 * (i / 255) ** inv) for i in range(256)])
    if s.sharpness != 1.0:
        im = ImageEnhance.Sharpness(im).enhance(s.sharpness)
    if s.invert:
        im = ImageOps.invert(im)
    return im


# --------------------------------------------------------------------------- dither

# (dx, dy, weight) - error-diffusion kernels
_KERNELS = {
    "floyd_steinberg": [(1, 0, 7 / 16), (-1, 1, 3 / 16), (0, 1, 5 / 16), (1, 1, 1 / 16)],
    # Atkinson spreads only 6/8 of the error: punchier, lighter, great on thermal paper.
    "atkinson": [(1, 0, 1 / 8), (2, 0, 1 / 8), (-1, 1, 1 / 8), (0, 1, 1 / 8), (1, 1, 1 / 8), (0, 2, 1 / 8)],
    "stucki": [(1, 0, 8 / 42), (2, 0, 4 / 42),
               (-2, 1, 2 / 42), (-1, 1, 4 / 42), (0, 1, 8 / 42), (1, 1, 4 / 42), (2, 1, 2 / 42),
               (-2, 2, 1 / 42), (-1, 2, 2 / 42), (0, 2, 4 / 42), (1, 2, 2 / 42), (2, 2, 1 / 42)],
}


def _error_diffusion(gray: Image.Image, kernel) -> np.ndarray:
    """Generic error diffusion. Returns bool array, True = white.

    The left-to-right part runs in Python (it is inherently sequential); the
    error pushed to the rows below is added with numpy, one row at a time.
    """
    w, h = gray.size
    pad = 2
    buf = np.zeros((h + 2, w + 2 * pad), dtype=np.float64)
    buf[:h, pad:pad + w] = np.asarray(gray, dtype=np.float64)
    same_row = [(dx, wt) for dx, dy, wt in kernel if dy == 0]
    below = [(dx, dy, wt) for dx, dy, wt in kernel if dy > 0]
    (r1, w1), (r2, w2) = (same_row + [(1, 0.0), (2, 0.0)])[:2]
    out = np.zeros((h, w), dtype=bool)

    for y in range(h):
        row = buf[y].tolist()
        errs = [0.0] * w
        white = [False] * w
        for x in range(w):
            i = x + pad
            old = row[i]
            if old >= 128.0:
                white[x] = True
                e = old - 255.0
            else:
                e = old
            errs[x] = e
            row[i + r1] += e * w1
            row[i + r2] += e * w2
        out[y] = white
        err = np.asarray(errs)
        for dx, dy, wt in below:
            buf[y + dy, pad + dx:pad + dx + w] += err * wt
    return out


def _bayer_matrix(n: int = 8) -> np.ndarray:
    m = np.array([[0]])
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return m


def dither(gray: Image.Image, s: Settings) -> Image.Image:
    """8-bit grayscale -> mode "1" image (0 = black dot, 255 = paper)."""
    arr = np.asarray(gray, dtype=np.uint8)
    if s.dither == "threshold":
        white = arr >= s.threshold
    elif s.dither == "bayer":
        m = _bayer_matrix(8)
        thr = (m + 0.5) * (256.0 / m.size)
        h, w = arr.shape
        tiled = np.tile(thr, (h // 8 + 1, w // 8 + 1))[:h, :w]
        white = arr >= tiled
    else:
        white = _error_diffusion(gray, _KERNELS[s.dither])
    return Image.fromarray(white.astype(np.uint8) * 255, "L").convert("1", dither=Image.NONE)


def render(gray: Image.Image, s: Settings) -> Image.Image:
    """Full pipeline: working image -> printable 1-bit image."""
    return dither(adjust(gray, s), s)


def printed_size_mm(img: Image.Image, dpi: int = 180) -> tuple[float, float]:
    return img.width / dpi * 25.4, img.height / dpi * 25.4
