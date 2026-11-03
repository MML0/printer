"""Run with:  python3 -m unittest discover tests"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from photo_printer import commands, imaging, jobs, printers  # noqa: E402


def decode(stream: bytes):
    """Minimal ESC/POS interpreter: returns (raster bands, text, cut?) and fails on unknown bytes."""
    i, bands, text, cut = 0, [], bytearray(), False
    fixed = {b"\x1b@": 0, b"\x1ba": 1, b"\x1bE": 1, b"\x1d!": 1, b"\x1bd": 1, b"\x1dV": 1}
    while i < len(stream):
        two = stream[i:i + 2]
        if stream[i:i + 3] == b"\x1dv0":
            m, xl, xh, yl, yh = stream[i + 3:i + 8]
            assert m == 0
            wb, rows = xl + 256 * xh, yl + 256 * yh
            data = stream[i + 8:i + 8 + wb * rows]
            assert len(data) == wb * rows, "truncated raster data"
            bits = np.unpackbits(np.frombuffer(data, np.uint8).reshape(rows, wb), axis=1)
            bands.append(bits)
            i += 8 + wb * rows
        elif two in fixed:
            if two == b"\x1dV":
                cut = True
            i += 2 + fixed[two]
        elif stream[i] < 0x20 and stream[i] != 0x0A:
            raise AssertionError(f"unexpected control byte 0x{stream[i]:02x} at {i}")
        else:
            text.append(stream[i])
            i += 1
    return bands, text.decode("ascii"), cut


def sample_photo() -> Image.Image:
    photos = sorted((ROOT / "imgs").glob("*.jpg"))
    if photos:
        return imaging.load_image(photos[0])
    y, x = np.mgrid[0:600, 0:800]
    return Image.fromarray(((x + y) % 256).astype(np.uint8), "L")


class SettingsTest(unittest.TestCase):
    def test_clamps_untrusted_input(self):
        s = imaging.Settings.from_dict({"brightness": "9", "width": "5000", "dither": "nope",
                                        "rotate": 45, "invert": "true", "gamma": None, "junk": 1})
        self.assertEqual(s.brightness, 3.0)
        self.assertEqual(s.width, 1024)
        self.assertEqual(s.dither, "floyd_steinberg")
        self.assertEqual(s.rotate, 0)
        self.assertTrue(s.invert)
        self.assertEqual(s.gamma, 1.0)


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gray = sample_photo()

    def test_every_dither_gives_printable_bitmap(self):
        for name in imaging.DITHERS:
            out = imaging.render(self.gray, imaging.Settings(dither=name))
            self.assertEqual(out.mode, "1", name)
            self.assertEqual(out.width, 512, name)
            black = 1 - np.asarray(out).mean()
            self.assertTrue(0.02 < black < 0.98, f"{name}: {black:.2f} black looks wrong")

    def test_brightness_makes_print_lighter(self):
        dark = imaging.render(self.gray, imaging.Settings(brightness=0.7))
        light = imaging.render(self.gray, imaging.Settings(brightness=1.4))
        self.assertGreater(np.asarray(light).mean(), np.asarray(dark).mean())

    def test_error_diffusion_preserves_tone(self):
        flat = Image.new("L", (256, 64), 64)  # 25% gray -> ~75% black dots
        for name in ("floyd_steinberg", "stucki", "bayer"):
            out = imaging.dither(flat, imaging.Settings(dither=name))
            self.assertAlmostEqual(np.asarray(out).mean(), 64 / 255, delta=0.03, msg=name)

    def test_max_black_reduces_dots(self):
        full = 1 - np.asarray(imaging.render(self.gray, imaging.Settings())).mean()
        limited = 1 - np.asarray(imaging.render(self.gray, imaging.Settings(max_black=0.8))).mean()
        self.assertLess(limited, full)
        solid = imaging.dither(imaging.adjust(Image.new("L", (64, 64), 0),
                                              imaging.Settings(width=64, max_black=0.8, auto_contrast=False)),
                               imaging.Settings(width=64))
        self.assertAlmostEqual(1 - np.asarray(solid).mean(), 0.8, delta=0.03)

    def test_rotate_90_swaps_orientation(self):
        tall = Image.new("L", (300, 600), 128)
        out = imaging.render(tall, imaging.Settings(rotate=90))
        self.assertEqual(out.size, (512, 256))


class EscPosTest(unittest.TestCase):
    def test_raster_round_trip(self):
        gray = sample_photo()
        for width in (512, 500, 384):  # 500 is not a multiple of 8 -> padded
            bmp = imaging.render(gray, imaging.Settings(width=width))
            bands, _, cut = decode(commands.image_job(bmp, band=100))
            self.assertTrue(cut)
            bits = np.vstack(bands)[:, :width]
            expected = ~np.asarray(bmp, dtype=bool)
            np.testing.assert_array_equal(bits.astype(bool), expected)
            self.assertTrue(all(b.shape[0] <= 100 for b in bands))

    def test_job_starts_with_init(self):
        bmp = Image.new("1", (16, 4), 0)
        job = commands.image_job(bmp, cut=False)
        self.assertTrue(job.startswith(b"\x1b@"))
        self.assertNotIn(b"\x1dV", job)

    def test_test_page_is_clean(self):
        bands, text, cut = decode(jobs.test_page(512))
        self.assertEqual(sum(b.shape[0] for b in bands), 40 + 56 * len(imaging.DITHERS))
        self.assertTrue(all(b.shape[0] <= commands.DEFAULT_BAND for b in bands))
        self.assertIn("PHOTO PRINTER TEST", text)
        self.assertTrue(cut)

    def test_status_parsing(self):
        ok = commands.parse_status({1: 0x16, 2: 0x12, 3: 0x12, 4: 0x12})
        self.assertTrue(ok["online"])
        self.assertIsNone(commands.blocking_problem(ok))
        cover = commands.parse_status({1: 0x1E, 2: 0x16, 3: 0x12, 4: 0x12})
        self.assertIn("cover", commands.blocking_problem(cover))
        paper = commands.parse_status({1: 0x1E, 2: 0x32, 3: 0x12, 4: 0x72})
        self.assertIn("paper", commands.blocking_problem(paper))
        silent = commands.parse_status({1: None, 2: None, 3: None, 4: None})
        self.assertFalse(silent["responded"])
        self.assertIsNone(commands.blocking_problem(silent))
        garbage = commands.parse_status({1: 0xFF, 2: 0x00})
        self.assertFalse(garbage["responded"])

    def test_offline_without_reason_warns_but_does_not_block(self):
        st = commands.parse_status({1: 0x1E, 2: None, 3: None, 4: None})
        self.assertIsNone(commands.blocking_problem(st))
        self.assertIn("offline", commands.status_warning(st))
        self.assertIn("online: NO", commands.describe(st))
        self.assertIn("no status reply", commands.describe(commands.parse_status({1: None})))
        self.assertEqual(commands.parse_port_status(0x18)["paper_empty"], False)
        self.assertEqual(commands.parse_port_status(0x38)["paper_empty"], True)


class JobTest(unittest.TestCase):
    def test_print_photo_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "job.bin")
            job = jobs.print_photo(printers.open_printer(f"file:{out}"), sample_photo(),
                                   imaging.Settings(dither="atkinson"))
            self.assertEqual(job["result"], "sent")
            bands, _, _ = decode(Path(out).read_bytes())
            self.assertEqual(sum(b.shape[0] for b in bands), job["height"])

    def test_refuses_absurdly_long_print(self):
        tall = Image.new("1", (512, 5000), 1)
        with self.assertRaises(printers.PrinterError):
            jobs.print_bitmap(printers.FilePrinter(os.devnull), tall)

    def test_blocks_when_out_of_paper(self):
        class NoPaper(printers.FilePrinter):
            def status(self):
                return commands.parse_status({1: 0x1E, 2: 0x32, 3: 0x12, 4: 0x72})
        with self.assertRaises(printers.PrinterError) as ctx:
            jobs.print_bitmap(NoPaper(os.devnull), Image.new("1", (512, 10), 1))
        self.assertIn("paper", str(ctx.exception))


try:
    import usb.core  # noqa: F401
    HAVE_PYUSB = True
except ImportError:
    HAVE_PYUSB = False


@unittest.skipUnless(HAVE_PYUSB, "pyusb not installed")
class UsbTransportTest(unittest.TestCase):
    """USB logic with a fake device (no hardware needed)."""

    def make(self, fail_after=None, max_per_write=None, replies=None):
        import usb.core
        received = bytearray()

        class InEp:
            bEndpointAddress = 0x81

            def read(self, n, timeout):
                if not replies:
                    raise usb.core.USBTimeoutError("Operation timed out")
                return bytes([replies.pop(0)])

        class Ep:
            bEndpointAddress = 0x02

            def write(self, data, timeout):
                if fail_after is not None and len(received) >= fail_after:
                    raise usb.core.USBTimeoutError("Operation timed out")
                data = bytes(data)[:max_per_write] if max_per_write else bytes(data)
                received.extend(data)
                return len(data)

        p = printers.UsbPrinter()
        dev = type("Dev", (), {"idVendor": 0x1504, "idProduct": 0x2B})()
        p._open = lambda: (dev, 0, Ep(), InEp() if replies is not None else None, False)
        p._close = lambda *a: None
        return p, received

    def test_sends_everything_even_with_short_writes(self):
        p, received = self.make(max_per_write=1000)
        data = bytes(range(256)) * 100
        p.send(data)
        self.assertEqual(bytes(received), data)

    def bixolon_like(self, stale=b""):
        """Fake that behaves like the real SRP-350III on macOS: answers arrive a few
        reads late, reads return empty until then, old answers can still be queued."""
        answers = {1: 0x16, 2: 0x12, 3: 0x12, 4: 0x1E}
        ready, pending = bytearray(stale), []

        class InEp:
            bEndpointAddress = 0x81

            def read(self, n, timeout):
                for item in pending:
                    item[0] -= 1
                while pending and pending[0][0] <= 0:
                    ready.append(pending.pop(0)[1])
                out = bytes(ready)
                ready.clear()
                return out

        class OutEp:
            bEndpointAddress = 0x02

            def write(self, data, timeout):
                data = bytes(data)
                if data[:2] == b"\x10\x04":
                    pending.append([3, answers[data[2]]])
                return len(data)

        p = printers.UsbPrinter()
        p._open = lambda: (object(), 0, OutEp(), InEp(), False)
        p._close = lambda *a: None
        return p

    def test_segments_join_to_the_plain_job(self):
        bmp = Image.fromarray((np.random.default_rng(1).random((100, 512)) > 0.5).astype(np.uint8) * 255).convert("1")
        segs = commands.image_segments(bmp)
        self.assertGreater(len(segs), 1)
        self.assertEqual(b"".join(seg for seg, _ in segs), commands.image_job(bmp))
        bands, _, cut = decode(b"".join(seg for seg, _ in segs))
        self.assertEqual(sum(b.shape[0] for b in bands), 100)
        self.assertTrue(cut)

    def test_light_rows_need_no_break_and_dark_rows_rest(self):
        light = np.full(400, 0.04)                       # 16 rows' worth in total
        self.assertEqual(commands.segment_rows(light), [(0, 400)])
        dark = np.full(200, 0.6)
        dark[30] = 0.1                                   # a light row near the first limit
        runs = commands.segment_rows(dark)
        self.assertEqual(runs[0], (0, 31))               # breaks right after the light row
        self.assertEqual(runs[-1][1], 200)
        self.assertTrue(all(b == c for (_, b), (c, _) in zip(runs, runs[1:])))
        for y0, y1 in runs:
            self.assertLessEqual(dark[y0:y1].sum(), commands.BLACK_BUDGET + 1)

    def test_pause_grows_with_black_and_last_segment_does_not_wait(self):
        a = np.full((120, 512), 255, dtype=np.uint8)
        a[:, :256] = 0                                   # 50% black everywhere
        segs = commands.image_segments(Image.fromarray(a).convert("1"))
        pauses = [p for _, p in segs]
        self.assertEqual(pauses[-1], 0.0)
        self.assertAlmostEqual(pauses[0], commands.PAUSE_PER_BLACK_ROW * commands.BLACK_BUDGET)

    def test_paced_send_sleeps_between_segments(self):
        p, received = self.make()
        slept, orig = [], printers.time.sleep
        printers.time.sleep = slept.append
        try:
            p.send([(b"a" * 10, 0.0), (b"b" * 10, 0.25), (b"c" * 10, 0.5)])
        finally:
            printers.time.sleep = orig
        self.assertEqual(bytes(received), b"a" * 10 + b"b" * 10 + b"c" * 10)
        self.assertEqual(slept, [0.25, 0.5])

    def test_status_with_late_answers_and_leftovers(self):
        p = self.bixolon_like(stale=bytes([0x16, 0x12, 0x12, 0x1E, 0x1E]))
        st = p.status()
        self.assertTrue(st["responded"])
        self.assertTrue(st["online"])
        self.assertFalse(st["cover_open"])   # was misread as OPEN before the fix
        self.assertFalse(st["paper_end"])
        self.assertTrue(st["paper_near_end"])
        self.assertIsNone(commands.blocking_problem(st))
        self.assertIsNone(st["port"])  # fake device has no control endpoint

    def test_status_twice_in_a_row_stays_aligned(self):
        p = self.bixolon_like()
        for _ in range(3):
            self.assertEqual(p.status()["raw"], {1: "0x16", 2: "0x12", 3: "0x12", 4: "0x1e"})

    def test_debug_status_lists_every_question(self):
        lines = self.bixolon_like(stale=b"\x1e").debug_status()
        self.assertIn("cleared 1 leftover byte(s): 1e", "\n".join(lines))
        self.assertTrue(any("round 2  DLE EOT 4: 0x1e" in line for line in lines))

    def test_status_survives_silent_printer(self):
        p, _ = self.make(replies=[])
        self.assertFalse(p.status()["responded"])

    def test_timeout_is_reported_as_partial(self):
        p, _ = self.make(fail_after=8192)
        orig = printers.list_usb
        printers.list_usb = lambda: [{"vid": 0x1504, "pid": 0x2B}]  # printer still attached
        try:
            with self.assertRaises(printers.PrinterError) as ctx:
                p.send(b"x" * 20000)
        finally:
            printers.list_usb = orig
        self.assertTrue(ctx.exception.partial)
        self.assertIn("8192 of 20000", str(ctx.exception))
        self.assertIn("Check paper/cover", str(ctx.exception))

    def test_printer_vanishing_mid_job_is_explained(self):
        p, _ = self.make(fail_after=57344)
        orig = printers.list_usb
        printers.list_usb = lambda: []  # printer rebooted and dropped off USB
        try:
            with self.assertRaises(printers.PrinterError) as ctx:
                p.send(b"x" * 58283)
        finally:
            printers.list_usb = orig
        self.assertIn("disconnected in the middle", str(ctx.exception))
        self.assertTrue(ctx.exception.partial)


if __name__ == "__main__":
    unittest.main()
