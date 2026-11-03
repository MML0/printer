"""Ways to reach the printer, plus `detect()` which reports everything it can see.

Printer specs (use with --printer):
    auto                      USB first, then a CUPS queue that looks like a BIXOLON
    usb                       first BIXOLON (or any USB printer-class device)
    usb:1504:0031             a specific USB vendor:product id (hex)
    cups:QUEUE_NAME           a macOS printer added in System Settings (raw mode)
    net:192.168.1.50[:9100]   Ethernet models
    serial:/dev/cu.usbmodemX  USB virtual-COM mode / serial models
    file:out.bin              dry run: write the bytes to a file
"""
from __future__ import annotations

import glob
import os
import platform
import re
import select
import shutil
import socket
import subprocess

from . import commands

BIXOLON_VID = 0x1504
USB_CLASS_PRINTER = 7
NAME_HINTS = ("bixolon", "srp-350", "srp350", "srp_350")

USB_CHUNK = 4096
USB_TIMEOUT_MS = 20000  # per chunk; a paused printer (no paper) triggers this


class PrinterError(Exception):
    """Something went wrong talking to the printer. `partial` = some bytes may have printed."""

    def __init__(self, msg: str, partial: bool = False):
        super().__init__(msg)
        self.partial = partial


class Printer:
    name = "printer"

    def send(self, data: bytes) -> None:
        raise NotImplementedError

    def status(self) -> dict | None:
        """Decoded DLE EOT status, or None if this connection can't read back."""
        return None


# --------------------------------------------------------------------------- USB

def usb_backend():
    """pyusb backend, looking in Homebrew/MacPorts folders that pyusb misses on macOS."""
    try:
        import usb.backend.libusb1 as libusb1
    except ImportError:
        return None
    be = libusb1.get_backend()
    if be is not None:
        return be
    for path in ("/opt/homebrew/lib/libusb-1.0.dylib", "/usr/local/lib/libusb-1.0.dylib",
                 "/opt/local/lib/libusb-1.0.dylib"):
        if os.path.exists(path):
            be = libusb1.get_backend(find_library=lambda _name, p=path: p)
            if be is not None:
                return be
    return None


def _usb_strings(dev) -> tuple[str, str]:
    import usb.util
    out = []
    for idx in (dev.iManufacturer, dev.iProduct):
        try:
            out.append(usb.util.get_string(dev, idx) or "" if idx else "")
        except Exception:  # permissions / device busy: strings are optional
            out.append("")
    return out[0], out[1]


def _is_printer_class(dev) -> bool:
    try:
        for cfg in dev:
            for intf in cfg:
                if intf.bInterfaceClass == USB_CLASS_PRINTER:
                    return True
    except Exception:
        pass
    return False


def list_usb() -> list[dict]:
    """All USB devices pyusb can see (empty list if pyusb/libusb is missing)."""
    be = usb_backend()
    if be is None:
        return []
    import usb.core
    devices = []
    for dev in usb.core.find(find_all=True, backend=be):
        maker, product = _usb_strings(dev)
        devices.append({
            "vid": dev.idVendor, "pid": dev.idProduct,
            "manufacturer": maker, "product": product,
            "printer_class": _is_printer_class(dev),
            "bixolon": dev.idVendor == BIXOLON_VID or any(h in f"{maker} {product}".lower() for h in NAME_HINTS),
            "_dev": dev,
        })
    return devices


class UsbPrinter(Printer):
    """Direct USB via libusb. The device is opened per job, so unplug/replug just works."""

    def __init__(self, vid: int | None = None, pid: int | None = None):
        self.vid, self.pid = vid, pid
        self.name = "usb"

    def _find(self):
        if usb_backend() is None:
            raise PrinterError(
                "USB needs pyusb + libusb. Run:  brew install libusb  and  pip install pyusb")
        devs = list_usb()
        if self.vid is not None:
            devs = [d for d in devs if d["vid"] == self.vid and (self.pid is None or d["pid"] == self.pid)]
        else:
            devs = sorted((d for d in devs if d["bixolon"] or d["printer_class"]),
                          key=lambda d: not d["bixolon"])
        if not devs:
            raise PrinterError("No USB printer found. Is it switched on and plugged in? "
                               "Run `python3 thermal.py detect`.")
        d = devs[0]
        self.name = f"usb:{d['vid']:04x}:{d['pid']:04x} {d['product'] or ''}".strip()
        return d["_dev"]

    def _open(self):
        import usb.core
        import usb.util
        dev = self._find()
        try:
            cfg = dev.get_active_configuration()
        except usb.core.USBError:
            cfg = None
        if cfg is None:
            dev.set_configuration()
            cfg = dev.get_active_configuration()

        intfs = list(cfg)
        intf = next((i for i in intfs if i.bInterfaceClass == USB_CLASS_PRINTER), None)

        def bulk(i, direction):
            return usb.util.find_descriptor(i, custom_match=lambda e: (
                usb.util.endpoint_direction(e.bEndpointAddress) == direction
                and usb.util.endpoint_type(e.bmAttributes) == usb.util.ENDPOINT_TYPE_BULK))

        if intf is None or bulk(intf, usb.util.ENDPOINT_OUT) is None:
            intf = next((i for i in intfs if bulk(i, usb.util.ENDPOINT_OUT) is not None), None)
        if intf is None:
            raise PrinterError("This USB device has no bulk OUT endpoint, so it isn't a printer port.")

        n = intf.bInterfaceNumber
        detached = False
        try:
            if dev.is_kernel_driver_active(n):  # Linux only; macOS raises NotImplementedError
                dev.detach_kernel_driver(n)
                detached = True
        except (NotImplementedError, usb.core.USBError):
            pass
        try:
            usb.util.claim_interface(dev, n)
        except usb.core.USBError as e:
            raise PrinterError(
                f"Can't claim the printer's USB interface ({e}). Another program is using it - "
                "quit the BIXOLON utility / pause the CUPS queue, or use --printer cups:NAME.")
        return dev, n, bulk(intf, usb.util.ENDPOINT_OUT), bulk(intf, usb.util.ENDPOINT_IN), detached

    @staticmethod
    def _close(dev, n, detached):
        import usb.util
        try:
            usb.util.release_interface(dev, n)
            if detached:
                dev.attach_kernel_driver(n)
        except Exception:
            pass
        usb.util.dispose_resources(dev)

    def send(self, data: bytes) -> None:
        import usb.core
        dev, n, ep_out, _ep_in, detached = self._open()
        sent = 0
        try:
            while sent < len(data):
                chunk = data[sent:sent + USB_CHUNK]
                written = ep_out.write(chunk, USB_TIMEOUT_MS)
                if written <= 0:
                    raise usb.core.USBError("wrote 0 bytes")
                sent += written
        except usb.core.USBError as e:
            raise PrinterError(
                f"USB transfer stopped after {sent} of {len(data)} bytes ({e}). "
                "Check paper/cover. Part of the image may already be printed.",
                partial=sent > 0)
        finally:
            self._close(dev, n, detached)

    def status(self) -> dict | None:
        import usb.core
        dev, n, ep_out, ep_in, detached = self._open()
        try:
            if ep_in is None:
                return None
            _drain(lambda: bytes(ep_in.read(64, 50)), usb.core.USBError)
            raw = {}
            for q, cmd in commands.STATUS_QUERIES.items():
                ep_out.write(cmd, 1000)
                try:
                    resp = bytes(ep_in.read(64, 500))
                    raw[q] = resp[-1] if resp else None
                except usb.core.USBError:
                    raw[q] = None
            return commands.parse_status(raw)
        finally:
            self._close(dev, n, detached)


def _drain(read, errors, limit: int = 20):
    for _ in range(limit):
        try:
            if not read():
                return
        except errors:
            return


# --------------------------------------------------------------------------- CUPS

def list_cups() -> list[dict]:
    if not shutil.which("lpstat"):
        return []
    r = subprocess.run(["lpstat", "-v"], capture_output=True, text=True)
    queues = []
    for line in r.stdout.splitlines():
        m = re.match(r"device for (\S+):\s*(.*)", line)
        if m:
            q, uri = m.group(1), m.group(2)
            queues.append({"queue": q, "uri": uri,
                           "bixolon": any(h in f"{q} {uri}".lower() for h in NAME_HINTS)})
    return queues


class CupsPrinter(Printer):
    """Send raw ESC/POS through a macOS print queue (`lp -o raw`). No libusb needed."""

    def __init__(self, queue: str):
        self.queue = queue
        self.name = f"cups:{queue}"

    def send(self, data: bytes) -> None:
        if not shutil.which("lp"):
            raise PrinterError("`lp` not found (CUPS isn't available on this computer).")
        r = subprocess.run(["lp", "-d", self.queue, "-o", "raw"], input=data, capture_output=True)
        if r.returncode != 0:
            raise PrinterError(f"CUPS refused the job: {r.stderr.decode(errors='replace').strip()}")

    def status(self) -> dict | None:
        r = subprocess.run(["lpstat", "-p", self.queue], capture_output=True, text=True)
        if r.returncode != 0:
            raise PrinterError(f"CUPS queue '{self.queue}' not found. See `lpstat -v`.")
        return None  # CUPS raw queues can't read the printer's status bytes


# --------------------------------------------------------------------------- network / serial / file

class NetPrinter(Printer):
    def __init__(self, host: str, port: int = 9100):
        self.host, self.port = host, port
        self.name = f"net:{host}:{port}"

    def _connect(self):
        try:
            return socket.create_connection((self.host, self.port), timeout=5)
        except OSError as e:
            raise PrinterError(f"Can't reach {self.host}:{self.port} ({e}).")

    def send(self, data: bytes) -> None:
        with self._connect() as s:
            s.settimeout(30)
            try:
                s.sendall(data)
            except OSError as e:
                raise PrinterError(f"Network send failed ({e}).", partial=True)

    def status(self) -> dict | None:
        with self._connect() as s:
            s.settimeout(0.5)
            raw = {}
            for q, cmd in commands.STATUS_QUERIES.items():
                s.sendall(cmd)
                try:
                    resp = s.recv(64)
                    raw[q] = resp[-1] if resp else None
                except OSError:
                    raw[q] = None
            return commands.parse_status(raw)


class SerialPrinter(Printer):
    """USB virtual-COM (/dev/cu.usbmodem*) or real serial port, using only the stdlib."""

    def __init__(self, path: str):
        self.path = path
        self.name = f"serial:{path}"

    def _open(self):
        import termios
        import tty
        try:
            fd = os.open(self.path, os.O_RDWR | os.O_NOCTTY)
        except OSError as e:
            raise PrinterError(f"Can't open {self.path} ({e}).")
        tty.setraw(fd)  # no newline translation: image bytes must pass untouched
        attrs = termios.tcgetattr(fd)
        attrs[4] = attrs[5] = termios.B115200  # ignored by USB virtual-COM, needed by real RS-232
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        return fd

    def send(self, data: bytes) -> None:
        fd = self._open()
        sent = 0
        try:
            while sent < len(data):
                sent += os.write(fd, data[sent:sent + USB_CHUNK])
        except OSError as e:
            raise PrinterError(f"Serial write failed after {sent} bytes ({e}).", partial=sent > 0)
        finally:
            os.close(fd)

    def status(self) -> dict | None:
        fd = self._open()
        try:
            raw = {}
            for q, cmd in commands.STATUS_QUERIES.items():
                os.write(fd, cmd)
                ready, _, _ = select.select([fd], [], [], 0.5)
                resp = os.read(fd, 64) if ready else b""
                raw[q] = resp[-1] if resp else None
            return commands.parse_status(raw)
        finally:
            os.close(fd)


class FilePrinter(Printer):
    def __init__(self, path: str):
        self.path = path
        self.name = f"file:{path}"

    def send(self, data: bytes) -> None:
        with open(self.path, "wb") as fh:
            fh.write(data)


# --------------------------------------------------------------------------- choose / detect

def open_printer(spec: str = "auto") -> Printer:
    spec = (spec or "auto").strip()
    kind, _, arg = spec.partition(":")
    kind = kind.lower()
    if kind == "usb":
        if not arg:
            return UsbPrinter()
        vid, _, pid = arg.partition(":")
        return UsbPrinter(int(vid, 16), int(pid, 16) if pid else None)
    if kind == "cups":
        return CupsPrinter(arg)
    if kind == "net":
        host, _, port = arg.partition(":")
        return NetPrinter(host, int(port or 9100))
    if kind == "serial":
        return SerialPrinter(arg)
    if kind == "file":
        return FilePrinter(arg or "out.bin")
    if kind != "auto":
        raise PrinterError(f"Unknown printer spec '{spec}'.")

    if usb_backend() is not None and any(d["bixolon"] or d["printer_class"] for d in list_usb()):
        return UsbPrinter()
    cups = [q for q in list_cups() if q["bixolon"]]
    if cups:
        return CupsPrinter(cups[0]["queue"])
    raise PrinterError(
        "Couldn't find the printer automatically. Run `python3 thermal.py detect` and then pass "
        "--printer usb / cups:NAME / serial:/dev/cu.xxx explicitly.")


def serial_ports() -> list[str]:
    pats = ("/dev/cu.usb*", "/dev/tty.usb*", "/dev/ttyUSB*", "/dev/ttyACM*")
    return sorted({p for pat in pats for p in glob.glob(pat)})


def macos_usb_report() -> list[str]:
    """USB devices as macOS itself sees them - works without libusb."""
    if platform.system() != "Darwin" or not shutil.which("ioreg"):
        return []
    r = subprocess.run(["ioreg", "-p", "IOUSB", "-l", "-w0"], capture_output=True, text=True)
    found, name, vid, pid = [], None, None, None

    def flush():
        if name:
            found.append(f"{name}  (vendor 0x{vid or 0:04x}, product 0x{pid or 0:04x})")

    for line in r.stdout.splitlines():
        m = re.search(r"\+-o (.+?)@", line)
        if m:
            flush()
            name, vid, pid = m.group(1).strip(), None, None
        m = re.search(r'"idVendor" = (\d+)', line)
        if m:
            vid = int(m.group(1))
        m = re.search(r'"idProduct" = (\d+)', line)
        if m:
            pid = int(m.group(1))
    flush()
    return found


def detect() -> str:
    """Plain-text report of every route to the printer this computer has."""
    lines = [f"System: {platform.system()} {platform.release()} ({platform.machine()}), "
             f"Python {platform.python_version()}", ""]

    lines.append("1) USB devices seen by macOS (no extra software needed):")
    mac = macos_usb_report()
    if platform.system() != "Darwin":
        lines.append("   (not macOS - skipped)")
    for entry in mac:
        mark = "  <-- BIXOLON" if (f"0x{BIXOLON_VID:04x}" in entry or any(h in entry.lower() for h in NAME_HINTS)) else ""
        lines.append(f"   - {entry}{mark}")
    if platform.system() == "Darwin" and not mac:
        lines.append("   (none found)")

    lines += ["", "2) USB via pyusb/libusb (what `--printer usb` uses):"]
    try:
        import usb  # noqa: F401
        have_pyusb = True
    except ImportError:
        have_pyusb = False
    if not have_pyusb:
        lines.append("   pyusb is not installed  ->  pip install -r requirements.txt")
    elif usb_backend() is None:
        lines.append("   libusb not found  ->  brew install libusb")
    else:
        devs = list_usb()
        for d in devs:
            tag = "  <-- BIXOLON" if d["bixolon"] else ("  <-- printer" if d["printer_class"] else "")
            label = " ".join(x for x in (d["manufacturer"], d["product"]) if x) or "(no name)"
            lines.append(f"   - {d['vid']:04x}:{d['pid']:04x}  {label}{tag}")
        if not devs:
            lines.append("   (no devices)")

    lines += ["", "3) Printers added to macOS / CUPS (what `--printer cups:NAME` uses):"]
    queues = list_cups()
    for q in queues:
        lines.append(f"   - {q['queue']}  ->  {q['uri']}" + ("  <-- BIXOLON" if q["bixolon"] else ""))
    if not queues:
        lines.append("   (none)")

    lines += ["", "4) Serial / virtual-COM ports (what `--printer serial:PATH` uses):"]
    ports = serial_ports()
    lines += [f"   - {p}" for p in ports] or ["   (none)"]

    lines += ["", "Auto choice:"]
    try:
        p = open_printer("auto")
        if isinstance(p, UsbPrinter):
            p._find()
        lines.append(f"   --printer auto  ->  {p.name}")
    except PrinterError as e:
        lines.append(f"   {e}")
    return "\n".join(lines)

