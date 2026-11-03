#!/usr/bin/env python3
"""One-step setup + printer check. Run it, then send the output to Claude.

    python3 setup_check.py

It will:
  1. show your Mac / Python details
  2. install libusb with Homebrew (if Homebrew is there and libusb is missing)
  3. create .venv and install the Python packages
  4. look for the printer on USB, CUPS and serial, and read its status
  5. offer to print a test page
Everything is also saved to setup-report.txt.
"""
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(ROOT, ".venv")
VPY = os.path.join(VENV, "bin", "python")
REPORT = os.path.join(ROOT, "setup-report.txt")
LIBUSB_PATHS = ("/opt/homebrew/lib/libusb-1.0.dylib", "/usr/local/lib/libusb-1.0.dylib",
                "/opt/local/lib/libusb-1.0.dylib")
_lines = []


def log(msg=""):
    print(msg, flush=True)
    _lines.append(msg)


def run(cmd, timeout=600, show=True, tail=15):
    """Run a command, log its (last) output lines, return (exit_code, output)."""
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout + r.stderr).strip()
        code = r.returncode
    except FileNotFoundError:
        out, code = f"command not found: {cmd[0]}", 127
    except subprocess.TimeoutExpired:
        out, code = f"timed out after {timeout}s", 124
    if show and out:
        lines = out.splitlines()
        if len(lines) > tail:
            log(f"     ... ({len(lines) - tail} lines hidden)")
        for line in lines[-tail:]:
            log("     " + line)
    return code, out


def step(title):
    log("")
    log("=" * 64)
    log(title)
    log("=" * 64)


# ---------------------------------------------------------------- probe (runs inside .venv)

def probe():
    """Low-level USB details of anything that looks like a printer."""
    sys.path.insert(0, ROOT)
    from photo_printer import printers
    be = printers.usb_backend()
    print(f"libusb backend: {'found' if be else 'NOT FOUND'}")
    if not be:
        return
    import usb.core
    import usb.util
    devs = [d for d in printers.list_usb() if d["bixolon"] or d["printer_class"]]
    if not devs:
        print("No printer-like USB device visible to libusb.")
    for d in devs:
        dev = d["_dev"]
        print(f"Device {d['vid']:04x}:{d['pid']:04x}  '{d['manufacturer']}' '{d['product']}'  "
              f"USB {dev.bcdUSB:04x} class {dev.bDeviceClass}")
        try:
            for cfg in dev:
                print(f"  config {cfg.bConfigurationValue}")
                for intf in cfg:
                    print(f"    interface {intf.bInterfaceNumber} alt {intf.bAlternateSetting} "
                          f"class {intf.bInterfaceClass}/{intf.bInterfaceSubClass}/{intf.bInterfaceProtocol}")
                    for ep in intf:
                        kind = {0: "control", 1: "iso", 2: "bulk", 3: "interrupt"}[ep.bmAttributes & 3]
                        way = "IN" if ep.bEndpointAddress & 0x80 else "OUT"
                        print(f"      endpoint 0x{ep.bEndpointAddress:02x} {kind} {way} max {ep.wMaxPacketSize}")
                    if intf.bInterfaceClass == 7:
                        try:  # IEEE-1284 device ID: names the model and command set
                            raw = bytes(dev.ctrl_transfer(0xA1, 0, cfg.bConfigurationValue - 1,
                                                          intf.bInterfaceNumber << 8 | intf.bAlternateSetting,
                                                          1024, 2000))
                            print(f"      device id: {raw[2:].decode('ascii', 'replace')}")
                        except Exception as e:
                            print(f"      device id: (could not read: {e})")
        except Exception as e:
            print(f"  (could not read descriptors: {e})")


# ---------------------------------------------------------------- main

def main():
    log("BIXOLON photo printer - setup check")
    ok = {"venv": False, "libusb": False}

    step("1. This computer")
    log(f"  OS:      {platform.system()} {platform.mac_ver()[0] or platform.release()} ({platform.machine()})")
    log(f"  Python:  {platform.python_version()}  at {sys.executable}")
    log(f"  Folder:  {ROOT}")
    if sys.version_info < (3, 9):
        log("  [!!] Python 3.9 or newer is needed. Install it from python.org or `brew install python`.")
        return finish(ok)
    run(["git", "log", "--oneline", "-1"])

    step("2. libusb (for direct USB printing)")
    brew = shutil.which("brew") or next((p for p in ("/opt/homebrew/bin/brew", "/usr/local/bin/brew")
                                         if os.path.exists(p)), None)
    have = [p for p in LIBUSB_PATHS if os.path.exists(p)]
    if have:
        log(f"  [OK] libusb found: {have[0]}")
    elif platform.system() != "Darwin":
        log("  [--] not macOS: install libusb with your package manager (e.g. apt install libusb-1.0-0)")
    elif brew:
        log("  installing libusb with Homebrew ...")
        code, _ = run([brew, "install", "libusb"])
        have = [p for p in LIBUSB_PATHS if os.path.exists(p)]
        log(f"  [OK] libusb installed: {have[0]}" if have else "  [!!] libusb install failed (see above)")
    else:
        log("  [!!] Homebrew not found, so libusb can't be installed automatically.")
        log("       Direct USB won't work yet; the script will still look for other routes.")
    ok["libusb"] = bool(have)

    step("3. Python environment (.venv)")
    if not os.path.exists(VPY):
        log("  creating .venv ...")
        run([sys.executable, "-m", "venv", VENV])
    if not os.path.exists(VPY):
        log("  [!!] could not create .venv (see above)")
        return finish(ok)
    log("  installing packages (pillow, numpy, pyusb) ...")
    run([VPY, "-m", "pip", "install", "--quiet", "--upgrade", "pip"], show=False)
    code, _ = run([VPY, "-m", "pip", "install", "--quiet", "-r", "requirements.txt"])
    code, out = run([VPY, "-c", "import PIL, numpy, usb; print('pillow', PIL.__version__, "
                                "'| numpy', numpy.__version__, '| pyusb', usb.__version__)"])
    ok["venv"] = code == 0
    log("  [OK] packages ready" if ok["venv"] else "  [!!] packages missing (see above)")
    if not ok["venv"]:
        return finish(ok)

    step("4. Looking for the printer")
    run([VPY, "thermal.py", "detect"], timeout=60, tail=60)

    step("5. USB details")
    run([VPY, os.path.abspath(__file__), "--probe"], timeout=60, tail=60)

    step("6. Printer status (detailed)")
    code, out = run([VPY, "thermal.py", "status", "--debug"], timeout=60, tail=30)
    found = code == 0

    step("7. Test page")
    if not found:
        log("  [--] skipped: printer not reachable yet")
    elif not sys.stdin.isatty():
        log("  [--] skipped (not interactive). Run:  .venv/bin/python thermal.py test")
    else:
        log("  Before answering: lid closed, paper loaded, green light on?")
        ans = input("  Print a test page now? It uses about 15 cm of paper. [y/N] ").strip().lower()
        _lines.append(f"  Print a test page now? -> {ans or 'n'}")
        if ans in ("y", "yes"):
            # --no-check: send it even if the status reply looks wrong, to see what really happens
            code, _ = run([VPY, "thermal.py", "test", "--no-check"], timeout=120)
            if code == 0:
                ans = input("  Did a page come out with text and 5 gray bars? [y/n] ").strip().lower()
                _lines.append(f"  Page came out OK? -> {ans}")
        else:
            log("  skipped")
    return finish(ok)


def finish(ok):
    step("Done")
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")
    log(f"  Report saved to: {REPORT}")
    log("  Copy everything above (or that file) and send it to Claude.")
    return 0


if __name__ == "__main__":
    if "--probe" in sys.argv:
        probe()
    else:
        sys.exit(main())
