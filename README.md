# printer

Print photos on a **BIXOLON SRP-350III** (80 mm thermal receipt printer, 180 dpi, 512 dots wide)
from a Mac, with live preview and controls for dithering, brightness, contrast, gamma and sharpness.

## 1. Install (once)

```bash
brew install libusb
```

```bash
cd printer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No Homebrew? Skip `libusb` and use the CUPS route in step 2 instead.

## 2. Check the Mac can see the printer

Turn the printer on, plug in USB, then:

```bash
python3 thermal.py detect
```

It lists the printer four ways: what macOS sees on USB, what libusb sees, print queues
added in System Settings, and serial ports. The `Auto choice` line at the bottom says which one
the app will use. If auto can't find it, pick one by hand with `--printer`:

| `--printer`               | when to use it                                                        |
|---------------------------|-----------------------------------------------------------------------|
| `usb`                     | direct USB through libusb (best: can read paper/cover status)         |
| `cups:QUEUE`              | printer added in System Settings → Printers (no libusb needed)         |
| `serial:/dev/cu.usbmodemX`| printer set to USB virtual-COM mode                                    |
| `net:192.168.1.50`        | Ethernet model                                                        |
| `file:out.bin`            | dry run, writes the bytes to a file                                   |

## 3. Test print

```bash
python3 thermal.py status
```

```bash
python3 thermal.py test
```

The test page prints an edge ruler (both thick end bars must be visible, otherwise lower
`--width`) and the same gray ramp with all five dither modes, so you can see on real paper
which one you like.

## 4. The app

```bash
python3 thermal.py app
```

Opens `http://127.0.0.1:8350` in your browser. Drop a photo, move the sliders, press **Print**.

## Command line

```bash
python3 thermal.py preview photo.jpg --dither atkinson --gamma 1.3
```

```bash
python3 thermal.py print photo.jpg --dither floyd_steinberg --brightness 1.1 --contrast 1.2
```

All settings: `--width --brightness --contrast --gamma --sharpness --dither --threshold
--rotate --invert --no-autocontrast`, plus `--no-cut` and `--band` for printing.

## How it avoids failed prints

- **Same code for preview and print**: what you see is the exact bitmap that gets sent.
- **Status check before every job** (USB, network, serial): refuses to start if the cover is
  open, the paper is out, or the cutter is jammed, and tells you which.
- **USB is opened per job**: unplugging and replugging the printer between prints is fine.
- **Endpoints are discovered automatically** instead of hard-coded USB addresses.
- **Image sent in bands** of 256 rows (`GS v 0`), in 4 KB USB chunks with a 20 s timeout.
- **No blind retries**: if a transfer dies mid-way the error says how many bytes went out
  and that the paper may hold half a print, so you don't get duplicates.
- **One job at a time** in the app, and anything longer than ~56 cm is refused.
- Every job is logged to `print-jobs.log`.

## Tips for photos on thermal paper

- Thermal dots spread a little, so prints come out **darker than the preview**. Start with
  `gamma` 1.2-1.4.
- **Floyd-Steinberg** is the safe default for photos, **Atkinson** gives a punchy retro look
  and stays lighter, **Bayer** gives a regular pattern, **Threshold** is for logos and text.
- Use **Rotate 90°** for landscape photos so they fill the roll width.
- Physical print darkness is a printer setting (DIP switches, see the density table in the
  SRP-350III user manual; switch the printer off first). It is separate from the brightness slider.

## Tests

```bash
python3 -m unittest discover tests
```
# printer
