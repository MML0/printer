#!/usr/bin/env python3
"""Photo printing for the BIXOLON SRP-350III (and other ESC/POS printers).

    python3 thermal.py detect                  # can this Mac see the printer?
    python3 thermal.py status                  # paper / cover / error state
    python3 thermal.py test                    # print a calibration page
    python3 thermal.py preview photo.jpg       # write photo-preview.png, no printing
    python3 thermal.py print photo.jpg --dither atkinson --brightness 1.1
    python3 thermal.py app                     # open the web app in your browser
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from photo_printer import imaging, jobs, printers
from photo_printer.commands import blocking_problem


def add_image_args(p: argparse.ArgumentParser) -> None:
    d = imaging.Settings()
    g = p.add_argument_group("image settings")
    g.add_argument("--width", type=int, default=d.width, help="dots (default 512 = full 80 mm roll)")
    g.add_argument("--brightness", type=float, default=d.brightness, help="1.0 = unchanged")
    g.add_argument("--contrast", type=float, default=d.contrast, help="1.0 = unchanged")
    g.add_argument("--gamma", type=float, default=d.gamma, help=">1 lightens midtones")
    g.add_argument("--sharpness", type=float, default=d.sharpness, help="1.0 = unchanged")
    g.add_argument("--dither", choices=imaging.DITHERS, default=d.dither)
    g.add_argument("--threshold", type=int, default=d.threshold, help="for --dither threshold")
    g.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=d.rotate)
    g.add_argument("--invert", action="store_true")
    g.add_argument("--no-autocontrast", dest="auto_contrast", action="store_false")


def settings_from(args) -> imaging.Settings:
    return imaging.Settings.from_dict(vars(args))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--printer", default="auto",
                    help="auto | usb | usb:VID:PID | cups:QUEUE | net:IP | serial:/dev/cu.X | file:out.bin")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("detect", help="list every way this computer can reach the printer")
    sub.add_parser("status", help="read paper/cover/error status (USB, network, serial)")
    t = sub.add_parser("test", help="print a calibration page")
    t.add_argument("--width", type=int, default=512)
    sub.add_parser("cut", help="feed a little paper and cut")

    pv = sub.add_parser("preview", help="save the dithered result as PNG (no printing)")
    pv.add_argument("image")
    pv.add_argument("-o", "--output")
    add_image_args(pv)

    pr = sub.add_parser("print", help="print a photo")
    pr.add_argument("image")
    pr.add_argument("--no-cut", dest="cut", action="store_false")
    pr.add_argument("--band", type=int, default=256, help="rows per raster command (try 64 if prints glitch)")
    add_image_args(pr)

    app = sub.add_parser("app", help="start the web app")
    app.add_argument("--port", type=int, default=8350)
    app.add_argument("--no-browser", action="store_true")

    args = ap.parse_args(argv)

    try:
        if args.cmd == "detect":
            print(printers.detect())
            return 0

        if args.cmd == "preview":
            s = settings_from(args)
            bitmap = imaging.render(imaging.load_image(args.image), s)
            out = args.output or f"{Path(args.image).stem}-preview.png"
            bitmap.save(out)
            w_mm, h_mm = imaging.printed_size_mm(bitmap)
            print(f"Saved {out}: {bitmap.width}x{bitmap.height} dots = {w_mm:.0f} x {h_mm:.0f} mm on paper")
            return 0

        if args.cmd == "app":
            from photo_printer import webapp
            webapp.serve(args.printer, args.port, open_browser=not args.no_browser)
            return 0

        printer = printers.open_printer(args.printer)

        if args.cmd == "status":
            st = printer.status()
            if st is None:
                print(f"{printer.name}: this connection can't read printer status (that's normal for CUPS).")
            else:
                print(f"{printer.name}:\n{json.dumps(st, indent=2)}")
                problem = blocking_problem(st)
                print("\n" + (problem or ("Ready to print." if st["responded"] else
                                          "No status reply - printing may still work.")))
            return 0
        if args.cmd == "test":
            job = jobs.print_test_page(printer, args.width)
        elif args.cmd == "cut":
            job = jobs.feed_and_cut(printer)
        else:  # print
            job = jobs.print_photo(printer, args.image, settings_from(args), cut=args.cut, band=args.band)
        print(f"Sent job {job['id']} to {job['printer']} ({job['bytes']} bytes"
              + (f", {job['length_mm']} mm of paper" if "length_mm" in job else "") + ").")
        return 0

    except printers.PrinterError as e:
        print(f"Printer error: {e}", file=sys.stderr)
        if e.partial:
            print("The job may be half-printed: check the paper before printing again.", file=sys.stderr)
        return 2
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
