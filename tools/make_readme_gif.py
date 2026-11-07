#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_readme_gif.py
==================

Records figure A of the atlas page (default view: interval, Gauss-Legendre,
n = 2, "dancing" polynomial) as an animated GIF for the README.

How it works
------------
* Opens the built page site/index.html (run `python tools/build_atlas.py`
  first) in headless Chromium via Playwright.
* Freezes the page clock (Playwright clock API) and advances it frame by
  frame, so every frame shows exactly the intended moment of the animation,
  independent of the speed of the machine.
* Crops the canvas of figure A to the part that contains the drawing.
* The animation never repeats exactly (incommensurate frequencies), so the
  clip is played forwards and then backwards, which gives a seamless loop.

Requirements: pip install playwright pillow ; a Chromium or Chrome browser
(`python -m playwright install chromium`, or pass --chrome /path/to/chrome).

Usage
-----
   python tools/build_atlas.py
   python tools/make_readme_gif.py --out assets/figure_a_interval_gauss_n2.gif

Style rule of this file: one command per line.
"""

# Imports.
import argparse
import io
import os
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

# Repository root (this file lives in tools/).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def record(page_path, chrome, seconds, fps, crop_height, width):
    """Return the list of frames (PIL images) of figure A."""
    # Frames.
    frames = []
    # Milliseconds between two frames.
    step = 1000.0 / fps
    # Number of frames.
    count = int(round(seconds * fps))
    # Start Playwright.
    with sync_playwright() as pw:
        # Browser options.
        opts = {"args": ["--no-sandbox"]}
        # Use a given browser executable if requested.
        if chrome:
            opts["executable_path"] = chrome
        # Launch.
        browser = pw.chromium.launch(**opts)
        # New page; light theme and English as in the default view.
        page = browser.new_page(viewport={"width": 1300, "height": 1000}, device_scale_factor=1, color_scheme="light", locale="en-US")
        # Freeze the clock before the page script starts.
        page.clock.install(time=0)
        # Load the page.
        page.goto("file://" + os.path.abspath(page_path))
        # Make sure the English texts are shown.
        page.click("#l-en")
        # Let the page settle (layout, fonts, first frames).
        page.clock.run_for(500)
        # Canvas of figure A.
        canvas = page.locator("#cv")
        # Previous screenshot (to detect frames that were not repainted yet).
        prev = None
        # Number of retaken screenshots.
        retakes = 0
        # Loop over the frames.
        for _ in range(count):
            # Advance the animation by one frame interval.
            page.clock.run_for(int(round(step)))
            # Screenshot of the canvas.
            png = canvas.screenshot()
            # Chrome sometimes has not composited the new canvas content yet: wait briefly (real time) and retake.
            tries = 0
            # Retake while the image is unchanged.
            while png == prev and tries < 10:
                # Real-time pause.
                page.wait_for_timeout(60)
                # Retake.
                png = canvas.screenshot()
                # Count.
                tries += 1
                # Statistics.
                retakes += 1
            # Remember the screenshot.
            prev = png
            # Decode and store.
            frames.append(Image.open(io.BytesIO(png)).convert("RGB"))
        # Report the retakes.
        print("frames: %d, retaken screenshots: %d" % (count, retakes))
        # Close the browser.
        browser.close()
    # Crop height: given, or the lowest drawn row over all frames plus a margin.
    height = crop_height if crop_height > 0 else auto_crop_height(frames)
    # Crop (the lower part of the canvas is empty in 1D) and optionally downscale.
    out = []
    # Loop over the frames.
    for img in frames:
        # Crop.
        img = img.crop((0, 0, img.width, min(height, img.height)))
        # Optional downscaling.
        if width and width < img.width:
            img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
        # Store.
        out.append(img)
    # Return the frames.
    return out


def auto_crop_height(frames, margin=18):
    """Lowest row that differs from the background in any frame, plus a margin."""
    # Lowest drawn row found so far.
    lowest = 0
    # Loop over the frames.
    for img in frames:
        # Grey values.
        a = np.asarray(img.convert("L"), dtype=float)
        # Background: the centre of the bottom rows (empty in every view).
        bg = np.median(a[-20:-8, a.shape[1] // 4: 3 * a.shape[1] // 4])
        # Rows with clearly drawn pixels, ignoring the frame border columns.
        rows = np.where(np.any(np.abs(a[:-8, 8:-8] - bg) > 30, axis=1))[0]
        # Update.
        if rows.size:
            lowest = max(lowest, int(rows.max()))
    # Add the margin.
    return lowest + margin


def save_gif(frames, path, fps):
    """Write a forwards-backwards looping GIF with one shared palette."""
    # Forwards, then backwards without repeating the end frames.
    seq = frames + frames[-2:0:-1]
    # Shared adaptive palette from a few representative frames (stable colours, small file).
    probe = Image.new("RGB", (seq[0].width, seq[0].height * 4))
    # Paste four frames spread over the clip.
    for k in range(4):
        # Frame index.
        idx = k * (len(frames) - 1) // 3
        # Paste.
        probe.paste(frames[idx], (0, k * seq[0].height))
    # Palette image with 64 colours.
    pal = probe.quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    # Quantize every frame with the shared palette (no dithering: flat colours compress well).
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in seq]
    # Create the output folder.
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Write the GIF.
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(round(1000.0 / fps)), loop=0, optimize=True, disposal=1)


def main():
    """Parse the arguments, record and save."""
    # Argument parser.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Built page.
    ap.add_argument("--page", default=os.path.join(ROOT, "site", "index.html"))
    # Output file.
    ap.add_argument("--out", default=os.path.join(ROOT, "assets", "figure_a_interval_gauss_n2.gif"))
    # Browser executable (empty: Playwright's own Chromium).
    ap.add_argument("--chrome", default="")
    # Length of the forward clip in seconds.
    ap.add_argument("--seconds", type=float, default=7.0)
    # Frames per second.
    ap.add_argument("--fps", type=float, default=15.0)
    # Height of the cropped canvas part in pixels (0 = automatic: lowest drawn row plus a margin).
    ap.add_argument("--crop", type=int, default=0)
    # Output width in pixels (0 = canvas width).
    ap.add_argument("--width", type=int, default=0)
    # Parse.
    args = ap.parse_args()
    # Record.
    frames = record(args.page, args.chrome, args.seconds, args.fps, args.crop, args.width)
    # Save.
    save_gif(frames, args.out, args.fps)
    # Report.
    print("%s: %d frames, %.0f kB" % (args.out, 2 * len(frames) - 2, os.path.getsize(args.out) / 1024.0))


# Entry point.
if __name__ == "__main__":
    main()
