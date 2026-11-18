"""Pack the rendered flame frames into one flipbook atlas.

    python3 tools/pack_flames.py

Reads blender/flame_frames/*.png (rendered at 2x), writes
assets/flames.webp + assets/flames-meta.json, and a contact sheet preview.
Also checks that every cell's border is black, so no faint rectangles show
when the page adds the flames on top of the scene.
"""
import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
SRC = os.path.join(ROOT, "blender", "flame_frames")
OUT = os.path.join(ROOT, "assets")
COLS = 8


def main():
    with open(os.path.join(SRC, "frames.json")) as f:
        info = json.load(f)
    n, cw, ch = info["frames"], info["cw"], info["ch"]
    rows = (n + COLS - 1) // COLS
    atlas = Image.new("RGB", (COLS * cw, rows * ch), (0, 0, 0))
    worst_edge = 0
    frames = []
    for i in range(n):
        im = Image.open(os.path.join(SRC, "flame_%03d.png" % i)).convert("RGB")
        frames.append(np.asarray(im.resize((cw, ch), Image.LANCZOS)).astype(np.int16))
    # where the fire actually starts, measured from the render itself:
    # the lowest row that is bright in a typical frame
    bottoms = [int(np.nonzero(f.max(axis=(1, 2)) > 60)[0].max()) for f in frames]
    base_row = float(np.median(bottoms)) + 1
    for i, a in enumerate(frames):
        # soften anything touching the cell edge to black over the outer 6px
        fade = np.ones((ch, cw))
        ramp = np.linspace(0, 1, 7)[:6]
        for k in range(6):
            fade[k, :] *= ramp[k]
            fade[:, k] *= ramp[k]
            fade[:, cw - 1 - k] *= ramp[k]
        # the burning strip is a hard-edged block in the render: fade the
        # flame in over the 34px above it, so it rises softly out of the edge
        rise = np.clip((base_row - np.arange(ch)) / 34.0, 0, 1)
        fade *= (rise * rise * (3 - 2 * rise))[:, None]
        # tall tongues that reach the top of the cell thin out instead of clipping
        top = np.clip(np.arange(ch) / 22.0, 0, 1)
        fade *= (top * top * (3 - 2 * top))[:, None]
        edge = max(a[0].max(), a[:, 0].max(), a[:, -1].max())
        worst_edge = max(worst_edge, int(edge))
        a = (a * fade[:, :, None]).astype(np.uint8)
        atlas.paste(Image.fromarray(a), ((i % COLS) * cw, (i // COLS) * ch))

    path = os.path.join(OUT, "flames.webp")
    atlas.save(path, "WEBP", quality=82, method=6)
    meta = {"cols": COLS, "rows": rows, "cw": cw, "ch": ch, "frames": n,
            "fps": info["fps"], "base": round((base_row - 14) / ch, 4), "source": info["source"]}
    with open(os.path.join(OUT, "flames-meta.json"), "w") as f:
        json.dump(meta, f)
    atlas.save(os.path.join(OUT, "preview-flames.png"))
    print("fire base at row %.0f of %d" % (base_row, ch))
    print("atlas %dx%d, %d frames, %d KB" % (atlas.width, atlas.height, n, os.path.getsize(path) // 1024))
    print("brightest pixel on a top/side cell edge before fading: %d / 255" % worst_edge)

    # decode what the page will actually get and look at the borders
    back = np.asarray(Image.open(path).convert("RGB")).astype(int)
    top = max(back[r * ch].max() for r in range(rows))
    side = max(back[:, c * cw].max() for c in range(COLS))
    print("after WebP: brightest top-edge pixel %d, left-edge pixel %d (page ignores anything below 9)" % (top, side))


if __name__ == "__main__":
    main()
