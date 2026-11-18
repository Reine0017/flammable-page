"""Inline the baked assets into index.html as data URIs.

    python3 tools/embed_assets.py            # quiet smoulder (default)
    python3 tools/embed_assets.py --flames   # also inline the flame flipbook

The page is opened straight from disk, where browsers refuse file images
as WebGL textures, so everything rides inside the HTML (like the font).
Rewrites only the block between /*ASSETS-BEGIN*/ and /*ASSETS-END*/.
"""
import base64
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
ASSETS = os.path.join(ROOT, "assets")
PAGE = os.path.join(ROOT, "index.html")
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp"}


def data_uri(name):
    path = os.path.join(ASSETS, name)
    with open(path, "rb") as f:
        raw = f.read()
    mime = MIME[os.path.splitext(name)[1]]
    return "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode("ascii")), len(raw)


def main():
    total = 0
    out = {}
    for key, name in (("arrival", "burn-arrival.png"), ("ramp", "char-ramp.png"), ("char", "char-texture.jpg")):
        out[key], size = data_uri(name)
        total += size
        print("%-8s %-22s %5d KB" % (key, name, size // 1024))
    with open(os.path.join(ASSETS, "burn-meta.json")) as f:
        out["meta"] = json.load(f)

    out["flames"] = None
    out["flamesMeta"] = None
    # flames are off by default (the look is a quiet smoulder);
    # pass --flames to inline the flipbook and switch them back on
    for name in ("flames.webp", "flames.jpg", "flames.png"):
        if "--flames" in sys.argv and os.path.exists(os.path.join(ASSETS, name)):
            out["flames"], size = data_uri(name)
            total += size
            with open(os.path.join(ASSETS, "flames-meta.json")) as f:
                out["flamesMeta"] = json.load(f)
            print("%-8s %-22s %5d KB" % ("flames", name, size // 1024))
            break

    block = "/*ASSETS-BEGIN*/\nconst BURN_ASSETS = %s;\n/*ASSETS-END*/" % json.dumps(out)
    with open(PAGE, encoding="utf-8") as f:
        html = f.read()
    new, n = re.subn(r"/\*ASSETS-BEGIN\*/.*?/\*ASSETS-END\*/", lambda m: block, html, flags=re.S)
    if n != 1:
        raise SystemExit("expected exactly one ASSETS block in index.html, found %d" % n)
    with open(PAGE, "w", encoding="utf-8") as f:
        f.write(new)
    print("assets %d KB raw -> index.html is %d KB" % (total // 1024, len(new.encode()) // 1024))
    print("arrival map source:", out["meta"].get("source"))


if __name__ == "__main__":
    main()
