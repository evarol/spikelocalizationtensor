import base64
import html
import json
import os
import sys


def build(gallery_dir, out_name="index_standalone.html"):
    meta_path = os.path.join(gallery_dir, "index.html")
    pngs = []
    for root, _, files in os.walk(gallery_dir):
        for f in sorted(files):
            if f.endswith(".png"):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, gallery_dir)
                pngs.append((rel, full))
    pngs.sort()
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>", html.escape(os.path.basename(gallery_dir)),
        " - standalone gallery</title>",
        "<style>body{font-family:sans-serif;background:#fff;color:#14171c;",
        "margin:20px}h2{margin:18px 0 6px}img{max-width:100%;height:auto;",
        "border:1px solid #ddd;margin:4px 0}p{color:#555}</style></head><body>",
        "<h1>", html.escape(os.path.basename(gallery_dir)), "</h1>",
        "<p>Self-contained gallery: all figures embedded, works offline and",
        " inside Jupyter. Source: index.html (interactive browser).</p>",
    ]
    for rel, full in pngs:
        with open(full, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        parts.append("<h2>%s</h2>" % html.escape(rel))
        parts.append("<img src='data:image/png;base64,%s'>" % b64)
    parts.append("</body></html>")
    out = os.path.join(gallery_dir, out_name)
    with open(out, "w") as fh:
        fh.write("".join(parts))
    print("wrote %s (%.1f MB, %d figures)" % (out, os.path.getsize(out) / 1e6, len(pngs)))


if __name__ == "__main__":
    build(sys.argv[1])
