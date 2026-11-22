# Copyright (c) 2026 aditya-forge. All rights reserved. Proprietary; see LICENSE.
"""Write the brand assets from the geometry in app/portal/icons.py.

    python scripts/build_brand_assets.py

Outputs app/assets/logo.svg (README, docs) and app/assets/favicon.png (browser tab).
The favicon is drawn at 8x and downsampled, so the line stays crisp at 16 and 32 px.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from portal.icons import LOGO_BLUE, LOGO_GRID, LOGO_NODE, LOGO_STROKES, LOGO_TILE_RADIUS, logo  # noqa: E402

ASSETS = ROOT / "app" / "assets"


def render_png(size: int = 256, oversample: int = 8) -> Image.Image:
    big = size * oversample
    k = big / LOGO_GRID
    tile = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    ImageDraw.Draw(tile).rounded_rectangle(
        [0, 0, big - 1, big - 1], radius=LOGO_TILE_RADIUS * k, fill=LOGO_BLUE
    )
    for points, width, opacity in LOGO_STROKES:
        layer = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        ImageDraw.Draw(layer).line(
            [(x * k, y * k) for x, y in points],
            width=round(width * k),
            fill=(255, 255, 255, 255),
            joint="curve",
        )
        if opacity < 1:
            layer.putalpha(layer.getchannel("A").point(lambda a: round(a * opacity)))
        tile = Image.alpha_composite(tile, layer)
    cx, cy, side = LOGO_NODE or (0, 0, 0)
    if side > 0:
        ImageDraw.Draw(tile).rectangle(
            [(cx - side / 2) * k, (cy - side / 2) * k, (cx + side / 2) * k - 1, (cy + side / 2) * k - 1],
            fill=(255, 255, 255, 255),
        )
    return tile.resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    # one element per line in the file; the inline version in the app stays compact
    svg = logo(64).replace("><", ">\n  <").replace("\n  </svg>", "\n</svg>")
    (ASSETS / "logo.svg").write_text(svg + "\n", encoding="utf-8")
    render_png(256).save(ASSETS / "favicon.png", optimize=True)
    print(f"wrote {ASSETS / 'logo.svg'} and {ASSETS / 'favicon.png'}")


if __name__ == "__main__":
    main()
