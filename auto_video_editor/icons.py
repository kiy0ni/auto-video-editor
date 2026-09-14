"""Icons and small illustrations drawn with Pillow (crisp on HiDPI screens, recolorable)."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageOps

_SS = 4  # supersampling factor for smooth edges


def _rgba(color: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16), alpha


@lru_cache(maxsize=512)
def icon(name: str, color: str, px: int = 40) -> Image.Image:
    """Draw a 24-unit grid icon at ``px`` pixels."""
    n = px * _SS
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    u = n / 24.0
    width = max(1, int(round(2.0 * u)))
    c = _rgba(color)

    def pts(*points):
        return [(x * u, y * u) for x, y in points]

    def box(x0, y0, x1, y1):
        return [x0 * u, y0 * u, x1 * u, y1 * u]

    def line(*points):
        scaled = pts(*points)
        d.line(scaled, fill=c, width=width, joint="curve")
        r = width / 2
        for x, y in (scaled[0], scaled[-1]):
            d.ellipse([x - r, y - r, x + r, y + r], fill=c)

    if name == "film":
        d.rounded_rectangle(box(3, 5, 21, 19), radius=3 * u, outline=c, width=width)
        d.polygon(pts((10, 9), (10, 15), (15.5, 12)), fill=c)
    elif name == "reel":
        d.rounded_rectangle(box(3, 4.5, 14, 8), radius=1.7 * u, fill=c)
        d.rounded_rectangle(box(7, 10.25, 21, 13.75), radius=1.7 * u, fill=c)
        d.rounded_rectangle(box(3, 16, 11, 19.5), radius=1.7 * u, fill=c)
    elif name == "phone":
        d.rounded_rectangle(box(6.5, 2.5, 17.5, 21.5), radius=2.8 * u, outline=c, width=width)
        d.polygon(pts((10.5, 8.5), (10.5, 14.5), (15, 11.5)), fill=c)
        line((10.5, 18.3), (13.5, 18.3))
    elif name == "wave":
        for x, h in zip((4, 8, 12, 16, 20), (5, 11, 17, 9, 4)):
            d.rounded_rectangle(box(x - 1.15, 12 - h / 2, x + 1.15, 12 + h / 2), radius=1.15 * u, fill=c)
    elif name == "export":
        line((12, 15), (12, 3.5))
        line((7.5, 8), (12, 3.5), (16.5, 8))
        line((4, 13.5), (4, 20), (20, 20), (20, 13.5))
    elif name == "star":
        points = []
        for i in range(10):
            radius = 9.8 if i % 2 == 0 else 4.3
            angle = math.radians(-90 + i * 36)
            points.append((12 + radius * math.cos(angle), 12.8 + radius * math.sin(angle)))
        d.polygon(pts(*points), fill=c)
    elif name == "folder":
        line((3, 7), (3, 19), (21, 19), (21, 9.5), (12, 9.5), (10, 6.5), (3, 6.5), (3, 7))
    elif name == "play":
        d.polygon(pts((7, 4.5), (7, 19.5), (19.5, 12)), fill=c)
    elif name == "stop":
        d.rounded_rectangle(box(6, 6, 18, 18), radius=2.5 * u, fill=c)
    elif name == "spark":
        d.polygon(pts((11, 2.5), (13, 9), (19.5, 11), (13, 13), (11, 19.5), (9, 13), (2.5, 11), (9, 9)), fill=c)
        d.polygon(pts((19, 15), (19.9, 17.6), (22.5, 18.5), (19.9, 19.4), (19, 22), (18.1, 19.4), (15.5, 18.5),
                      (18.1, 17.6)), fill=c)
    elif name == "check":
        line((5, 12.5), (10, 17.5), (19, 7))
    elif name == "terminal":
        d.rounded_rectangle(box(3, 4, 21, 20), radius=2.8 * u, outline=c, width=width)
        line((7, 9), (10, 12), (7, 15))
        line((12.5, 15), (17, 15))
    elif name == "clock":
        d.ellipse(box(3, 3, 21, 21), outline=c, width=width)
        line((12, 7.5), (12, 12), (15.5, 14))
    elif name == "crop":
        line((6, 2.5), (6, 18), (21.5, 18))
        line((2.5, 6), (18, 6), (18, 21.5))
    elif name == "captions":
        d.rounded_rectangle(box(2.5, 5, 21.5, 19), radius=2.8 * u, outline=c, width=width)
        line((6.5, 13.5), (11, 13.5))
        line((13.5, 13.5), (17.5, 13.5))
        line((6.5, 10), (15, 10))
    elif name == "x":
        line((6, 6), (18, 18))
        line((18, 6), (6, 18))
    elif name == "gamepad":
        d.rounded_rectangle(box(2.5, 7, 21.5, 17.5), radius=5 * u, outline=c, width=width)
        line((6.5, 12.25), (10.5, 12.25))
        line((8.5, 10.25), (8.5, 14.25))
        d.ellipse(box(14.3, 9.8, 16.5, 12), fill=c)
        d.ellipse(box(16.8, 12.3, 19, 14.5), fill=c)
    elif name == "mic":
        d.rounded_rectangle(box(9, 2.5, 15, 14), radius=3 * u, outline=c, width=width)
        d.arc(box(5.5, 5.5, 18.5, 17.5), start=0, end=180, fill=c, width=width)
        line((12, 17.5), (12, 21))
        line((8.5, 21), (15.5, 21))
    elif name == "camera":
        d.rounded_rectangle(box(2.5, 7, 21.5, 20), radius=3 * u, outline=c, width=width)
        line((8, 7), (9.5, 4.5), (14.5, 4.5), (16, 7))
        d.ellipse(box(8.2, 9.7, 15.8, 17.3), outline=c, width=width)
    elif name == "layers":
        d.rounded_rectangle(box(2.5, 5, 15.5, 14.5), radius=2.2 * u, outline=c, width=width)
        d.rounded_rectangle(box(12.5, 9, 21.5, 21.5), radius=2.2 * u, fill=c)
    elif name == "sliders":
        line((4, 7), (20, 7))
        line((4, 17), (20, 17))
        d.ellipse(box(6.5, 4, 12.5, 10), fill=c)
        d.ellipse(box(12.5, 14, 18.5, 20), fill=c)
    return img.resize((px, px), Image.LANCZOS)


def _gradient(size: Tuple[int, int], top_left: str, bottom_right: str) -> Image.Image:
    w, h = size
    horizontal = Image.linear_gradient("L").rotate(90).resize((w, h))  # 0 on the left, 255 on the right
    vertical = Image.linear_gradient("L").resize((w, h))
    mask = Image.blend(horizontal, vertical, 0.5)  # diagonal: 0 at top-left, 255 at bottom-right
    return Image.composite(Image.new("RGBA", size, _rgba(bottom_right)), Image.new("RGBA", size, _rgba(top_left)), mask)


@lru_cache(maxsize=8)
def logo(px: int = 128) -> Image.Image:
    n = px * _SS
    img = _gradient((n, n), "#8b5cf6", "#ec4899")
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, n - 1, n - 1], radius=int(n * 0.27), fill=255)
    img.putalpha(mask)
    d = ImageDraw.Draw(img)
    d.polygon([(n * 0.37, n * 0.28), (n * 0.37, n * 0.72), (n * 0.75, n * 0.50)], fill=(255, 255, 255, 255))
    for i, (x0, x1) in enumerate(((0.20, 0.30), (0.20, 0.27), (0.20, 0.30))):
        y = n * (0.36 + i * 0.14)
        d.rounded_rectangle([n * x0, y - n * 0.025, n * x1, y + n * 0.025], radius=n * 0.025,
                            fill=(255, 255, 255, 170))
    return img.resize((px, px), Image.LANCZOS)


def _scene(size: Tuple[int, int]) -> Image.Image:
    """A small stylized landscape used as a stand-in video frame."""
    w, h = size
    img = _gradient((w, h), "#4338ca", "#f472b6").convert("RGB")
    d = ImageDraw.Draw(img)
    d.ellipse([w * 0.62, h * 0.16, w * 0.80, h * 0.16 + w * 0.18], fill="#fde68a")
    d.polygon([(0, h), (0, h * 0.62), (w * 0.28, h * 0.38), (w * 0.52, h * 0.66), (w * 0.52, h)], fill="#312e81")
    d.polygon([(w * 0.35, h), (w * 0.70, h * 0.48), (w, h * 0.72), (w, h)], fill="#1e1b4b")
    return img


def _person(d: ImageDraw.ImageDraw, cx: float, cy: float, s: float, color: str) -> None:
    d.ellipse([cx - 0.19 * s, cy - 0.62 * s, cx + 0.19 * s, cy - 0.24 * s], fill=color)
    d.rounded_rectangle([cx - 0.40 * s, cy - 0.16 * s, cx + 0.40 * s, cy + 0.55 * s], radius=0.28 * s, fill=color)


@lru_cache(maxsize=16)
def layout_preview(layout: str, px_w: int = 128, px_h: int = 224) -> Image.Image:
    """Phone-shaped preview of a shorts layout."""
    W, H = px_w * 2, px_h * 2
    if layout == "blur":
        frame = _scene((W, int(W * 9 / 16)))
        canvas = ImageOps.fit(frame, (W, H)).filter(ImageFilter.GaussianBlur(W * 0.06))
        canvas = Image.blend(canvas, Image.new("RGB", (W, H), "#000000"), 0.25)
        canvas.paste(frame, (0, (H - frame.height) // 2))
    elif layout == "split":
        canvas = Image.new("RGB", (W, H), "#1f2937")
        top = int(H / 3)
        cam = Image.new("RGB", (W, top), "#334155")
        _person(ImageDraw.Draw(cam), W * 0.5, top * 0.62, top * 0.62, "#f9a8d4")
        canvas.paste(cam, (0, 0))
        canvas.paste(ImageOps.fit(_scene((W * 2, H)), (W, H - top)), (0, top))
        ImageDraw.Draw(canvas).line([(0, top), (W, top)], fill="#0f172a", width=max(2, W // 60))
    elif layout == "auto":
        canvas = ImageOps.fit(_scene((W * 2, H)), (W, H)).filter(ImageFilter.GaussianBlur(W * 0.03))
        canvas = Image.blend(canvas, Image.new("RGB", (W, H), "#000000"), 0.35).convert("RGBA")
        glyph = icon("spark", "#ffffff", int(W * 0.5))
        canvas.alpha_composite(glyph, ((W - glyph.width) // 2, (H - glyph.height) // 2))
        canvas = canvas.convert("RGB")
    else:
        canvas = ImageOps.fit(_scene((W * 2, H)), (W, H))
        if layout == "smart":
            d = ImageDraw.Draw(canvas)
            _person(d, W * 0.5, H * 0.68, W * 0.62, "#f9a8d4")
            r = W * 0.30
            d.rounded_rectangle([W * 0.5 - r, H * 0.36, W * 0.5 + r, H * 0.36 + 2 * r], radius=W * 0.05,
                                outline="#fde68a", width=max(2, W // 45))
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, W - 1, H - 1], radius=int(W * 0.14), fill=255)
    out = canvas.convert("RGBA")
    out.putalpha(mask)
    return out.resize((px_w, px_h), Image.LANCZOS)


def rounded_image(img: Image.Image, size: Tuple[int, int], radius: int) -> Image.Image:
    out = ImageOps.fit(img.convert("RGB"), size, Image.LANCZOS).convert("RGBA")
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=255)
    out.putalpha(mask)
    return out


@lru_cache(maxsize=16)
def placeholder(size: Tuple[int, int], background: str, foreground: str, radius: int = 18) -> Image.Image:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=_rgba(background))
    glyph = icon("film", foreground, min(size) // 3)
    img.alpha_composite(glyph, ((size[0] - glyph.width) // 2, (size[1] - glyph.height) // 2))
    return img
