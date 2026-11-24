"""Flow / residual / support / label visualization (numpy + Pillow)."""

from __future__ import annotations

import colorsys

import numpy as np
from PIL import Image, ImageDraw


def make_color_wheel(bins: int = 55) -> np.ndarray:
    """Middlebury-style hue wheel, returned as Nx3 uint8 RGB."""
    pts = []
    # red -> yellow -> green -> cyan -> blue -> magenta -> red
    stops = [
        (0.0, (255, 0, 0)),
        (1 / 6, (255, 255, 0)),
        (2 / 6, (0, 255, 0)),
        (3 / 6, (0, 255, 255)),
        (4 / 6, (0, 0, 255)),
        (5 / 6, (255, 0, 255)),
        (1.0, (255, 0, 0)),
    ]
    n = bins * 6
    for i in range(n):
        t = i / (n - 1)
        for k in range(len(stops) - 1):
            t0, c0 = stops[k]
            t1, c1 = stops[k + 1]
            if t0 <= t <= t1:
                f = (t - t0) / (t1 - t0)
                pts.append(tuple(int(c0[j] + (c1[j] - c0[j]) * f) for j in range(3)))
                break
    return np.array(pts, dtype=np.uint8)


WHEEL = make_color_wheel()


def flow_to_rgb(
    u: np.ndarray,
    v: np.ndarray,
    max_motion: float | None = None,
    support: np.ndarray | None = None,
) -> Image.Image:
    """Encode direction as hue, magnitude as saturation/brightness."""
    mag = np.hypot(u, v)
    if max_motion is None:
        valid = mag[np.isfinite(mag)]
        max_motion = float(np.percentile(valid, 95)) if valid.size else 1.0
        max_motion = max(max_motion, 1e-6)

    angle = np.arctan2(v, u)  # [-pi, pi], y down: hue rotates accordingly
    n = WHEEL.shape[0]
    idx = ((angle + np.pi) / (2 * np.pi) * n).astype(int) % n
    rgb = WHEEL[idx].astype(np.float64)

    norm = np.clip(mag / max_motion, 0.0, 1.0)
    # White background; stronger motion -> saturated color.
    rgb = 255.0 * (1.0 - norm[..., None]) + rgb * norm[..., None]

    if support is not None:
        s = np.clip(support, 0.15, 1.0)[..., None]
        rgb = rgb * s + 255.0 * (1.0 - s)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def scalar_heatmap(z: np.ndarray, vmax: float | None = None) -> Image.Image:
    """Map scalar >=0 to a black-red-orange-yellow heat map."""
    z = np.asarray(z, dtype=np.float64)
    if vmax is None:
        vmax = float(np.percentile(z[np.isfinite(z)], 95)) if z.size else 1.0
        vmax = max(vmax, 1e-6)
    t = np.clip(z / vmax, 0.0, 1.0)
    r = np.clip(255 * np.minimum(1.5 * t, 1.0), 0, 255)
    g = np.clip(255 * np.minimum(1.5 * t - 0.5, 1.0), 0, 255)
    b = np.clip(255 * (1.5 * t - 1.0), 0, 255)
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def support_to_rgb(score: np.ndarray, reliable: np.ndarray | None = None) -> Image.Image:
    """Grayscale support score with reliable pixels tinted green."""
    s = (np.clip(score, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([s, s, s], axis=-1)
    if reliable is not None:
        m = reliable
        rgb[m, 1] = np.maximum(rgb[m, 1], 200)
        rgb[m, 0] = (rgb[m, 0] * 0.4).astype(np.uint8)
        rgb[m, 2] = (rgb[m, 2] * 0.4).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


LABEL_COLORS = [
    (0, 0, 0, 0),
    (230, 57, 70, 150),    # foreground id 1 red
    (58, 134, 255, 150),   # id 2 blue
    (255, 190, 11, 150),   # held: amber
]


def labels_overlay(
    base: Image.Image | np.ndarray,
    labels: np.ndarray,
    held: np.ndarray | None = None,
) -> Image.Image:
    if isinstance(base, np.ndarray):
        base = Image.fromarray(base.astype(np.uint8)).convert("RGBA")
    else:
        base = base.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    arr = np.array(overlay)
    for lid in np.unique(labels):
        if lid == 0:
            continue
        color = LABEL_COLORS[int(lid) % len(LABEL_COLORS)]
        arr[labels == lid] = color
    if held is not None and held.any():
        arr[held] = (255, 190, 11, 120)
    overlay = Image.fromarray(arr, "RGBA")
    out = Image.alpha_composite(base, overlay)
    return out.convert("RGB")


def draw_vectors(
    base: Image.Image | np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    stride: int = 16,
    scale: float = 1.0,
    color=(0, 255, 120),
    bad: np.ndarray | None = None,
) -> Image.Image:
    if isinstance(base, np.ndarray):
        img = Image.fromarray(base.astype(np.uint8)).convert("RGB")
    else:
        img = base.convert("RGB")
    draw = ImageDraw.Draw(img)
    h, w = u.shape
    for y in range(stride // 2, h, stride):
        for x in range(stride // 2, w, stride):
            dx = float(u[y, x]) * scale
            dy = float(v[y, x]) * scale
            if abs(dx) + abs(dy) < 0.4:
                continue
            col = (255, 80, 80) if bad is not None and not bad[y, x] else color
            ex, ey = x + dx, y + dy
            draw.line([(x, y), (ex, ey)], fill=col, width=1)
            ang = np.arctan2(dy, dx)
            hl = 4.0
            for da in (2.6, -2.6):
                hx = ex - hl * np.cos(ang + da)
                hy = ey - hl * np.sin(ang + da)
                draw.line([(ex, ey), (hx, hy)], fill=col, width=1)
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(255, 255, 255))
    return img


def to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    import base64
    import io

    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
