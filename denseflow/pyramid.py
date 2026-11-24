"""Coordinate-correct Gaussian pyramid and flow prolongation."""

from __future__ import annotations

import numpy as np

from .coords import pixel_grid, sample_bilinear


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with reflect padding (2D or HxWxC)."""
    if sigma <= 0:
        return img.astype(np.float64, copy=True)
    k = gaussian_kernel1d(sigma)
    r = k.size // 2
    img = img.astype(np.float64, copy=False)

    def blur_axis(a: np.ndarray, axis: int) -> np.ndarray:
        pad_width = [(0, 0)] * a.ndim
        pad_width[axis] = (r, r)
        padded = np.pad(a, pad_width, mode="reflect")
        out = np.zeros_like(a)
        for i, kv in enumerate(k):
            sl = [slice(None)] * a.ndim
            sl[axis] = slice(i, i + a.shape[axis])
            out += kv * padded[tuple(sl)]
        return out

    out = blur_axis(img, axis=0)
    out = blur_axis(out, axis=1)
    return out


def level_size(h: int, w: int, level: int) -> tuple[int, int]:
    scale = 2.0**-level
    return max(1, int(np.ceil(h * scale))), max(1, int(np.ceil(w * scale)))


def build_pyramid(
    gray: np.ndarray,
    num_levels: int = 4,
    blur_sigma: float = 1.1,
) -> list[np.ndarray]:
    """Build an octave pyramid by blur + coordinate-aligned decimation.

    Level k is sampled at full coordinates ``2**k * j`` from a blurred
    version of level k-1; centers stay aligned (coarse center j maps to
    fine center 2j), which is what makes the x_k = s_k*x_0 rule exact.
    """
    pyr = [gray.astype(np.float64, copy=False)]
    for level in range(1, num_levels):
        prev = gaussian_blur(pyr[-1], blur_sigma)
        h, w = prev.shape
        nh, nw = level_size(gray.shape[0], gray.shape[1], level)
        # Coarse center (j, i) maps to prev-level center (2j, 2i).
        out = prev[0 : 2 * nh : 2, 0 : 2 * nw : 2]
        if out.shape != (nh, nw):  # pragma: no cover - defensive
            ys, xs = np.mgrid[0:nh, 0:nw]
            out, _ = sample_bilinear(prev, xs * 2.0, ys * 2.0)
        pyr.append(out)
    return pyr


def prolong_flow(
    u: np.ndarray,
    v: np.ndarray,
    out_h: int,
    out_w: int,
    scale_from: float,
    scale_to: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample flow to a finer/coarser grid and transform displacement units.

    Spatial sampling uses ``x_from = x_to * scale_from / scale_to`` and
    displacement values are multiplied by ``scale_to / scale_from`` so the
    physical displacement in level-0 pixels is preserved:
    ``d = s * d0  =>  d_to = d_from * s_to / s_from``.
    """
    xs, ys = pixel_grid(out_h, out_w)
    coord_ratio = scale_from / scale_to
    sx = xs * coord_ratio
    sy = ys * coord_ratio
    uu, valid_u = sample_bilinear(u, sx, sy, border="edge")
    vv, valid_v = sample_bilinear(v, sx, sy, border="edge")
    value_ratio = scale_to / scale_from
    uu = uu * value_ratio
    vv = vv * value_ratio
    if not (np.all(valid_u) and np.all(valid_v)):
        # Outside the coarse support: fall back to nearest edge value is
        # already produced by clamped bilinear; only flag nothing here.
        pass
    return uu, vv


def resize_image_like(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Coordinate-aligned bilinear resize to an explicit size."""
    h, w = img.shape[:2]
    xs, ys = pixel_grid(out_h, out_w)
    sx = xs * (w - 1) / max(1, out_w - 1) if out_w > 1 else np.zeros_like(xs)
    sy = ys * (h - 1) / max(1, out_h - 1) if out_h > 1 else np.zeros_like(ys)
    out, _ = sample_bilinear(img, sx, sy)
    return out
