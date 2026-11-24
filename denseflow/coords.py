"""Coordinate conventions and coordinate-correct sampling/warping.

Coordinate convention (single source of truth, also shown in the UI):

* Array image ``I[i, j]``: ``i`` is the row, ``j`` the column.
* Pixel *centers* have integer Cartesian coordinates ``(x=j, y=i)``; the
  origin is the center of the top-left pixel.  ``x`` points right,
  ``y`` points down.  One unit is one pixel pitch of the image at hand.
  The sensor area therefore spans ``[-0.5, W-0.5] x [-0.5, H-0.5]``.
* Forward flow ``F_{A->B}(x) = (u, v)`` maps the center ``x`` in A to
  ``x + (u, v)`` in B.  Arrays are stored as ``u[i, j]`` (x / columns)
  and ``v[i, j]`` (y / rows).
* Backward flow ``B_{B->A}(x) = (bu, bv)`` maps a B center to
  ``x + (bu, bv)`` in A.
* Reprojection of previous frame A into B's grid uses the backward flow::

      A_hat_B(x) = sample_bilinear(A, x + B_{B->A}(x))

* A source center is *valid* when it lies inside ``[0, W-1] x [0, H-1]``.
  Out-of-frame samples are edge-clamped for the returned values but carry
  ``valid = False``; callers must turn those into zero support.

Pyramid note: at level k with scale ``s_k = 2**-k`` relative to level 0,
the same physical point has level-k coordinate ``x_k = s_k * x_0``.
Prolonging a displacement field from level k to a finer level j therefore
multiplies displacement *values* by ``s_k / s_j`` (2 per octave) while
resampling their spatial coordinates with the matching mapping.  Merely
upscaling the array is not enough.
"""

from __future__ import annotations

import numpy as np

CONVENTION = {
    "pixel_center": "integer (x=column, y=row), origin at top-left pixel center",
    "x_axis": "right",
    "y_axis": "down",
    "forward_flow": "x_B = x_A + F_A->B(x_A)",
    "backward_flow": "x_A = x_B + B_B->A(x_B)",
    "reprojection": "A_hat_B(x_B) = sample(A, x_B + B_B->A(x_B))",
    "valid_range": "source center in [0,W-1] x [0,H-1]",
}


def pixel_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Return integer-center coordinate grids xs, ys (float64)."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    return xs, ys


def sample_bilinear(
    img: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    cval: float = 0.0,
    border: str = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``img`` at source coordinates ``(sx, sy)`` with bilinear interp.

    Parameters
    ----------
    img : HxW or HxWxC array
    sx, sy : arrays of arbitrary shape giving source center coordinates.
    border : ``"constant"`` zeroes samples whose source center is outside
        ``[0, W-1] x [0, H-1]`` (returned ``valid`` is False there);
        ``"edge"`` replicates the border instead (used for flow prolongation).

    Returns
    -------
    values : sampled values, shape ``sx.shape`` or ``sx.shape + (C,)``.
    valid : bool array, True where the source center is inside the image.
    """
    h, w = img.shape[:2]
    sx = np.asarray(sx, dtype=np.float64)
    sy = np.asarray(sy, dtype=np.float64)

    valid = (sx >= 0.0) & (sx <= w - 1.0) & (sy >= 0.0) & (sy <= h - 1.0)

    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    # Clamp gather indices; invalid outputs are replaced by cval below.
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    wx = sx - x0
    wy = sy - y0
    wx = np.clip(wx, 0.0, 1.0)
    wy = np.clip(wy, 0.0, 1.0)

    if img.ndim == 3:
        wx = wx[..., None]
        wy = wy[..., None]
        valid_out = valid
    else:
        valid_out = valid

    img_f = img.astype(np.float64, copy=False)
    Ia = img_f[y0c, x0c]
    Ib = img_f[y0c, x1c]
    Ic = img_f[y1c, x0c]
    Id = img_f[y1c, x1c]

    top = Ia + (Ib - Ia) * wx
    bot = Ic + (Id - Ic) * wx
    out = top + (bot - top) * wy

    if border == "edge":
        return out, valid

    fill = np.full_like(out, cval)
    if img.ndim == 3:
        out = np.where(valid[..., None], out, fill)
    else:
        out = np.where(valid, out, fill)
    return out, valid


def sample_nearest(
    labels: np.ndarray, sx: np.ndarray, sy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor sampling for discrete label maps."""
    h, w = labels.shape[:2]
    valid = (sx >= 0.0) & (sx <= w - 1.0) & (sy >= 0.0) & (sy <= h - 1.0)
    ix = np.clip(np.rint(sx).astype(np.int64), 0, w - 1)
    iy = np.clip(np.rint(sy).astype(np.int64), 0, h - 1)
    out = labels[iy, ix]
    out = np.where(valid, out, 0)
    return out, valid


def spatial_gradient(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradients ``(gx=d/dx columns, gy=d/dy rows)`` with edge handling."""
    gy, gx = np.gradient(g.astype(np.float64, copy=False))
    return gx, gy


def warp_forward(
    img: np.ndarray, u: np.ndarray, v: np.ndarray, cval: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Warp img onto the grid of ``u/v`` using source ``x + (u, v)``.

    For a forward flow A->B this samples B at ``x_A + F(x_A)`` back onto
    A's grid (used by the linearization).  For a backward flow B->A it
    reprojects A onto B's grid, which is the reprojection check.
    """
    h, w = u.shape
    xs, ys = pixel_grid(h, w)
    return sample_bilinear(img, xs + u, ys + v, cval=cval)


def sample_flow_at(
    u: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinearly sample a flow field at arbitrary coordinates."""
    su, valid_u = sample_bilinear(u, x, y)
    sv, valid_v = sample_bilinear(v, x, y)
    return su, sv, valid_u & valid_v


def forward_backward_error(
    fu: np.ndarray,
    fv: np.ndarray,
    bu: np.ndarray,
    bv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cycle-consistency error evaluated on the A grid.

    ``err_A(x) = || F(x) + B(x + F(x)) ||``.  Also returns the validity of
    the sampled backward vector.  Occluded points tend to fail this test.
    """
    h, w = fu.shape
    xs, ys = pixel_grid(h, w)
    bx, by, valid = sample_flow_at(bu, bv, xs + fu, ys + fv)
    ex = fu + bx
    ey = fv + by
    err = np.hypot(ex, ey)
    err = np.where(valid, err, np.nan)
    return err, valid


def backward_cycle_error(
    fu: np.ndarray,
    fv: np.ndarray,
    bu: np.ndarray,
    bv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cycle-consistency error evaluated on the B grid.

    ``err_B(x) = || B(x) + F(x + B(x)) ||``.  This is the correct trust map
    for gating a backward warp that propagates labels from A onto B's grid:
    the gate is indexed by B pixel.
    """
    h, w = bu.shape
    xs, ys = pixel_grid(h, w)
    fx, fy, valid = sample_flow_at(fu, fv, xs + bu, ys + bv)
    ex = bu + fx
    ey = bv + fy
    err = np.hypot(ex, ey)
    err = np.where(valid, err, np.nan)
    return err, valid
