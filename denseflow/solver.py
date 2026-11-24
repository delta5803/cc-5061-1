"""Variational dense optical-flow kernel.

The kernel minimizes a robust data term plus weighted spatial
regularization on the displacement field ``w = (u, v)``::

    E(w) = sum_i Psi_d( I2(x_i + w_i) - I1(x_i) )
           + alpha * sum_{edges i-j} p_ij *
             [ Psi_s(u_i-u_j) + Psi_s(v_i-v_j) ]

with Charbonnier penalty ``Psi(z; c) = sqrt(z*z + c*c) - c``.

Minimization is Gauss-Newton / IRLS inside a coordinate-correct coarse to
fine pyramid.  Every update is an inspectable 2x2 pixel solve with a
weighted-Laplacian Jacobi split; convergence history (true objective,
per-outer displacement correction, inner residual) is returned.

Nothing here treats an all-zero field as known static: a separate support
map marks local image evidence, photometric fit, bounds and eventually
forward/backward consistency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .coords import spatial_gradient, warp_forward
from .pyramid import build_pyramid, prolong_flow


@dataclass
class SolveConfig:
    num_levels: int = 4
    outer_iters: int = 4
    inner_iters: int = 40
    alpha: float = 160.0          # regularization weight (0..255 intensities)
    c_data: float = 10.0          # Charbonnier scale for residuals
    c_smooth: float = 0.35        # Charbonnier scale for flow jumps (px)
    relax: float = 1.0            # Jacobi relaxation
    tol_outer: float = 2.5e-2     # max outer correction (level px) for convergence
    tol_inner: float = 2.5e-3
    blur_sigma: float = 1.1
    texture_tau: float = 8.0      # structure-tensor lambda_min midpoint
    residual_sigma: float = 12.0  # intensity residual for support confidence
    fb_tol: float = 1.5           # px cycle error considered consistent
    reliable_residual: float = 16.0

    def to_dict(self) -> dict:
        return asdict(self)


def charbonnier(z: np.ndarray, c: float) -> np.ndarray:
    return np.sqrt(z * z + c * c) - c


def irls_weight(z: np.ndarray, c: float) -> np.ndarray:
    return 1.0 / np.sqrt(z * z + c * c)


def objective(
    u: np.ndarray,
    v: np.ndarray,
    i1: np.ndarray,
    i2: np.ndarray,
    alpha: float,
    c_data: float,
    c_smooth: float,
) -> dict:
    """Evaluate the exact (non-quadratic) objective and its components."""
    i2w, valid = warp_forward(i2, u, v)
    r = i2w - i1
    data = np.sum(charbonnier(r[valid], c_data)) if valid.any() else 0.0

    du_h = u[:, 1:] - u[:, :-1]
    du_v = u[1:, :] - u[:-1, :]
    dv_h = v[:, 1:] - v[:, :-1]
    dv_v = v[1:, :] - v[:-1, :]
    smooth_u = np.sum(charbonnier(du_h, c_smooth)) + np.sum(charbonnier(du_v, c_smooth))
    smooth_v = np.sum(charbonnier(dv_h, c_smooth)) + np.sum(charbonnier(dv_v, c_smooth))
    smooth = smooth_u + smooth_v
    total = data + alpha * smooth
    n_valid = int(valid.sum())
    return {
        "total": float(total),
        "data": float(data),
        "smooth": float(smooth),
        "total_per_pixel": float(total / max(1, i1.size)),
        "rms_residual_valid": float(
            np.sqrt(np.mean(r[valid] ** 2)) if n_valid else float("nan")
        ),
        "valid_fraction": float(valid.mean()),
    }


def box_filter(a: np.ndarray, radius: int = 2) -> np.ndarray:
    """Fast uniform box filter via integral image (reflect-ish edges)."""
    r = int(radius)
    ap = np.pad(a, r, mode="reflect")
    ii = np.cumsum(np.cumsum(ap, axis=0), axis=1)
    ii = np.pad(ii, ((1, 0), (1, 0)), mode="constant")
    h, w = a.shape
    y0 = np.arange(h)
    y1 = y0 + 2 * r + 1
    x0 = np.arange(w)
    x1 = x0 + 2 * r + 1
    y0i = y0[:, None]
    y1i = y1[:, None]
    x0i = x0[None, :]
    x1i = x1[None, :]
    s = (
        ii[y1i, x1i]
        - ii[y0i, x1i]
        - ii[y1i, x0i]
        + ii[y0i, x0i]
    )
    return s / float((2 * r + 1) ** 2)


def texture_evidence(g: np.ndarray, radius: int = 2) -> np.ndarray:
    """Minimum eigenvalue of the local structure tensor."""
    gx, gy = spatial_gradient(g)
    jxx = box_filter(gx * gx, radius)
    jyy = box_filter(gy * gy, radius)
    jxy = box_filter(gx * gy, radius)
    disc = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    return 0.5 * (jxx + jyy - disc)


def _laplacian_parts(
    z: np.ndarray, ph: np.ndarray, pv: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted degree d and weighted neighbor sum nbz."""
    h, w = z.shape
    d = np.zeros_like(z)
    nb = np.zeros_like(z)

    d[:, :-1] += ph
    d[:, 1:] += ph
    nb[:, :-1] += ph * z[:, 1:]
    nb[:, 1:] += ph * z[:, :-1]

    d[:-1, :] += pv
    d[1:, :] += pv
    nb[:-1, :] += pv * z[1:, :]
    nb[1:, :] += pv * z[:-1, :]
    return d, nb


def solve_level(
    i1: np.ndarray,
    i2: np.ndarray,
    cfg: SolveConfig,
    u: np.ndarray,
    v: np.ndarray,
    level: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run outer linearization / inner Jacobi updates at one resolution."""
    h, w = i1.shape
    u = u.copy()
    v = v.copy()
    history: list[dict] = []
    converged = False

    for outer in range(cfg.outer_iters):
        e_before = objective(u, v, i1, i2, cfg.alpha, cfg.c_data, cfg.c_smooth)

        # Warp I2 by the current estimate, then linearize the data term.
        i2w, valid = warp_forward(i2, u, v)
        r0 = i2w - i1
        ix, iy = spatial_gradient(i2w)

        qd = irls_weight(r0, cfg.c_data)
        qd = np.where(valid, qd, 0.0)

        # Robust smoothness edge weights frozen for this outer step.
        du_e_h = u[:, 1:] - u[:, :-1]
        du_e_v = u[1:, :] - u[:-1, :]
        dv_e_h = v[:, 1:] - v[:, :-1]
        dv_e_v = v[1:, :] - v[:-1, :]
        ph = irls_weight(du_e_h, cfg.c_smooth) + irls_weight(dv_e_h, cfg.c_smooth)
        pv = irls_weight(du_e_v, cfg.c_smooth) + irls_weight(dv_e_v, cfg.c_smooth)
        # Two components share one edge weight; average keeps scale stable.
        ph *= 0.5
        pv *= 0.5

        d_u, nb_u0 = _laplacian_parts(u, ph, pv)
        d_v, nb_v0 = _laplacian_parts(v, ph, pv)
        lu0 = d_u * u - nb_u0
        lv0 = d_v * v - nb_v0
        degree = 0.5 * (d_u + d_v)

        du = np.zeros_like(u)
        dv = np.zeros_like(v)
        inner_last = float("inf")
        for inner in range(cfg.inner_iters):
            _, nb_du = _laplacian_parts(du, ph, pv)
            _, nb_dv = _laplacian_parts(dv, ph, pv)

            a11 = qd * ix * ix + cfg.alpha * degree
            a22 = qd * iy * iy + cfg.alpha * degree
            a12 = qd * ix * iy

            b1 = -qd * ix * r0 - cfg.alpha * lu0 + cfg.alpha * nb_du
            b2 = -qd * iy * r0 - cfg.alpha * lv0 + cfg.alpha * nb_dv

            det = a11 * a22 - a12 * a12
            det_safe = np.where(det > 1e-14, det, 1e-14)
            du_new = (b1 * a22 - b2 * a12) / det_safe
            dv_new = (a11 * b2 - a12 * b1) / det_safe
            if cfg.relax != 1.0:
                du_new = (1.0 - cfg.relax) * du + cfg.relax * du_new
                dv_new = (1.0 - cfg.relax) * dv + cfg.relax * dv_new
            if not np.all(det > 1e-14):
                # Fully degenerate 2x2 blocks (should not happen interior
                # because alpha*degree > 0) fall back to smooth update.
                bad = det <= 1e-14
                du_new = np.where(bad, nb_du / np.maximum(degree, 1e-12), du_new)
                dv_new = np.where(bad, nb_dv / np.maximum(degree, 1e-12), dv_new)

            delta = max(
                float(np.max(np.abs(du_new - du))),
                float(np.max(np.abs(dv_new - dv))),
            )
            du, dv = du_new, dv_new
            inner_last = delta
            if delta < cfg.tol_inner:
                break

        u = u + du
        v = v + dv
        corr = float(max(np.max(np.abs(du)), np.max(np.abs(dv))))
        e_after = objective(u, v, i1, i2, cfg.alpha, cfg.c_data, cfg.c_smooth)
        rec = {
            "outer": outer,
            "inner_iters": inner + 1,
            "inner_final_delta": inner_last,
            "outer_correction_max": corr,
            "energy_before": e_before["total_per_pixel"],
            "energy_after": e_after["total_per_pixel"],
            "data_per_pixel": e_after["data"] / max(1, i1.size),
            "smooth_per_pixel": cfg.alpha * e_after["smooth"] / max(1, i1.size),
            "rms_residual": e_after["rms_residual_valid"],
        }
        history.append(rec)
        if corr < cfg.tol_outer:
            converged = True
            break

    final = objective(u, v, i1, i2, cfg.alpha, cfg.c_data, cfg.c_smooth)
    info = {
        "level": level,
        "height": h,
        "width": w,
        "converged": converged,
        "outer_iterations": len(history),
        "history": history,
        "final_objective": final,
    }
    return u, v, info


def estimate_flow(
    i1: np.ndarray,
    i2: np.ndarray,
    cfg: SolveConfig | None = None,
    initial_full: tuple[np.ndarray, np.ndarray] | None = None,
    compute_support: bool = True,
) -> dict:
    """Full coarse-to-fine estimation for one direction.

    ``i1``/``i2`` are 2D float arrays in ``[0, 255]``.
    """
    if cfg is None:
        cfg = SolveConfig()
    i1 = i1.astype(np.float64, copy=False)
    i2 = i2.astype(np.float64, copy=False)
    h, w = i1.shape

    max_levels = int(np.floor(np.log2(min(h, w)))) + 1
    num_levels = max(1, min(cfg.num_levels, max_levels))
    pyr1 = build_pyramid(i1, num_levels, cfg.blur_sigma)
    pyr2 = build_pyramid(i2, num_levels, cfg.blur_sigma)

    u = v = None
    level_reports: list[dict] = []
    level_fields: list[dict] = []

    for k in range(num_levels - 1, -1, -1):
        g1, g2 = pyr1[k], pyr2[k]
        lh, lw = g1.shape
        scale_k = 2.0**-k
        if u is None:
            if initial_full is not None:
                u, v = prolong_flow(
                    initial_full[0], initial_full[1], lh, lw, 1.0, scale_k
                )
            else:
                u = np.zeros((lh, lw), dtype=np.float64)
                v = np.zeros((lh, lw), dtype=np.float64)
        else:
            # Existing field lives at the previously solved level k+1
            # whose scale is scale_k / 2; prolong into scale_k.
            u, v = prolong_flow(u, v, lh, lw, scale_k / 2.0, scale_k)

        u, v, info = solve_level(g1, g2, cfg, u, v, level=k)
        level_reports.append(info)
        level_fields.append(
            {"level": k, "scale": scale_k, "u": u.copy(), "v": v.copy()}
        )

    support = None
    if compute_support:
        support = compute_support_map(i1, i2, u, v, cfg)

    final_info = level_reports[0] if level_reports else {}
    return {
        "u": u,
        "v": v,
        "support": support,
        "levels": list(reversed(level_reports)),
        "level_fields": list(reversed(level_fields)),
        "config": cfg.to_dict(),
        "converged": bool(final_info.get("converged", False)),
    }


def compute_support_map(
    i1: np.ndarray,
    i2: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    cfg: SolveConfig,
) -> dict:
    """Local evidence / photometric fit / validity support arrays."""
    i2w, valid = warp_forward(i2, u, v)
    r = i2w - i1
    absr = np.abs(r)
    gx, gy = spatial_gradient(0.5 * (i1 + i2w))
    jxx = box_filter(gx * gx, 2)
    jyy = box_filter(gy * gy, 2)
    jxy = box_filter(gx * gy, 2)
    disc = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    lam_min = 0.5 * (jxx + jyy - disc)
    lam_max = 0.5 * (jxx + jyy + disc)
    lam = lam_min
    evidence = np.clip(lam / (lam + cfg.texture_tau), 0.0, 1.0)
    photo = np.exp(-(absr**2) / (2.0 * cfg.residual_sigma**2))
    score = np.clip(valid.astype(np.float64) * evidence * photo, 0.0, 1.0)
    # Two reliability paths: 2D texture (lambda_min high) with a moderate
    # fit, OR a tight photometric fit on oriented edge structure (lambda_max
    # high) which covers repeated / 1-D stripes suffering the aperture issue.
    reliable = (
        valid
        & (absr < cfg.reliable_residual)
        & ((evidence > 0.5)
           | ((lam_max > 3.0) & (absr < 8.0)))
    )
    return {
        "score": score,
        "evidence": evidence,
        "texture_lambda": lam_min,
        "texture_lambda_max": lam_max,
        "photo": photo,
        "residual": absr,
        "valid": valid,
        "reliable_photo": reliable,
    }
