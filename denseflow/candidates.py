"""Cross-scale motion candidates and branch selection.

Repeated texture plus occlusion can make a wrong displacement branch
photometrically good and forward/backward self-consistent.  Instead of
committing to one coarse minimum, this module:

1. enumerates integer-shift block-matching candidates at a coarse level;
2. keeps K minima separated in displacement space;
3. refines every candidate into a dense branch with the variational kernel;
4. scores branches with photometric residual, FB consistency, an
   occlusion-boundary ordering cue driven by the foreground mask, and an
   optional label-claim cue.

The boundary cue projects B pixels back into the A foreground mask with
the candidate backward flow.  A correct background branch predicts the
newly exposed strip on the retreating side of the propagated silhouette;
a period-shifted self-consistent branch predicts it in the wrong place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coords import forward_backward_error, pixel_grid, sample_bilinear
from .metrics import reprojection_residual
from .solver import SolveConfig, box_filter, estimate_flow, solve_level


def block_match_shifts(
    i1: np.ndarray,
    i2: np.ndarray,
    region: np.ndarray | None = None,
    level: int = 2,
    radius: int = 14,
    window: int = 7,
    top_k: int = 5,
    min_separation: float = 6.0,
    pyr1: list[np.ndarray] | None = None,
    pyr2: list[np.ndarray] | None = None,
) -> list[dict]:
    """Enumerate integer shifts at pyramid ``level`` and return K minima.

    Cost is mean windowed SAD over ``region`` mapped to that level.
    Returned shifts are in FULL-RES pixels (``shift / scale``).
    """
    if pyr1 is None or pyr2 is None:
        from .pyramid import build_pyramid

        n = level + 1
        pyr1 = build_pyramid(i1, n)
        pyr2 = build_pyramid(i2, n)
    g1, g2 = pyr1[level], pyr2[level]
    lh, lw = g1.shape
    scale = 2.0**-level

    if region is not None:
        xs, ys = pixel_grid(lh, lw)
        rsmall, _ = sample_bilinear(region.astype(np.float64),
                                    xs / scale, ys / scale)
        rmask = rsmall > 0.5
    else:
        rmask = np.ones((lh, lw), bool)

    m = window // 2
    valid_center = np.zeros_like(rmask)
    valid_center[m:lh - m, m:lw - m] = True
    rmask &= valid_center
    if rmask.sum() < 16:
        rmask = valid_center

    costs: list[tuple[float, int, int]] = []
    xs, ys = pixel_grid(lh, lw)
    br = max(1, window // 2)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            # A point x appears at B x+shift, so compare B(x+shift) to A(x).
            b_s, _ = sample_bilinear(g2, xs + dx, ys + dy, border="edge")
            sad = np.abs(b_s - g1)
            win = box_filter(sad, br)
            c = float(win[rmask].mean())
            costs.append((c, dx, dy))

    costs.sort(key=lambda z: z[0])
    chosen: list[dict] = []
    for c, dx, dy in costs:
        if all((dx - q["shift_level"][0]) ** 2 + (dy - q["shift_level"][1]) ** 2
               >= min_separation**2 for q in chosen):
            chosen.append({
                "shift_level": [int(dx), int(dy)],
                "shift_full": [dx / scale, dy / scale],
                "cost": c,
            })
        if len(chosen) >= top_k:
            break
    cmin = chosen[0]["cost"] if chosen else 0.0
    cmax = chosen[-1]["cost"] if chosen else 1.0
    for q in chosen:
        q["cost_norm"] = (q["cost"] - cmin) / max(1e-9, cmax - cmin)
    return chosen


def global_median_shift(u: np.ndarray, v: np.ndarray,
                        mask: np.ndarray | None = None) -> tuple[float, float]:
    if mask is None or mask.sum() < 20:
        return 0.0, 0.0
    return float(np.median(u[mask])), float(np.median(v[mask]))


def motion_side_band(
    mask_b: np.ndarray,
    d_f: tuple[float, float],
    width: int = 18,
    side: str = "trailing",
) -> np.ndarray:
    """Exterior band of ``mask_b`` lying on the trailing/leading side.

    For an exterior pixel y, walk along +/- the unit foreground motion
    direction; if the mask is hit within ``width`` pixels, y belongs to
    that side's band.  The trailing side is ``-d_f`` (where motion uncovers
    background); the leading side is ``+d_f`` (newly occluded background).
    """
    m = np.asarray(mask_b, bool)
    h, w = m.shape
    dx, dy = d_f
    mag = float(np.hypot(dx, dy))
    out = np.zeros((h, w), bool)
    if mag < 1e-6:
        return out
    ux, uy = dx / mag, dy / mag
    sign = 1.0 if side == "trailing" else -1.0
    xs, ys = pixel_grid(h, w)
    hit_dist = np.full((h, w), np.inf)
    for r in range(1, width + 1):
        qx = xs + sign * ux * r
        qy = ys + sign * uy * r
        inside, valid = sample_bilinear(m.astype(np.float64), qx, qy)
        hit = (inside > 0.5) & valid & (~m) & np.isinf(hit_dist)
        hit_dist[hit] = r
    out = np.isfinite(hit_dist)
    return out


def boundary_score(
    bu: np.ndarray,
    bv: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    trailing_band: np.ndarray | None = None,
) -> dict:
    """Score how well a backward branch explains newly exposed boundary.

    Predicted reveal set R: B pixels whose backward source lands inside the
    A foreground mask while the B pixel itself is outside mask_b.  Compare
    with the observed trailing-side exterior band.
    """
    h, w = bu.shape
    xs, ys = pixel_grid(h, w)
    ma_at_src, valid = sample_bilinear(mask_a.astype(np.float64),
                                       xs + bu, ys + bv)
    src_inside = ma_at_src > 0.5
    outside_b = mask_b < 0.5 if mask_b.dtype != bool else ~mask_b
    predicted_reveal = src_inside & outside_b & valid

    if trailing_band is None:
        observed = outside_b
    else:
        observed = np.asarray(trailing_band, bool) & outside_b

    inter = np.logical_and(predicted_reveal, observed).sum()
    n_pred = max(1, predicted_reveal.sum())
    n_obs = max(1, observed.sum())
    precision = inter / n_pred
    recall = inter / n_obs
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "boundary_f1": float(f1),
        "boundary_precision": float(precision),
        "boundary_recall": float(recall),
        "predicted_reveal_pixels": int(predicted_reveal.sum()),
    }


def strong_edge_band(gray: np.ndarray, grad_thr: float = 70.0) -> np.ndarray:
    """Thresholded Sobel edges dilated into a 1 px alignment band."""
    from PIL import Image, ImageFilter

    g = np.asarray(gray, np.float64)
    if g.ndim == 3:
        g = g.mean(axis=2)
    p = np.pad(g, 1, mode="edge")
    gx = ((p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:])
          - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2]))
    gy = ((p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
          - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:]))
    mag = np.hypot(gx, gy)
    thr = max(grad_thr, 2.2 * float(np.median(mag)) + 12.0)
    edge = mag > thr
    return np.asarray(
        Image.fromarray((edge.astype(np.uint8) * 255))
        .filter(ImageFilter.MaxFilter(3))) > 0


def edge_alignment_score(
    i2: np.ndarray,
    fu: np.ndarray,
    fv: np.ndarray,
    mask_a: np.ndarray,
    grad_thr: float = 70.0,
) -> dict:
    """Align the warped foreground contour with strong edges observed in B.

    This is the "choose with the newly exposed boundary" cue that remains
    informative when interior photometry is ambiguous (flat object) or
    self-similar (repeated texture): the candidate forward field warps the
    A silhouette into B, and a correct motion lands that contour on strong
    image edges of frame B.  Returns contour precision within a 1px band.
    """
    from PIL import Image, ImageFilter

    h, w = fu.shape
    xs, ys = pixel_grid(h, w)
    warped, valid = sample_bilinear(mask_a.astype(np.float64),
                                    xs - fu, ys - fv)
    wm = (warped > 0.5) & valid
    if wm.sum() < 30:
        return {"edge_align": 0.0}
    core_im = Image.fromarray((wm.astype(np.uint8) * 255)).filter(
        ImageFilter.MinFilter(5))
    core = np.asarray(core_im) > 0
    contour = wm & ~core
    if contour.sum() < 12:
        return {"edge_align": 0.0}

    band = strong_edge_band(i2, grad_thr)
    return {"edge_align": float(band[contour].mean())}


def claim_score(
    fu: np.ndarray,
    fv: np.ndarray,
    mask_a: np.ndarray,
    claim_mask_b: np.ndarray,
) -> dict:
    """Dice between the A mask warped by a forward branch and a label claim."""
    h, w = fu.shape
    xs, ys = pixel_grid(h, w)
    # Forward "splat" approximated by sampling A mask at x - F(x): this
    # answers which A label maps onto B pixel x (inverse lookup of the
    # forward field, exact for pure translation branches).
    warped, valid = sample_bilinear(mask_a.astype(np.float64),
                                    xs - fu, ys - fv)
    wm = (warped > 0.5) & valid
    cm = np.asarray(claim_mask_b, bool)
    inter = np.logical_and(wm, cm).sum()
    dice = 2 * inter / max(1, wm.sum() + cm.sum())
    iou = inter / max(1, np.logical_or(wm, cm).sum())
    return {"claim_dice": float(dice), "claim_iou": float(iou)}


@dataclass
class BranchResult:
    candidate: dict
    fwd: dict
    bwd: dict
    fb_error: np.ndarray
    scores: dict


def score_fields(
    fwd: dict,
    bwd: dict,
    i1: np.ndarray,
    i2: np.ndarray,
    cfg: SolveConfig,
    mask_a: np.ndarray | None = None,
    mask_b: np.ndarray | None = None,
    trailing_band: np.ndarray | None = None,
    claim_mask_b: np.ndarray | None = None,
    gt_flow: np.ndarray | None = None,
    gt_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    fb_err, fb_valid = forward_backward_error(
        fwd["u"], fwd["v"], bwd["u"], bwd["v"]
    )
    fb_ok = fb_valid & (fb_err < cfg.fb_tol)
    scores = {
        "fb_consistent_fraction": float(np.nan_to_num(fb_ok.mean())),
        "fb_median_error": float(np.nanmedian(fb_err)) if np.isfinite(fb_err).any() else None,
    }
    rp = reprojection_residual(i1, i2, bwd["u"], bwd["v"])
    scores["reproj_mean"] = rp["mean"]
    scores["reproj_median"] = rp["median"]

    if mask_a is not None and mask_b is not None:
        scores.update(boundary_score(
            bwd["u"], bwd["v"], mask_a, mask_b, trailing_band
        ))
    if mask_a is not None:
        scores.update(edge_alignment_score(i2, fwd["u"], fwd["v"], mask_a))
    if claim_mask_b is not None and mask_a is not None:
        scores.update(claim_score(fwd["u"], fwd["v"], mask_a, claim_mask_b))
    if gt_flow is not None:
        from .metrics import endpoint_error

        e = endpoint_error(fwd["u"], fwd["v"],
                           gt_flow[..., 0], gt_flow[..., 1], gt_valid)
        scores["epe_median"] = e["median"]
        scores["epe_mean"] = e["mean"]
    return fb_err, scores


def existing_branch(
    fwd: dict,
    bwd: dict,
    fb_error: np.ndarray,
    scores: dict,
    label: str = "pyramid",
) -> BranchResult:
    return BranchResult(
        candidate={"shift_full": [0.0, 0.0], "label": label},
        fwd=fwd, bwd=bwd, fb_error=fb_error, scores=scores,
    )


def refine_candidate(
    i1: np.ndarray,
    i2: np.ndarray,
    shift_full: tuple[float, float],
    cfg: SolveConfig,
    mask_a: np.ndarray | None = None,
    mask_b: np.ndarray | None = None,
    trailing_band: np.ndarray | None = None,
    claim_mask_b: np.ndarray | None = None,
    gt_flow: np.ndarray | None = None,
    gt_valid: np.ndarray | None = None,
) -> BranchResult:
    init_f = (
        np.full(i1.shape, shift_full[0], np.float64),
        np.full(i1.shape, shift_full[1], np.float64),
    )
    fwd = estimate_flow(i1, i2, cfg, initial_full=init_f)
    init_b = (
        np.full(i1.shape, -shift_full[0], np.float64),
        np.full(i1.shape, -shift_full[1], np.float64),
    )
    bwd = estimate_flow(i2, i1, cfg, initial_full=init_b)

    fb_err, scores = score_fields(
        fwd, bwd, i1, i2, cfg, mask_a, mask_b, trailing_band,
        claim_mask_b, gt_flow, gt_valid,
    )
    return BranchResult(
        candidate={"shift_full": list(shift_full)},
        fwd=fwd, bwd=bwd, fb_error=fb_err, scores=scores,
    )


def refine_seed(
    i1: np.ndarray,
    i2: np.ndarray,
    seed_u: np.ndarray,
    seed_v: np.ndarray,
    cfg: SolveConfig,
    seed_bwd_u: np.ndarray | None = None,
    seed_bwd_v: np.ndarray | None = None,
    mask_a: np.ndarray | None = None,
    mask_b: np.ndarray | None = None,
    trailing_band: np.ndarray | None = None,
    claim_mask_b: np.ndarray | None = None,
    gt_flow: np.ndarray | None = None,
    gt_valid: np.ndarray | None = None,
    label: str = "fused",
    shift_full: tuple[float, float] | None = None,
) -> BranchResult:
    """Refine a spatially composite seed field (region-specific motions).

    Unlike :func:`refine_candidate`, this does NOT rebuild a coarse pyramid:
    a small fast foreground region can vanish at the coarsest level, where
    regularization then erases the injected displacement before it can
    recover.  The composite warm start is therefore polished directly at
    full resolution (the seed itself is already a valid dense field).
    """
    from .solver import compute_support_map

    u, v, info = solve_level(i1, i2, cfg,
                             np.asarray(seed_u, np.float64),
                             np.asarray(seed_v, np.float64), level=0)
    support = compute_support_map(i1, i2, u, v, cfg)
    fwd = {
        "u": u, "v": v, "support": support,
        "levels": [info], "level_fields": [{"level": 0, "scale": 1.0, "u": u, "v": v}],
        "config": cfg.to_dict(), "converged": bool(info.get("converged", False)),
    }

    if seed_bwd_u is None or seed_bwd_v is None:
        bu0, bv0 = -np.asarray(seed_u, np.float64), -np.asarray(seed_v, np.float64)
    else:
        bu0, bv0 = np.asarray(seed_bwd_u, np.float64), np.asarray(seed_bwd_v, np.float64)
    bu, bv, binfo = solve_level(i2, i1, cfg, bu0, bv0, level=0)
    bsupport = compute_support_map(i2, i1, bu, bv, cfg)
    bwd = {
        "u": bu, "v": bv, "support": bsupport,
        "levels": [binfo],
        "level_fields": [{"level": 0, "scale": 1.0, "u": bu, "v": bv}],
        "config": cfg.to_dict(), "converged": bool(binfo.get("converged", False)),
    }

    fb_err, scores = score_fields(
        fwd, bwd, i1, i2, cfg, mask_a, mask_b, trailing_band,
        claim_mask_b, gt_flow, gt_valid,
    )
    if shift_full is None:
        shift_full = (float(np.median(u)), float(np.median(v)))
    return BranchResult(
        candidate={"shift_full": [float(shift_full[0]), float(shift_full[1])],
                   "label": label},
        fwd=fwd, bwd=bwd, fb_error=fb_err, scores=scores,
    )


def choose_branch(branches: list[BranchResult],
                  use_boundary: bool = True,
                  use_claim: bool = True) -> tuple[int, dict]:
    """Select a branch; return index and explanation of the decision rule.

    First apply a photometric gate (branches whose median reprojection
    residual is far above the best are discarded), then rank the survivors
    by boundary-ordering / label-claim evidence and FB consistency.
    """
    if not branches:
        return -1, {"reason": "no candidates"}

    def _val(name, default):
        out = []
        for b in branches:
            x = b.scores.get(name)
            out.append(default if x is None else x)
        return np.array(out, float)

    reproj = _val("reproj_median", 1e9)
    reproj_mean = _val("reproj_mean", 50.0)
    fb = _val("fb_consistent_fraction", 0.0)
    boundary = np.array([b.scores.get("boundary_f1", 0.0) for b in branches])
    edge = np.array([b.scores.get("edge_align", 0.0) for b in branches])
    claim = np.array([b.scores.get("claim_dice", 0.0) for b in branches])

    best_photo = float(reproj.min())
    gate_tol = max(2.5, 1.6 * best_photo + 1.0)
    photo_ok = reproj <= best_photo + gate_tol
    if not photo_ok.any():
        photo_ok = np.ones(len(branches), bool)
    fb_ok = fb >= 0.75
    eligible = photo_ok & fb_ok
    if not eligible.any():
        eligible = photo_ok

    rank_score = 0.15 * fb
    if use_boundary and boundary.max() > 0:
        rank_score = rank_score + 0.4 * boundary
    if use_boundary and edge.max() > 0:
        rank_score = rank_score + 1.4 * edge
    if use_claim and claim.max() > 0:
        rank_score = rank_score + 1.6 * claim
    # Aggregate photometric fit: the mean residual carries the cost of
    # boundary-misalignment outliers that the median deliberately hides.
    rank_score = rank_score + 0.8 * np.exp(-reproj_mean / 4.0)
    # Tiny photometric tie-break.
    rank_score = rank_score - 0.02 * _norm(reproj)
    rank_score[~eligible] = -1e9

    idx = int(np.argmax(rank_score))
    naive_idx = int(np.argmin(_norm(reproj) - 0.5 * _norm(fb)))
    explanation = {
        "selected": idx,
        "naive_photo_fb_index": naive_idx,
        "boundary_used": bool(use_boundary and boundary.max() > 0),
        "edge_used": bool(use_boundary and edge.max() > 0),
        "claim_used": bool(use_claim and claim.max() > 0),
        "photo_gate_residual_tol": gate_tol,
        "eligible": [bool(x) for x in eligible],
        "rank_score": [float(x) for x in rank_score],
        "boundary_f1": [float(x) for x in boundary],
        "edge_align": [float(x) for x in edge],
        "claim_dice": [float(x) for x in claim],
        "reproj_median": [float(x) for x in reproj],
        "reproj_mean": [float(x) for x in reproj_mean],
        "fb_fraction": [float(x) for x in fb],
    }
    return idx, explanation


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)
