"""Endpoint error, angular error, reprojection residual and support stats."""

from __future__ import annotations

import numpy as np

from .coords import warp_forward


def endpoint_error(
    u: np.ndarray,
    v: np.ndarray,
    gt_u: np.ndarray,
    gt_v: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict:
    epe = np.hypot(u - gt_u, v - gt_v)
    if mask is None:
        mask = np.ones_like(epe, dtype=bool)
    m = np.asarray(mask, dtype=bool)
    vals = epe[m]
    if vals.size == 0:
        return {"mean": None, "median": None, "p90": None, "inlier_1px": None, "n": 0}
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90)),
        "max": float(vals.max()),
        "inlier_1px": float((vals <= 1.0).mean()),
        "n": int(vals.size),
    }


def angular_error(
    u: np.ndarray,
    v: np.ndarray,
    gt_u: np.ndarray,
    gt_v: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict:
    """Middlebury angular error in degrees (homogeneous coordinates)."""
    num = 1.0 + u * gt_u + v * gt_v
    den = np.sqrt(1.0 + u * u + v * v) * np.sqrt(1.0 + gt_u * gt_u + gt_v * gt_v)
    ae = np.degrees(np.arccos(np.clip(num / np.maximum(den, 1e-12), -1.0, 1.0)))
    if mask is None:
        mask = np.ones_like(ae, dtype=bool)
    vals = ae[mask]
    if vals.size == 0:
        return {"mean": None, "median": None}
    return {"mean": float(vals.mean()), "median": float(np.median(vals))}


def reprojection_residual(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    bu: np.ndarray,
    bv: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict:
    """Warp prev frame onto curr grid using backward flow; measure residual."""
    reprojected, valid = warp_forward(prev_gray, bu, bv)
    diff = reprojected - curr_gray
    absr = np.abs(diff)
    use = valid if mask is None else (valid & np.asarray(mask, bool))
    vals = absr[use]
    if vals.size == 0:
        return {"mean": None, "median": None, "p95": None, "valid_fraction": 0.0}
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95)),
        "rms": float(np.sqrt((diff[use] ** 2).mean())),
        "valid_fraction": float(use.mean()),
        "map": absr,
        "valid": use,
        "reprojected": reprojected,
    }


def support_summary(support: dict, fb_error: np.ndarray | None = None) -> dict:
    score = support["score"]
    reliable = support["reliable_photo"].copy()
    out = {
        "mean_score": float(score.mean()),
        "photo_reliable_fraction": float(reliable.mean()),
        "valid_fraction": float(support["valid"].mean()),
        "textured_fraction": float((support["evidence"] > 0.5).mean()),
    }
    if fb_error is not None:
        fb_ok = np.isfinite(fb_error) & (fb_error < 1.5)
        out["fb_consistent_fraction"] = float(fb_ok.mean())
        out["fb_median_error"] = float(np.nanmedian(fb_error)) if np.isfinite(fb_error).any() else None
        out["fully_reliable_fraction"] = float((reliable & fb_ok).mean())
    return out


def label_scores(
    pred: np.ndarray,
    gt: np.ndarray,
    foreground_ids=(1,),
    decoy_mask: np.ndarray | None = None,
) -> dict:
    """IoU/precision/recall for foreground labels; optional same-color decoy FP."""
    pf = np.isin(pred, list(foreground_ids))
    gf = np.isin(gt, list(foreground_ids))
    inter = np.logical_and(pf, gf).sum()
    union = np.logical_or(pf, gf).sum()
    iou = float(inter / union) if union else None
    precision = float(inter / pf.sum()) if pf.sum() else None
    recall = float(inter / gf.sum()) if gf.sum() else None
    out = {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "pred_foreground_pixels": int(pf.sum()),
        "gt_foreground_pixels": int(gf.sum()),
    }
    if decoy_mask is not None:
        dm = np.asarray(decoy_mask, bool)
        wrong = pf & dm
        out["same_color_decoy_pixels"] = int(dm.sum())
        out["same_color_decoy_false_foreground"] = int(wrong.sum())
        out["same_color_decoy_fp_fraction"] = float(wrong.sum() / max(1, dm.sum()))
    return out
