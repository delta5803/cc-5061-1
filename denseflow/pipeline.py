"""High-level pair analysis shared by the web kernel API and experiment runner."""

from __future__ import annotations

import base64

import numpy as np

from .candidates import (
    block_match_shifts,
    choose_branch,
    motion_side_band,
    refine_candidate,
)
from .coords import CONVENTION, forward_backward_error
from .labels import propagate_once, shift_mask, trusted_mask
from .metrics import (
    angular_error,
    endpoint_error,
    reprojection_residual,
    support_summary,
)
from .solver import SolveConfig, estimate_flow
from . import visualize as vz


def to_gray(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            w = np.array([0.299, 0.587, 0.114])
            return arr[..., :3].astype(np.float64) @ w
        return arr[..., 0].astype(np.float64)
    return arr.astype(np.float64)


def vector_grid(u: np.ndarray, v: np.ndarray, stride: int = 22,
                reliable: np.ndarray | None = None,
                coord_scale: float = 1.0,
                value_scale: float = 1.0) -> list[list]:
    """Subsample the dense field.

    ``coord_scale`` maps native level coords to full-res (1/scale); the same
    factor multiplies displacement values because d_full = d_level / scale.
    Returns rows ``[x, y, dx, dy, ok]`` in full-res coordinates/units.
    """
    h, w = u.shape
    rows = []
    for y in range(stride // 2, h, stride):
        for x in range(stride // 2, w, stride):
            ok = 1 if reliable is None or reliable[y, x] else 0
            rows.append([
                round((x + 0.5) * coord_scale - 0.5, 1),
                round((y + 0.5) * coord_scale - 0.5, 1),
                float(u[y, x]) * value_scale,
                float(v[y, x]) * value_scale,
                ok,
            ])
    return rows


def b64_f32(a: np.ndarray) -> str:
    return base64.b64encode(np.asarray(a, np.float32).tobytes()).decode("ascii")


def analyze_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    params: dict | SolveConfig | None = None,
    mask_a: np.ndarray | None = None,
    claim_mask_b: np.ndarray | None = None,
    run_candidates: bool = True,
    candidate_region: str = "auto",
    max_branches: int = 4,
    motion_hint: tuple[float, float] | None = None,
    gt_flow: np.ndarray | None = None,
    gt_valid: np.ndarray | None = None,
) -> dict:
    """Run the full kernel on an adjacent frame pair.

    All returned numpy arrays stay in the result for in-process callers;
    use :func:`api_payload` for a JSON/browser-ready dictionary.
    """
    if isinstance(params, SolveConfig):
        cfg = params
    else:
        p = dict(params or {})
        allowed = {f for f in SolveConfig.__dataclass_fields__}  # type: ignore[attr-defined]
        cfg = SolveConfig(**{k: v for k, v in p.items() if k in allowed})

    a = to_gray(img_a)
    b = to_gray(img_b)
    h, w = a.shape

    fwd = estimate_flow(a, b, cfg)
    bwd = estimate_flow(b, a, cfg)

    fb_error, fb_valid = forward_backward_error(
        fwd["u"], fwd["v"], bwd["u"], bwd["v"]
    )
    fb_ok = fb_valid & (fb_error < cfg.fb_tol)
    trusted = trusted_mask(fwd.get("support"), fb_error, cfg.fb_tol)
    bwd_trusted = trusted_mask(bwd.get("support"), None, cfg.fb_tol)

    reproj = reprojection_residual(a, b, bwd["u"], bwd["v"])
    sup_summary = support_summary(fwd["support"], fb_error)

    gt_summary = None
    if gt_flow is not None:
        gt_summary = {
            "epe": endpoint_error(fwd["u"], fwd["v"],
                                  gt_flow[..., 0], gt_flow[..., 1], gt_valid),
            "angular": angular_error(fwd["u"], fwd["v"],
                                     gt_flow[..., 0], gt_flow[..., 1], gt_valid),
        }

    # ---- optional foreground-label propagation on the base branch --------
    label_info = None
    mask_b_base = None
    d_f = (0.0, 0.0)
    if mask_a is not None:
        ma = np.asarray(mask_a) > 0
        labels_b, assigned = propagate_once(
            ma.astype(np.uint8), bwd["u"], bwd["v"], trusted
        )
        mask_b_base = labels_b > 0
        if ma.sum() >= 10:
            d_f = (
                float(np.median(fwd["u"][ma])),
                float(np.median(fwd["v"][ma])),
            )
        predicted_claim = shift_mask(ma, *d_f)
        if claim_mask_b is not None:
            predicted_claim = predicted_claim | (np.asarray(claim_mask_b) > 0)
        label_info = {
            "mask_a": ma,
            "labels_b": labels_b,
            "assigned": assigned,
            "predicted_claim": predicted_claim,
            "d_f": d_f,
        }

    # ---- cross-scale candidates -----------------------------------------
    branches = []
    selection = None
    candidate_meta = []
    if run_candidates:
        from PIL import ImageFilter, Image

        from .candidates import existing_branch, score_fields

        ma = label_info["mask_a"] if label_info is not None else None

        # Foreground displacement from masked block matching.  This rescues
        # small/fast objects that disappear at the coarsest pyramid level.
        fg_shift = None
        fg_cost = None
        if ma is not None and ma.sum() >= 30:
            eroded = np.array(
                Image.fromarray(ma.astype(np.uint8) * 255)
                .filter(ImageFilter.MinFilter(5))
            ) > 0
            fg_region = eroded if eroded.sum() >= 20 else ma
            fg_cands = block_match_shifts(
                a, b, region=fg_region,
                level=min(2, cfg.num_levels - 1),
                radius=16, top_k=3, min_separation=4,
            )
            if fg_cands:
                fg_shift = tuple(fg_cands[0]["shift_full"])
                fg_cost = fg_cands[0]["cost"]

        # Decide the foreground velocity used for boundary geometry: prefer
        # the sub-pixel base median when it is well supported inside the mask;
        # otherwise the label tracker's trusted velocity hint, then masked
        # block matching.
        hint_shift = (tuple(float(z) for z in motion_hint)
                      if motion_hint is not None else None)
        if ma is not None and ma.any():
            base_trusted_frac = float(trusted[ma].mean())
            base_mag = abs(d_f[0]) + abs(d_f[1])
            if base_trusted_frac < 0.5:
                d_f = hint_shift if hint_shift is not None else (
                    fg_shift if fg_shift is not None else d_f)
            elif base_mag < 0.6 and fg_shift is not None and fg_cost < 12:
                d_f = fg_shift
            if label_info is not None:
                label_info["d_f"] = d_f
                label_info["predicted_claim"] = shift_mask(ma, *d_f)
                if claim_mask_b is not None:
                    label_info["predicted_claim"] |= np.asarray(claim_mask_b) > 0

        mask_b_geom = shift_mask(ma, *d_f) if ma is not None else mask_b_base
        trailing = None
        if ma is not None and (abs(d_f[0]) + abs(d_f[1])) > 0.5:
            trailing = motion_side_band(mask_b_geom, d_f, width=20)

        claim_for_score: np.ndarray | None
        if claim_mask_b is not None:
            # An external label claim is the independent, stateful cue; do
            # not union it with this call's own displacement prediction.
            claim_for_score = np.asarray(claim_mask_b) > 0
        else:
            claim_for_score = (label_info["predicted_claim"]
                               if label_info is not None else None)
        base_fb_err, base_scores = score_fields(
            fwd, bwd, a, b, cfg,
            mask_a=ma, mask_b=mask_b_geom, trailing_band=trailing,
            claim_mask_b=claim_for_score,
            gt_flow=gt_flow, gt_valid=gt_valid,
        )
        branches.append(existing_branch(fwd, bwd, base_fb_err,
                                        base_scores, label="pyramid_base"))

        # Composite fused branches: trusted base field outside the foreground
        # mask, a candidate displacement inside it.  Refined directly at full
        # resolution so a small/fast region is not erased on top.  We keep
        # both the masked block-match shift and the label tracker's motion
        # hint when they disagree, and let the claim/boundary scores choose.
        fuse_shifts: list[tuple[str, tuple[float, float]]] = []
        if fg_shift is not None and ma is not None:
            fuse_shifts.append(("foreground_fused", tuple(fg_shift)))
        if hint_shift is not None and ma is not None:
            if all(abs(hint_shift[0] - s[1][0])
                   + abs(hint_shift[1] - s[1][1]) > 3.0
                   for s in fuse_shifts):
                fuse_shifts.append(("label_claim_fused", hint_shift))

        for fuse_label, fuse_shift in fuse_shifts:
            from .candidates import refine_seed

            fuse_region = np.array(
                Image.fromarray(ma.astype(np.uint8) * 255)
                .filter(ImageFilter.MaxFilter(9))
            ) > 0
            seed_u = fwd["u"].copy()
            seed_v = fwd["v"].copy()
            seed_u[fuse_region] = fuse_shift[0]
            seed_v[fuse_region] = fuse_shift[1]

            seed_bu = bwd["u"].copy()
            seed_bv = bwd["v"].copy()
            if mask_b_geom is not None:
                fuse_region_b = np.array(
                    Image.fromarray(mask_b_geom.astype(np.uint8) * 255)
                    .filter(ImageFilter.MaxFilter(9))
                ) > 0
                seed_bu[fuse_region_b] = -fuse_shift[0]
                seed_bv[fuse_region_b] = -fuse_shift[1]

            fused = refine_seed(
                a, b, seed_u, seed_v, cfg,
                seed_bwd_u=seed_bu, seed_bwd_v=seed_bv,
                mask_a=ma, mask_b=mask_b_geom, trailing_band=trailing,
                claim_mask_b=claim_for_score,
                gt_flow=gt_flow, gt_valid=gt_valid,
                label=fuse_label,
                shift_full=fuse_shift,
            )
            fused.candidate["region"] = "foreground"
            if fuse_label == "foreground_fused":
                fused.candidate["match_cost"] = fg_cost
            branches.append(fused)

        regions: list[tuple[str, np.ndarray | None]] = []
        if ma is not None and candidate_region in ("auto", "background"):
            dil = np.array(
                Image.fromarray((~ma).astype(np.uint8) * 255)
                .filter(ImageFilter.MaxFilter(9))
            ) > 0
            bg = ~dil
            if bg.sum() > 200:
                regions.append(("background", bg))
        if candidate_region in ("auto", "global") or not regions:
            regions.append(("global", None))

        seen = {(0, 0)}
        raw_cands = []
        match_level = min(2, cfg.num_levels - 1)
        rad = max(10, int(np.ceil(18 * 2 ** match_level / 4)))

        if fg_shift is not None:
            key = (round(fg_shift[0] / 4), round(fg_shift[1] / 4))
            if key not in seen:
                seen.add(key)
                raw_cands.append({"shift_full": list(fg_shift),
                                  "cost": fg_cost, "region": "foreground"})

        for rname, region in regions:
            cs = block_match_shifts(
                a, b, region=region, level=match_level,
                radius=14 if rname == "background" else max(10, rad),
                top_k=max_branches + 2, min_separation=6,
            )
            for c in cs:
                key = (round(c["shift_full"][0] / 4),
                       round(c["shift_full"][1] / 4))
                if key in seen:
                    continue
                seen.add(key)
                c["region"] = rname
                raw_cands.append(c)

        raw_cands = raw_cands[:max_branches - 1]

        for c in raw_cands:
            br = refine_candidate(
                a, b, tuple(c["shift_full"]), cfg,
                mask_a=ma,
                mask_b=mask_b_geom,
                trailing_band=trailing,
                claim_mask_b=claim_for_score,
                gt_flow=gt_flow, gt_valid=gt_valid,
            )
            br.candidate.update(c)
            branches.append(br)

        sel_idx, expl = choose_branch(branches)
        selection = expl
        selection["index"] = sel_idx

        # Re-propagate labels with the selected backward branch if it is
        # not the base branch (base labels already computed).
        if label_info is not None and sel_idx >= 0:
            from .coords import backward_cycle_error

            sel_b = branches[sel_idx].bwd
            sb_err, _ = backward_cycle_error(
                branches[sel_idx].fwd["u"], branches[sel_idx].fwd["v"],
                sel_b["u"], sel_b["v"],
            )
            strusted = trusted_mask(sel_b.get("support"), sb_err, cfg.fb_tol)
            labels_sel, assigned_sel = propagate_once(
                label_info["mask_a"].astype(np.uint8),
                sel_b["u"], sel_b["v"], strusted,
            )
            label_info["labels_b_selected"] = labels_sel
            label_info["assigned_selected"] = assigned_sel
            label_info["b_trusted_selected"] = strusted

        for i, br in enumerate(branches):
            candidate_meta.append({
                "index": i,
                "shift_full": [float(br.candidate["shift_full"][0]),
                               float(br.candidate["shift_full"][1])],
                "label": br.candidate.get("label"),
                "region": br.candidate.get("region"),
                "match_cost": br.candidate.get("cost"),
                "scores": _jsonable(br.scores),
                "selected": i == sel_idx,
            })

    return {
        "width": w,
        "height": h,
        "convention": CONVENTION,
        "config": cfg.to_dict(),
        "gray_a": a,
        "gray_b": b,
        "fwd": fwd,
        "bwd": bwd,
        "fb_error": fb_error,
        "fb_ok": fb_ok,
        "trusted": trusted,
        "reproj": reproj,
        "summary": {
            "support": sup_summary,
            "reprojection": {k: v for k, v in reproj.items()
                             if k not in ("map", "valid", "reprojected")},
            "ground_truth": gt_summary,
            "forward_converged": fwd["converged"],
            "backward_converged": bwd["converged"],
        },
        "label_info": label_info,
        "branches": branches,
        "candidates": candidate_meta,
        "selection": selection,
    }


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def level_payloads(res: dict, stride: int = 26) -> list[dict]:
    """Vector grids + image URLs for every pyramid scale of the fwd field."""
    out = []
    level_fields = res["fwd"]["level_fields"]
    for lf in level_fields:
        k = lf["level"]
        scale = lf["scale"]
        u, v = lf["u"], lf["v"]
        native_stride = max(4, int(round(stride * scale)))
        grid = vector_grid(u, v, stride=native_stride,
                           coord_scale=1.0 / scale, value_scale=1.0 / scale)
        flow_img = vz.flow_to_rgb(u, v)
        # Residual at this level against the matching pyramid image is not
        # retained; show flow + vector grid and level convergence history.
        rep = next((x for x in res["fwd"]["levels"] if x["level"] == k), {})
        out.append({
            "level": k,
            "scale": scale,
            "native_width": u.shape[1],
            "native_height": u.shape[0],
            "vectors": grid,
            "flow_url": vz.to_data_url(flow_img),
            "converged": rep.get("converged"),
            "history": rep.get("history", []),
            "final_objective": rep.get("final_objective"),
        })
    return out


def _view_payload(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
    fwd: dict,
    bwd: dict,
    cfg: SolveConfig,
    stride: int,
    fb_error: np.ndarray,
    label_mask_b: np.ndarray | None = None,
) -> dict:
    h, w = gray_a.shape
    sup = fwd["support"]
    reproj = reprojection_residual(gray_a, gray_b, bwd["u"], bwd["v"])
    fb_ok = np.isfinite(fb_error) & (fb_error < cfg.fb_tol)
    reliable = sup["reliable_photo"] & fb_ok
    grid = vector_grid(fwd["u"], fwd["v"], stride=stride, reliable=reliable)

    images = {
        "flow": vz.to_data_url(vz.flow_to_rgb(
            fwd["u"], fwd["v"], support=sup["score"])),
        "residual": vz.to_data_url(vz.scalar_heatmap(reproj["map"], vmax=40)),
        "support": vz.to_data_url(vz.support_to_rgb(sup["score"], reliable)),
        "reprojected": vz.to_data_url(
            vz.Image.fromarray(np.clip(reproj["reprojected"], 0, 255).astype(np.uint8))
            .convert("RGB")),
        "vectors": vz.to_data_url(vz.draw_vectors(
            np.stack([gray_a] * 3, -1).astype(np.uint8),
            fwd["u"], fwd["v"], stride=stride, bad=reliable)),
        "labels": None,
    }
    if label_mask_b is not None:
        images["labels"] = vz.to_data_url(vz.labels_overlay(
            np.stack([gray_b] * 3, -1).astype(np.uint8), label_mask_b))

    fields = {
        "u": b64_f32(fwd["u"]),
        "v": b64_f32(fwd["v"]),
        "bu": b64_f32(bwd["u"]),
        "bv": b64_f32(bwd["v"]),
        "residual": b64_f32(reproj["map"]),
        "support": b64_f32(sup["score"]),
        "fb_error": b64_f32(np.nan_to_num(fb_error, nan=-1.0)),
    }
    summary = {
        "support": support_summary(sup, fb_error),
        "reprojection": {k: v for k, v in reproj.items()
                         if k not in ("map", "valid", "reprojected")},
        "converged": bool(fwd.get("converged", False)),
    }
    return {
        "width": w, "height": h, "stride": stride,
        "vectors": grid, "images": images,
        "fields_b64": fields, "view_summary": summary,
    }


def branch_payload(res: dict, index: int, stride: int = 24) -> dict:
    """Browser payload for one preserved cross-scale candidate branch."""
    br = res["branches"][index]
    label_mask = None
    if res["label_info"] is not None:
        from .coords import backward_cycle_error

        li = res["label_info"]
        berr, _ = backward_cycle_error(
            br.fwd["u"], br.fwd["v"], br.bwd["u"], br.bwd["v"])
        trusted = trusted_mask(br.bwd.get("support"), berr,
                               res["config"]["fb_tol"])
        labels_b, _ = propagate_once(
            li["mask_a"].astype(np.uint8), br.bwd["u"], br.bwd["v"], trusted
        )
        label_mask = labels_b
    view = _view_payload(
        res["gray_a"], res["gray_b"], br.fwd, br.bwd,
        SolveConfig(**{k: v for k, v in res["config"].items()
                       if k in SolveConfig.__dataclass_fields__}),  # type: ignore[attr-defined]
        stride, br.fb_error, label_mask,
    )
    view["branch_index"] = index
    view["scores"] = _jsonable(br.scores)
    return view


def api_payload(res: dict, stride: int = 24) -> dict:
    """Browser-ready JSON-able payload including binary field blobs and PNGs."""
    sel_idx = res["selection"]["index"] if res.get("selection") else 0
    active = sel_idx if 0 <= sel_idx < len(res["branches"]) else 0

    label_mask = None
    if res["label_info"] is not None:
        label_mask = res["label_info"].get(
            "labels_b_selected", res["label_info"]["labels_b"])

    if res["branches"]:
        ab = res["branches"][active]
        view = _view_payload(
            res["gray_a"], res["gray_b"], ab.fwd, ab.bwd,
            SolveConfig(**{k: v for k, v in res["config"].items()
                           if k in SolveConfig.__dataclass_fields__}),  # type: ignore[attr-defined]
            stride, ab.fb_error, label_mask,
        )
    else:
        view = _view_payload(
            res["gray_a"], res["gray_b"], res["fwd"], res["bwd"],
            SolveConfig(**{k: v for k, v in res["config"].items()
                           if k in SolveConfig.__dataclass_fields__}),  # type: ignore[attr-defined]
            stride, res["fb_error"], label_mask,
        )

    payload = {
        **view,
        "active_branch": active,
        "convention": res["convention"],
        "config": res["config"],
        "summary": _jsonable(res["summary"]),
        "levels": [],
        "candidates": res["candidates"],
        "selection": res["selection"],
    }
    try:
        payload["levels"] = level_payloads(res)
    except Exception as exc:  # pragma: no cover
        payload["levels_error"] = str(exc)
    return payload
