"""End-to-end experiment runner: every built-in sequence, every adjacent pair.

This module is the single path that produces *actual* per-frame results:

* runs the variational dense-flow kernel in both directions;
* runs cross-scale candidate branching / selection where relevant;
* drives the LabelTracker closed loop (mask + claim + motion hint);
* writes PNG views, per-pair JSON (with binary float fields for the hover
  inspector), ``manifest.json`` and ``summary.csv``.

The stdlib server reads these artifacts to drive the playback UI, and can
re-run a single pair through :func:`rebuild_pair_result` for branch switching.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

from . import synth, visualize as vz
from .candidates import choose_branch  # noqa: F401  (re-export convenience)
from .coords import backward_cycle_error
from .labels import LabelTracker, trusted_mask
from .metrics import angular_error, endpoint_error, label_scores
from .pipeline import _view_payload, analyze_pair, level_payloads
from .solver import SolveConfig


DEFAULT_ALPHA = 120.0

# Sequences that get the full candidate / label treatment.
TRACKER_SEQUENCES = {"texture_reveal", "local_motion", "repeat_occlude"}
CANDIDATE_SEQUENCES = {"texture_reveal", "repeat_occlude", "local_motion"}


def _rel(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def _save_data_url(url: str, path: str) -> None:
    if not url or not url.startswith("data:"):
        return
    b64 = url.split(",", 1)[1]
    with open(path, "wb") as fh:
        fh.write(base64.b64decode(b64))


def _u8_b64(mask: np.ndarray | None) -> str | None:
    if mask is None:
        return None
    a = np.asarray(mask, np.uint8)
    return base64.b64encode(a.tobytes()).decode("ascii")


def _u8_unb64(s: str | None, h: int, w: int) -> np.ndarray | None:
    if not s:
        return None
    return np.frombuffer(base64.b64decode(s), np.uint8).reshape(h, w)


def _gt_motion_metrics(res: dict, seq: synth.Sequence, t: int) -> dict:
    out: dict = {}
    if t >= len(seq.flows):
        return out
    gt = seq.flows[t]
    gv = seq.valid[t] if t < len(seq.valid) else None
    fwd = res["branches"][res["selection"]["index"]].fwd \
        if res.get("branches") and res.get("selection") else res["fwd"]
    out["global"] = {
        "epe": endpoint_error(fwd["u"], fwd["v"], gt[..., 0], gt[..., 1], gv),
        "angular": angular_error(fwd["u"], fwd["v"],
                                 gt[..., 0], gt[..., 1], gv),
    }
    labels_t = seq.labels[t] if t < len(seq.labels) else None
    if labels_t is not None and (labels_t > 0).sum() >= 20:
        m = labels_t > 0
        out["foreground"] = {
            "epe": endpoint_error(fwd["u"], fwd["v"],
                                  gt[..., 0], gt[..., 1], m),
            "angular": angular_error(fwd["u"], fwd["v"],
                                     gt[..., 0], gt[..., 1], m),
            "pixels": int(m.sum()),
        }
        bg = (~m) if gv is None else ((~m) & gv)
        if bg.sum() >= 20:
            out["background"] = {
                "epe": endpoint_error(fwd["u"], fwd["v"],
                                      gt[..., 0], gt[..., 1], bg),
            }
    return out


def run_sequence(
    name: str,
    out_root: str,
    cfg: SolveConfig | None = None,
    verbose: bool = True,
) -> dict:
    cfg = cfg or SolveConfig(num_levels=4, alpha=DEFAULT_ALPHA)
    seq = synth.REGISTRY[name]()
    seq_dir = os.path.join(out_root, name)
    os.makedirs(seq_dir, exist_ok=True)
    h, w = seq.frames[0].shape[:2]

    for i, fr in enumerate(seq.frames):
        Image = vz.Image
        Image.fromarray(fr).save(os.path.join(seq_dir, f"frame_{i:03d}.png"))

    use_tracker = name in TRACKER_SEQUENCES and (seq.labels[0] > 0).any()
    run_cand = name in CANDIDATE_SEQUENCES
    tracker: LabelTracker | None = None
    if use_tracker:
        tracker = LabelTracker(hold_ttl=6)
        tracker.seed(seq.labels[0])
        tracker.seed_appearance(seq.frames[0], 1, tol=95)

    pair_index: list[dict] = []
    rows: list[dict] = []

    for t in range(len(seq.frames) - 1):
        t0 = time.time()
        a = seq.frames[t].astype(np.float64).mean(axis=2)
        b = seq.frames[t + 1].astype(np.float64).mean(axis=2)

        mask_a = None
        claim_b = None
        hint = None
        if tracker is not None:
            mask_img = tracker.composite_label_image((h, w))
            if mask_img.any():
                mask_a = mask_img
            claim = tracker.predict_claim()
            if claim.any():
                claim_b = claim
            hint = tracker.motion_hint(1)
        elif run_cand and (seq.labels[t] > 0).any():
            mask_a = seq.labels[t]

        res = analyze_pair(
            a, b, cfg,
            mask_a=mask_a,
            claim_mask_b=claim_b,
            motion_hint=hint,
            run_candidates=run_cand,
            max_branches=5,
            gt_flow=seq.flows[t] if t < len(seq.flows) else None,
            gt_valid=seq.valid[t] if t < len(seq.valid) else None,
        )
        sel_idx = res["selection"]["index"] if res.get("selection") else 0
        br = res["branches"][sel_idx] if res.get("branches") else None

        # ---- tracker update with the selected branch ----------------------
        tracker_out = None
        label_metrics = None
        if tracker is not None and br is not None:
            berr, _ = backward_cycle_error(
                br.fwd["u"], br.fwd["v"], br.bwd["u"], br.bwd["v"])
            b_trust = trusted_mask(br.bwd.get("support"), berr, cfg.fb_tol)
            claim_dice = br.scores.get("claim_dice")
            tracker_out = tracker.step(
                t + 1,
                mask_a if mask_a is not None
                else np.zeros((h, w), np.uint8),
                br.bwd["u"], br.bwd["v"], b_trust,
                fu=br.fwd["u"], fv=br.fwd["v"],
                fwd_trusted=br.fwd["support"]["reliable_photo"],
                claim_evidence=({"dice": claim_dice}
                                if claim_dice is not None else None),
                current_rgb=seq.frames[t + 1],
            )
            state_lab = tracker_out["labels"]
            label_metrics = {
                "state": label_scores(state_lab, seq.labels[t + 1],
                                      decoy_mask=seq.decoy_mask),
                "trusted": label_scores(tracker_out["trusted_labels"],
                                         seq.labels[t + 1],
                                         decoy_mask=seq.decoy_mask),
            }
            st = tracker.states.get(1)
            label_metrics["tracker"] = {
                "status": st.status if st else None,
                "coverage": st.coverage if st else None,
                "held_frames": st.held_frames if st else None,
                "velocity": (list(st.velocity) if st else None),
            }

        # ---- view payload + PNG artifacts --------------------------------
        active = br if br is not None else None
        view = _view_payload(
            a, b,
            active.fwd if active else res["fwd"],
            active.bwd if active else res["bwd"],
            cfg, stride=24,
            fb_error=active.fb_error if active else res["fb_error"],
        )
        pair_stem = f"pair_{t:03d}"
        img_urls = {}
        for key, url in view["images"].items():
            if url is None:
                continue
            fname = f"{pair_stem}_{key}.png"
            _save_data_url(url, os.path.join(seq_dir, fname))
            img_urls[key] = fname

        # Label overlay with held region in amber.
        if tracker_out is not None:
            lab_img = vz.labels_overlay(
                seq.frames[t + 1], tracker_out["labels"],
                held=tracker_out["held_map"])
            lab_fname = f"{pair_stem}_labels_held.png"
            lab_img.save(os.path.join(seq_dir, lab_fname))
            img_urls["labels_held"] = lab_fname

        # Per-branch flow thumbnails.
        branch_files = []
        for bi, bch in enumerate(res.get("branches", [])):
            fim = vz.flow_to_rgb(bch.fwd["u"], bch.fwd["v"],
                                 support=bch.fwd["support"]["score"])
            fn = f"{pair_stem}_branch{bi}_flow.png"
            fim.save(os.path.join(seq_dir, fn))
            branch_files.append(fn)

        gt_metrics = _gt_motion_metrics(res, seq, t)

        context = {
            "params": cfg.to_dict(),
            "motion_hint": list(hint) if hint else None,
            "mask_a_u8": _u8_b64(mask_a),
            "claim_b_u8": _u8_b64(claim_b),
            "height": h, "width": w,
        }

        pair_json = {
            "sequence": name,
            "pair": t,
            "frame_a": t,
            "frame_b": t + 1,
            "frame_url_a": f"frame_{t:03d}.png",
            "frame_url_b": f"frame_{t+1:03d}.png",
            "image_urls": img_urls,
            "branch_image_urls": branch_files,
            "width": w,
            "height": h,
            "stride": view["stride"],
            "vectors": view["vectors"],
            "fields_b64": view["fields_b64"],
            "view_summary": view["view_summary"],
            "convention": res["convention"],
            "config": res["config"],
            "summary": res["summary"],
            "candidates": res["candidates"],
            "selection": res.get("selection"),
            "levels": _safe_levels(res),
            "gt_metrics": gt_metrics,
            "label_metrics": label_metrics,
            "tracker_events": tracker_out["events"] if tracker_out else [],
            "context": context,
            "elapsed_sec": round(time.time() - t0, 2),
        }
        with open(os.path.join(seq_dir, f"{pair_stem}.json"), "w") as fh:
            json.dump(pair_json, fh)

        compact = {
            "pair": t,
            "json_url": f"{name}/{pair_stem}.json",
            "frame_a_url": f"{name}/frame_{t:03d}.png",
            "frame_b_url": f"{name}/frame_{t+1:03d}.png",
            "main_view_url": f"{name}/{img_urls.get('flow', '')}",
            "selected_branch": sel_idx,
            "elapsed_sec": pair_json["elapsed_sec"],
        }
        if gt_metrics.get("global", {}).get("epe"):
            compact["epe_median"] = gt_metrics["global"]["epe"]["median"]
        if label_metrics:
            lm = label_metrics["state"]
            compact["label_iou"] = lm["iou"]
            compact["decoy_fp"] = lm.get(
                "same_color_decoy_false_foreground")
            compact["tracker_status"] = label_metrics["tracker"]["status"]
        pair_index.append(compact)

        row = {
            "sequence": name, "pair": t,
            "selected_branch": sel_idx,
            "epe_median": _dig(gt_metrics, "global", "epe", "median"),
            "epe_mean": _dig(gt_metrics, "global", "epe", "mean"),
            "fg_epe_median": _dig(gt_metrics, "foreground", "epe", "median"),
            "ae_median": _dig(gt_metrics, "global", "angular", "median"),
            "reproj_median": _dig(res["summary"], "reprojection", "median"),
            "support_fully_reliable": _dig(
                res["summary"], "support", "fully_reliable_fraction"),
            "label_iou": (label_metrics or {}).get("state", {}).get("iou")
                         if label_metrics else None,
            "decoy_fp": (label_metrics or {}).get("state", {}).get(
                "same_color_decoy_false_foreground")
                        if label_metrics else None,
            "tracker_status": (label_metrics or {}).get("tracker", {}).get(
                "status") if label_metrics else None,
        }
        rows.append(row)
        if verbose:
            print(f"  [{name}] pair {t}: {pair_json['elapsed_sec']}s "
                  f"sel={sel_idx} epe={row['epe_median']} "
                  f"fg={row['fg_epe_median']} iou={row['label_iou']} "
                  f"status={row['tracker_status']}")

    report = {
        "name": name,
        "title": name.replace("_", " "),
        "n_frames": len(seq.frames),
        "n_pairs": len(seq.frames) - 1,
        "meta": seq.meta,
        "events_hint": seq.events,
        "decoy_present": seq.decoy_mask is not None,
        "pairs": pair_index,
    }
    return {"report": report, "rows": rows}


def _safe_levels(res: dict) -> list:
    try:
        return level_payloads(res)
    except Exception as exc:  # pragma: no cover
        return [{"error": str(exc)}]


def _dig(d: dict, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def large_displacement_study(out_root: str) -> dict:
    """Quantify the pyramid-level-dependent large displacement failure."""
    study_dir = os.path.join(out_root, "_studies")
    os.makedirs(study_dir, exist_ok=True)
    results = []
    for dx in (16.0, 32.0, 52.0, 80.0):
        seq = synth.gen_large_displacement(dx=dx)
        a = seq.frames[0].astype(float).mean(2)
        b = seq.frames[1].astype(float).mean(2)
        for levels in (3, 5):
            cfg = SolveConfig(num_levels=levels, alpha=DEFAULT_ALPHA)
            t0 = time.time()
            res = analyze_pair(a, b, cfg, run_candidates=False)
            e = endpoint_error(res["fwd"]["u"], res["fwd"]["v"],
                               seq.flows[0][..., 0], seq.flows[0][..., 1],
                               seq.valid[0])
            results.append({
                "dx": dx, "levels": levels,
                "epe_median": e["median"], "epe_mean": e["mean"],
                "inlier_1px": e["inlier_1px"],
                "elapsed_sec": round(time.time() - t0, 2),
            })
            fim = vz.flow_to_rgb(res["fwd"]["u"], res["fwd"]["v"])
            fim.save(os.path.join(
                study_dir, f"large_dx{int(dx)}_L{levels}_flow.png"))
    with open(os.path.join(study_dir, "large_displacement.json"), "w") as fh:
        json.dump({"results": results}, fh, indent=2)
    return {"results": results}


def run_all(out_root: str = "experiments/output") -> dict:
    os.makedirs(out_root, exist_ok=True)
    all_reports = []
    all_rows: list[dict] = []
    for name in synth.REGISTRY:
        print(f"== sequence {name} ==")
        out = run_sequence(name, out_root)
        all_reports.append(out["report"])
        all_rows.extend(out["rows"])

    print("== large displacement study ==")
    study = large_displacement_study(out_root)
    for r in study["results"]:
        print(f"  dx={r['dx']:>4} levels={r['levels']} "
              f"EPEmed={r['epe_median']:.2f} inlier={r['inlier_1px']:.2f}")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "convention_note": "pixel centers integer (x=col,y=row); see denseflow/coords.py",
        "sequences": all_reports,
        "large_displacement_study": study,
    }
    with open(os.path.join(out_root, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    if all_rows:
        keys = list(all_rows[0].keys())
        with open(os.path.join(out_root, "summary.csv"), "w", newline="") as fh:
            wri = csv.DictWriter(fh, fieldnames=keys)
            wri.writeheader()
            wri.writerows(all_rows)
    with open(os.path.join(out_root, "summary.json"), "w") as fh:
        json.dump({"rows": all_rows,
                   "large_displacement": study["results"]}, fh, indent=2)
    return manifest


# ---------------------------------------------------------------------------
# On-demand single-pair rebuild for the server /api/rebranch endpoint.
# ---------------------------------------------------------------------------

def rebuild_pair_result(seq_name: str, pair: int, pair_json: dict) -> dict:
    """Re-run analyze_pair for an artifact pair using its stored context."""
    seq = synth.REGISTRY[seq_name]()
    a = seq.frames[pair].astype(float).mean(2)
    b = seq.frames[pair + 1].astype(float).mean(2)
    ctx = pair_json.get("context", {})
    h, w = ctx.get("height", a.shape[0]), ctx.get("width", a.shape[1])
    mask_a = _u8_unb64(ctx.get("mask_a_u8"), h, w)
    claim = _u8_unb64(ctx.get("claim_b_u8"), h, w)
    hint = ctx.get("motion_hint")
    hint_t = tuple(hint) if hint else None
    params = ctx.get("params", {})
    cfg = SolveConfig(**{k: v for k, v in params.items()
                         if k in SolveConfig.__dataclass_fields__})
    res = analyze_pair(
        a, b, cfg,
        mask_a=mask_a,
        claim_mask_b=(claim > 0) if claim is not None else None,
        motion_hint=hint_t,
        run_candidates=bool(pair_json.get("candidates")),
        max_branches=5,
        gt_flow=seq.flows[pair] if pair < len(seq.flows) else None,
        gt_valid=seq.valid[pair] if pair < len(seq.valid) else None,
    )
    return res


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "experiments/output"
    run_all(root)
    print(f"wrote artifacts to {root}")
