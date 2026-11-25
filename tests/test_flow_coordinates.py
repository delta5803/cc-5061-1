"""Executable coordinate-convention checks (pytest or direct).

Run either::

    python3 tests/test_flow_coordinates.py
    python3 -m pytest tests/test_flow_coordinates.py -q

These tests generate every frame they use; they never read
``experiments/output`` or hand-edited data, so they pass in a clean
checkout.  One test additionally runs ``check_flow_coordinates.py`` as a
subprocess from an empty working directory to prove cwd/data independence.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from denseflow.coord_audit import (  # noqa: E402
    LevelFrame,
    analytic_motion_pair,
    check_scale_crop_roundtrip,
    run_audit,
    scale_and_crop,
)
from denseflow.coords import (  # noqa: E402
    forward_backward_error,
    pixel_grid,
    sample_bilinear,
)
from denseflow.pyramid import build_pyramid, prolong_flow  # noqa: E402


# ---------------------------------------------------------------------------
# Convention algebra
# ---------------------------------------------------------------------------

def test_forward_backward_worked_example():
    # A center x=10, forward +5 -> B center 15; backward at 15 is -5 -> A 10.
    x_a = np.array([10.0, 0.0])
    f = np.array([5.0, 0.0])
    x_b = x_a + f
    b_at_b = -f
    x_a_rt = x_b + b_at_b
    assert tuple(x_b) == (15.0, 0.0)
    assert np.allclose(x_a_rt, [10.0, 0.0])
    # The source/target-confused reading (use +F as the B-grid sample offset)
    # lands at 20 and must not be confused with the reprojection source.
    assert not np.allclose(x_b + f, x_a_rt)


def test_level_frame_octave_is_continuous_rule():
    fr = LevelFrame.octave(2, 64, 80)
    assert (fr.native_h, fr.native_w) == (16, 20)
    x0, y0 = fr.to_full(3.0, 5.0)
    assert (x0, y0) == (12.0, 20.0)          # 2**2 * index
    xl, yl = fr.to_level(x0, y0)
    assert (xl, yl) == (3.0, 5.0)
    du, dv = fr.displacement_to_full(1.5, -0.5)
    assert (du, dv) == (6.0, -2.0)           # displacement scales by 4


def test_prolong_flow_matches_frame_displacement():
    u = np.full((8, 8), 1.5)
    v = np.zeros((8, 8))
    uu, vv = prolong_flow(u, v, 32, 32, 0.25, 1.0)
    assert np.allclose(uu, 6.0)
    fr = LevelFrame.octave(2, 32, 32)
    du, _ = fr.displacement_to_full(1.5, 0.0)
    assert du == 6.0


def test_noninteger_scale_crop_explicit_origin_roundtrip():
    rep = check_scale_crop_roundtrip()
    assert rep["ok"], rep.get("traceback", rep["detail"])
    m = rep["metrics"]
    assert m["affine_forward_chase_max_err"] < 1e-9
    assert m["affine_backward_chase_max_err"] < 1e-9
    # Explicit origin registers the image marker accurately ...
    assert m["explicit_origin_marker_err"] < 1.7
    # ... while dropping the crop origin misses by roughly that origin.
    assert m["naive_origin_free_marker_err"] > 10.0


def test_scale_and_crop_validity_is_explicit():
    img = np.zeros((40, 50))
    out, valid, fr = scale_and_crop(img, 0.5, 20, 20, crop_x_scaled=0,
                                    crop_y_scaled=0)
    assert out.shape == (20, 20)
    assert valid.dtype == bool
    # Asking for a crop window outside the scaled image yields invalid cells.
    out2, valid2, _ = scale_and_crop(img, 0.5, 20, 20,
                                     crop_x_scaled=40, crop_y_scaled=0)
    assert valid2.sum() < valid.sum()
    assert out2.shape == (20, 20)


# ---------------------------------------------------------------------------
# Analytic generator: direction + explicit missing support
# ---------------------------------------------------------------------------

def test_analytic_direction_signs_and_nan_support():
    pair = analytic_motion_pair(128, 160, dx=10.0, dy=-6.0, zoom=1.04)
    assert np.median(pair["fwd_u"][:, int(0.62 * 160):int(0.9 * 160)]) > 8
    assert np.median(pair["fwd_v"][int(0.1 * 128):int(0.4 * 128), :]) < -4
    # Out-of-frame regions exist and are encoded as NaN + validity mask,
    # never as a finite zero motion that could be read as "proven static".
    assert (~pair["fwd_valid"]).sum() > 200
    assert (~pair["bwd_valid"]).sum() > 200
    assert np.isnan(pair["gt_fwd_nan"][..., 0]).sum() == \
        int((~pair["fwd_valid"]).sum())
    assert np.isnan(pair["gt_bwd_nan"][..., 1]).sum() == \
        int((~pair["bwd_valid"]).sum())
    z = analytic_motion_pair(128, 160, dx=0.0, dy=0.0, zoom=1.08)
    assert np.median(z["fwd_u"][:, 120:140]) > 1.0
    assert np.median(z["fwd_u"][:, 20:40]) < -1.0


def test_analytic_backward_field_inverts_forward_map():
    pair = analytic_motion_pair(96, 128, dx=7.0, dy=4.0, zoom=1.03)
    xs, ys = pixel_grid(96, 128)
    # x_A = x_B + B_B->A(x_B), then forward map must return to x_B.
    xa = xs + pair["bwd_u"]
    ya = ys + pair["bwd_v"]
    cx, cy = pair["center"]
    xb_rt = cx + pair["zoom"] * (xa - cx) + pair["dx"]
    yb_rt = cy + pair["zoom"] * (ya - cy) + pair["dy"]
    err = np.hypot(xb_rt - xs, yb_rt - ys)
    assert np.max(err[pair["bwd_valid"]]) < 1e-9


def test_analytic_reprojection_residual_zero_on_support():
    pair = analytic_motion_pair(96, 128, dx=6.0, dy=-3.0, zoom=1.02)
    xs, ys = pixel_grid(96, 128)
    reproj, valid = sample_bilinear(
        pair["a"], xs + pair["bwd_u"], ys + pair["bwd_v"])
    use = pair["bwd_valid"] & valid
    assert np.median(np.abs(reproj[use] - pair["b"][use])) < 1e-6


# ---------------------------------------------------------------------------
# Real kernel + full audit orchestration
# ---------------------------------------------------------------------------

def test_full_quick_audit_passes_and_writes_evidence(tmp_path):
    report = run_audit(out_dir=tmp_path / "evidence", quick=True)
    failed = [c["id"] for c in report["checks"] if not c["ok"]]
    assert report["ok"], json.dumps(
        [{"id": c["id"], "detail": c["detail"],
          "tb": c.get("traceback", "")[-800:]}
         for c in report["checks"] if not c["ok"]],
        ensure_ascii=False, indent=2)
    assert not failed
    jf = tmp_path / "evidence" / "coord_check_report.json"
    assert jf.is_file()
    on_disk = json.loads(jf.read_text(encoding="utf-8"))
    assert on_disk["passed"] == on_disk["n_checks"]


def test_audit_cli_runs_from_empty_cwd(tmp_path):
    """The coordinate check must pass from a clean directory with no data."""
    ev = tmp_path / "ev"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "check_flow_coordinates.py"),
         "--quick", "--out", str(ev)],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=240,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert (ev / "coord_check_report.json").is_file()
    report = json.loads((ev / "coord_check_report.json").read_text())
    assert report["ok"] is True
    # Flow direction and reprojection evidence must carry real numbers.
    kernel = next(c for c in report["checks"]
                  if c["id"] == "kernel_flow_direction_reprojection")
    assert kernel["metrics"]["median_u_interior"] > 3.5
    assert kernel["metrics"]["reproj_median_gray"] < 3.0


def test_fb_error_is_nan_outside_support_not_zero():
    # Coordinate-level guarantee: invalid cycle samples become NaN, so a zero
    # error can never be claimed where the backward sample is out of frame.
    fu = np.full((5, 5), 4.0)
    fv = np.zeros((5, 5))
    bu = np.full((5, 5), -4.0)
    bv = np.zeros((5, 5))
    err, valid = forward_backward_error(fu, fv, bu, bv)
    assert np.isnan(err[~valid]).all()
    assert not np.any(err[~valid] == 0.0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"ALL FLOW-COORDINATE CHECKS PASSED ({len(fns)})")
