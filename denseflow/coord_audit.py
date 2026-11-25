"""Executable coordinate-convention audit for DenseFlow.

This module is the maintenance check for:

* forward / backward displacement signs (source vs target grid);
* pyramid scale **and per-level grid origin** bookkeeping, including
  non-integer scaling followed by a crop;
* analytic translation+zoom pair generation with exact ground truth and
  explicit invalid-support masks (unsupported is never encoded as zero
  motion);
* real-kernel flow direction and reprojection-residual evidence;
* documentation consistency with :data:`denseflow.coords.CONVENTION`.

Everything checked here is generated in memory (or under an output
directory given by the caller).  The audit never reads hand-edited data or
``experiments/output`` artifacts, so it runs in a clean checkout.

Canonical CLI::

    python3 check_flow_coordinates.py            # full evidence run
    python3 check_flow_coordinates.py --quick     # fast (server uses this)
    python3 -m denseflow.coord_audit --out /tmp/coord_evidence

The chosen pyramid convention is an explicit affine frame per level
(:class:`LevelFrame`).  For the aligned integer-octave pyramid its origin is
``(0, 0)`` and it collapses to the continuous-plane rule
``x_k = s_k * x_0``.  The measurement in
:func:`check_scale_crop_roundtrip` is the recorded reason the origin term is
retained: after a non-integer scale + crop the origin-free rule registers
the same marker several full pixels away.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .coords import CONVENTION, pixel_grid, sample_bilinear

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Per-level affine coordinate frame (the executable pyramid convention)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LevelFrame:
    """Affine map between a pyramid level's center grid and level 0.

    Full-plane pixel-center coordinates ``(x0, y0)`` relate to level
    coordinates ``(xk, yk)`` by::

        x0 = ax * xk + ox
        y0 = ay * yk + oy

    and displacements (differences of points, origin cancels) by
    ``d0 = (ax * dxk, ay * dyk)``.  For an isotropic scale ``s`` relative to
    level 0, ``ax = ay = 1 / s``.
    """

    level: int
    native_w: int
    native_h: int
    scale: float = 1.0
    ax: float = 1.0
    ay: float = 1.0
    ox: float = 0.0
    oy: float = 0.0
    kind: str = "identity"

    # ---- constructors ----------------------------------------------------
    @classmethod
    def octave(cls, level: int, full_h: int, full_w: int) -> "LevelFrame":
        """Aligned stride-decimated octave: coarse index k <-> full 2**k*k."""
        s = 2.0**-level
        a = float(2**level)
        # ceil sizes agree with denseflow.pyramid.level_size
        nh = max(1, int(math.ceil(full_h * s)))
        nw = max(1, int(math.ceil(full_w * s)))
        return cls(level=level, native_w=nw, native_h=nh, scale=s,
                   ax=a, ay=a, ox=0.0, oy=0.0,
                   kind="octave_aligned_shared_origin")

    @classmethod
    def scaled_crop(
        cls,
        scale: float,
        out_w: int,
        out_h: int,
        crop_x_scaled: float = 0.0,
        crop_y_scaled: float = 0.0,
    ) -> "LevelFrame":
        """Non-integer scale followed by a crop.

        ``crop_x_scaled`` / ``crop_y_scaled`` is the crop window origin in
        the *scaled* image's center coordinates.  Output center ``j``
        samples the full plane at ``(crop + j) / scale``.
        """
        if scale <= 0:
            raise ValueError("scale must be positive")
        a = 1.0 / float(scale)
        return cls(
            level=-1, native_w=int(out_w), native_h=int(out_h),
            scale=float(scale), ax=a, ay=a,
            ox=crop_x_scaled * a, oy=crop_y_scaled * a,
            kind="noninteger_scale_then_crop_explicit_origin",
        )

    # ---- mappings --------------------------------------------------------
    def to_full(self, xk, yk):
        xk = np.asarray(xk, dtype=np.float64)
        yk = np.asarray(yk, dtype=np.float64)
        return self.ax * xk + self.ox, self.ay * yk + self.oy

    def to_level(self, x0, y0):
        x0 = np.asarray(x0, dtype=np.float64)
        y0 = np.asarray(y0, dtype=np.float64)
        return (x0 - self.ox) / self.ax, (y0 - self.oy) / self.ay

    def displacement_to_full(self, dxk, dyk):
        return self.ax * np.asarray(dxk, float), self.ay * np.asarray(dyk, float)

    def displacement_to_level(self, dx0, dy0):
        return np.asarray(dx0, float) / self.ax, np.asarray(dy0, float) / self.ay

    def naive_origin_free_to_full(self, xk, yk):
        """The tempting but incomplete ``x0 = xk / s`` map (drops crop origin)."""
        a = 1.0 / self.scale
        return a * np.asarray(xk, float), a * np.asarray(yk, float)

    def to_dict(self) -> dict:
        return asdict(self)


def scale_and_crop(
    img: np.ndarray,
    scale: float,
    out_w: int,
    out_h: int,
    crop_x_scaled: float = 0.0,
    crop_y_scaled: float = 0.0,
    cval: float = 0.0,
):
    """Actually resample ``img`` through the matching :class:`LevelFrame`.

    Returns ``(out, valid, frame)``.  Output pixel ``(j, i)`` reads the full
    plane at ``frame.to_full(j, i)``; out-of-bounds reads are invalid, never
    fabricated observations.
    """
    frame = LevelFrame.scaled_crop(
        scale, out_w, out_h, crop_x_scaled, crop_y_scaled)
    xs, ys = pixel_grid(out_h, out_w)
    sx, sy = frame.to_full(xs, ys)
    out, valid = sample_bilinear(img, sx, sy, cval=cval)
    return out, valid, frame


# ---------------------------------------------------------------------------
# Analytic translation + zoom pair generator
# ---------------------------------------------------------------------------


def smooth_analytic_texture(h: int, w: int, seed: int = 11) -> np.ndarray:
    """Low-frequency pattern where exact bilinear round trips stay tight."""
    rng = np.random.default_rng(seed)
    xs, ys = pixel_grid(h, w)
    img = (128.0
           + 52.0 * np.sin(xs / 21.0 + 0.6) * np.cos(ys / 25.0 - 0.4)
           + 26.0 * np.sin((xs + 1.7 * ys) / 13.0)
           + 14.0 * np.cos((xs - ys) / 9.0))
    img += 6.0 * gaussian_noise(h, w, rng, sigma=2.0)
    return np.clip(img, 0.0, 255.0)


def gaussian_noise(h: int, w: int, rng, sigma: float = 2.0) -> np.ndarray:
    from .pyramid import gaussian_blur

    n = gaussian_blur(rng.normal(0.0, 1.0, (h, w)), sigma)
    return n / (n.std() + 1e-9)


def analytic_motion_pair(
    h: int = 128,
    w: int = 160,
    dx: float = 8.0,
    dy: float = -5.0,
    zoom: float = 1.0,
    cx: float | None = None,
    cy: float | None = None,
    base: np.ndarray | None = None,
    seed: int = 7,
):
    """Generate A/B with known translation + zoom-about-center motion.

    Forward map (A center -> B plane)::

        W(x) = c + zoom * (x - c) + (dx, dy)
        F_A->B(x) = W(x) - x

    The B image is rendered with the inverse map
    ``x_A = c + (x_B - c - (dx,dy)) / zoom`` via bilinear sampling.

    Returns a dict with images, analytic fwd/bwd fields, validity masks and
    explicit NaN-masked fields.  Pixels without support are *not* given a
    zero GT motion: the ``*_nan`` arrays carry NaN there.
    """
    if base is None:
        base = smooth_analytic_texture(h, w, seed)
    a = np.asarray(base, dtype=np.float64)
    if a.shape != (h, w):
        raise ValueError(f"base shape {a.shape} != ({h},{w})")
    if cx is None:
        cx = (w - 1) / 2.0
    if cy is None:
        cy = (h - 1) / 2.0

    xs, ys = pixel_grid(h, w)

    # Forward field on the A grid.
    wx = cx + zoom * (xs - cx) + dx
    wy = cy + zoom * (ys - cy) + dy
    fu = wx - xs
    fv = wy - ys
    fwd_valid = (wx >= 0.0) & (wx <= w - 1.0) & (wy >= 0.0) & (wy <= h - 1.0)

    # Backward field on the B grid + actual rendered B.
    ax = cx + (xs - cx - dx) / zoom
    ay = cy + (ys - cy - dy) / zoom
    bu = ax - xs
    bv = ay - ys
    bwd_valid = (ax >= 0.0) & (ax <= w - 1.0) & (ay >= 0.0) & (ay <= h - 1.0)
    b, sample_valid = sample_bilinear(a, ax, ay)
    bwd_valid = bwd_valid & sample_valid

    def nan_mask(u, v, valid):
        un = np.where(valid, u, np.nan)
        vn = np.where(valid, v, np.nan)
        return np.stack([un, vn], axis=-1)

    return {
        "a": a,
        "b": b,
        "height": h,
        "width": w,
        "center": (float(cx), float(cy)),
        "dx": float(dx),
        "dy": float(dy),
        "zoom": float(zoom),
        "fwd_u": fu,
        "fwd_v": fv,
        "fwd_valid": fwd_valid,
        "bwd_u": bu,
        "bwd_v": bv,
        "bwd_valid": bwd_valid,
        "gt_fwd": np.stack([fu, fv], axis=-1),
        "gt_bwd": np.stack([bu, bv], axis=-1),
        "gt_fwd_nan": nan_mask(fu, fv, fwd_valid),
        "gt_bwd_nan": nan_mask(bu, bv, bwd_valid),
    }


def marker_image(h: int, w: int, px: float, py: float,
                 radius: float = 4.0, amp: float = 255.0) -> np.ndarray:
    """Dark image with one soft bright marker centered at subpixel (px, py)."""
    xs, ys = pixel_grid(h, w)
    img = np.zeros((h, w), np.float64)
    d2 = (xs - px) ** 2 + (ys - py) ** 2
    img += amp * np.exp(-d2 / (2.0 * radius**2))
    return img


def marker_centroid(img: np.ndarray, thresh_frac: float = 0.35):
    """Intensity-weighted centroid of the bright blob (subpixel index coords)."""
    m = img > thresh_frac * float(img.max())
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    wts = img[m]
    return float((xs * wts).sum() / wts.sum()), float((ys * wts).sum() / wts.sum())


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check(cid, title, fn):
    t0 = time.time()
    try:
        metrics, thresholds, evidence, detail = fn()
        ok = bool(metrics.pop("_ok", True))
        return {
            "id": cid, "title": title, "ok": ok,
            "metrics": _jsonable(metrics),
            "thresholds": _jsonable(thresholds),
            "evidence": evidence or [],
            "detail": detail or "",
            "elapsed_sec": round(time.time() - t0, 3),
        }
    except Exception as exc:  # a broken check must report failure, not fake pass
        return {
            "id": cid, "title": title, "ok": False,
            "metrics": {}, "thresholds": {}, "evidence": [],
            "detail": f"CHECK RAISED: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_sec": round(time.time() - t0, 3),
        }


def check_convention_constants():
    def body():
        assert CONVENTION["forward_flow"] == "x_B = x_A + F_A->B(x_A)"
        assert CONVENTION["backward_flow"] == "x_A = x_B + B_B->A(x_B)"
        # Worked source/target example: A center 10 moves +5 -> B center 15.
        x_a, f = 10.0, 5.0
        x_b = x_a + f
        b_at_b = -5.0
        x_a_back = x_b + b_at_b
        assert x_b == 15.0 and x_a_back == 10.0
        # The source/target-confused reading would sample at 15 + 5 = 20.
        confused_source = x_b + f
        assert confused_source == 20.0 and confused_source != x_a_back
        metrics = {"x_a": x_a, "F": f, "x_b": x_b,
                   "B_at_xb": b_at_b, "reproj_source": x_a_back,
                   "confused_wrong_source": confused_source}
        return metrics, {"roundtrip_exact": True}, [], (
            "前向 x_B=x_A+F；在 x_B=15 处重投影必须用后向 B=-5 采样 A@10，"
            "把前向 +5 当成 B 网格采样偏移会错误地取 A@20（源/目标混淆）。")
    return _check("convention_constants",
                  "前后向位移符号与源/目标坐标示例", body)


def check_octave_alignment():
    def body():
        h = w = 64
        px, py = 40.0, 24.0
        img = marker_image(h, w, px, py, radius=3.0) + 40.0
        from .pyramid import build_pyramid

        pyr = build_pyramid(img, 4)
        rows = []
        max_err = 0.0
        for k in range(4):
            fr = LevelFrame.octave(k, h, w)
            assert pyr[k].shape == (fr.native_h, fr.native_w)
            cen = marker_centroid(pyr[k] - 40.0)
            fx, fy = fr.to_full(*cen)
            err = math.hypot(fx - px, fy - py)
            max_err = max(max_err, err)
            rows.append({"level": k, "native": [fr.native_w, fr.native_h],
                         "centroid_level": [round(cen[0], 3), round(cen[1], 3)],
                         "mapped_full": [round(fx, 3), round(fy, 3)],
                         "err_full_px": round(err, 3)})
        return ({"max_marker_error_full_px": round(max_err, 3), "levels": rows,
                 "_ok": max_err < 1.5},
                {"max_marker_error_full_px_lt": 1.5}, [],
                "整数倍八层下粗格中心 j 对应全图 2^k j，实测标记质心经 "
                "LevelFrame 映射回全图误差 < 1.5 px，原点为 (0,0)。")
    return _check("octave_pyramid_alignment",
                  "整数八层金字塔格点原点实测对齐", body)


def check_scale_crop_roundtrip():
    """The measured origin decision + non-integer scale/crop point chase."""
    def body():
        h, w = 120, 160
        q = 0.6
        cx_s, cy_s = 9.0, 7.0          # crop origin in scaled pixels
        ow, oh = 64, 48
        fr = LevelFrame.scaled_crop(q, ow, oh, cx_s, cy_s)

        # 1) exact affine motion point chase, both directions.
        rng = np.random.default_rng(0)
        p = np.stack([rng.uniform(30, w - 30, 200),
                      rng.uniform(20, h - 20, 200)], axis=1)
        d = np.array([8.0, -5.0])
        pk_x, pk_y = fr.to_level(p[:, 0], p[:, 1])
        dk_x, dk_y = fr.displacement_to_level(d[0], d[1])
        qx, qy = pk_x + dk_x, pk_y + dk_y
        back_x, back_y = fr.to_full(qx, qy)
        fwd_err = np.max(np.hypot(back_x - (p[:, 0] + d[0]),
                                  back_y - (p[:, 1] + d[1])))
        # backward chase from Q = P + d with -d
        q0x, q0y = p[:, 0] + d[0], p[:, 1] + d[1]
        qk_x, qk_y = fr.to_level(q0x, q0y)
        bk_x, bk_y = fr.displacement_to_level(-d[0], -d[1])
        px_rt, py_rt = fr.to_full(qk_x + bk_x, qk_y + bk_y)
        bwd_err = np.max(np.hypot(px_rt - p[:, 0], py_rt - p[:, 1]))

        # 2) image-content marker registration: explicit origin vs naive.
        mx, my = 72.0, 54.0
        marker = marker_image(h, w, mx, my, radius=4.0)
        level_img, valid, fr2 = scale_and_crop(
            marker, q, ow, oh, cx_s, cy_s)
        cen = marker_centroid(level_img)
        assert cen is not None, "marker lost by scale+crop"
        ex_x, ex_y = fr2.to_full(*cen)
        naive_x, naive_y = fr2.naive_origin_free_to_full(*cen)
        explicit_marker_err = math.hypot(ex_x - mx, ex_y - my)
        naive_marker_err = math.hypot(naive_x - mx, naive_y - my)
        expected_origin_bias = math.hypot(fr2.ox, fr2.oy)

        ok = (fwd_err < 1e-9 and bwd_err < 1e-9
              and explicit_marker_err < 1.7
              and naive_marker_err > 0.8 * expected_origin_bias - 1.0)
        metrics = {
            "scale": q, "crop_origin_scaled": [cx_s, cy_s],
            "frame_origin_full": [round(fr.ox, 3), round(fr.oy, 3)],
            "affine_forward_chase_max_err": float(fwd_err),
            "affine_backward_chase_max_err": float(bwd_err),
            "marker_true_full": [mx, my],
            "marker_centroid_level": [round(cen[0], 3), round(cen[1], 3)],
            "explicit_origin_mapped": [round(ex_x, 2), round(ex_y, 2)],
            "explicit_origin_marker_err": round(explicit_marker_err, 3),
            "naive_origin_free_mapped": [round(naive_x, 2), round(naive_y, 2)],
            "naive_origin_free_marker_err": round(naive_marker_err, 3),
            "expected_origin_bias": round(expected_origin_bias, 3),
            "_ok": ok,
        }
        thresholds = {"affine_chase_err_lt": 1e-9,
                      "explicit_marker_err_lt_px": 1.7,
                      "naive_err_approximately_origin_bias": True}
        detail = (
            f"q={q} 非整比缩放后从缩放图 ({cx_s},{cy_s}) 裁切：显式原点 "
            f"o=({fr.ox:.1f},{fr.oy:.1f}) 使运动点正/反向往返误差为 0，"
            f"标记实测落回误差 {explicit_marker_err:.2f} px；丢掉裁切原点的"
            f"“连续平面”读法偏差 {naive_marker_err:.2f} px"
            f"（≈原点偏置 {expected_origin_bias:.1f} px）。"
            "故约定：每层显式保留仿射原点；整数八层时它为 0，退化为 "
            "x_k=s_k x_0。")
        return metrics, thresholds, [], detail
    return _check("noninteger_scale_crop_roundtrip",
                  "非整比缩放+裁切的坐标往返与原点选择实测", body)


def check_analytic_direction_and_support():
    def body():
        h, w = 128, 160
        pair = analytic_motion_pair(h, w, dx=10.0, dy=-6.0, zoom=1.05)
        fu, fv = pair["fwd_u"], pair["fwd_v"]
        # Right-of-center points move right for dx>0; above-center points
        # move up for dy<0; zoom>1 pushes right-half points further right.
        right = (slice(None), slice(int(w * 0.62), int(w * 0.9)))
        top = (slice(int(h * 0.1), int(h * 0.4)), slice(None))
        med_u_right = float(np.median(fu[right]))
        med_v_top = float(np.median(fv[top]))
        # Out-of-frame support must exist for a 10 px move and be explicit.
        n_fwd_invalid = int((~pair["fwd_valid"]).sum())
        n_bwd_invalid = int((~pair["bwd_valid"]).sum())
        nan_fwd = int(np.isnan(pair["gt_fwd_nan"][..., 0]).sum())
        nan_bwd = int(np.isnan(pair["gt_bwd_nan"][..., 0]).sum())
        assert n_fwd_invalid > 200 and n_bwd_invalid > 200
        assert nan_fwd == n_fwd_invalid and nan_bwd == n_bwd_invalid
        # The raw analytic field is finite everywhere (formula), so the
        # validity mask / NaN field is the only honest support channel.
        assert np.isfinite(pair["gt_fwd"]).all()
        ok = med_u_right > 8.0 and med_v_top < -4.0
        metrics = {"median_u_right_strip": round(med_u_right, 3),
                   "median_v_top_strip": round(med_v_top, 3),
                   "fwd_invalid_px": n_fwd_invalid,
                   "bwd_invalid_px": n_bwd_invalid,
                   "nan_fwd_px": nan_fwd, "nan_bwd_px": nan_bwd, "_ok": ok}
        return metrics, {"median_u_right_gt": 8.0, "median_v_top_lt": -4.0,
                         "invalid_px_gt": 200,
                         "nan_count_equals_invalid_count": True}, [], (
            "解析 GT：dx=+10 时右侧条带 u>0（流向向右），dy=-6 时上方条带 "
            "v<0（流向向上）；平移+缩放产生的出画像素在 gt_*_nan 中为 NaN，"
            "并由 *_valid 显式表达，绝不把缺失支持写成 0 运动。")
    return _check("analytic_direction_support",
                  "解析平移+缩放的流向符号与缺失支持显式化", body)


def check_analytic_reprojection():
    def body():
        h, w = 128, 160
        pair = analytic_motion_pair(h, w, dx=9.0, dy=5.0, zoom=1.04)
        a, b = pair["a"], pair["b"]
        # Reproject A onto B using the *analytic* backward field.
        reprojected, valid = _reproject_analytic(a, pair)
        diff = np.abs(reprojected - b)
        inside = pair["bwd_valid"] & valid
        med = float(np.median(diff[inside]))
        p95 = float(np.percentile(diff[inside], 95))
        # Outside support the rendered B is fill and must stay excluded.
        outside = ~pair["bwd_valid"]
        n_out = int(outside.sum())
        ok = med < 2.0 and p95 < 12.0 and n_out > 200
        metrics = {"reproj_median_gray": round(med, 4),
                   "reproj_p95_gray": round(p95, 3),
                   "supported_px": int(inside.sum()),
                   "excluded_unsupported_px": n_out, "_ok": ok}
        return metrics, {"median_lt_gray": 2.0, "p95_lt_gray": 12.0}, [], (
            "用解析后向流把 A 重投影到 B：支持区内残差中位≈0，出画/新露出 "
            "区被 bwd_valid 排除而不是按零流参与统计。")
    return _check("analytic_reprojection",
                  "解析后向流重投影残差（源/目标坐标核对）", body)


def _reproject_analytic(a, pair):
    xs, ys = pixel_grid(pair["height"], pair["width"])
    return sample_bilinear(a, xs + pair["bwd_u"], ys + pair["bwd_v"])


def check_kernel_flow(quick: bool = True):
    """Real DenseFlow kernel: direction signs + measured reprojection."""
    def body():
        from .metrics import reprojection_residual
        from .pipeline import to_gray
        from .solver import SolveConfig, estimate_flow
        from .synth import rich_texture

        if quick:
            h, w = 128, 160
            dx, dy = 5.0, -3.0
            cfg = SolveConfig(num_levels=4, outer_iters=6, inner_iters=60,
                              alpha=120.0)
            epe_limit = 0.5
        else:
            h, w = 160, 200
            dx, dy = 5.0, -3.0
            cfg = SolveConfig(num_levels=4, outer_iters=5, inner_iters=60,
                              alpha=120.0)
            epe_limit = 0.25
        base = to_gray(rich_texture(h, w, seed=2))
        pair = analytic_motion_pair(h, w, dx=dx, dy=dy, zoom=1.0, base=base)
        a, b = pair["a"], pair["b"]

        fwd = estimate_flow(a, b, cfg)
        bwd = estimate_flow(b, a, cfg)
        u, v = fwd["u"], fwd["v"]
        m = np.zeros((h, w), bool)
        m[int(h * 0.18):int(h * 0.82), int(w * 0.18):int(w * 0.82)] = True
        med_u = float(np.median(u[m]))
        med_v = float(np.median(v[m]))
        epe_med = float(np.median(np.hypot(u[m] - dx, v[m] - dy)))

        rep = reprojection_residual(a, b, bwd["u"], bwd["v"])
        rep_med = rep["median"]
        rep_p95 = rep["p95"]

        # Boundary out-of-frame band: backward support validity must be False
        # exactly where x_B + B(B) leaves A (coordinate-level assertion),
        # and the support score must collapse there rather than claim 0 flow.
        bu, bv = bwd["u"], bwd["v"]
        xs, ys = pixel_grid(h, w)
        src_x, src_y = xs + bu, ys + bv
        coord_in = ((src_x >= 0) & (src_x <= w - 1)
                    & (src_y >= 0) & (src_y <= h - 1))
        support_valid = bwd["support"]["valid"]
        validity_agreement = float((coord_in == support_valid).mean())
        # GT newly-exposed left strip for dx>0 (B's left edge has no A source).
        exposed = ~pair["bwd_valid"]
        exposed_frac_unsupported = float((~support_valid[exposed]).mean()) \
            if exposed.any() else None

        dir_ok = med_u > 3.5 and med_v < -1.8
        epe_ok = epe_med < epe_limit
        rep_ok = rep_med is not None and rep_med < 3.0
        valid_ok = validity_agreement > 0.999
        exposed_ok = (exposed_frac_unsupported is not None
                      and exposed_frac_unsupported > 0.9)
        ok = dir_ok and epe_ok and rep_ok and valid_ok and exposed_ok
        metrics = {
            "size": [w, h], "quick": quick,
            "gt_dx_dy": [dx, dy],
            "median_u_interior": round(med_u, 3),
            "median_v_interior": round(med_v, 3),
            "interior_epe_median": round(epe_med, 3),
            "reproj_median_gray": None if rep_med is None else round(rep_med, 3),
            "reproj_p95_gray": None if rep_p95 is None else round(rep_p95, 3),
            "reproj_valid_fraction": round(rep["valid_fraction"], 4),
            "support_valid_vs_coordinate_agreement": round(validity_agreement, 5),
            "exposed_strip_frac_marked_unsupported":
                None if exposed_frac_unsupported is None
                else round(exposed_frac_unsupported, 4),
            "_ok": ok,
        }
        thresholds = {"median_u_gt": 3.5, "median_v_lt": -1.8,
                      "epe_median_lt": epe_limit,
                      "reproj_median_lt_gray": 3.0,
                      "validity_agreement_gt": 0.999,
                      "exposed_strip_unsupported_frac_gt": 0.9}
        detail = (
            f"真实内核在解析平移 ({dx},{dy}) 对上：内部中位流 "
            f"({med_u:.2f},{med_v:.2f})，方向与 GT 一致；EPE 中位 "
            f"{epe_med:.3f} px；后向重投影残差中位 {rep_med:.2f} 灰度级；"
            f"出画新露出带 {exposed_frac_unsupported * 100:.1f}% 被标记为无效支持"
            "（不是零运动）。")
        ev = _write_kernel_evidence_images(a, b, u, v, rep)
        return metrics, thresholds, ev, detail
    return _check("kernel_flow_direction_reprojection",
                  "稠密光流内核的流向与重投影残差执行证据", body)


def _write_kernel_evidence_images(a, b, u, v, rep):
    ev = []
    out_dir = _EVIDENCE_DIR[0]
    if out_dir is None:
        return ev
    try:
        from . import visualize as vz

        base = np.stack([a] * 3, -1).astype(np.uint8)
        p_vec = out_dir / "coord_check_vectors.png"
        vz.draw_vectors(base, u, v, stride=14).save(p_vec)
        p_rep = out_dir / "coord_check_reprojected.png"
        vz.Image.fromarray(
            np.clip(rep["reprojected"], 0, 255).astype(np.uint8)
        ).convert("RGB").save(p_rep)
        p_res = out_dir / "coord_check_residual.png"
        vz.scalar_heatmap(rep["map"], vmax=40).save(p_res)
        ev = [str(p_vec), str(p_rep), str(p_res)]
    except Exception:
        pass
    return ev


_EVIDENCE_DIR: list[Path | None] = [None]


def check_documentation():
    def body():
        readme = REPO_ROOT / "README.md"
        text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
        required = [
            "x_B = x_A + F_A→B(x_A)",
            "x_A = x_B + B_B→A(x_B)",
            "sample(A, x_B + B_B→A(x_B))",
            "x_k = s_k x_0",
            "LevelFrame",
            "非整比",
            "往返",
            "缺失支持",
            "check_flow_coordinates.py",
            "导入相邻帧",
            "坐标自检",
        ]
        missing = [s for s in required if s not in text]
        # The UI modal must carry the same source/target formulas.
        appjs = REPO_ROOT / "web" / "app.js"
        js = appjs.read_text(encoding="utf-8") if appjs.is_file() else ""
        js_missing = [s for s in [
            "x_B = x_A + F_A->B(x_A)",
            "x_A = x_B + B_B->A(x_B)",
            "/api/coord-check",
        ] if s not in js]
        ok = not missing and not js_missing
        return ({"readme_missing_anchors": missing, "appjs_missing_anchors": js_missing,
                 "_ok": ok},
                {"required_readme_anchors": required}, [],
                "README 与前端约定弹窗/自检入口包含同一套前后向、金字塔和"
                "缺失支持口径。")
    return _check("documentation_consistency",
                  "文档/界面与坐标实现的口径一致性", body)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def run_audit(out_dir: str | Path | None = None,
              quick: bool = True,
              run_kernel: bool = True) -> dict:
    """Run every coordinate check; write JSON + image evidence if possible."""
    if out_dir is None:
        out_dir = REPO_ROOT / "experiments" / "output" / "coord_check"
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        _EVIDENCE_DIR[0] = out_dir
        evidence_dir = str(out_dir)
        writable = True
    except OSError:
        import tempfile

        out_dir = Path(tempfile.mkdtemp(prefix="coord_check_"))
        _EVIDENCE_DIR[0] = out_dir
        evidence_dir = str(out_dir)
        writable = False

    checks = [
        check_convention_constants(),
        check_octave_alignment(),
        check_scale_crop_roundtrip(),
        check_analytic_direction_and_support(),
        check_analytic_reprojection(),
    ]
    if run_kernel:
        checks.append(check_kernel_flow(quick=quick))
    checks.append(check_documentation())

    n_fail = sum(1 for c in checks if not c["ok"])
    report = {
        "ok": n_fail == 0,
        "passed": len(checks) - n_fail,
        "failed": n_fail,
        "n_checks": len(checks),
        "quick": quick,
        "evidence_dir": evidence_dir,
        "evidence_dir_in_repo": writable,
        "origin_decision": (
            "explicit per-level affine LevelFrame(ax,ay,ox,oy); octave "
            "pyramid has ox=oy=0 and equals continuous x_k=s_k*x_0"),
        "checks": checks,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        (out_dir / "coord_check_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    return report


def print_report(report: dict) -> None:
    print("=" * 72)
    print("DenseFlow 光流坐标检查 / flow-coordinate audit")
    print("=" * 72)
    for c in report["checks"]:
        mark = "PASS" if c["ok"] else "FAIL"
        print(f"[{mark}] {c['id']} — {c['title']} ({c['elapsed_sec']}s)")
        if c.get("detail"):
            print(f"       {c['detail']}")
        if not c["ok"] and c.get("traceback"):
            for line in c["traceback"].splitlines()[-4:]:
                print(f"       ! {line}")
    print("-" * 72)
    print(f"结果: {report['passed']}/{report['n_checks']} 通过, "
          f"{report['failed']} 失败")
    print(f"证据目录: {report['evidence_dir']}")
    print(f"JSON: {Path(report['evidence_dir']) / 'coord_check_report.json'}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DenseFlow coordinate audit")
    ap.add_argument("--out", default=None, help="evidence output directory")
    ap.add_argument("--quick", action="store_true",
                    help="smaller/faster kernel check (this is the default)")
    ap.add_argument("--full", action="store_true",
                    help="full-size kernel accuracy check")
    ap.add_argument("--no-kernel", action="store_true",
                    help="skip the real solver check")
    args = ap.parse_args(argv)
    report = run_audit(out_dir=args.out, quick=not args.full,
                       run_kernel=not args.no_kernel)
    print_report(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
