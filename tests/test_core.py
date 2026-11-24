"""Coordinate/solver sanity checks. Run: python3 tests/test_core.py"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from denseflow.coords import pixel_grid, sample_bilinear  # noqa: E402
from denseflow.pyramid import build_pyramid, prolong_flow  # noqa: E402
from denseflow.solver import SolveConfig, estimate_flow  # noqa: E402


def make_textured(h=160, w=200, seed=0):
    rng = np.random.default_rng(seed)
    # Smooth multiscale texture: filtered noise + shapes.
    base = rng.normal(128, 40, size=(h, w))
    from denseflow.pyramid import gaussian_blur

    base = gaussian_blur(base, 2.2)
    base = (base - base.mean()) / (base.std() + 1e-9) * 45 + 128
    xx, yy = pixel_grid(h, w)
    base += 35 * np.sin(xx / 11.0) * np.cos(yy / 13.0)
    for cx, cy, rad, amp in [(50, 50, 22, 70), (150, 110, 30, -60), (110, 40, 14, 80)]:
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad**2
        base[m] += amp
    return np.clip(base, 0, 255)


def shift_image(img, dx, dy):
    """Render B(x) = A(x - d) for a scene moving by d (forward flow d)."""
    h, w = img.shape
    xs, ys = pixel_grid(h, w)
    out, valid = sample_bilinear(img, xs - dx, ys - dy)
    return out, valid


def test_sampler_identity():
    img = np.arange(12, dtype=float).reshape(3, 4)
    xs, ys = pixel_grid(3, 4)
    out, valid = sample_bilinear(img, xs, ys)
    assert np.allclose(out, img)
    assert valid.all()
    v, ok = sample_bilinear(img, np.array([-0.5, 0.0, 3.0]), np.array([0.0, 0.0, 0.0]))
    assert ok.tolist() == [False, True, True]
    print("ok sampler identity / bounds")


def test_pyramid_flow_scaling():
    img = make_textured(64, 64)
    pyr = build_pyramid(img, 3)
    assert pyr[2].shape == (16, 16), pyr[2].shape
    u = np.full((16, 16), 1.5)  # level-2 px -> 6 full px
    uu, vv = prolong_flow(u, u * 0, 32, 32, 0.25, 0.5)
    assert uu.shape == (32, 32)
    assert np.allclose(uu, 3.0), uu[0, 0]
    uu0, _ = prolong_flow(u, u * 0, 64, 64, 0.25, 1.0)
    assert np.allclose(uu0, 6.0)
    print("ok pyramid sizes and displacement value transform")


def test_integer_translation():
    a = make_textured()
    dx, dy = 5.0, -3.0
    b, _ = shift_image(a, dx, dy)
    cfg = SolveConfig(num_levels=4, outer_iters=5, inner_iters=60, alpha=120.0)
    res = estimate_flow(a, b, cfg)
    u, v = res["u"], res["v"]
    err = np.hypot(u - dx, v - dy)
    interior = np.zeros_like(err, bool)
    interior[12:-12, 12:-12] = True
    med = np.median(err[interior])
    mean = err[interior].mean()
    print(f"translation ({dx},{dy}): median EPE={med:.3f} mean={mean:.3f} "
          f"converged={res['converged']}")
    assert med < 0.25, med
    assert mean < 0.6, mean


def test_subpixel_translation():
    a = make_textured(160, 200, seed=3)
    dx, dy = 2.25, 1.5
    b, _ = shift_image(a, dx, dy)
    cfg = SolveConfig(num_levels=4, outer_iters=6, inner_iters=60, alpha=120.0)
    res = estimate_flow(a, b, cfg)
    err = np.hypot(res["u"] - dx, res["v"] - dy)
    m = np.zeros_like(err, bool)
    m[14:-14, 14:-14] = True
    med = np.median(err[m])
    print(f"subpixel ({dx},{dy}): median EPE={med:.3f}")
    assert med < 0.35, med


def test_zero_flow_not_proven():
    a = make_textured(96, 96, seed=4)
    cfg = SolveConfig(num_levels=3)
    res = estimate_flow(a, a.copy(), cfg)
    # On a textureless image support must be low even though flow is zero.
    flat = np.full((96, 96), 128.0)
    res2 = estimate_flow(flat, flat.copy(), SolveConfig(num_levels=3))
    rel = res2["support"]["reliable_photo"]
    assert rel.mean() < 0.05, rel.mean()
    assert np.max(np.abs(res2["u"])) < 1e-6
    print(f"ok textureless zero field unsupported (reliable frac={rel.mean():.3f})")


if __name__ == "__main__":
    test_sampler_identity()
    test_pyramid_flow_scaling()
    test_integer_translation()
    test_subpixel_translation()
    test_zero_flow_not_proven()
    print("ALL CORE TESTS PASSED")
