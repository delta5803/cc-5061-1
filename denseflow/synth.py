"""Synthetic sequences with exact ground-truth flow / labels / masks.

All geometry uses the project coordinate convention: pixel centers are
integer ``(x=j, y=i)``.  Motion is rendered through
:func:`denseflow.coords.sample_bilinear` with the inverse map, and the GT
forward flow is taken from the same transform, so EPE numbers are not
guessed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from .coords import pixel_grid, sample_bilinear
from .pyramid import gaussian_blur


# ---------------------------------------------------------------------------
# texture / shape helpers
# ---------------------------------------------------------------------------

def rich_texture(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Multi-octave smooth noise: solvable, natural-looking texture rather
    # than Nyquist noise that decorrelates under sub-pixel resampling.
    n1 = gaussian_blur(rng.normal(0, 1, (h, w)), 5.0)
    n1 /= n1.std() + 1e-9
    n2 = gaussian_blur(rng.normal(0, 1, (h, w)), 2.0)
    n2 /= n2.std() + 1e-9
    img = 132 + 34 * n1 + 12 * n2
    xs, ys = pixel_grid(h, w)
    img += 18 * np.sin(xs / 13.0 + 1.3) * np.cos(ys / 15.0)
    img += 10 * np.sin((xs + ys) / 23.0)
    img = np.clip(img, 0, 255)
    rgb = np.repeat(img[..., None], 3, axis=-1)
    # A few painted shapes give strong edges/corners.
    pil = Image.fromarray(rgb.astype(np.uint8), "RGB")
    d = ImageDraw.Draw(pil)
    rng2 = np.random.default_rng(seed + 100)
    for _ in range(14):
        x0 = int(rng2.integers(0, w - 40))
        y0 = int(rng2.integers(0, h - 40))
        s = int(rng2.integers(18, 46))
        col = tuple(int(c) for c in rng2.integers(30, 230, 3))
        kind = rng2.integers(0, 3)
        if kind == 0:
            d.ellipse([x0, y0, x0 + s, y0 + s], fill=col)
        elif kind == 1:
            d.rectangle([x0, y0, x0 + s, y0 + int(s * 0.7)], fill=col)
        else:
            d.polygon([(x0, y0 + s), (x0 + s // 2, y0), (x0 + s, y0 + s)], fill=col)
    return np.array(pil).astype(np.float64)


def stripe_background(h: int, w: int, period: int = 40) -> np.ndarray:
    xs, ys = pixel_grid(h, w)
    phase = np.mod(xs, period)
    base = np.where(phase < period * 0.42, 214, 70).astype(np.float64)
    # Slight vertical shading + thin horizontal reference lines (these do
    # not break horizontal periodic-shift ambiguity).
    base += 10 * np.sin(ys / 26.0)
    rgb = np.repeat(base[..., None], 3, axis=-1)
    pil = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(pil)
    for y in range(20, h, 54):
        d.line([(0, y), (w, y)], fill=(30, 30, 30), width=2)
    return np.array(pil).astype(np.float64)


def make_object_patch(size: int = 92, kind: str = "cross") -> tuple[np.ndarray, np.ndarray]:
    """Return RGBA-ish RGB patch + soft binary mask for the occluder."""
    patch = np.zeros((size, size, 3), np.uint8)
    mask_img = Image.new("L", (size, size), 0)
    pd = ImageDraw.Draw(mask_img)
    if kind == "rounded":
        pd.rounded_rectangle([4, 4, size - 5, size - 5], radius=18, fill=255)
    else:
        pd.ellipse([4, 4, size - 5, size - 5], fill=255)
    mask = np.array(mask_img) > 0

    pil = Image.fromarray(patch, "RGB")
    d = ImageDraw.Draw(pil)
    d.rounded_rectangle([4, 4, size - 5, size - 5], radius=18,
                        fill=(15, 110, 120), outline=(230, 240, 240), width=3)
    d.line([(size * 0.3, size * 0.25), (size * 0.7, size * 0.75)],
           fill=(255, 220, 90), width=8)
    d.line([(size * 0.7, size * 0.25), (size * 0.3, size * 0.75)],
           fill=(255, 220, 90), width=8)
    arr = np.array(pil).astype(np.float64)
    return arr, mask


def textured_disk(radius: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    s = radius * 2 + 4
    ys, xs = np.mgrid[0:s, 0:s] - s / 2
    m = xs**2 + ys**2 <= radius**2
    rgb = np.zeros((s, s, 3))
    red = np.array([214.0, 69.0, 65.0])
    darkred = np.array([140.0, 26.0, 30.0])
    if mode == "stripes":
        t = (np.sin(xs / 3.0) > 0)[..., None]
        rgb = np.repeat(np.where(t, 232.0, 150.0), 3, axis=2)
        rgb = 0.55 * rgb + 0.45 * red
    elif mode == "dots":
        rgb[:] = red
        gx = np.mod(xs + 100, 11) - 5.5
        gy = np.mod(ys + 100, 11) - 5.5
        dots = (gx**2 + gy**2) < 8
        rgb[dots] = darkred
    else:
        rgb[:] = red
    return np.clip(rgb, 0, 255), m


def paste(canvas: np.ndarray, patch: np.ndarray, mask: np.ndarray, cx: float, cy: float):
    """Paste patch centered at (cx, cy); returns updated canvas + full mask."""
    h, w = canvas.shape[:2]
    s = patch.shape[0]
    x0 = int(round(cx - s / 2))
    y0 = int(round(cy - s / 2))
    full_mask = np.zeros((h, w), bool)
    py0 = max(0, y0)
    px0 = max(0, x0)
    py1 = min(h, y0 + s)
    px1 = min(w, x0 + s)
    if py1 <= py0 or px1 <= px0:
        return canvas, full_mask
    qy0 = py0 - y0
    qx0 = px0 - x0
    sub_m = mask[qy0:qy0 + (py1 - py0), qx0:qx0 + (px1 - px0)]
    region = canvas[py0:py1, px0:px1]
    pm = sub_m[..., None]
    region[:] = np.where(pm, patch[qy0:qy0 + (py1 - py0), qx0:qx0 + (px1 - px0)], region)
    full_mask[py0:py1, px0:px1] = sub_m
    return canvas, full_mask


# ---------------------------------------------------------------------------
# sequence container
# ---------------------------------------------------------------------------

@dataclass
class Sequence:
    name: str
    frames: list[np.ndarray] = field(default_factory=list)          # uint8 RGB
    flows: list[np.ndarray] = field(default_factory=list)           # GT HxWx2 per pair
    valid: list[np.ndarray] = field(default_factory=list)
    labels: list[np.ndarray] = field(default_factory=list)
    decoy_mask: np.ndarray | None = None
    events: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def pair_gray(self, i: int):
        a = self.frames[i].astype(np.float64).mean(axis=2)
        b = self.frames[i + 1].astype(np.float64).mean(axis=2)
        return a, b


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------

def gen_translation(h: int = 240, w: int = 320, n: int = 5) -> Sequence:
    margin = 70
    canvas = rich_texture(h + 2 * margin, w + 2 * margin, seed=11)
    dx, dy = 5.0, -3.0
    seq = Sequence("translation", meta={"gt_flow": [dx, dy], "kind": "global translation"})
    origins = []
    for t in range(n):
        ox = margin - dx * t
        oy = margin - dy * t
        origins.append((ox, oy))
        frame = canvas[int(round(oy)):int(round(oy)) + h,
                       int(round(ox)):int(round(ox)) + w]
        seq.frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    for t in range(n - 1):
        # origin decreases by (dx,dy) -> content shifts by +(dx,dy)
        f = np.zeros((h, w, 2))
        f[..., 0] = dx
        f[..., 1] = dy
        seq.flows.append(f)
        seq.valid.append(np.ones((h, w), bool))
        seq.labels.append(np.zeros((h, w), np.uint8))
    seq.labels.append(np.zeros((h, w), np.uint8))
    return seq


def gen_rotation(h: int = 240, w: int = 320, n: int = 5) -> Sequence:
    margin = 80
    base = rich_texture(h + 2 * margin, w + 2 * margin, seed=23)
    ch, cw = (h - 1) / 2.0, (w - 1) / 2.0
    angle_step = math.radians(2.2)
    xs, ys = pixel_grid(h, w)
    X = xs - cw
    Y = ys - ch
    seq = Sequence("rotation", meta={"angle_step_deg": 2.2, "center": [cw, ch]})

    for t in range(n):
        theta = -angle_step * t  # render inverse rotation
        cos, sin = math.cos(theta), math.sin(theta)
        sx = cw + cos * X - sin * Y + margin
        sy = ch + sin * X + cos * Y + margin
        out, valid = sample_bilinear(base, sx, sy)
        seq.frames.append(np.clip(out, 0, 255).astype(np.uint8))

    for t in range(n - 1):
        phi = angle_step
        cos, sin = math.cos(phi), math.sin(phi)
        u = (cos - 1) * X - sin * Y
        v = sin * X + (cos - 1) * Y
        seq.flows.append(np.stack([u, v], axis=-1))
        # Corners may sample outside the margined base for cumulative angle.
        theta_t = -angle_step * t
        ct, st = math.cos(theta_t), math.sin(theta_t)
        bx = cw + ct * X + st * Y + margin
        by = ch - st * X + ct * Y + margin
        ok = (bx >= 0) & (bx < base.shape[1] - 1) & (by >= 0) & (by < base.shape[0] - 1)
        seq.valid.append(ok)
        seq.labels.append(np.zeros((h, w), np.uint8))
    seq.labels.append(np.zeros((h, w), np.uint8))
    return seq


def _shift_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = mask.shape
    xs, ys = pixel_grid(h, w)
    out, _ = sample_bilinear(mask.astype(np.float64), xs - dx, ys - dy)
    return out > 0.5


def gen_local_motion(h: int = 240, w: int = 320) -> Sequence:
    bg = rich_texture(h, w, seed=41)
    patch, pmask = make_object_patch(96)
    cy = 128.0
    xs_pos = [70.0, 84.0, 98.0, 112.0, 126.0]
    dx = xs_pos[1] - xs_pos[0]
    seq = Sequence("local_motion", meta={"object_flow": [dx, 0.0], "bg_flow": [0, 0]})
    obj_masks = []
    for cx in xs_pos:
        frame = bg.copy()
        frame, m = paste(frame, patch, pmask, cx, cy)
        seq.frames.append(np.clip(frame, 0, 255).astype(np.uint8))
        obj_masks.append(m)
        seq.labels.append(m.astype(np.uint8))

    for t in range(len(xs_pos) - 1):
        f = np.zeros((h, w, 2))
        m0 = obj_masks[t]
        f[m0, 0] = dx
        seq.flows.append(f)
        # A-grid correspondence validity:
        shifted_obj = _shift_mask(m0, dx, 0.0)
        m1 = obj_masks[t + 1]
        valid = np.ones((h, w), bool)
        # background pixels that land inside object t+1 are newly occluded
        occ = (~m0) & shifted_obj & m1
        valid &= ~occ
        # object pixels must land on object t+1 inside frame
        bad_obj = m0 & ~(shifted_obj & m1)
        valid &= ~bad_obj
        seq.valid.append(valid)
    return seq


def gen_large_displacement(h: int = 240, w: int = 320, dx: float = 52.0) -> Sequence:
    a = rich_texture(h, w, seed=7)
    xs, ys = pixel_grid(h, w)
    b, valid = sample_bilinear(a, xs - dx, ys)
    # black strip where content left the frame
    seq = Sequence(
        "large_displacement",
        frames=[np.clip(a, 0, 255).astype(np.uint8), np.clip(b, 0, 255).astype(np.uint8)],
        meta={"gt_flow": [dx, 0.0]},
    )
    f = np.zeros((h, w, 2))
    f[..., 0] = dx
    seq.flows.append(f)
    gt_valid = valid & (xs + dx <= w - 1)
    seq.valid.append(gt_valid)
    seq.labels = [np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)]
    return seq


def gen_repeat_occlude(h: int = 240, w: int = 320, shift: float = 12.0) -> Sequence:
    bg = stripe_background(h, w, period=40)
    patch, pmask = make_object_patch(92, kind="rounded")
    positions = [112.0, 112.0 + shift]
    cy = 120.0
    seq = Sequence(
        "repeat_occlude",
        meta={
            "background_flow": [0, 0],
            "foreground_flow": [shift, 0],
            "period": 40,
            "wrong_candidates": [[40, 0], [-40, 0]],
        },
    )
    masks = []
    frames_rgb = []
    for cx in positions:
        fr = bg.copy()
        fr, m = paste(fr, patch, pmask, cx, cy)
        frames_rgb.append(np.clip(fr, 0, 255).astype(np.uint8))
        masks.append(m)
    seq.frames = frames_rgb
    f = np.zeros((h, w, 2))
    f[masks[0], 0] = shift
    seq.flows.append(f)
    # Valid A correspondences: object overlap + bg not newly occluded.
    valid = np.ones((h, w), bool)
    shifted = _shift_mask(masks[0], shift, 0)
    valid &= ~((~masks[0]) & shifted & masks[1])
    valid &= ~(masks[0] & ~(shifted & masks[1]))
    seq.valid.append(valid)
    seq.labels = [masks[0].astype(np.uint8), masks[1].astype(np.uint8)]
    seq.events = [{
        "type": "repeat_plus_occlusion",
        "detail": "periodic stripes admit +/-40px branches; label boundary selects 0",
    }]
    return seq


def gen_texture_reveal(h: int = 240, w: int = 320) -> Sequence:
    rng = np.random.default_rng(99)
    bg = rich_texture(h, w, seed=55) * 0.55 + 110
    bg = np.clip(bg, 0, 255)
    pil = Image.fromarray(bg.astype(np.uint8), "RGB")
    d = ImageDraw.Draw(pil)
    # Same-color decoy patch: shares the object's red fill, never moves.
    decoy = Image.new("L", (w, h), 0)
    dd = ImageDraw.Draw(decoy)
    dd.ellipse([250, 26, 306, 82], fill=255)
    d.ellipse([250, 26, 306, 82], fill=(214, 69, 65), outline=(120, 30, 30), width=3)
    bg = np.array(pil).astype(np.float64)
    decoy_mask = np.array(decoy) > 0
    radius = 22
    vx = 12.0
    n = 16
    cy = 176.0
    # Bar always composited on top: gives correct partial occlusion.
    bar = (140, 140, 224, 214)
    bar_mask = np.zeros((h, w), bool)
    bar_mask[bar[1]:bar[3], bar[0]:bar[2]] = True
    seq = Sequence(
        "texture_reveal",
        decoy_mask=decoy_mask,
        meta={"velocity": [vx, 0.0], "hold_frames": 4,
              "occluded_frames": [10, 11, 12], "reappear_frame": 14,
              "note": "texture changes while behind bar; label held then reclaimed"},
    )
    prev_mask = None
    for t in range(n):
        cx = 48.0 + vx * t
        mode = "stripes" if t <= 11 else "dots"
        patch, omask = textured_disk(radius, mode)
        frame = bg.copy()
        frame, full = paste(frame, patch, omask, cx, cy)
        # Bar is in front: repaint it and subtract it from the label mask.
        p2 = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))
        dd2 = ImageDraw.Draw(p2)
        dd2.rounded_rectangle(bar, radius=10, fill=(45, 52, 64),
                              outline=(20, 24, 30), width=3)
        frame = np.array(p2).astype(np.float64)
        full = full & ~bar_mask
        seq.frames.append(np.clip(frame, 0, 255).astype(np.uint8))
        seq.labels.append(full.astype(np.uint8))
        if t > 0:
            f = np.zeros((h, w, 2))
            if prev_mask is not None:
                f[prev_mask, 0] = vx
            seq.flows.append(f)
            seq.valid.append(np.ones((h, w), bool))
        prev_mask = full
    seq.events = [
        {"frame": 11, "type": "texture_change_while_occluded"},
        {"frame": 11, "type": "label_held"},
        {"frame": 14, "type": "reappear_claim_constrains_branch"},
    ]
    return seq


REGISTRY = {
    "translation": gen_translation,
    "rotation": gen_rotation,
    "local_motion": gen_local_motion,
    "large_displacement": gen_large_displacement,
    "repeat_occlude": gen_repeat_occlude,
    "texture_reveal": gen_texture_reveal,
}


def generate_all() -> dict[str, Sequence]:
    return {name: fn() for name, fn in REGISTRY.items()}
