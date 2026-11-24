"""Foreground-label propagation driven by trusted flow.

Rules implemented here:

* A label moves to the current frame by nearest-neighbour backward warp
  through the backward flow, but ONLY where the flow is trusted (bounds,
  photometric support and forward/backward consistency).
* In multi-solution / occluded regions the propagator does NOT invent
  assignments.  A label whose trusted coverage collapses is put in the
  ``held`` state: its mask is extrapolated by its last trusted velocity for
  a limited number of frames and exposed as a *claim* region.
* When the object is seen again, trusted assignments inside the claim
  region reclaim the label.  The claim mask is also fed back to the
  cross-scale branch selector, so the label constrains which flow branch
  is accepted.
* Colour alone never assigns foreground.  :func:`same_color_pixels` exists
  to audit whether a same-colour background decoy nevertheless ends up
  labelled (it should not).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .coords import pixel_grid, sample_nearest


def trusted_mask(support: dict | None,
                 fb_error: np.ndarray | None,
                 fb_tol: float = 1.5) -> np.ndarray:
    if support is None:
        h, w = fb_error.shape if fb_error is not None else (1, 1)
        trusted = np.ones((h, w), bool)
    else:
        valid = support["valid"]
        # Path 1: strong 2-D texture evidence.
        core = support["score"] > 0.35
        # Path 2: tight photometric fit on oriented edge/stripe structure.
        # This rescues the aperture case (1-D repeated texture) where
        # lambda_min is small but the solved correspondence fits exactly.
        lam_max = support.get("texture_lambda_max")
        if lam_max is not None:
            edge_fit = (support["photo"] > 0.80) & (lam_max > 3.0)
            core = core | edge_fit
        trusted = valid & core
    if fb_error is not None:
        trusted &= np.isfinite(fb_error) & (fb_error < fb_tol)
    return trusted


def propagate_once(
    prev_labels: np.ndarray,
    bu: np.ndarray,
    bv: np.ndarray,
    trusted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp prev labels into the current grid through backward flow.

    Returns ``(new_labels, assigned)``; untrusted / out-of-bounds pixels
    stay 0 even if the nearest source carried a label.
    """
    h, w = bu.shape
    xs, ys = pixel_grid(h, w)
    warped, valid = sample_nearest(prev_labels, xs + bu, ys + bv)
    assigned = valid & trusted & (warped > 0)
    new_labels = np.where(assigned, warped, 0).astype(prev_labels.dtype)
    return new_labels, assigned


def label_velocity(
    fu: np.ndarray,
    fv: np.ndarray,
    prev_mask: np.ndarray,
    fwd_trusted: np.ndarray | None = None,
) -> tuple[float, float]:
    m = np.asarray(prev_mask, bool)
    if fwd_trusted is not None:
        m = m & fwd_trusted
    if m.sum() < 10:
        m = np.asarray(prev_mask, bool)
    if m.sum() < 5:
        return 0.0, 0.0
    return float(np.median(fu[m])), float(np.median(fv[m]))


def shift_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = mask.shape
    xs, ys = pixel_grid(h, w)
    # A point at target x came from x - d for a +d moving mask.
    from .coords import sample_bilinear

    out, _ = sample_bilinear(mask.astype(np.float64), xs - dx, ys - dy)
    return out > 0.5


def mask_core(mask: np.ndarray, erode: int = 6, min_pixels: int = 30) -> np.ndarray:
    """Eroded mask interior; falls back to weaker erosion for small labels.

    Velocity medians over a full mask are biased low by the solver's
    boundary transition ramp.  The eroded core carries the true motion.
    """
    from PIL import Image, ImageFilter

    m = np.asarray(mask, bool)
    for r in (erode, 3, 0):
        if r == 0:
            return m
        out = np.array(
            Image.fromarray(m.astype(np.uint8) * 255)
            .filter(ImageFilter.MinFilter(2 * r + 1))
        ) > 0
        if out.sum() >= min_pixels:
            return out
    return m


def same_color_pixels(rgb: np.ndarray, color, tol: float = 42.0) -> np.ndarray:
    """Audit helper: pixels within Euclidean RGB distance of ``color``."""
    c = np.asarray(color, float)
    dist = np.sqrt(((rgb.astype(float) - c) ** 2).sum(axis=-1))
    return dist < tol


@dataclass
class LabelState:
    label_id: int
    velocity: tuple[float, float] = (0.0, 0.0)
    velocity_known: bool = False
    status: str = "active"          # active | held | lost
    held_frames: int = 0
    last_mask: np.ndarray | None = None
    expected_size: int = 0
    coverage: float = 1.0
    color_center: np.ndarray | None = None
    color_tol: float = 70.0
    history: list[dict] = field(default_factory=list)


class LabelTracker:
    """Stateful multi-frame label holder / reclaimer."""

    def __init__(self, hold_ttl: int = 6, reacquire_coverage: float = 0.18,
                 partial_coverage: float = 0.08, motion_tol: float = 3.0,
                 claim_reclaim_dice: float = 0.55,
                 min_color_fraction: float = 0.05):
        self.hold_ttl = hold_ttl
        self.reacquire_coverage = reacquire_coverage
        self.partial_coverage = partial_coverage
        self.motion_tol = motion_tol
        self.claim_reclaim_dice = claim_reclaim_dice
        self.states: dict[int, LabelState] = {}
        self.events: list[dict] = []

    def seed(self, labels: np.ndarray):
        h, w = labels.shape
        for lid in np.unique(labels):
            if lid == 0:
                continue
            m = labels == lid
            self.states[int(lid)] = LabelState(
                label_id=int(lid), last_mask=m, expected_size=int(m.sum())
            )

    def seed_appearance(self, rgb: np.ndarray, label_id: int = 1,
                        tol: float = 70.0):
        """Remember the median RGB colour inside a seeded label mask.

        Colour is used ONLY as independent evidence inside a motion-derived
        claim region at reappearance.  It never assigns foreground by itself
        (a same-colour decoy outside the claim stays unlabelled).
        """
        st = self.states.get(int(label_id))
        if st is None or st.last_mask is None:
            return
        px = np.asarray(rgb)[..., :3][st.last_mask]
        if len(px) >= 20:
            st.color_center = np.median(px.astype(float), axis=0)
            st.color_tol = tol

    def color_fraction(self, rgb: np.ndarray, region: np.ndarray,
                       label_id: int = 1) -> float:
        st = self.states.get(int(label_id))
        if st is None or st.color_center is None or rgb is None:
            return 0.0
        m = np.asarray(region, bool)
        if m.sum() < 20:
            return 0.0
        dist = np.linalg.norm(
            np.asarray(rgb)[..., :3].astype(float) - st.color_center, axis=-1)
        return float(np.mean(dist[m] < st.color_tol))

    def snap_reappearance(
        self,
        rgb_b: np.ndarray,
        prior_mask: np.ndarray,
        label_id: int = 1,
        radius_x: int = 12,
        radius_y: int = 6,
    ) -> tuple[np.ndarray, dict]:
        """Locally align a held template with newly visible evidence in B.

        Held extrapolation accumulates small velocity errors, so at first
        reappearance the motion prior can sit several pixels behind the
        object.  Dense flow on a thin newly exposed sliver cannot repair
        this by itself.  Search a bounded translation window and score each
        template placement by (1) alignment of its contour with strong B
        edges ("newly exposed boundary") and (2) learned foreground colour
        inside the template.  Colour never acts alone; the gates below also
        require a real edge alignment gain.  Returns ``(mask, info)``; when
        evidence is not decisive the original prior is returned.
        """
        from PIL import Image, ImageFilter

        from .candidates import strong_edge_band

        st = self.states.get(int(label_id))
        prior = np.asarray(prior_mask, bool)
        info: dict = {"snapped": False, "dx": 0, "dy": 0}
        if st is None or st.color_center is None or prior.sum() < 50:
            return prior, info

        rgb = np.asarray(rgb_b)[..., :3].astype(float)
        edge_band = strong_edge_band(rgb.mean(axis=2))
        dist = np.linalg.norm(rgb - st.color_center, axis=-1)
        color_mask = dist < st.color_tol

        def contour(m: np.ndarray) -> np.ndarray:
            core = np.asarray(
                Image.fromarray(m.astype(np.uint8) * 255)
                .filter(ImageFilter.MinFilter(5))) > 0
            return m & ~core

        def score_at(dx: float, dy: float) -> tuple[float, float, float]:
            m = shift_mask(prior, dx, dy)
            c = contour(m)
            if c.sum() < 12:
                return -1.0, 0.0, 0.0
            edge_hit = float(edge_band[c].mean())
            color_hit = float(color_mask[m].mean()) if m.any() else 0.0
            return edge_hit + 1.2 * color_hit, edge_hit, color_hit

        s0, e0, c0 = score_at(0.0, 0.0)
        info.update(base_score=round(s0, 4), base_edge=round(e0, 3),
                    base_color=round(c0, 3))
        best = (s0, 0, 0, e0, c0)
        for dx in range(-radius_x, radius_x + 1):
            for dy in range(-radius_y, radius_y + 1):
                if dx == 0 and dy == 0:
                    continue
                s, e, c = score_at(float(dx), float(dy))
                if s > best[0] + 1e-9 or (
                        abs(s - best[0]) <= 1e-9
                        and abs(dx) + abs(dy) < abs(best[1]) + abs(best[2])):
                    best = (s, dx, dy, e, c)
        s, dx, dy, e, c = best
        info.update(score=round(s, 4), edge=round(e, 3), color=round(c, 3))
        if (s >= s0 + 0.12 and c >= 0.12 and e >= 0.30
                and abs(dx) + abs(dy) > 0):
            info["snapped"] = True
            info["dx"], info["dy"] = int(dx), int(dy)
            return shift_mask(prior, float(dx), float(dy)), info
        return prior, info

    def composite_label_image(self, shape: tuple[int, int]) -> np.ndarray:
        """All non-lost masks (including held extrapolations) as label image.

        Coordinate contract: after ``step(frame_index=k)`` / ``seed``, every
        ``last_mask`` lives in frame ``k`` coordinates, so the returned image
        is a valid ``mask_a`` for the pair starting at that frame.
        """
        h, w = shape
        out = np.zeros((h, w), np.uint8)
        for lid, st in self.states.items():
            if st.status == "lost" or st.last_mask is None:
                continue
            out[st.last_mask] = lid
        return out

    def predict_claim(self) -> np.ndarray:
        """Union of velocity-extrapolated masks for the NEXT frame."""
        any_st = next(iter(self.states.values()), None)
        if any_st is None or any_st.last_mask is None:
            return np.zeros((1, 1), bool)
        h, w = any_st.last_mask.shape
        claim = np.zeros((h, w), bool)
        for st in self.states.values():
            if st.status == "lost" or st.last_mask is None:
                continue
            if not st.velocity_known:
                continue
            claim |= shift_mask(st.last_mask, *st.velocity)
        return claim

    def motion_hint(self, label_id: int | None = None
                    ) -> tuple[float, float] | None:
        """Last trusted velocity, usable as a cross-scale branch hint."""
        if not self.states:
            return None
        if label_id is not None:
            st = self.states.get(label_id)
            if st is None or not st.velocity_known or st.status == "lost":
                return None
            return st.velocity
        active = [st for st in self.states.values()
                  if st.status != "lost" and st.velocity_known]
        if not active:
            return None
        return active[0].velocity

    def step(
        self,
        frame_index: int,
        prev_labels: np.ndarray,
        bu: np.ndarray,
        bv: np.ndarray,
        trusted: np.ndarray,
        fu: np.ndarray | None = None,
        fv: np.ndarray | None = None,
        fwd_trusted: np.ndarray | None = None,
        claim_evidence: dict | None = None,
        current_rgb: np.ndarray | None = None,
        min_color_fraction: float = 0.05,
    ) -> dict:
        """Advance one pair.  Returns labels + held/claim maps + events.

        Known ids missing from ``prev_labels`` (e.g. a fully occluded object
        whose returned visible-label image is empty) are rebuilt from the
        state's coordinate-consistent ``last_mask`` before propagation, so
        the id is never silently dropped.
        """
        h, w = prev_labels.shape

        # Register ids present in the passed image but unknown to us.
        for raw_id in np.unique(prev_labels):
            lid = int(raw_id)
            if lid != 0 and lid not in self.states:
                self.states[lid] = LabelState(label_id=lid)

        # Rebuild missing ids from their last (possibly held) mask.
        effective = prev_labels
        missing = [lid for lid, st in self.states.items()
                   if not np.any(prev_labels == lid)
                   and st.last_mask is not None]
        if missing:
            effective = prev_labels.copy()
            for lid in missing:
                effective[self.states[lid].last_mask] = lid

        moved, assigned = propagate_once(effective, bu, bv, trusted)
        final_labels = moved.copy()
        held_map = np.zeros((h, w), bool)
        claim_map = np.zeros((h, w), bool)
        frame_events: list[dict] = []
        snap_infos: dict[int, dict] = {}

        for lid, st in list(self.states.items()):
            prev_mask = effective == lid
            if prev_mask.sum() == 0:
                continue
            if st.expected_size == 0:
                st.expected_size = int(prev_mask.sum())

            # Motion-consistency gate on the B grid: once the label has a
            # trusted velocity, a backward vector must approximately equal
            # -velocity.  This stops a static occluder/background at the old
            # location from claiming the label under a wrong zero branch.
            label_hits = moved == lid
            if st.velocity_known:
                vx0, vy0 = st.velocity
                tol = max(self.motion_tol, 0.35 * float(np.hypot(vx0, vy0)))
                disagree = np.hypot(bu + vx0, bv + vy0) > tol
                bad = label_hits & disagree
                final_labels[bad] = 0
                assigned[bad] = False
                label_hits = final_labels == lid

            got = int(label_hits.sum())
            coverage = got / max(1, st.expected_size)
            st.coverage = coverage

            claim_region = shift_mask(prev_mask, *st.velocity)

            # A strong label-claim score from the branch selector offers an
            # alternative reclaim path at first reappearance, when the newly
            # exposed pixels cannot yet match the occluded A texture and
            # backward trusted coverage is legitimately ~0.
            claim_dice = None
            if claim_evidence is not None:
                if claim_evidence.get("label") in (None, lid):
                    claim_dice = claim_evidence.get("dice")
                elif lid in claim_evidence:
                    claim_dice = claim_evidence[lid].get("dice")
            # Independent reappearance evidence: the motion-derived claim
            # region must actually show the learned foreground appearance in
            # the current frame.  A claim Dice alone is circular (the branch
            # may have been seeded from the same hint).
            color_frac = self.color_fraction(
                current_rgb, claim_region, lid) if current_rgb is not None else None
            appearance_ok = (
                color_frac is not None and color_frac >= min_color_fraction
            )
            claim_reclaim = (
                claim_dice is not None
                and claim_dice >= self.claim_reclaim_dice
                and appearance_ok
            )
            observed = coverage >= self.reacquire_coverage

            # Only a genuinely observed frame may update the motion.  A
            # claim-only reclaim at first reappearance keeps the held
            # velocity: backward coverage is impossible there and the raw
            # median could be dominated by the occluder.
            if observed and fu is not None and fv is not None:
                core = mask_core(prev_mask)
                # Measure velocity only where the forward field itself is
                # photo-supported, otherwise pixels over the occluder/bar
                # drag the median down and the held prior accumulates lag.
                if fwd_trusted is not None:
                    trusted_core = core & np.asarray(fwd_trusted, bool)
                    if int(trusted_core.sum()) >= 30:
                        core = trusted_core
                vx = float(np.median(fu[core]))
                vy = float(np.median(fv[core]))
                if np.isfinite(vx) and np.isfinite(vy) \
                        and abs(vx) + abs(vy) > 1e-6:
                    if not st.velocity_known:
                        st.velocity = (vx, vy)
                        st.velocity_known = True
                    else:
                        st.velocity = (0.6 * vx + 0.4 * st.velocity[0],
                                       0.6 * vy + 0.4 * st.velocity[1])

            predicted = shift_mask(prev_mask, *st.velocity)
            claim_map |= predicted

            if observed or claim_reclaim:
                reemerging = st.status in ("held", "lost")
                snap_info = None
                solid = (final_labels == lid) | predicted
                # Claim-only first reappearance: dense assignments on the
                # thin visible sliver are unreliable, so align the solid
                # template to the newly exposed edges/colour in B.
                if (reemerging and claim_reclaim and not observed
                        and current_rgb is not None):
                    snapped, snap_info = self.snap_reappearance(
                        current_rgb, predicted, lid)
                    snap_infos[lid] = snap_info
                    if snap_info.get("snapped"):
                        solid = snapped
                if reemerging:
                    how = ("label claim + appearance"
                           if claim_reclaim and not observed
                           else "trusted coverage")
                    detail = (
                        f"via {how}; coverage={coverage:.2f}, "
                        f"claim_dice={claim_dice}, "
                        f"color_frac={None if color_frac is None else round(color_frac, 2)}"
                    )
                    if snap_info and snap_info.get("snapped"):
                        detail += (
                            f"; snap=({snap_info['dx']:+d},{snap_info['dy']:+d})"
                            f" edge={snap_info['edge']:.2f}"
                            f" color={snap_info['color']:.2f}"
                        )
                    frame_events.append({
                        "frame": frame_index, "label": lid,
                        "type": "reclaimed", "detail": detail,
                    })
                    self.events.append(frame_events[-1])
                st.status = "active"
                st.held_frames = 0
                # Keep a solid region: trusted assignments can be holey on
                # 1-D texture plateaus, so union with the velocity prediction
                # (or the evidence-snapped template at first reappearance).
                st.last_mask = solid
                st.expected_size = max(st.expected_size,
                                       int(st.last_mask.sum()), got)
            else:
                # Trusted coverage collapsed: occlusion / multi-solution.
                was_lost = st.status == "lost"
                if st.status not in ("held", "lost"):
                    frame_events.append({
                        "frame": frame_index, "label": lid,
                        "type": "held",
                        "detail": f"trusted coverage={coverage:.2f}; "
                                  f"velocity={tuple(round(z, 2) for z in st.velocity)}",
                    })
                    self.events.append(frame_events[-1])
                st.held_frames += 1
                # Keep the extrapolated claim, but do NOT paint it as fact.
                held_map |= predicted
                if st.held_frames > self.hold_ttl:
                    st.status = "lost"
                    if not was_lost:
                        evt = {"frame": frame_index, "label": lid, "type": "lost"}
                        frame_events.append(evt)
                        self.events.append(evt)
                else:
                    st.status = "held"
                    claim_map |= predicted
                st.last_mask = predicted
            st.history.append({
                "frame": frame_index, "coverage": coverage,
                "status": st.status, "held_frames": st.held_frames,
            })

        # Solid state labels: active ids get their filled state mask, held
        # predictions stay ONLY in held_map (amber), never painted as fact.
        state_labels = np.zeros((h, w), prev_labels.dtype)
        for lid, st in self.states.items():
            if st.status == "active" and st.last_mask is not None:
                state_labels[st.last_mask] = lid

        # Held claims are exposed separately; never written into labels.
        return {
            "labels": state_labels,
            "trusted_labels": final_labels,
            "assigned": assigned,
            "held_map": held_map,
            "claim_map": claim_map,
            "events": frame_events,
            "snap_info": snap_infos,
        }
