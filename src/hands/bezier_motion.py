"""Procedural, human-like motion via cubic Bézier curves.

The ``hands`` package turns a *desired move* ("go from A to B") into a smooth,
organically-eased sequence of waypoints that the sandbox agent follows one
frame at a time.

Design goals
------------
* **Framework-agnostic.** This module imports only ``numpy`` (+ stdlib). It
  knows nothing about Pygame, ADB, or the arena, so the same kinematics can
  drive a simulated agent, a UI animation, or a path planner.
* **Organic, not robotic.** We build a cubic Bézier for the *path shape* (a
  gentle curve, not a dead-straight segment) and apply an easing profile for
  the *speed* (accelerate out of rest, decelerate into the target).
* **Speed is controlled, not incidental.** A naïve Bézier is traversed with a
  non-uniform relationship between the parameter ``t`` and arc length. We
  reparameterize by arc length so the easing function genuinely governs how
  fast the agent moves.

Run ``python src/hands/bezier_motion.py`` for a self-test, or add ``--plot``
to render a demo PNG of the path and speed profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import numpy as np

Vector = np.ndarray  # shape (2,), float
EasingFn = Callable[[float], float]


# --------------------------------------------------------------------------- #
# Easing functions
# --------------------------------------------------------------------------- #
# Each maps normalized progress p in [0, 1] -> eased progress in [0, 1] with
# f(0) == 0 and f(1) == 1. These shape the *speed* profile along the path.

def linear(p: float) -> float:
    return p


def ease_in_quad(p: float) -> float:
    return p * p


def ease_out_quad(p: float) -> float:
    return 1.0 - (1.0 - p) * (1.0 - p)


def ease_in_out_quad(p: float) -> float:
    if p < 0.5:
        return 2.0 * p * p
    return 1.0 - ((-2.0 * p + 2.0) ** 2) / 2.0


def ease_in_out_cubic(p: float) -> float:
    if p < 0.5:
        return 4.0 * p * p * p
    return 1.0 - ((-2.0 * p + 2.0) ** 3) / 2.0


def ease_in_out_sine(p: float) -> float:
    return -(math.cos(math.pi * p) - 1.0) / 2.0


def smoothstep(p: float) -> float:
    return p * p * (3.0 - 2.0 * p)


def smootherstep(p: float) -> float:
    return p * p * p * (p * (p * 6.0 - 15.0) + 10.0)


EASING: dict[str, EasingFn] = {
    "linear": linear,
    "ease_in_quad": ease_in_quad,
    "ease_out_quad": ease_out_quad,
    "ease_in_out_quad": ease_in_out_quad,
    "ease_in_out_cubic": ease_in_out_cubic,
    "ease_in_out_sine": ease_in_out_sine,
    "smoothstep": smoothstep,
    "smootherstep": smootherstep,
}


def resolve_easing(easing: str | EasingFn) -> EasingFn:
    """Accept either an easing name or a callable and return the callable."""
    if callable(easing):
        return easing
    try:
        return EASING[easing]
    except KeyError:
        raise ValueError(
            f"Unknown easing {easing!r}. Available: {sorted(EASING)}"
        ) from None


# --------------------------------------------------------------------------- #
# Cubic Bézier curve
# --------------------------------------------------------------------------- #

def _as_vec(p) -> Vector:
    arr = np.asarray(p, dtype=float)
    if arr.shape != (2,):
        raise ValueError(f"Expected a 2D point, got shape {arr.shape}")
    return arr


@dataclass(eq=False)
class CubicBezier:
    """A cubic Bézier curve defined by four control points P0..P3.

    ``point_at`` / ``velocity_at`` accept a scalar or a 1-D array of ``t`` in
    [0, 1]. An internal arc-length table (built once at construction) lets us
    map normalized arc length -> ``t`` for constant-speed traversal.
    """

    p0: Vector
    p1: Vector
    p2: Vector
    p3: Vector
    _arc_t: np.ndarray = field(default=None, repr=False)
    _arc_s: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.p0 = _as_vec(self.p0)
        self.p1 = _as_vec(self.p1)
        self.p2 = _as_vec(self.p2)
        self.p3 = _as_vec(self.p3)
        self._build_arc_table()

    def point_at(self, t):
        """Position(s) on the curve. Scalar t -> (2,), array t -> (N, 2)."""
        t = np.asarray(t, dtype=float)
        mt = 1.0 - t
        b0 = mt ** 3
        b1 = 3.0 * mt ** 2 * t
        b2 = 3.0 * mt * t ** 2
        b3 = t ** 3
        return (b0[..., None] * self.p0
                + b1[..., None] * self.p1
                + b2[..., None] * self.p2
                + b3[..., None] * self.p3)

    def velocity_at(self, t):
        """First derivative dP/dt (the tangent / un-normalized velocity)."""
        t = np.asarray(t, dtype=float)
        mt = 1.0 - t
        d0 = 3.0 * mt ** 2
        d1 = 6.0 * mt * t
        d2 = 3.0 * t ** 2
        return (d0[..., None] * (self.p1 - self.p0)
                + d1[..., None] * (self.p2 - self.p1)
                + d2[..., None] * (self.p3 - self.p2))

    def _build_arc_table(self, samples: int = 256) -> None:
        ts = np.linspace(0.0, 1.0, samples)
        pts = self.point_at(ts)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self._arc_t = ts
        self._arc_s = np.concatenate([[0.0], np.cumsum(seg)])

    @property
    def length(self) -> float:
        """Approximate total arc length of the curve."""
        return float(self._arc_s[-1])

    def t_for_arclength(self, s):
        """Map normalized arc length s in [0, 1] to curve parameter t."""
        total = self.length
        if total == 0.0:
            return np.zeros_like(np.asarray(s, dtype=float))
        target = np.asarray(s, dtype=float) * total
        return np.interp(target, self._arc_s, self._arc_t)

    def point_at_arclength(self, s):
        """Position at normalized arc length s in [0, 1] (constant-speed)."""
        return self.point_at(self.t_for_arclength(s))


def organic_curve(
    start,
    end,
    *,
    curvature: float = 0.25,
    jitter: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> CubicBezier:
    """Build a gently curving cubic Bézier between ``start`` and ``end``.

    The two interior control points sit ~1/3 and ~2/3 along the straight line,
    then are pushed perpendicular to it so the path bows out instead of being a
    dead-straight segment. ``curvature`` scales the bow as a fraction of the
    segment length; ``jitter`` adds Gaussian variation so repeated moves are
    not identical.
    """
    rng = rng or np.random.default_rng()
    p0 = _as_vec(start)
    p3 = _as_vec(end)
    delta = p3 - p0
    dist = float(np.linalg.norm(delta))
    if dist == 0.0:
        return CubicBezier(p0, p0.copy(), p0.copy(), p0.copy())

    direction = delta / dist
    normal = np.array([-direction[1], direction[0]])

    sign = rng.choice([-1.0, 1.0]) if (jitter or curvature) else 1.0
    offset = curvature * dist * sign

    def jit() -> float:
        return float(rng.normal(0.0, jitter * dist)) if jitter else 0.0

    p1 = p0 + direction * (dist / 3.0) + normal * (offset + jit())
    p2 = p0 + direction * (2.0 * dist / 3.0) + normal * (offset + jit())
    return CubicBezier(p0, p1, p2, p3)


# --------------------------------------------------------------------------- #
# Motion planning
# --------------------------------------------------------------------------- #

@dataclass(eq=False)
class MotionPlan:
    """A time-sampled motion: positions, timestamps, and per-frame velocity."""

    positions: np.ndarray   # (N, 2)
    times: np.ndarray       # (N,)  seconds
    velocities: np.ndarray  # (N, 2)  units / second
    curve: CubicBezier

    @property
    def speeds(self) -> np.ndarray:
        return np.linalg.norm(self.velocities, axis=1)

    def __len__(self) -> int:
        return len(self.positions)

    def __iter__(self) -> Iterator[tuple[float, np.ndarray]]:
        for t, p in zip(self.times, self.positions):
            yield float(t), p


def plan_motion(
    start,
    end,
    *,
    duration: float = 1.0,
    fps: int = 60,
    easing: str | EasingFn = "ease_in_out_cubic",
    curvature: float = 0.25,
    jitter: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> MotionPlan:
    """Plan a smooth move from ``start`` to ``end``.

    Returns ``round(duration * fps)`` frame-by-frame samples following an
    organic Bézier path, with an easing profile applied over arc length so the
    agent accelerates out of ``start`` and decelerates into ``end``.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")

    ease = resolve_easing(easing)
    curve = organic_curve(start, end, curvature=curvature, jitter=jitter, rng=rng)

    n = max(2, int(round(duration * fps)))
    times = np.linspace(0.0, duration, n)
    progress = times / duration                                # linear in time
    eased = np.clip([ease(float(p)) for p in progress], 0.0, 1.0)

    ts = curve.t_for_arclength(eased)                          # arc len -> t
    positions = curve.point_at(ts)

    # Per-frame velocity via finite differences (units per second).
    velocities = np.zeros_like(positions)
    dt = np.diff(times)
    velocities[1:] = np.diff(positions, axis=0) / dt[:, None]
    velocities[0] = velocities[1]

    return MotionPlan(positions=positions, times=times,
                      velocities=velocities, curve=curve)


def joystick_stream(plan: MotionPlan, *, max_speed: Optional[float] = None) -> np.ndarray:
    """Convert a plan into per-frame twin-stick vectors in [-1, 1]^2.

    Each row is a (dx, dy) direction whose magnitude encodes how hard the stick
    is pushed (speed / max_speed). Handy for feeding a simulated controller in
    the arena instead of teleporting the agent.
    """
    speeds = plan.speeds
    peak = max_speed if max_speed is not None else (float(speeds.max()) or 1.0)
    mags = np.clip(speeds / peak, 0.0, 1.0)
    dirs = np.zeros_like(plan.velocities)
    nonzero = speeds > 1e-9
    dirs[nonzero] = plan.velocities[nonzero] / speeds[nonzero, None]
    return dirs * mags[:, None]


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #

def _self_test(plot: bool = False) -> None:
    start, end = (50.0, 300.0), (650.0, 300.0)
    rng = np.random.default_rng(42)
    plan = plan_motion(start, end, duration=1.0, fps=60,
                       easing="ease_in_out_cubic", curvature=0.30,
                       jitter=0.02, rng=rng)

    assert len(plan) == 60, len(plan)
    np.testing.assert_allclose(plan.positions[0], start, atol=1e-6)
    np.testing.assert_allclose(plan.positions[-1], end, atol=1e-6)
    assert np.all(np.diff(plan.times) > 0), "timestamps must be increasing"

    speeds = plan.speeds
    mid = len(speeds) // 2
    assert speeds[mid] > speeds[2], "expected acceleration toward the middle"
    assert speeds[mid] > speeds[-3], "expected deceleration toward the end"

    js = joystick_stream(plan)
    assert js.shape == (60, 2)
    assert np.all(np.abs(js) <= 1.0 + 1e-9), "joystick must stay in [-1, 1]"

    print("bezier_motion self-test OK")
    print(f"  frames            : {len(plan)}")
    print(f"  path length       : {plan.curve.length:.1f} px")
    print(f"  start -> end      : {plan.positions[0].round(1).tolist()} -> "
          f"{plan.positions[-1].round(1).tolist()}")
    print(f"  speed min/mid/max : {speeds.min():.1f} / {speeds[mid]:.1f} / "
          f"{speeds.max():.1f} px/s")
    print(f"  peak joystick mag : {np.linalg.norm(js, axis=1).max():.3f}")

    if plot:
        _plot(plan)


def _plot(plan: MotionPlan) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    pts = plan.positions
    ax1.plot(pts[:, 0], pts[:, 1], "-o", ms=3, label="sampled path")
    cp = np.array([plan.curve.p0, plan.curve.p1, plan.curve.p2, plan.curve.p3])
    ax1.plot(cp[:, 0], cp[:, 1], "x--", color="grey", alpha=0.6,
             label="control polygon")
    ax1.set_title("Bézier path"); ax1.set_aspect("equal")
    ax1.invert_yaxis(); ax1.legend()

    ax2.plot(plan.times, plan.speeds)
    ax2.set_title("Speed profile (ease-in-out)")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("px / s")
    fig.tight_layout()
    out = "bezier_motion_demo.png"
    fig.savefig(out, dpi=110)
    print(f"  wrote {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bézier motion self-test")
    parser.add_argument("--plot", action="store_true",
                        help="render a demo PNG (requires matplotlib)")
    args = parser.parse_args()
    _self_test(plot=args.plot)
