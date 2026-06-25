"""Unit tests for hands.bezier_motion (run with `pytest`)."""

import numpy as np
import pytest

from hands.bezier_motion import (
    CubicBezier,
    organic_curve,
    plan_motion,
    joystick_stream,
    ease_in_out_cubic,
    resolve_easing,
)


def test_endpoints_are_exact():
    plan = plan_motion((0, 0), (100, 40), duration=1.0, fps=30,
                       rng=np.random.default_rng(0))
    np.testing.assert_allclose(plan.positions[0], (0, 0), atol=1e-6)
    np.testing.assert_allclose(plan.positions[-1], (100, 40), atol=1e-6)
    assert len(plan) == 30


def test_times_strictly_increasing():
    plan = plan_motion((0, 0), (10, 10), rng=np.random.default_rng(1))
    assert np.all(np.diff(plan.times) > 0)


def test_ease_in_out_accelerates_then_decelerates():
    plan = plan_motion((0, 0), (500, 0), duration=1.0, fps=60,
                       easing="ease_in_out_cubic", curvature=0.0,
                       rng=np.random.default_rng(2))
    speeds = plan.speeds
    mid = len(speeds) // 2
    assert speeds[mid] > speeds[1]
    assert speeds[mid] > speeds[-2]


def test_joystick_within_unit_box():
    plan = plan_motion((0, 0), (300, 200), rng=np.random.default_rng(3))
    js = joystick_stream(plan)
    assert js.shape == (len(plan), 2)
    assert np.all(np.abs(js) <= 1.0 + 1e-9)


def test_easing_endpoints_normalized():
    assert ease_in_out_cubic(0.0) == 0.0
    assert ease_in_out_cubic(1.0) == 1.0


def test_resolve_easing_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_easing("does_not_exist")


def test_zero_length_curve_is_stable():
    curve = organic_curve((5, 5), (5, 5))
    assert curve.length == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(curve.point_at(0.5), (5, 5))


def test_arclength_reparam_is_constant_speed_for_straight_line():
    # A straight Bézier sampled by arc length should have ~uniform spacing.
    curve = CubicBezier((0, 0), (10, 0), (20, 0), (30, 0))
    s = np.linspace(0, 1, 11)
    pts = curve.point_at_arclength(s)
    gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert np.allclose(gaps, gaps[0], rtol=0.05)
