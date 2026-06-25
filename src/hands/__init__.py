"""hands: kinematics & action execution for the sandbox agent.

Currently provides Bézier-curve motion planning (procedural animation /
action smoothing). Future home for the simulated controller driver.
"""

from .bezier_motion import (
    CubicBezier,
    MotionPlan,
    organic_curve,
    plan_motion,
    joystick_stream,
    EASING,
    resolve_easing,
)

__all__ = [
    "CubicBezier",
    "MotionPlan",
    "organic_curve",
    "plan_motion",
    "joystick_stream",
    "EASING",
    "resolve_easing",
]
