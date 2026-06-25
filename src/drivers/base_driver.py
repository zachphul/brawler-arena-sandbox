"""Hardware Abstraction Layer (HAL): the ``BaseDriver`` contract.

Every environment the agent runs in — the Pygame arena, the dependency-free
mock, or any future *offline* sandbox — is exposed through this one interface.
The brain (behavior tree) and eyes (vision) depend only on ``BaseDriver``; they
must never import Pygame (or any other environment library) directly. To add an
environment: write a driver, register it in ``drivers/registry.py``, point
``config.DRIVER_NAME`` at it, and the same AI loop runs unchanged.

This sandbox is intentionally self-contained — drivers wrap simulated / offline
environments only.
"""

from __future__ import annotations

import abc
from typing import Sequence

import numpy as np

Frame = np.ndarray         # (H, W, 3) uint8, RGB
Vector = Sequence[float]   # (x, y), each component in [-1, 1]


class BaseDriver(abc.ABC):
    """Abstract sense/act interface between an agent and an environment."""

    # -- required contract --------------------------------------------------- #
    @property
    @abc.abstractmethod
    def resolution(self) -> tuple[int, int]:
        """``(width, height)`` in pixels of frames from :meth:`get_screen_frame`."""

    @abc.abstractmethod
    def get_screen_frame(self) -> Frame:
        """Return the current screen as an ``(H, W, 3)`` ``uint8`` RGB array."""

    @abc.abstractmethod
    def tap_coordinate(self, x: float, y: float) -> None:
        """Discrete tap/click at pixel ``(x, y)`` — e.g. fire or press a button."""

    @abc.abstractmethod
    def hold_joystick(self, vector: Vector) -> None:
        """Set the movement stick to ``(x, y)`` in ``[-1, 1]``; persists until changed.

        NOTE: this is a *continuous* 2-D vector, not a ``(direction, duration)``
        pair — the behavior tree streams Bézier-smoothed vectors, which a coarse
        string direction can't express.
        """

    @abc.abstractmethod
    def read_hud_text(self) -> dict[str, str]:
        """Return on-screen HUD strings, e.g. ``{'hp': 'HP: 100', 'ammo': 'AMMO: 3'}``."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the environment to a fresh episode."""

    # -- optional hooks (sensible defaults) ---------------------------------- #
    def step(self, dt: float) -> None:
        """Advance a *simulated* environment by ``dt`` seconds.

        No-op for environments that advance on their own (e.g. a real-time feed),
        so the same orchestrator loop drives both kinds.
        """

    @property
    def should_close(self) -> bool:
        """True when the environment requests shutdown (e.g. window closed)."""
        return False

    def entity_boxes(self) -> list[tuple[int, float, float, float, float]]:
        """Ground-truth ``(class_id, cx, cy, w, h)`` boxes, for sim envs that can
        provide them (used by ``OraclePerception``). Not available for real feeds.
        """
        raise NotImplementedError(
            f"{type(self).__name__} provides no ground-truth entity_boxes()"
        )

    def close(self) -> None:
        """Release driver resources. Optional — default is a no-op."""
