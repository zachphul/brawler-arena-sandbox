"""Backwards-compatible shim — the canonical HAL now lives in ``drivers/base_driver.py``.

Kept so existing imports (`from arena.driver_interface import BaseDriver`) keep working.
"""

from drivers.base_driver import BaseDriver, Frame, Vector

__all__ = ["BaseDriver", "Frame", "Vector"]
