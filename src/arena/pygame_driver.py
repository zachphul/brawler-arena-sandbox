"""Backwards-compatible shim — ``PygameDriver`` now lives in ``drivers/pygame_driver.py``."""

from drivers.pygame_driver import PygameDriver

__all__ = ["PygameDriver"]
