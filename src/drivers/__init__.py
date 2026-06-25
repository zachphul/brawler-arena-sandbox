"""drivers: the HAL contract + selectable offline environment drivers."""

from .base_driver import BaseDriver, Frame, Vector
from .registry import DRIVERS, get_driver

__all__ = ["BaseDriver", "Frame", "Vector", "DRIVERS", "get_driver"]
