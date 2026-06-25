"""eyes: perception — frame capture, YOLO object detection, and OCR.

Exposes the vision pipeline. Heavy deps load lazily, so importing this package
is cheap; only constructing a ``VisionEngine`` needs the 3.12 vision stack.
"""

from .oracle import OraclePerception
from .vision_pipeline import CLASS_NAMES, Detection, FrameState, VisionEngine

__all__ = ["CLASS_NAMES", "Detection", "FrameState", "VisionEngine", "OraclePerception"]
