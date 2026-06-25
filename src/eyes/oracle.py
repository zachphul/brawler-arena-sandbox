"""Ground-truth perception for any sim driver (no trained model needed).

Builds a :class:`FrameState` from a driver's ground-truth ``entity_boxes()`` and
``read_hud_text()``, shaped exactly like :class:`VisionEngine`'s output — so the
behavior tree can run against the Pygame arena or the mock driver immediately.
"""

from __future__ import annotations

import time

from .vision_pipeline import CLASS_NAMES, Detection, FrameState, _first_int


class OraclePerception:
    """Drop-in replacement for ``VisionEngine`` that reads ground truth."""

    def __init__(self, driver):
        self.driver = driver

    def analyze(self, frame=None) -> FrameState:
        detections = []
        for cid, cx, cy, w, h in self.driver.entity_boxes():
            name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
            detections.append(Detection(cid, name, 1.0,
                                        (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        hud = self.driver.read_hud_text()
        return FrameState(
            detections=detections,
            hp=_first_int(hud.get("hp", ""), "HP"),
            ammo=_first_int(hud.get("ammo", ""), "AMMO"),
            hud_text=" ".join(hud.values()),
            timestamp=time.time(),
        )

    def close(self) -> None:
        pass
