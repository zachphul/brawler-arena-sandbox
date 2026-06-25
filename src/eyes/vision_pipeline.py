"""Vision pipeline: turn a raw HAL frame into a structured ``FrameState``.

For each frame the :class:`VisionEngine` runs two analyzers **in parallel**:

* :meth:`VisionEngine.process_entities` — YOLOv8 detection of player, enemy,
  projectile, gem, and power-cube boxes.
* :meth:`VisionEngine.process_hud` — crops the HUD ROI (top-left) and runs
  EasyOCR to read the HP and Ammo integers.

The results are merged into a :class:`FrameState` that the ``brain`` layer can
consume. Heavy dependencies (``ultralytics``, ``easyocr``, ``torch``) are
imported lazily inside ``__init__`` so this module imports fine even where they
are not installed — constructing a ``VisionEngine`` is what requires the 3.12
vision environment.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Must match the dataset class ids (arena.entity_boxes / dataset_generator).
CLASS_NAMES = ["Player", "Enemy", "Projectile", "Gem", "PowerCube"]


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class FrameState:
    """Everything the brain needs to know about one frame."""

    detections: list[Detection] = field(default_factory=list)
    hp: Optional[int] = None
    ammo: Optional[int] = None
    hud_text: str = ""
    timestamp: float = 0.0

    def of_class(self, name: str) -> list[Detection]:
        return [d for d in self.detections if d.class_name == name]

    @property
    def enemies(self) -> list[Detection]:
        return self.of_class("Enemy")

    @property
    def player(self) -> Optional[Detection]:
        players = self.of_class("Player")
        return max(players, key=lambda d: d.confidence) if players else None


def _first_int(text: str, label: str) -> Optional[int]:
    """Pull the first integer following ``label`` (tolerant of OCR noise)."""
    match = re.search(rf"{label}\D*?(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


class VisionEngine:
    """Runs YOLO + OCR over frames and returns a merged :class:`FrameState`."""

    def __init__(self, model_path="models/best.pt", *, hud_roi=(0, 0, 220, 100),
                 ammo_roi=(80, 34, 140, 66), hud_pad=10, conf=0.25,
                 ocr_langs=("en",), use_gpu=False, class_names=None):
        self.model_path = str(model_path)
        self.hud_roi = tuple(hud_roi)
        self.ammo_roi = tuple(ammo_roi)  # tight box around just the AMMO digits
        self.hud_pad = hud_pad
        self.conf = conf
        self.class_names = list(class_names) if class_names else CLASS_NAMES

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - env guard
            raise ImportError(
                "ultralytics is required for VisionEngine. In the 3.12 env: "
                "pip install ultralytics easyocr torch"
            ) from exc
        self._yolo = YOLO(self.model_path)

        try:
            import easyocr
        except ImportError as exc:  # pragma: no cover - env guard
            raise ImportError(
                "easyocr is required for VisionEngine. In the 3.12 env: "
                "pip install easyocr"
            ) from exc
        self._reader = easyocr.Reader(list(ocr_langs), gpu=use_gpu)

        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vision")

    # -- analyzers ----------------------------------------------------------- #
    def process_entities(self, frame: np.ndarray) -> list[Detection]:
        """YOLOv8 inference -> list of :class:`Detection`."""
        # HAL frames are RGB; ultralytics treats numpy input as BGR (cv2
        # convention) and flips internally, so feed BGR to match training.
        bgr = np.ascontiguousarray(frame[:, :, ::-1])
        result = self._yolo.predict(bgr, conf=self.conf, verbose=False)[0]
        detections: list[Detection] = []
        for box in result.boxes:
            cid = int(box.cls[0])
            name = self.class_names[cid] if cid < len(self.class_names) else str(cid)
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(Detection(cid, name, float(box.conf[0]), (x1, y1, x2, y2)))
        return detections

    def _hud_crop(self, frame: np.ndarray, roi) -> np.ndarray:
        """Crop ``roi`` with ``hud_pad`` padding, clamped to the frame.

        EasyOCR drops/garbles glyphs that touch a tight crop edge, so we give it
        a little breathing room around the HUD.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = roi
        pad = self.hud_pad
        return frame[max(0, y1 - pad):min(h, y2 + pad),
                     max(0, x1 - pad):min(w, x2 + pad)]

    def _read_ammo_value(self, frame: np.ndarray) -> Optional[int]:
        """Digit-only OCR of just the AMMO value ROI (excludes the label text).

        The general HUD pass tends to drop the small single-digit AMMO; OCRing a
        tight crop of only the value, with an integer ``allowlist``, recovers it
        without the surrounding letters contaminating the read.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.ammo_roi
        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if crop.size == 0:
            return None
        # EasyOCR's detector misses tiny isolated glyphs; upscale ~4x (nearest)
        # and loosen the text thresholds so the lone digit is found.
        crop = np.repeat(np.repeat(crop, 4, axis=0), 4, axis=1)
        digits = self._reader.readtext(crop, detail=0, allowlist="0123456789",
                                       text_threshold=0.5, low_text=0.3)
        numbers = [int(n) for d in digits for n in re.findall(r"\d+", d)]
        return numbers[0] if numbers else None

    def process_hud(self, frame: np.ndarray) -> tuple[Optional[int], Optional[int], str]:
        """Crop the padded HUD ROI and OCR it -> (hp, ammo, raw_text).

        HP comes from the general (labelled) read of the padded HUD crop. If that
        read also yields AMMO, great; otherwise fall back to a tight digit-only
        read of just the AMMO value ROI (see :meth:`_read_ammo_value`).
        """
        crop = self._hud_crop(frame, self.hud_roi)
        lines = self._reader.readtext(crop, detail=0, paragraph=False)
        text = " ".join(lines)
        hp = _first_int(text, "HP")
        ammo = _first_int(text, "AMMO")
        if ammo is None:
            ammo = self._read_ammo_value(frame)
        return hp, ammo, text

    # -- combined ------------------------------------------------------------ #
    def analyze(self, frame: np.ndarray) -> FrameState:
        """Run entity + HUD analysis in parallel and merge into a FrameState."""
        fut_entities = self._pool.submit(self.process_entities, frame)
        fut_hud = self._pool.submit(self.process_hud, frame)
        detections = fut_entities.result()
        hp, ammo, hud_text = fut_hud.result()
        return FrameState(detections=detections, hp=hp, ammo=ammo,
                          hud_text=hud_text, timestamp=time.time())

    def close(self) -> None:
        self._pool.shutdown(wait=False)

    def __enter__(self) -> "VisionEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
