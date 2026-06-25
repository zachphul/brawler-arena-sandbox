"""Sandbox configuration — pick the environment driver and perception here.

Swapping environments is just: add a driver in ``drivers/``, register it in
``drivers/registry.py``, and change ``DRIVER_NAME`` below (or pass ``--driver``).
"""

# Which registered driver to run. See drivers/registry.py -> DRIVERS.
DRIVER_NAME = "pygame"      # "pygame" | "mock"

# Perception backend: "oracle" = ground truth (no model); "vision" = trained YOLO + OCR.
PERCEPTION = "oracle"

# Path to the trained YOLO weights (used only when PERCEPTION == "vision").
MODEL_PATH = "models/best.pt"

# Run without a window (off-screen). Required for the mock driver / CI.
HEADLESS = False

# RNG seed for reproducible runs.
SEED = 7
