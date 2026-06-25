"""Pure-logic tests for the vision pipeline (no torch/ultralytics needed).

These exercise the OCR parsing and FrameState query helpers, so they run on the
lightweight 3.14 env. Inference itself is validated separately once the 3.12
vision stack + a trained model are available.
"""

from eyes.vision_pipeline import CLASS_NAMES, Detection, FrameState, _first_int


def test_class_names_match_dataset():
    assert CLASS_NAMES == ["Player", "Enemy", "Projectile", "Gem", "PowerCube"]


def test_first_int_labeled():
    assert _first_int("HP: 100 AMMO: 3", "HP") == 100
    assert _first_int("HP: 100 AMMO: 3", "AMMO") == 3


def test_first_int_is_tolerant_and_case_insensitive():
    assert _first_int("HP  87", "HP") == 87
    assert _first_int("hp:5", "HP") == 5
    assert _first_int("AMMO: 12 GEMS: 4", "AMMO") == 12


def test_first_int_missing_returns_none():
    assert _first_int("GEMS: 4", "HP") is None


def test_detection_center():
    d = Detection(1, "Enemy", 0.9, (10.0, 20.0, 30.0, 40.0))
    assert d.center == (20.0, 30.0)


def test_framestate_queries():
    dets = [
        Detection(0, "Player", 0.90, (0, 0, 10, 10)),
        Detection(1, "Enemy", 0.80, (50, 50, 70, 70)),
        Detection(1, "Enemy", 0.95, (80, 80, 100, 100)),
    ]
    fs = FrameState(detections=dets, hp=100, ammo=3)
    assert len(fs.enemies) == 2
    assert fs.of_class("Gem") == []
    assert fs.player is not None and fs.player.class_name == "Player"
    assert fs.hp == 100 and fs.ammo == 3


def test_framestate_defaults_are_empty():
    fs = FrameState()
    assert fs.detections == []
    assert fs.hp is None and fs.ammo is None
    assert fs.player is None
