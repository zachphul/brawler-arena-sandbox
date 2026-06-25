"""Tests for the YOLO dataset generator (normalization + engine box extraction)."""

import pathlib
import sys

import pytest

# The generator lives under tools/, not on the src path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

import dataset_generator as dg  # noqa: E402
from arena.simulation import Arena  # noqa: E402


def test_class_names_match_spec():
    assert dg.CLASS_NAMES == ["Player", "Enemy", "Projectile", "Gem", "PowerCube"]


def test_to_yolo_centered_box():
    # 100x60 box centered in an 800x600 frame.
    box = dg.to_yolo(400, 300, 100, 60, 800, 600)
    assert box == pytest.approx((0.5, 0.5, 0.125, 0.1))


def test_to_yolo_values_stay_normalized_at_edge():
    box = dg.to_yolo(5, 300, 40, 40, 800, 600)  # half hangs off the left edge
    assert box is not None
    assert all(0.0 <= v <= 1.0 for v in box)
    # Clipped width should be < the full 40/800.
    assert box[2] < 40 / 800


def test_to_yolo_fully_offscreen_is_none():
    assert dg.to_yolo(-100, -100, 10, 10, 800, 600) is None


def test_entity_boxes_have_player_enemy_and_positive_size():
    arena = Arena(seed=0, headless=True)
    try:
        boxes = arena.entity_boxes()
        ids = [b[0] for b in boxes]
        assert 0 in ids and 1 in ids
        for _cid, _cx, _cy, w, h in boxes:
            assert w > 0 and h > 0
    finally:
        arena.close()


def test_harvest_writes_paired_split_and_yaml(tmp_path):
    stats = dg.harvest(20, tmp_path, seed=5, val_split=0.2, clean=True)
    assert stats["train"] + stats["val"] == 20
    assert stats["val"] == 4  # every 5th frame
    totals, issues = dg.verify(tmp_path)
    assert issues == []
    assert totals == {"train": 16, "val": 4}
    assert (tmp_path / "dataset.yaml").exists()
    # Player + Enemy appear in every frame.
    assert stats["class_counts"][0] == 20
    assert stats["class_counts"][1] == 20
