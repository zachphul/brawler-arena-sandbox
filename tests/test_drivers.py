"""Tests for the driver registry and the dependency-free MockDriver.

The MockDriver path uses no Pygame and no torch, so the full
eyes(oracle) -> brain -> hands loop is exercised here with zero heavy deps.
"""

import numpy as np
import pytest

from brain.tactics import BrawlerBot
from drivers.base_driver import BaseDriver
from drivers.registry import DRIVERS, get_driver
from eyes.oracle import OraclePerception


def test_registry_has_expected_drivers():
    assert "pygame" in DRIVERS and "mock" in DRIVERS


def test_get_driver_unknown_raises():
    with pytest.raises(ValueError):
        get_driver("does_not_exist")


def test_mock_driver_satisfies_contract():
    d = get_driver("mock", seed=0)
    assert isinstance(d, BaseDriver)
    w, h = d.resolution
    frame = d.get_screen_frame()
    assert frame.shape == (h, w, 3) and frame.dtype == np.uint8
    hud = d.read_hud_text()
    assert hud["hp"].startswith("HP: ") and hud["ammo"].startswith("AMMO: ")
    ids = [b[0] for b in d.entity_boxes()]
    assert 0 in ids and 1 in ids  # player + enemy


def test_mock_hold_joystick_moves_player_right():
    d = get_driver("mock", seed=0)
    x0 = d.entity_boxes()[0][1]
    d.hold_joystick((1.0, 0.0))
    for _ in range(10):
        d.step(1 / 60)
    assert d.entity_boxes()[0][1] > x0


def test_mock_tap_consumes_ammo():
    d = get_driver("mock", seed=0)
    d.tap_coordinate(10, 10)
    assert d.read_hud_text()["ammo"] == "AMMO: 2"


def test_mock_reset_restores_state():
    d = get_driver("mock", seed=0)
    d.tap_coordinate(0, 0)
    d.hold_joystick((1.0, 0.0))
    for _ in range(30):
        d.step(1 / 60)
    d.reset()
    assert d.read_hud_text()["ammo"] == "AMMO: 3"


def test_full_loop_runs_on_mock_via_oracle():
    # The whole stack on a no-Pygame, no-torch environment.
    driver = get_driver("mock", seed=1)
    perception = OraclePerception(driver)
    bot = BrawlerBot(driver)
    for _ in range(40):
        driver.step(1 / 60)
        state = perception.analyze(driver.get_screen_frame())
        bot.tick(state)
    assert bot.root.tip() is not None  # the tree always selects a leaf action
