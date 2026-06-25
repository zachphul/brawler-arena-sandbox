"""Headless tests for the arena engine + Pygame HAL driver.

Forces SDL's dummy video/audio drivers so the suite runs with no window.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest

from arena.driver_interface import BaseDriver
from arena.pygame_driver import PygameDriver
from arena.simulation import Arena, WanderAgent


@pytest.fixture
def arena():
    a = Arena(seed=1, headless=True)
    try:
        yield a
    finally:
        a.close()


def test_driver_implements_interface(arena):
    driver = PygameDriver(arena)
    assert isinstance(driver, BaseDriver)
    assert driver.resolution == arena.size


def test_frame_shape_and_dtype(arena):
    driver = PygameDriver(arena)
    arena.set_driver(driver)
    arena.step(1 / 60)
    frame = driver.get_screen_frame()
    w, h = arena.size
    assert frame.shape == (h, w, 3)
    assert frame.dtype == np.uint8


def test_hold_joystick_moves_player_right(arena):
    driver = PygameDriver(arena)
    arena.set_driver(driver)
    x0 = arena.player.pos[0]
    driver.hold_joystick((1.0, 0.0))
    for _ in range(10):
        arena.step(1 / 60)
    assert arena.player.pos[0] > x0


def test_joystick_vector_is_clipped(arena):
    driver = PygameDriver(arena)
    arena.set_driver(driver)
    driver.hold_joystick((5.0, -9.0))
    assert np.all(np.abs(arena.player.move_input) <= 1.0)


def test_read_hud_text_format(arena):
    driver = PygameDriver(arena)
    arena.set_driver(driver)
    arena.step(1 / 60)
    hud = driver.read_hud_text()
    assert hud["hp"].startswith("HP: ")
    assert hud["ammo"].startswith("AMMO: ")
    assert hud["hp"].split(": ")[1].isdigit()


def test_tap_consumes_ammo_and_spawns_projectile(arena):
    driver = PygameDriver(arena)
    arena.set_driver(driver)
    arena.player.ammo = 3
    before = len(arena.projectiles)
    driver.tap_coordinate(arena.size[0] - 10, arena.player.pos[1])
    assert arena.player.ammo == 2
    assert len(arena.projectiles) == before + 1


def test_tap_with_no_ammo_is_noop(arena):
    driver = PygameDriver(arena)
    arena.set_driver(driver)
    arena.player.ammo = 0
    driver.tap_coordinate(0, 0)
    assert arena.player.ammo == 0
    assert len(arena.projectiles) == 0


def test_runs_with_injected_agent(arena):
    driver = PygameDriver(arena)
    arena.set_driver(driver)
    agent = WanderAgent(resolution=arena.size, seed=2, fps=arena.fps)
    frames = arena.run(agent=agent, max_frames=60)
    assert frames == 60
    assert arena.player.hp > 0  # shouldn't die to the enemy in 1 second
