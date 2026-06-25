"""Behavior-tree tests: verify the priority Selector picks the right action.

Uses a fake driver (records commands) and hand-built FrameStates, so these run
on the lightweight 3.14 env (only py_trees + numpy needed).
"""

import numpy as np

from arena.driver_interface import BaseDriver
from brain.tactics import BrawlerBot
from eyes.vision_pipeline import Detection, FrameState


class FakeDriver(BaseDriver):
    def __init__(self, resolution=(800, 600)):
        self._res = resolution
        self.last_joystick = None
        self.taps = []

    @property
    def resolution(self):
        return self._res

    def get_screen_frame(self):
        w, h = self._res
        return np.zeros((h, w, 3), dtype=np.uint8)

    def tap_coordinate(self, x, y):
        self.taps.append((x, y))

    def hold_joystick(self, vector):
        self.last_joystick = np.asarray(vector, dtype=float)

    def read_hud_text(self):
        return {}

    def reset(self):
        self.last_joystick = None
        self.taps = []


def det(class_id, name, center, size=(32, 32)):
    cx, cy = center
    w, h = size
    return Detection(class_id, name, 1.0, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))


def player_at(center=(400, 300)):
    return det(0, "Player", center)


def tick_once(bot, **state_kwargs):
    bot.tick(FrameState(**state_kwargs))
    tip = bot.root.tip()
    return tip.name if tip is not None else None


def test_low_health_with_enemy_evades():
    driver = FakeDriver()
    bot = BrawlerBot(driver)
    action = tick_once(bot, hp=10,
                       detections=[player_at((400, 300)), det(1, "Enemy", (450, 300))])
    assert action == "Evade"
    # moving away from the enemy -> negative x component of joystick
    assert driver.last_joystick is not None
    assert driver.last_joystick[0] <= 0.0


def test_survival_beats_loot():
    driver = FakeDriver()
    bot = BrawlerBot(driver)
    action = tick_once(bot, hp=10, detections=[
        player_at((400, 300)),
        det(1, "Enemy", (430, 300)),
        det(3, "Gem", (410, 300)),
    ])
    assert action == "Evade"


def test_healthy_collects_loot():
    driver = FakeDriver()
    bot = BrawlerBot(driver)
    action = tick_once(bot, hp=100,
                       detections=[player_at((400, 300)), det(3, "Gem", (600, 300))])
    assert action == "CollectLoot"
    assert driver.last_joystick is not None


def test_loot_beats_attack():
    driver = FakeDriver()
    bot = BrawlerBot(driver)
    action = tick_once(bot, hp=100, detections=[
        player_at((400, 300)),
        det(1, "Enemy", (440, 300)),     # in range
        det(4, "PowerCube", (600, 300)),
    ])
    assert action == "CollectLoot"
    assert driver.taps == []  # did not fire


def test_attacks_enemy_in_range_when_no_loot():
    driver = FakeDriver()
    bot = BrawlerBot(driver)
    action = tick_once(bot, hp=100,
                       detections=[player_at((400, 300)), det(1, "Enemy", (450, 300))])
    assert action == "AttackEnemy"
    assert len(driver.taps) == 1


def test_far_enemy_is_not_attacked_so_bot_wanders():
    driver = FakeDriver()
    bot = BrawlerBot(driver)
    action = tick_once(bot, hp=100,
                       detections=[player_at((400, 300)), det(1, "Enemy", (790, 590))])
    assert action == "Wander"
    assert driver.taps == []


def test_idle_wanders():
    driver = FakeDriver()
    bot = BrawlerBot(driver)
    action = tick_once(bot, hp=100, detections=[player_at((400, 300))])
    assert action == "Wander"
    assert driver.last_joystick is not None
