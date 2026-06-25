"""Pygame-arena driver: the ``BaseDriver`` bound to the offline ``Arena``.

The only place that knows both the HAL contract and Pygame internals. It owns an
``Arena`` instance and maps the abstract sense/act calls onto it.
"""

from __future__ import annotations

import numpy as np
import pygame

from drivers.base_driver import BaseDriver, Frame, Vector


class PygameDriver(BaseDriver):
    """Drives the agent's I/O against a running :class:`~arena.simulation.Arena`."""

    def __init__(self, arena) -> None:
        self._arena = arena
        self._should_close = False

    @property
    def arena(self):
        return self._arena

    @property
    def resolution(self) -> tuple[int, int]:
        return self._arena.size

    def get_screen_frame(self) -> Frame:
        # surfarray.array3d -> (W, H, 3) RGB; transpose to image-convention (H, W, 3).
        arr = pygame.surfarray.array3d(self._arena.screen)
        return np.transpose(arr, (1, 0, 2)).copy()

    def tap_coordinate(self, x: float, y: float) -> None:
        self._arena.register_tap(float(x), float(y))

    def hold_joystick(self, vector: Vector) -> None:
        self._arena.set_move_input(vector)

    def read_hud_text(self) -> dict[str, str]:
        return self._arena.hud_text()

    def entity_boxes(self):
        return self._arena.entity_boxes()

    def step(self, dt: float) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._should_close = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._should_close = True
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._arena.register_tap(*event.pos)
        self._arena.step(dt)
        if self._arena.game_over:
            self._should_close = True

    @property
    def should_close(self) -> bool:
        return self._should_close

    def reset(self) -> None:
        self._arena.reset()
        self._should_close = False

    def close(self) -> None:
        self._arena.close()
