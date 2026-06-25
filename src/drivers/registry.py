"""Driver registry: map a name -> a ready-to-use ``BaseDriver``.

Adding an environment is three steps:
  1. write ``drivers/<name>_driver.py`` implementing ``BaseDriver``,
  2. add a factory entry here,
  3. set ``config.DRIVER_NAME`` (or pass ``--driver``).

Factories (rather than bare classes) let each driver wire up its own
environment — e.g. the Pygame driver builds an ``Arena`` — while heavy imports
(Pygame) stay lazy so a mock-only / CI run needs nothing extra.
"""

from __future__ import annotations

from drivers.base_driver import BaseDriver


def _make_pygame(*, headless=False, seed=None, size=(800, 600), **_) -> BaseDriver:
    from arena.simulation import Arena
    from drivers.pygame_driver import PygameDriver
    return PygameDriver(Arena(size=size, seed=seed, headless=headless))


def _make_mock(*, seed=None, size=(800, 600), **_) -> BaseDriver:
    from drivers.mock_driver import MockDriver
    return MockDriver(size=size, seed=seed)


DRIVERS = {
    "pygame": _make_pygame,
    "mock": _make_mock,
}


def get_driver(name: str, **kwargs) -> BaseDriver:
    try:
        factory = DRIVERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown driver {name!r}. Available: {sorted(DRIVERS)}"
        ) from None
    return factory(**kwargs)
