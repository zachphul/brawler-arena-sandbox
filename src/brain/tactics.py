"""Behavior-tree tactics for the arena bot (py_trees).

The bot perceives the world only through a :class:`FrameState` (produced by the
``eyes`` layer) and acts only through a :class:`BaseDriver` (the HAL), so this
module is fully decoupled from both Pygame and YOLO.

Tree (priority Selector — survival > loot > attack > wander)::

    Selector "BrawlerTactics"
      ├─ Sequence "Survive"        : Inverter(CheckHealth) -> Evade
      ├─ CollectLoot               : move toward nearest gem / power-cube
      ├─ AttackEnemy               : if an enemy is in range, face + fire
      └─ Wander                    : roam (Bézier) when nothing else applies

Movement nodes use ``hands.bezier_motion`` to emit organic joystick streams.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import py_trees

from arena.driver_interface import BaseDriver
from eyes.vision_pipeline import FrameState
from hands.bezier_motion import joystick_stream, plan_motion

Status = py_trees.common.Status


# --------------------------------------------------------------------------- #
# Shared context + Bézier movement helper
# --------------------------------------------------------------------------- #
@dataclass
class BotContext:
    """Injected into every node: the HAL plus the latest perception + tunables."""

    driver: BaseDriver
    state: FrameState = field(default_factory=FrameState)
    low_health: int = 30
    attack_range: float = 160.0
    loot_classes: tuple[str, ...] = ("Gem", "PowerCube")


class BezierMover:
    """Turns a (start -> target) request into a rolling stream of joystick vectors.

    Keeps a short Bézier plan and emits its next eased joystick vector each tick,
    replanning when the plan is consumed or the target moves significantly. This
    gives organic, curved motion across ticks rather than a robotic straight line.
    """

    def __init__(self, *, fps=10, seed=0, retarget_dist=48.0, duration_range=(0.5, 0.9)):
        self.rng = np.random.default_rng(seed)
        self.fps = fps
        self.retarget = retarget_dist
        self.dmin, self.dmax = duration_range
        self._stream: np.ndarray | None = None
        self._i = 0
        self._target: np.ndarray | None = None

    def vector_toward(self, start, target) -> np.ndarray:
        start = np.asarray(start, dtype=float)
        target = np.asarray(target, dtype=float)
        if (self._stream is None or self._i >= len(self._stream)
                or self._target is None
                or np.linalg.norm(target - self._target) > self.retarget):
            self._replan(start, target)
        if self._stream is None or len(self._stream) == 0:
            return np.zeros(2)
        vec = self._stream[self._i]
        self._i += 1
        return vec

    def _replan(self, start, target) -> None:
        self._target = np.asarray(target, dtype=float).copy()
        self._i = 0
        if np.linalg.norm(self._target - start) < 1e-3:
            self._stream = np.zeros((1, 2))
            return
        plan = plan_motion(start, self._target,
                           duration=float(self.rng.uniform(self.dmin, self.dmax)),
                           fps=self.fps, easing="ease_in_out_sine",
                           curvature=0.20, jitter=0.05, rng=self.rng)
        self._stream = joystick_stream(plan)


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
class _BotNode(py_trees.behaviour.Behaviour):
    def __init__(self, ctx: BotContext, name: str):
        super().__init__(name=name)
        self.ctx = ctx

    @property
    def state(self) -> FrameState:
        return self.ctx.state

    def _self_pos(self) -> np.ndarray:
        player = self.state.player
        if player is not None:
            return np.array(player.center)
        w, h = self.ctx.driver.resolution
        return np.array([w / 2.0, h / 2.0])

    def _nearest(self, dets) -> object | None:
        if not dets:
            return None
        me = self._self_pos()
        return min(dets, key=lambda d: float(np.hypot(*(np.array(d.center) - me))))


class CheckHealth(_BotNode):
    """SUCCESS while healthy; FAILURE when HP < threshold (unknown HP = healthy)."""

    def update(self) -> Status:
        hp = self.state.hp
        if hp is None:
            return Status.SUCCESS
        return Status.SUCCESS if hp >= self.ctx.low_health else Status.FAILURE


class Evade(_BotNode):
    """Flee from the nearest enemy along a Bézier path. FAILURE if no enemy seen."""

    def __init__(self, ctx: BotContext):
        super().__init__(ctx, name="Evade")
        self._mover = BezierMover(seed=1)

    def update(self) -> Status:
        enemy = self._nearest(self.state.enemies)
        if enemy is None:
            return Status.FAILURE  # nothing to flee -> let lower priorities run
        me = self._self_pos()
        away = me - np.array(enemy.center)
        norm = float(np.linalg.norm(away))
        if norm < 1e-6:
            away = np.array([1.0, 0.0])
            norm = 1.0
        w, h = self.ctx.driver.resolution
        flee = me + (away / norm) * 220.0
        flee = np.clip(flee, [0, 0], [w, h])
        self.ctx.driver.hold_joystick(self._mover.vector_toward(me, flee))
        return Status.SUCCESS


class CollectLoot(_BotNode):
    """Move toward the nearest gem / power-cube. FAILURE if none visible."""

    def __init__(self, ctx: BotContext):
        super().__init__(ctx, name="CollectLoot")
        self._mover = BezierMover(seed=2)

    def update(self) -> Status:
        loot = [d for d in self.state.detections if d.class_name in self.ctx.loot_classes]
        target = self._nearest(loot)
        if target is None:
            return Status.FAILURE
        me = self._self_pos()
        self.ctx.driver.hold_joystick(self._mover.vector_toward(me, np.array(target.center)))
        return Status.SUCCESS


class AttackEnemy(_BotNode):
    """If an enemy is within range, stop, face it, and fire. FAILURE otherwise."""

    def __init__(self, ctx: BotContext):
        super().__init__(ctx, name="AttackEnemy")

    def update(self) -> Status:
        enemy = self._nearest(self.state.enemies)
        if enemy is None:
            return Status.FAILURE
        me = self._self_pos()
        if float(np.hypot(*(np.array(enemy.center) - me))) > self.ctx.attack_range:
            return Status.FAILURE
        self.ctx.driver.hold_joystick((0.0, 0.0))      # plant to aim
        self.ctx.driver.tap_coordinate(*enemy.center)   # face + fire
        return Status.SUCCESS


class Wander(_BotNode):
    """Roam toward random points (Bézier) so the bot is never idle."""

    def __init__(self, ctx: BotContext):
        super().__init__(ctx, name="Wander")
        self._mover = BezierMover(seed=3)
        self._rng = np.random.default_rng(7)
        self._target: np.ndarray | None = None

    def update(self) -> Status:
        me = self._self_pos()
        w, h = self.ctx.driver.resolution
        if self._target is None or float(np.hypot(*(self._target - me))) < 40.0:
            self._target = np.array([self._rng.uniform(0.1 * w, 0.9 * w),
                                     self._rng.uniform(0.1 * h, 0.9 * h)])
        self.ctx.driver.hold_joystick(self._mover.vector_toward(me, self._target))
        return Status.SUCCESS


# --------------------------------------------------------------------------- #
# Assembled bot
# --------------------------------------------------------------------------- #
class BrawlerBot:
    """Owns the BotContext and the py_trees BehaviourTree."""

    def __init__(self, driver: BaseDriver, *, low_health=30, attack_range=160.0):
        self.ctx = BotContext(driver=driver, low_health=low_health,
                              attack_range=attack_range)
        self.tree = py_trees.trees.BehaviourTree(self._build())

    def _build(self) -> py_trees.behaviour.Behaviour:
        survive = py_trees.composites.Sequence(name="Survive", memory=False)
        survive.add_children([
            py_trees.decorators.Inverter(name="HealthLow",
                                         child=CheckHealth(self.ctx, name="CheckHealth")),
            Evade(self.ctx),
        ])
        root = py_trees.composites.Selector(name="BrawlerTactics", memory=False)
        root.add_children([
            survive,
            CollectLoot(self.ctx),
            AttackEnemy(self.ctx),
            Wander(self.ctx),
        ])
        return root

    @property
    def root(self) -> py_trees.behaviour.Behaviour:
        return self.tree.root

    def tick(self, state: FrameState) -> None:
        """Update perception and tick the tree once."""
        self.ctx.state = state
        self.tree.tick()

    def ascii_tree(self) -> str:
        return py_trees.display.ascii_tree(self.root)
