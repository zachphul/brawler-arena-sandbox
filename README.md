# Brawler Arena Sandbox

A **self-contained AI research sandbox**. It builds a small top-down 2D *Brawler
Arena* in Pygame and runs a full perceive → decide → act agent against it —
entirely offline, against a simulated opponent.

> **Scope & ethics:** there is **no real game, no emulator, and no networking**.
> The arena is a pure *simulation target* for experimenting with computer vision,
> behavior trees, and procedural motion. The agent only ever sees rendered frames
> and acts through an abstract controller, so the same brain could later drive any
> visual environment without changing a line of decision logic.

## Why it's interesting

The agent is split into four decoupled layers that mirror a real embodied agent,
wired together by a **Hardware Abstraction Layer (HAL)** so the brain never knows
whether it's looking at a Pygame surface or a trained detector's output:

| Layer | Package | Responsibility |
|-------|---------|----------------|
| 👁️ **Eyes** | `src/eyes/` | Perception — turns a raw RGB frame into a structured `FrameState` via either ground-truth (`oracle`) or a trained **YOLOv8 + EasyOCR** pipeline (`vision`). |
| 🧠 **Brain** | `src/brain/` | Decision — a **py_trees** behavior tree: *survival → loot → attack → wander*. |
| ✋ **Hands** | `src/hands/` | Kinematics — **cubic Bézier** motion planning that emits organic, curved joystick streams instead of robotic straight lines. |
| 🌍 **World** | `src/arena/`, `src/drivers/` | The Pygame arena + the `BaseDriver` HAL and a swappable **driver registry**. |

Because every layer talks only to the `BaseDriver` contract and the `FrameState`
dataclass, you can swap the *whole environment* by adding one driver and changing
a config name — the perception, behavior tree, and motion code run unchanged.

## Architecture at a glance

```
                  ┌─────────────────────────── main.py (orchestrator, ~10 Hz) ──────────────────────────┐
                  │                                                                                       │
   driver.step(dt)│         frame ──► eyes.analyze() ──► FrameState ──► brain.tick() ──► hands ──► driver │
                  │            ▲                                                            │             │
                  └────────────┴──────────────── BaseDriver (HAL) ◄─────────────────────────┘            │
                                                     ▲   ▲
                              ┌──────────────────────┘   └───────────────────────┐
                        PygameDriver (rendered arena)                       MockDriver (headless, dep-free, CI)
```

**Behavior tree** (priority selector — first satisfied branch wins):

```
Selector "BrawlerTactics"
  ├─ Sequence "Survive"   : if health is low → Evade
  ├─ CollectLoot          : move toward nearest Gem / PowerCube
  ├─ AttackEnemy          : if an enemy is in range → face + fire
  └─ Wander               : roam (Bézier) when nothing else applies
```

**Two perception backends, one interface:**

- **`oracle`** — reads ground-truth entity boxes straight from the simulator. Zero
  ML dependencies; ideal for developing and testing the brain in isolation.
- **`vision`** — runs trained **YOLOv8** object detection + **EasyOCR** HUD reads
  on the rendered frame, so the agent perceives the world exactly as it would a
  real screen. Imported lazily, so oracle/mock runs need no `torch`.

## Project layout

```
brawler-arena-sandbox/
├─ main.py                     # orchestrator: eyes → brain → hands over a registry-selected driver
├─ config.py                   # pick driver / perception / headless / seed
├─ pyproject.toml              # Poetry project + optional `vision` dependency group
├─ src/
│  ├─ arena/                   # Pygame world, HUD, enemy bot, driver_interface (HAL re-export)
│  ├─ drivers/                 # BaseDriver contract, PygameDriver, MockDriver, registry
│  ├─ eyes/                    # oracle (ground truth) + vision_pipeline (YOLO + OCR) → FrameState
│  ├─ brain/                   # py_trees behavior tree (tactics.py)
│  └─ hands/                   # bezier_motion.py — Bézier kinematics + joystick streams
├─ tests/                      # pytest suite covering arena, drivers, tactics, vision, kinematics
└─ tools/                      # dataset_generator, train_yolo, predict_overlay, grab_frame, ...
```

## Setup

The core (arena + brain + kinematics) needs only `numpy` / `pygame` / `py-trees`.
The vision stack (`ultralytics`, `easyocr`) is torch-based and currently has **no
Python 3.14 wheels**, so use a **3.12** interpreter for the full install:

```bash
poetry env use 3.12
poetry install --with vision     # full stack (YOLO + OCR)
# or, lightweight core only:
poetry install
```

## Running

```bash
# Watch the bot play the rendered arena with ground-truth perception:
python main.py

# Headless smoke run against the dependency-free mock driver (great for CI):
python main.py --driver mock --headless --max-ticks 60

# Full vision pipeline (trained YOLO + OCR) — requires the 3.12 vision env:
python main.py --perception vision --model models/best.pt
```

| Flag | Meaning |
|------|---------|
| `--driver {pygame,mock}` | Which environment to run. |
| `--perception {oracle,vision}` | Ground-truth vs trained YOLO + OCR. |
| `--headless` | No window (off-screen); required for the mock driver / CI. |
| `--max-ticks` / `--max-seconds` | Bound the run for tests and benchmarks. |
| `--seed` | Deterministic, reproducible episodes. |

## Tooling (the vision workflow)

| Tool | Purpose |
|------|---------|
| `tools/dataset_generator.py` | Render labelled frames from the arena into a YOLO dataset. |
| `tools/train_yolo.py` | Train YOLOv8 on the generated sprites. |
| `tools/predict_overlay.py` | Visualize detections on captured frames. |
| `tools/check_hud_ocr.py` | Sanity-check the EasyOCR HUD reader. |
| `tools/grab_frame.py` / `tools/draw_labels.py` | Capture / annotate single frames. |

## Testing

```bash
pytest        # arena, drivers, tactics, vision pipeline, and Bézier kinematics
```

You can also exercise the kinematics module on its own:

```bash
python src/hands/bezier_motion.py          # numeric self-test
python src/hands/bezier_motion.py --plot   # renders a Bézier demo plot
```

## Tech stack

Python 3.10–3.13 · Pygame · NumPy · py_trees · Ultralytics YOLOv8 · EasyOCR ·
Matplotlib · Poetry · pytest

## License

© 2026 Zachary Phul. **All rights reserved.** The source is published here for
viewing and evaluation only (portfolio / job application). Please do not copy,
reuse, or redistribute it without permission. See [LICENSE](LICENSE).
