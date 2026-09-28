# CMOC instructions

- [Layout](#layout)
- [Setup and ROS build](#setup-and-ros-build)
- [Run the demo](#run-the-demo)
- [SPA and Search](#spa-and-search)
- [Graphs, schemas, and viewers](#graphs-schemas-and-viewers)
- [Offline review](#offline-review)
- [Troubleshooting](#troubleshooting)

## Layout

Paths below are relative to `~/ros2_ws/src/cmoc` unless stated otherwise.

| Path | Purpose |
| --- | --- |
| `demo.py` | Shared Sense–Plan–Act demo and CLI |
| `ros2/cmoc_interfaces`, `ros2/cmoc_perception` | ROS interfaces and VLM observation action server |
| `external/robokgnet` | Semantic KG submodule; canonical concept/action knowledge |
| `external/webots_ros2_simulation` | Webots/TIAGo submodule, navigation launch, maps, ROS actions |
| `episodic_memory` | Canonical scene graph and runtime SPA snapshots |
| `procedural_memory/planning` | Unified Planning wrappers and `bringing/`, `search/` PDDL domains |
| `schemas/task_frames`, `schemas/objects.json` | Task-frame and perception schemas |
| `utils/view_kg.py` | Existing browser viewer for graphs and schemas |
| `tests`, `ros2/cmoc_perception/test` | Application and ROS perception tests |
| `examples/visualization` | Tracked offline fixture outputs and provenance README |

See [README.md](README.md) for detailed PEL, action, and perception behavior.

## Setup and ROS build

Use ROS 2 Jazzy, its system Python, Webots, `webots_ros2`, Nav2, RViz, and
`laser_filters`. The simulator launch expects `/usr/local/webots/webots`.
Install ROS/Webots first; this is not a standalone pip application.

For a new checkout:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone --recurse-submodules https://github.com/Dorteel/cmoc.git
```

For an existing checkout:

```bash
cd ~/ros2_ws/src/cmoc
git submodule update --init --recursive
```

Build **from `~/ros2_ws`, with no CMOC virtual environment active**. Use a fresh
shell, or run `deactivate` first if necessary. ROS/ament build tooling needs the
system ROS Python environment; `.venv` is for application runtime.

The package names below were checked against their `package.xml` files. The
first two live in `ros2/`; the next two in matching simulator subdirectories;
`explore_lite_msgs` and `explore_lite` live in the simulator's
`third_party/explore_lite_msgs` and `third_party/explore` directories. Exploration
supports mapping; the normal Bring run uses the saved map.

```bash
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
rosdep install --from-paths src/cmoc/ros2 \
  src/cmoc/external/webots_ros2_simulation --ignore-src -r -y
colcon build --symlink-install \
  --packages-select \
  cmoc_interfaces cmoc_perception \
  simulation_actions navigate_to_position \
  explore_lite_msgs explore_lite
source install/setup.bash

cd ~/ros2_ws/src/cmoc
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt -r external/robokgnet/requirements.txt
```

The local wider workspace includes Protobuf with a `protobuf_BUILD_TESTS` CMake
option. When building that package in a broader workspace build, append
`--cmake-args -Dprotobuf_BUILD_TESTS=OFF`. It is unnecessary for the six-package
selection above, which does not build Protobuf.

Ollama must be installed and serving at `http://localhost:11434`. If it is not
already running as a service, run `ollama serve` in another terminal. Obtain the
local semantic model and fallback VLM:

```bash
ollama pull qwen3:1.7b
ollama pull qwen3-vl:2b
read -rsp 'Nebula API key: ' NEBULA_API_KEY; echo
export NEBULA_API_KEY
```

The demo starts its perception server automatically with Nebula as primary
backend and local Ollama as fallback. `.env` is not loaded automatically. The
semantic fallback uses local `qwen3:1.7b`; perception uses `qwen3-vl:2b`. See the
README for endpoint, retry, and provenance details.

## Run the demo

In each runtime terminal:

```bash
source ~/ros2_ws/install/setup.bash
cd ~/ros2_ws/src/cmoc
source .venv/bin/activate
```

Run one command at a time:

```bash
python3 demo.py                           # legacy, no plan action dispatch
python3 demo.py --execute                 # legacy, Nav2 and ROS actions
python3 demo.py --execute --step           # legacy, confirm each action
python3 demo.py --search                   # Search mode, no plan action dispatch
python3 demo.py --search --execute         # Search mode with execution
python3 demo.py --search --execute --step  # Search mode with confirmations
```

**Dry run is not offline:** without `--execute`, the demo still starts the
simulator, waits for Nav2, and uses live VLM sensing. `--step` requires `--execute`.
Search is off by default. Startup prints `Demo mode: LEGACY SCENE GRAPH` or
`Demo mode: SEARCH / PARTIAL OBSERVABILITY`. Ctrl+C stops the owned processes;
the simulator otherwise stays open after completion.

Other supported flags: `--no-simulator` uses an already running simulator;
`--test-navigation` visits KITCHEN and LIVING_ROOM_1 instead of SPA;
`--scenario {existing,empty,human-moves}` selects initial episodic setup
(default `existing`). `--teleport` exists and requires `--execute`, but is not
the recommended execution path here.

Current validation limitation: odometry-gated launch started TIAGo and one Nav2
stack, but recent legacy runs stalled during lifecycle activation on docking or
collision-monitor services. These commands match the CLI; successful end-to-end
execution is not currently established. Search live validation remains deferred.

## SPA and Search

The conceptual loop is:

```python
while not task_complete:
    state = sense(result)
    plan_ = plan(state)
    result = act(plan_)
```

- **Sense:** instruction and VLM observation produce an observation graph; actual
  observations update episodic memory.
- **Plan:** frame filling, Perceived Entity Linking (PEL), semantic/episodic
  grounding, then PDDL/Unified Planning produce a plan.
- **Act:** ROS2 actions dispatch Nav2 navigation and manipulation, producing an
  execution result (or a dry-run snapshot).

Legacy uses full existing scene-graph grounding for Bring and has no Search
recovery. Search mode retains Bring as the main intention, but invokes a Search
frame when the Theme cannot be grounded; a fresh pre-pick visibility check can
also interrupt Bring. It selects one candidate at a time from observation-backed
interaction evidence, RoboKGNet priors, then VLM/semantic-LLM suggestions grounded
to observed IDs. Failed candidates are exclusions for the current task only.

The KG facade is `knowledge_interface.py`, delegating to
`external/robokgnet/knowledge_interface.py` and its canonical
`robonet_graph/robokgconceptnet.json` and `robokgverbnet_framenet.json` data.
Search planning uses `observed_snapshot()` rather than simulator-seeded object
locations. Simulation geometry may resolve execution coordinates after selection.

Search plans `look-at`; a **fresh VLM observation**, not PDDL completion,
determines success. Successful evidence updates memory and replans Bring; the
stale Bring plan is discarded. Current `look-at` turns from the current position:
it does not navigate between candidate rooms and needs execution coordinates.

## Graphs, schemas, and viewers

| Artifact under `episodic_memory/` | Meaning |
| --- | --- |
| `scene_graph.json` | Canonical episodic scene graph; objects, relations, and saved evidence |
| `g1_sense.json` | G1: perception/sensed state |
| `g2_plan.json` | G2: reasoning, grounding, and planning snapshot |
| `g3_action.json` | G3: action/execution snapshot |

G2 is a lens over the knowledge used for planning, not a warehouse containing
the complete world model. Search G2 carries the selected candidate and small
plan. Snapshots are runtime outputs; missing files or an empty G3 mean the stage
has not produced a usable output yet.

From the repository root, run a viewer command and open its printed localhost URL:

```bash
python3 utils/view_kg.py scene_graph
python3 utils/view_kg.py scene_graph --tree --watch
python3 utils/view_kg.py g1 --watch
python3 utils/view_kg.py g2 --watch
python3 utils/view_kg.py g3 --watch
python3 utils/view_kg.py schemas --watch
```

The `scene_graph` shortcut reads `episodic_memory/scene_graph.json` directly; it
is not a generated SPA snapshot. Viewing does not expose knowledge to Search or
change the graph. `--tree` is supported for `scene_graph` or a path named
`scene_graph.json`. `--watch` waits for missing artifacts; live reload is always
enabled once the viewer starts. Ctrl+C closes the viewer.

Schema visualization is derived from the actual JSON files in
`schemas/task_frames/`, not a manually copied diagram. Currently these are
`bring-11.3.json`, `search.json`, and `search_result.json`. Search has required
`Agent`, `Theme`, `Location`, and boolean result field `Success`; its separate
result schema uses `theme`, `location`, and `success`. This guide changes no
schema semantics.

## Offline review

No ROS, Webots, Nav2, VLM, or API key is needed for the viewer. In a plain Python
environment, the only viewer dependency is `rdflib`:

```bash
cd ~/ros2_ws/src/cmoc
python3 -m venv /tmp/cmoc-viewer-venv
source /tmp/cmoc-viewer-venv/bin/activate
python3 -m pip install rdflib
python3 utils/view_kg.py scene_graph --tree
python3 utils/view_kg.py schemas
```

To inspect committed examples, run these individually:

```bash
python3 utils/view_kg.py examples/visualization/scene_graph.json --tree
python3 utils/view_kg.py examples/visualization/g1.json
python3 utils/view_kg.py examples/visualization/g2.json
python3 utils/view_kg.py examples/visualization/g3.json
```

These are fixture-based dry-run outputs, **not recorded VLM/robot success**;
G3 has no executed actions. See [example provenance](examples/visualization/README.md).
For your own saved outputs, pass their JSON paths or use the G1/G2/G3 shortcuts.
No new sample data is needed or generated by these instructions.

## Troubleshooting

**Workspace contamination:** duplicate packages, stale install paths, or strange
behavior after branch changes warrant checking discovery from `~/ros2_ws`:

```bash
cd ~/ros2_ws
colcon list
```

Do not keep another CMOC checkout/worktree under `~/ros2_ws/src/` unless it is
excluded with `COLCON_IGNORE`. Accidental `build/`, `install/`, or `log/`
directories inside `cmoc/` should not normally exist.

**Virtual environment during build:** if ament invokes `cmoc/.venv/bin/python3`
and cannot import ROS dependencies such as `catkin_pkg`, deactivate the venv and
clean/rebuild from the workspace root. Do not repair this by building in CMOC.

**Clean rebuild:** the following deletes workspace build outputs for all packages,
not source. Use it after checking the paths, when a clean rebuild is intended;
packages outside the selected six must be rebuilt separately if needed.

```bash
# In a fresh shell; deactivate first if a venv is active.
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
rm -rf ~/ros2_ws/build ~/ros2_ws/install ~/ros2_ws/log
colcon build --symlink-install --packages-select \
  cmoc_interfaces cmoc_perception simulation_actions navigate_to_position \
  explore_lite_msgs explore_lite
source install/setup.bash
cd ~/ros2_ws/src/cmoc
source .venv/bin/activate
```

A lifecycle-service startup stall is still unresolved; this guide does not claim
that rebuilding fixes it. No simulator or build was run to prepare this document.
