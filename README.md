# Common Model of Cognition (CMOC)

CMOC is a robotics prototype organized around Sense → Plan → Act. It reuses
RoboKGNet for static semantic knowledge, Webots/TIAGo for simulation, and the
existing CMOC perception action for VLM observations. The current SPA demo makes
a deterministic Bringing decision; it does **not** execute a delivery.

## Current architecture

```text
Sense
  → current TIAGo camera observation through /observe_with_vlm (Nebula)
  → G1: Episode + Instruction + raw Observation (observedBy robot)
Plan
  → merge observation into episodic memory
  → extract semantic task frame
  → Perceived Entity Linking (PEL) and concrete task grounding
  → G2: same Episode + PEL + task frame + Frame Element bindings
Act
  → currently prints the decision; no navigation/manipulation chain
  → G3: placeholder until action execution/result observations exist
```

- `external/robokgnet/` is a Git submodule. CMOC's `knowledge_interface.py`
  delegates to its explicitly qualified `external.robokgnet.knowledge_interface`
  API. Only RoboKGNet reads/writes its canonical concept/action JSON.
- `external/webots_ros2_simulation/` is a separate Git submodule. Its existing
  navigation launch starts Webots, TIAGo, controllers, map server, and Nav2.
- `observation.py` calls the existing `cmoc_perception` `/observe_with_vlm`
  action. The server snapshots the latest RGB camera image and validates JSON.
  Entities use `schemas/objects.json` inside the existing scene-graph envelope.
- `scene_graph_interface.py` owns episodic objects/relations. Observations upsert
  exact IDs, preserve unseen entities, replace superseded spatial knowledge, and
  recompute grounding. RGB perception does not invent metric world coordinates.
- `perceived_entity_linking.py` performs **Perceived Entity Linking (PEL)** on
  planning copies. The explicit demo IDs `person_1`, `pedestrian_1`, and `mannequin` resolve
  to one `user`, with relation endpoints rewritten and an `aliases` list retained.
  Other person/pedestrian IDs are not assumed to be the user. The existing
  mannequin-type fallback is preserved. G1 and stored episodic IDs remain raw.

PEL resolves normalized object types and the task Theme through RoboKGNet, with
spaces-to-underscores fallback. Only a unique concept match is linked; ambiguous
senses remain unresolved. Remembered entities are linked on demand by stored type.
One Theme instance is selected directly; multiple matches require finite robot
and candidate XYZ positions. Nearest Euclidean distance wins, with ID ordering
for ties. Missing positions never cause an arbitrary choice. Source must belong
to the selected instance through a unique room `in` relation.

The semantic `frame` keeps Theme=`fork`; concrete `bindings` may have
Theme=`FORK_1`. G2 contains both, plus `entity_links`, `theme_concept`, `type`, and
`issues`. The happy path returns `bring` only when required concrete entities
and Source are known. Otherwise it returns `incomplete`. LookFor execution and
Unified Planning integration into SPA are upcoming. Existing standalone planning
experiments under `procedural_memory/planning/` remain separate.

## Runtime artifacts

```text
episodic_memory/
    scene_graph.json       # accumulated episodic memory; initially the migrated demo graph
    scene_graph_2.json     # preserved older demo snapshot
    g1_sense.json          # raw instruction + current observation
    g2_plan.json           # planning graph after PEL and task grounding
    g3_action.json         # currently null; future post-action snapshot
    <world>.scene_graph.json  # optional scene-graph generator outputs
```

`graph_snapshots.py` centralizes repository-relative paths and atomic JSON
replacement. Each SPA cycle publishes G1 immediately after sensing, then the
accumulated episodic graph and G2 after planning, and G3 after Act. Snapshot files
are visualization/debug outputs, not additional sources of truth. Only the latest
cycle is retained. `scene_graph.json` is the persisted memory for the current run;
an `empty` run replaces it with the newly accumulated memory. Save a copy first
if you want to preserve a particular starting state.

Static RoboKGNet JSON stays in its submodule. The simulator's own `scene_graph.json`
stays there as static room/alignment geometry for navigation; it is not the
mutable CMOC episodic graph. No additional semantic-memory files are generated.

## Clone, build, and environment

```bash
git clone --recurse-submodules https://github.com/Dorteel/cmoc.git
# Existing clone, from CMOC:
git submodule update --init --recursive
```

Use the existing ROS 2 Jazzy/Webots installation with Nav2, RViz, and
`webots_ros2`. Upstream expects Webots at `/usr/local/webots/webots`.
From the ROS workspace root, build the existing packages and source them:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --base-paths src/cmoc/ros2 src/cmoc/external/webots_ros2_simulation/simulation_actions src/cmoc/external/webots_ros2_simulation/navigate_to_position --packages-select cmoc_interfaces cmoc_perception simulation_actions navigate_to_position
source install/setup.bash
cd src/cmoc
# Activate your ROS-compatible Python environment if using one:
source .venv/bin/activate
```

The scoped build avoids duplicate package discovery from an original sibling
simulator checkout. Python dependencies are listed in `requirements.txt`; the
viewer uses rdflib. ROS dependencies must remain importable in the environment.
Do not build inside `src/cmoc`; build/install/log outputs belong in the workspace.

Export the Nebula key without placing it in source or command-line arguments:

```bash
read -rsp 'Nebula API key: ' NEBULA_API_KEY; echo
export NEBULA_API_KEY
```

The demo automatically runs:

```bash
ros2 run cmoc_perception observe_with_vlm_server --ros-args -p backend:=nebula
```

No separate perception server is needed for the demo. The configured endpoint is
`https://nebula.cs.vu.nl/api/chat/completions`; model is
`SURF.Qwen3.5 122B A10B NVFP4`. The key is inherited by the child process; `.env`
is not automatically loaded. Missing credentials fail before SPA. Standalone
Ollama support remains available via `backend:=ollama`.

Setup waits for `/wheel/odom`, then ACTIVE lifecycle states from `bt_navigator`,
`planner_server`, and `controller_server` (120-second deadline), then perception
action availability (60 seconds). It prints `CMOC READY` before the first prompt.
`demo.py` initializes ROS once and shuts it down after owned-process cleanup.
Ctrl+C stops perception and simulation with process-group SIGINT/TERM/KILL handling.

## Useful commands

```bash
python demo.py
python demo.py --scenario empty
python demo.py --scenario human-moves
python demo.py --test-navigation
```

Enter at `Instruction [Bring me a fork]:` uses the default. Instruction persists
across cycles; every Sense call requests a new VLM observation. The preview runs
one cycle and leaves the simulator open until Ctrl+C. `existing` loads canonical
episodic memory; `empty` starts with no remembered objects; `human-moves` currently
initializes like `existing` without scripted movement. All use the same planning
logic. `spa_loop(episodic, robokg, semantic_memory, navigator, observation_count=2)`
can exercise subsequent cycles programmatically.

`--test-navigation` remains a separate KITCHEN → LIVING_ROOM_1 movement test,
followed by the legacy memory/search demo. Its Qwen semantic fallback may require
local Ollama with `qwen3:1.7b`. Normal SPA does not invoke that fallback.
`--no-simulator` uses an already running camera simulation and still starts its
own perception server; combine it with `--test-navigation` only against a prepared
aligned Nav2 setup.

### View live G1/G2

```bash
python utils/view_kg.py g1
python utils/view_kg.py g2
python utils/view_kg.py g2 --watch
python utils/view_kg.py g3
python utils/view_kg.py path/to/graph.ttl
```

The existing local browser viewer supports both RDF (Turtle/XML/JSON-LD) and
SPA JSON, retaining its pan/zoom/drag interface and one-second live reload.
`--format` still overrides RDF format detection. Shortcuts resolve independently
of the working directory. Missing snapshots produce a friendly message;
`--watch` opens the viewer and waits for their creation. G3 reports unavailable
while its value is null.

Snapshots are successive knowledge states of one SPA Episode. G1 keeps the raw
`scene_graph` unchanged and adds a JSON `context_graph` (nodes/relations):
`episode_1 → hasInstruction → instruction_1` (with text) and
`episode_1 → hasObservation → observation_1 → observedBy → robot`.
The Observation connects to each perceived entity through `observes`.
Later cycles retain the Episode ID and number their Observations.

G2 preserves that Episode/Observation and adds PEL concept links, canonical `user`
with alias metadata, and `hasFrame → BringingFrame_1`. Its four
`hasFrameElement` nodes (`Agent_FE`, `Theme_FE`, `Source_FE`, `Destination_FE`)
carry `semanticValue` and, when resolved, `bindsTo` edges to concrete entities.
Unresolved roles remain visible without `bindsTo`. The `frame` and `bindings`
dictionaries remain separate. G3 is reserved for the same Episode plus resulting
actions/observations; it remains null until Act is implemented. This RDF translation exists only
inside the viewer; saved knowledge remains JSON. Last valid graphs stay visible
while invalid/missing files are retried. The UI uses text content for labels.

### Simulator details and manual checks

The reused `launch/navigation.launch.py` accepts `map`, `params_file`,
`map_to_odom`, and `doors_config`. CMOC selects `maps/apartment_room_aligned/map.yaml`,
`nav2_params_jazzy.yaml`, and the transform from `alignment.yaml`, ignoring obsolete
absolute paths in its metadata. Ground-truth world odometry replaces AMCL.
The existing door-preparation mode opens all seven configured doors and confirms
settling before Nav2 starts. Generic `/open` behavior is unchanged.

Room goals use simulator floor bounds and the alignment transform. Sampling is
seedable, rejects unknown/occupied map cells, and checks conservative clearance.
Nav2 still determines reachability. No runtime geometry is fabricated.

Run only one apartment: the world's `oracle` controller owns `127.0.0.1:8765`.
Startup checks for a conflicting listener and reports its owner when discoverable;
it never kills unrelated processes. The supervisor belongs to the simulator's
owned launch process group.

Manual verification: run the demo, confirm robot/Nav2/perception readiness, enter
an instruction, then open G1/G2 viewers. Check mannequin→user and frame/binding
edges in G2 while G1 stays raw. For navigation, run `--test-navigation`, confirm
all doors settle, `/navigate_to_pose` exists, and both room goals succeed.
Ctrl+C and rerun to check cleanup. `/open` and `/pick` should remain available
while simulation runs.

### Direct Nebula diagnostic

```bash
python3 test_nebula.py --image-max-dimension 1024 --jpeg-quality 85
```

This reuses `cmoc_perception/vlm_http.py` without ROS, testing text, a local JPEG,
and image plus observation schema. It prints key presence only, timings, and
sanitized errors. Use Python with OpenCV installed (the ROS environment already has it).
Both the live camera and diagnostic use the same JPEG encoder: maximum dimension
1024 px, aspect ratio preserved, quality 85, no upscaling. The server exposes
`nebula_image_max_dimension` and `nebula_jpeg_quality` ROS parameters; the script
has matching CLI options. Logs show source/output dimensions and encoding,
JPEG/base64 bytes, serialized payload bytes, model, and per-attempt elapsed time;
image contents and credentials are not logged.

Timeout remains 120 seconds per HTTP attempt. Only a confirmed response-read
timeout gets one retry after 1 second; HTTP errors and connect/TLS timeouts do not.
The observation client allows 270 seconds for both attempts and stops its local
executor before destroying the action client, so callbacks cannot outlive it. Error diagnostics retain exception
causes and identify connection/read phases when possible; HTTP bodies are bounded
and keys redacted. Action availability alone does not prove API reachability.

## Focused tests

No Webots or Nebula is launched by these tests. Artifact writes are isolated in
temporary directories.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/test_graph_snapshots.py tests/test_pel.py tests/test_spa.py tests/test_demo_setup.py tests/test_navigation.py tests/test_perception_launcher.py tests/test_simulator_launcher.py -q
```

Additional transport tests: `tests/test_vlm_http.py`. Existing ROS perception tests
in `ros2/cmoc_perception/test/` use a local HTTP fixture after build/source.
