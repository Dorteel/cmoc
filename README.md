# Common Model of Cognition (CMOC)

CMOC is a robotics prototype organized around Sense → Plan → Act. It reuses
RoboKGNet for static semantic knowledge, Webots/TIAGo for simulation, and the
existing CMOC perception action for VLM observations. The current SPA demo makes
a grounded Bringing plan; execution is opt-in with `--execute`.

## Current architecture

```text
Sense
  → current TIAGo camera observation through /observe_with_vlm (Nebula → local Ollama fallback)
  → G1: Episode + Instruction + raw Observation (observedBy robot)
Plan
  → merge observation into episodic memory
  → extract semantic task frame
  → Perceived Entity Linking (PEL) and concrete task grounding
  → G2: same Episode + PEL + task frame + Frame Element bindings
  → Unified Planning: existing Bringing domain → ordered symbolic plan
Act
  → dry-run printout, or sequential ROS2 dispatch with --execute
  → G3: same Episode, ordered plan, attempted actions and result
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
spaces-to-underscores fallback. The demo normalizes `ForkConnector` to `fork`
and selects the existing utensil entry `fork.n.01`. Otherwise only a unique concept match is linked; ambiguous
senses remain unresolved. Remembered entities are linked on demand by stored type.
Theme candidates must also pass the simulator affordance check: their local PROTO
exposes `connectorModel` wired to a passive Connector. A type suffix alone is not
evidence of graspability. Connector types normalize lexically (`BookConnector` →
`book`, `ForkConnector` → `fork`). An exact normalized type may identify concrete
candidates when WordNet is ambiguous, without fabricating a semantic concept link.
No graspable match leaves Theme unbound and prevents planning/picking.
One graspable Theme instance is selected directly; multiple matches require finite robot
and candidate XYZ positions. Nearest Euclidean distance wins, with ID ordering
for ties. Missing positions never cause an arbitrary choice. Source must belong
to the selected instance through a unique room `in` relation.

The semantic `frame` keeps Theme=`fork`; concrete `bindings` may have
Theme=`FORK_1`. G2 contains both, plus `entity_links`, `theme_concept`, `type`, and
`issues`. The happy path returns `bring` only when required concrete entities
and Source are known. UP additionally requires known robot and destination rooms;
missing facts return an incomplete plan. `bringing_plan.py` reuses `Planner` and
`bringing/chatgpt/domain.pddl` (move/pick/place, no LookFor), with a 30-second solver
timeout. The demo assumes an initially empty gripper. No search/replanning is added.

## Runtime artifacts

```text
episodic_memory/
    scene_graph.json       # accumulated episodic memory; initially the migrated demo graph
    scene_graph_2.json     # preserved older demo snapshot
    g1_sense.json          # raw instruction + current observation
    g2_plan.json           # planning graph after PEL and task grounding
    g3_action.json         # task-only ordered plan, attempts and outcome
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
is not automatically loaded. Nebula gets at most two attempts; connection/timeout/HTTP
errors retry once when transient (timeout, connection, HTTP 5xx); exhausted attempts,
authentication errors or missing credentials trigger a logged
fallback to existing local Ollama `qwen3-vl:2b` at `http://localhost:11434`.
Ollama likewise gets at most two attempts with the same transient-error policy.
Ensure Ollama is running and the model is installed. Both failing aborts perception;
invalid schemas or schema-invalid model output remain explicit errors.
Standalone Ollama remains available via `backend:=ollama`.

The action result now has a separate `perception_provenance` JSON field. Rebuild
**both `cmoc_interfaces` and `cmoc_perception`**, then source the workspace before
running the demo; clients and server must use the updated interface.

Setup waits for `/wheel/odom`, then ACTIVE lifecycle states from `bt_navigator`,
`planner_server`, and `controller_server` (120-second deadline), then perception
action availability (60 seconds). It prints `CMOC READY` before the first prompt.
`demo.py` initializes ROS once and shuts it down after owned-process cleanup.
Ctrl+C stops perception and simulation with process-group SIGINT/TERM/KILL handling.

The single dispatcher in `plan_execution.py` maps UP `move` to
`RoomNavigator.go_to_room`, `pick` to `/pick` (`robot`, `object`), and `place` to
`/place_next_to` (`robot`, `object`, `target`). This simulator operation releases
the object at a clear lateral pose beside the target. Task identities map to
verified Webots names: `robot → TIAGo`, `user → pedestrian` (the generated memory
ID is `pedestrian_1`). Room targets keep existing room navigation, except when immediately followed by
`pick`: execution approaches that exact grounded Theme if its coordinates are
known. The symbolic room target and Source binding stay unchanged; unavailable
Theme coordinates retain room navigation. Positioned
entity targets use a pose 0.5 m before the target on the current robot→target
line, facing it. Within 0.5 m the robot keeps its position and turns toward the
target. Episodic world coordinates use the existing map alignment; current robot
coordinates come from `map → base_link` TF at execution time. Missing target
coordinates use the known-room fallback; unavailable robot TF stops the step
instead of using a stale pre-plan pose. The Bringing planner still requires known
rooms for its symbolic problem. Manipulation remains the
simulator's Supervisor fallback, not physical grasp planning. Action availability,
acceptance and completion are checked; timeout/failure stops later steps, with no
retry. Cancellation is best effort because the existing fallback servers may reject it.

## Useful commands

```bash
python demo.py                         # dry-run: no plan goals sent
python demo.py --execute --step        # confirm each step
python demo.py --execute               # sequential full execution
python demo.py --execute --teleport    # same goals, Webots teleport instead of Nav2 movement
python demo.py --scenario empty
python demo.py --scenario human-moves
python demo.py --test-navigation
```

Enter at `Instruction [Bring me a fork]:` uses the default. Instruction persists
across cycles; every Sense call requests a new VLM observation. Dry-run is the default. Live execution stops on the first failure or success,
then leaves the simulator open until Ctrl+C. `existing` loads canonical
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
`--watch` opens the viewer and waits for their creation. Use `python utils/view_kg.py g3 --watch`
to inspect the ordered plan and attempted-action statuses.

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
dictionaries remain separate. `frame_element_links` adds a third, semantic-grounding
layer: roles first reuse PEL links for canonical entities or matching episodic
types, then use RoboKGNet lexical resolution. Destination `user` thus inherits
the canonical pedestrian concept. Missing/ambiguous links stay null with `frame_element_issues` also listed
in `issues`; they do not change existing concrete plan selection.
`linkedTo` connects each resolved FE to its concept independently of `bindsTo`.

G1/G2 keep `perception_provenance` (`backend`, `model`, `fallback_used`); the Observation
shows `generatedByModel`, `perceptionBackend`, and `fallbackUsed`. G2 is bounded
to current perception, Episode/frame context, canonical user, referenced concepts,
and selected concrete bindings. Full memory stays in `episodic_memory/scene_graph.json`.
The G2 viewer renders only the compact task context (about 10–20 nodes), never
the full `scene_graph`. Instruction, model provenance and types are node details;
aliases remain in JSON without visible alias nodes.
G3 contains only the same Episode, plan, ordered actions, attempts/status, and
direct argument entities; it does not duplicate G1/G2 or world memory. This RDF translation exists only
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

Timeout remains configurable via `request_timeout_sec` (default 120 seconds per
HTTP attempt). The live server allows two attempts per backend, retrying only timeout, connection
and HTTP 5xx failures before falling back or failing. The standalone Nebula diagnostic retains its one read-timeout
retry. The observation client allows 510 seconds for the four possible primary/fallback attempts and stops its local
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

For transport/fallback and action-contract tests, use the ROS Python environment
with the rebuilt workspace sourced (local HTTP fixtures only):

```bash
ROS_DOMAIN_ID=184 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_vlm_http.py ros2/cmoc_perception/test/test_observe_with_vlm.py -q
```

`--teleport` requires `--execute`. It reuses Nav2 goal resolution and path validation
(Nav2 must remain available), then calls the existing Webots supervisor `move`
command. Pick/place are unchanged; teleport failure stops execution.

### Search recovery within SPA

Search is opt-in with `--search` (default: off). `python demo.py` uses full
scene-graph grounding and prints `Demo mode: LEGACY SCENE GRAPH`; it does not
create Search frames or perform the Search pre-pick visibility check.
`python demo.py --search` uses observation-backed grounding and prints
`Demo mode: SEARCH / PARTIAL OBSERVABILITY`. Both modes share the SPA loop,
Bring planner, Nav2/teleport selection, and pick/place execution.

```bash
python demo.py                                # legacy dry run
python demo.py --execute --teleport           # legacy teleport execution
python demo.py --search                       # Search dry run
python demo.py --search --execute --teleport  # Search teleport execution
```

The following recovery behavior applies only with `--search`.

`demo.spa_loop` preserves the Bring intention while Search temporarily plans one
candidate. An unresolved Theme requests Search. A previously observed Theme can
still use Bring; a fresh VLM visibility check before pick interrupts that plan if
the concrete Theme is missing. Other execution failures retain their existing
stop behavior.

Search uses `schemas/task_frames/search.json`: required `Agent`, `Theme`,
`Location`, and boolean `Success`, with optional `verb: search`. `Success` is a
result field, not a semantic role. `search_result.json` requires `theme`,
`location`, and boolean `success`, with optional `observed_ids`.

`procedural_memory/planning/search/domain.pddl` is separate from Bring. Its sole
action is `look-at(agent, location)`, with a `checked(location)` goal. The problem
wrapper uses opaque PDDL tokens to preserve punctuation in symbolic IDs. PDDL
completion does not mean the Theme was found. Act turns at the current position
through the existing navigation backend; only the next VLM observation establishes
Search success. This minimal behavior assumes the candidate is inspectable from
the current area; it does not navigate between candidate rooms. Missing execution
coordinates are reported as an execution error.

Candidate selection uses successful interaction evidence and RoboKGNet priors,
with specific locations before rooms within each tier. Unchecked VLM suggestions
and then LLM suggestions provide fallbacks. Suggestions must ground to observed
symbolic IDs; unresolved text is reported and never placed in PDDL. A failed
candidate is excluded only for the current task, without excluding its room or
writing a negative KG fact. Following failure, the VLM is asked for alternatives
based on the current image.

`KnowledgeInterface.observed_snapshot()` is the epistemic boundary: simulator
seed objects and relations are absent unless learned through `merge_observation`.
Saved runtime graphs include `observed_evidence` so those facts survive reload.
Search planning never receives the seed graph. Act may resolve the already chosen
candidate's coordinates from simulator geometry. Successful post-look evidence
uses existing `in`/`on` relations and is saved through the existing episodic
snapshot mechanism; no location histogram or probability is updated. Search G2
contains only the active frame, selected candidate, and its small plan.

Example recovery trace:

```text
Bring(fork)
  -> Search(worktop(1), false) -> look-at -> new VLM observation: absent
  -> exclude worktop(1) for this task; request visible alternatives
  -> Search(KITCHEN, false) -> look-at -> new VLM observation: found
  -> Search.Success = true; persist location evidence
  -> replan original Bring(fork) from updated state -> execute
```

The old Bring plan is discarded; it is never resumed after Search.
Regression coverage is in `tests/test_search_recovery.py`, including changed hidden
seed locations, observed-location updates, task-local exclusions, fresh sensing,
PDDL/schema checks, compact G2, and discarding a stale Bring plan.
