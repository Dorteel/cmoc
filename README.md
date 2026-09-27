# Common Model of Cognition

## Introduction

This repository is a small prototype that demonstrates a perception-first pipeline driven by prompt templates and JSON Schemas for controlled VLM output. It focuses on producing scene-graph outputs from images using a remote Nebula chat/completions API.

Key ideas:
- Keep prompts and output schemas separate so different prompts (neutral vs instruction-primed) can produce the same structured output.
- Use JSON Schema to enforce output structure for downstream consistency.

## Repository structure (important files)

- `core.py`: `PerceptionModule` and `PromptLibrary` — central API client and prompt loader.
- `main.py`: example runner that demonstrates neutral and instruction-primed scene-graph calls.
- `config.yaml`: configuration including `perception_module.model_name`.
- `prompts/`: prompt templates loaded by `PromptLibrary` (e.g. `prompts/perception/create_scene_graph_neutral.md`).
- `schemas/`: JSON Schemas used for response formats (e.g. `schemas/perception/create_scene_graph.json`).
- `tools/`: small helper scripts such as `test_perception_api.py` and `test_nebula_stages.py`.
- `test.png`: example image used by the demos and tests.

## PerceptionModule API

`PerceptionModule` (defined in `core.py`) provides a simple `perceive(...)` method with the following signature:

```
perceive(image_path=None, prompt_name="perception.create_scene_graph", schema_name=None, instruction=None)
```

- `image_path`: optional Path to an image to include.
- `prompt_name`: name of a prompt loaded from `prompts/` via `PromptLibrary` (e.g. `perception.create_scene_graph_neutral`).
- `schema_name`: optional dot-name of a JSON Schema under `schemas/` (e.g. `perception.create_scene_graph`). When provided the schema is attached as a `response_format` of type `json_schema`.
- `instruction`: optional instruction string injected into the user content to guide attention for instruction-primed prompts.

The method encodes images as base64 data URIs, loads prompts via `PromptLibrary`, and loads schemas from the `schemas/` folder when requested. It returns the raw model response content.

## Scene-graph schema and predicates

The repository includes a single scene-graph schema at `schemas/perception/create_scene_graph.json`. The schema defines two top-level arrays:

- `objects`: items with `id`, `label`, and `type` fields (all strings).
- `relations`: items with `subject`, `predicate`, and `object` fields.

Allowed `relation.predicate` values (exact vocabulary):

`in`, `on`, `under`, `next_to`, `touching`, `attached_to`, `held_by`, `above`, `in_front_of`, `behind`.

Prompts `prompts/perception/create_scene_graph_neutral.md` and `prompts/perception/create_scene_graph_instruction.md` provide neutral and instruction-primed templates respectively; both are intended to produce the same schema-shaped output.

## Example commands

Run the simple demo in `main.py` which performs both neutral and instruction-primed calls:

```bash
python main.py
```

Run the minimal test script that prints raw and parsed JSON (uses the refactored `perceive`):

```bash
python tools/test_perception_api.py
```

Run the three-stage Nebula diagnostic:

```bash
python tools/test_nebula_stages.py
```

## Instruction parser

`instruction_parser.py` provides a transparent first step before frame lookup:
NLTK sentence and word tokenization, POS tagging, WordNet lemmatization, and
simple verb and object candidates. It uses no imperative-mood detection or
dependency parsing.

Install NLTK and download its English tokenizer, tagger, and lemma resources once:

```bash
python -m pip install nltk
python -m nltk.downloader punkt_tab averaged_perceptron_tagger_eng wordnet
```

Run the three example instructions:

```bash
python instruction_parser.py
```

`parse_instruction("Bring me the red cup.")` returns a plain dictionary with a
`sentences` list. Each sentence record contains `sentence`, original `tokens`,
aligned `pos_tags` and `lemmas`, `main_verb` (`"bring"`), and `object_phrase`
(`"red cup"`). Empty input returns an empty list of sentence records.

Tokens are lowercased for tagging to reduce sentence-initial capitalization
errors. The first `VB*` token supplies the verb lemma; modal tags such as `MD`
are not verb candidates. After that verb, the parser skips personal pronouns
such as `me`, collects the first determiner/adjective/noun run, and omits
determiners from the returned phrase. Other token types end the phrase.
Without a verb, the candidates are `None` and `""`. These are intentionally
small heuristics; auxiliary verbs and complex instructions may need later work.

Run the focused tests with `python -m pytest test_instruction_parser.py`
(requires `pytest` and the NLTK resources above).

## Semantic bridge

`tools/semantic_bridge.py` exposes explicit alignments among NLTK FrameNet,
VerbNet, and WordNet. It reads the official
[SemLink2 class/member-to-frame mappings](https://github.com/cu-clear/semlink)
and follows the WordNet sense keys declared on VerbNet members.

Install the lexical resources once, then run the `Bringing` example:

```bash
python -m pip install nltk
python -m nltk.downloader framenet_v17 verbnet wordnet
python tools/semantic_bridge.py
```

The three NLTK corpora are local; each `show_link()` call fetches the small
SemLink2 `instances/vn-fn2.json` file from GitHub and requires network access.
There are no downloads at import time. Missing corpora raise NLTK's setup
instructions, and a failed SemLink2 fetch raises `RuntimeError`.

From Python, start with any of the three resources:

```python
from tools.semantic_bridge import show_link

result = show_link("framenet", "Bringing")
show_link("verbnet", "bring-11.3-1")
show_link("wordnet", "bring.v.01")
```

FrameNet accepts an exact frame name or numeric ID; VerbNet accepts a full class
ID or numeric short form such as `11.3-1`; WordNet accepts a synset name or sense
key such as `bring%2:38:00::`. Bare words are not disambiguated.

The function prints a concise summary and returns a plain dictionary:
`framenet`, `verbnet`, and `wordnet` contain sorted identifier lists; `links`
records the member-level paths; `missing` explains absent or unresolved mappings.
It keeps the requested source fixed and does not recursively expand neighbors.
Only direct class members are used, without inferred subclass inheritance.

SemLink2 targets VerbNet 3.3, while NLTK's bundled VerbNet is 2.1. Exact class
numbers and member names are matched; unavailable entries are reported without
fallbacks. WordNet links marked `?` are excluded and reported. Abbreviated verb
sense keys receive their standard empty `::` fields, but unresolved keys are
never replaced by another sense. A member may declare several WordNet senses:
these are linked candidates, not a claim that every sense is equivalent to the
FrameNet frame. No synonym search or semantic-similarity guessing is performed.

## VerbNet task-frame schemas

Generate an exploratory JSON Schema from a VerbNet class's direct thematic
roles (requires NLTK and its `verbnet` corpus):

```bash
python -m tools.verbnet_to_schema bring-11.3
```

This saves `schemas/task_frames/bring-11.3.json`. Every lowercase role key is
required and accepts a string entity ID or `null`; descriptions preserve
selectional restrictions and their logical groups without enforcing them.
The schema includes `verbnet_class` metadata. Calling
`verbnet_to_schema(class_id)` also returns the schema dictionary.

## Search strategy for unknown sources

Source unknown
→ query known Locations
→ semantic ranking
→ search highest-ranked location first

## Planning

`procedural_memory/planning/planner.py` reads PDDL and solves it through Unified
Planning, providing an abstraction over planner executables such as Metric-FF.
`Planner(planner_name=None).solve(domain_path, problem_path)` returns a plan or
raises `RuntimeError` for an unsuccessful solver result. By default, Unified
Planning selects a compatible installed engine; pass a name to choose one.
Fast Downward supports the Bringing example's negative precondition.

```bash
python3 -m pip install 'unified-planning[fast-downward]' pytest
python3 procedural_memory/planning/planner.py
python3 -m pytest tests/test_planner.py
```

This utility is standalone and has no ROS integration.

## Notes

- Keep a `.env` file with `NEBULA_API_KEY` for API access. Do not commit secrets.
- The repository intentionally keeps logic minimal: prompts are plain Markdown files loaded by `PromptLibrary`, and schemas are plain JSON files under `schemas/`.
- The semantic bridge is a standalone lexical utility; it is not yet connected to the perception pipeline.

## Live RDF knowledge graph viewer

Install `rdflib` (`python -m pip install rdflib`), then run from the repository root:

```bash
python -m tools.view_kg path/to/graph.ttl
```

The self-contained viewer starts a localhost web server on an available port and
opens your default browser. No frontend dependencies or internet connection are
needed for the visualization. Turtle (`.ttl`), RDF/XML (`.rdf`, `.xml`), and JSON-LD
(`.jsonld`) are supported; use `--format turtle`, `--format xml`, or
`--format json-ld` to override detection. JSON-LD documents with remote contexts
may require network access through RDFLib.

The file is checked roughly once per second, and the browser refreshes the graph
automatically. Failed reads or invalid RDF retain the last valid graph until a
successful reload. An initially missing or invalid file is retried as well.
Press Ctrl+C in the terminal to stop the server.

Drag nodes to arrange them, drag the background to pan, and scroll to zoom. Hover
or select a node for its URI, types, or full literal value. Predicate and resource
labels use local names. Types have stable URI-derived colors; for multiple types,
the first type in URI sort order determines the node color. The legend lists the
currently visible types. Blank nodes use dashed diamonds and the `_:identifier`
convention (their parser-assigned identifiers may change on reload); literals use
neutral rounded rectangles. This lightweight layout is intended for small debug
graphs rather than very large datasets.

## Scene graph from a Webots world

Extract a conservative reference graph from explicit names and DEF scene groups
in a saved world (requires `jsonschema`):

```bash
python -m tools.scene_graph_generator_from_simulation resources/simulation_worlds/setting_the_table_complete_apartment_tiago_ros2.wbt
```

The utility saves `<world>.scene_graph.json` beside the world and returns the
graph dictionary. Objects are validated against `schemas/objects.json`; relations
use the existing predicate schema and reference object IDs. Names take precedence
over DEF identifiers; duplicate names receive a deterministic source-path suffix.
Types come from explicit `model` values or Webots node types, except for named
direct spatial children of the object named `floor`: these become `Location`
objects with their stable IDs preserved. A direct Shape containing a Plane with
explicit width and height supplies numeric `qualities.area` in square metres
(`width * height`); missing geometry yields no area. Other Pose nodes are not
reclassified. Unknown qualities are omitted; objects with empty `qualities`
remain in the graph. Locations are numeric world-frame `[x, y, z]` coordinates
in metres, composed through the complete parent hierarchy using axis-angle
rotations and translations (and explicit Transform scaling). Unknown positions,
including offsets hidden inside external PROTO parents, are omitted. Orientations are numeric world-frame axis-angle arrays. Sizes are world-aligned
bounds computed from transformed Box/Plane corners (or exact Sphere bounds),
including explicit ancestor scaling; horizontal floor locations use two numbers.
Incomplete dimensions and hidden PROTO geometry are omitted. RGB colours retain
their existing labelled strings.
Known unnamed physical node types receive deterministic `type_N` IDs (for
example `pedestrian_1`); anonymous rendering and grouping helpers stay excluded.
Only explicit joint attachments produce relations. PROTO defaults,
texture colours, material guesses, and spatial/contact guesses are
omitted; this reads the saved file rather than a running simulation. External
PROTOs are not loaded, spatial `USE` instances warn and are skipped, and inline
PROTO definitions are rejected.

## Episodic spatial knowledge

`scene_graph_interface.KnowledgeInterface(scene_graph_path)` loads the quantitative scene graph once
and grounds `in` and `on` relations once, combining them with explicit relations
in memory without changing the JSON file. `query_theme_location(theme)` follows
only `in` relations and returns the first matching `Location` ID (or `None`),
ready to use as the frame's `Source`. Theme matching and `query_locations()`
remain available through the same interface.

Run `python scene_graph_interface.py scene_graph.json` to inspect the known rooms
for plates, wine glasses, and an unknown object.

## VerbNet semantic patterns

Run `python -m tools.verbnet_semantic_patterns bring-11.3` to inspect normalized
predicates across a class's direct frames (requires NLTK's `verbnet` corpus).
The summary reports invariants, strict-majority predicates, positive role-presence
frequency associations, and single-frame predicates. These are observations,
not logical implications. Event variables are normalized within each predicate;
raw semantics remain available through `analyze_class()` for cross-predicate
links. Run the embedded sanity test with
`python -m pytest tools/verbnet_semantic_patterns.py`.

## Observe one camera frame with a VLM

Simulation publishes sensor data; CMOC interprets it using VLMs. Only the nested
`ros2/cmoc_perception` (Python node) and `ros2/cmoc_interfaces` (action definition)
are ROS packages; the repository itself remains a standalone Python project.
`colcon` discovers both recursively from `~/ros2_ws`. Neither package depends on
`simulation_actions` or Webots. Set `camera_topic` to use another camera, including
a real robot camera.


`cmoc_perception` provides `/observe_with_vlm` with action type
`cmoc_interfaces/action/ObserveWithVLM`. It continuously caches the latest RGB
image and sends exactly one snapshot plus the goal `prompt` and caller-provided
`json_schema` (a JSON string) to the selected backend. No observation schema is
hard-coded in the server. The result contains `success` and a JSON-serialized
`response`; feedback reports `running_vlm`. Success means inference, JSON parsing,
and validation passed, even when a task-specific field such as `found` is false.
Missing images, unavailable endpoints, invalid schemas, and invalid responses abort
the goal with `success: false` and an explanation. Frames continue updating during
inference. Concurrent goals are rejected while one inference is active; canceling
an in-flight HTTP request is not supported. The cached frame can be old if the camera stops publishing.

The action type is now `cmoc_interfaces/action/ObserveWithVLM`. Rebuild and restart
existing servers and clients using this type; an empty `json_schema` is rejected. The server checks
schemas with `python3-jsonschema` before reading a frame, then strictly parses and
validates the VLM output. It does not strip Markdown fences, repair JSON, coerce
values, or retry invalid output. Schemas without `$schema` use Draft 2020-12.

When upgrading an already-built workspace, remove the old generated simulation
package first (`rm -rf ~/ros2_ws/build/simulation_actions ~/ros2_ws/install/simulation_actions`)
so its obsolete VLM executable and generated interface are not left installed.

Build and launch the camera simulation (terminal 1):

```bash
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select cmoc_interfaces cmoc_perception simulation_actions
source install/setup.bash
cd src/webots_ros2_simulation
ros2 launch ./launch/tiago_apartment_ros2.launch.py
```

Start Ollama (`ollama serve` if it is not already running), install a vision model,
and start the action server (terminal 2):

```bash
ollama pull qwen3-vl:2b
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run cmoc_perception observe_with_vlm_server --ros-args -p backend:=ollama
```

Test from terminal 3:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 action list
ros2 action send_goal /observe_with_vlm cmoc_interfaces/action/ObserveWithVLM \
  "{prompt: 'Do you see a coffee mug?', json_schema: '{\"type\":\"object\",\"properties\":{\"found\":{\"type\":\"boolean\"},\"description\":{\"type\":\"string\"}},\"required\":[\"found\",\"description\"],\"additionalProperties\":false}'}" \
  --feedback
```

To use Nebula, stop the action server in terminal 2, set the key in that shell,
and restart it (the key is never a ROS parameter):

```bash
read -rsp 'Nebula API key: ' NEBULA_API_KEY; echo
export NEBULA_API_KEY
ros2 run cmoc_perception observe_with_vlm_server --ros-args -p backend:=nebula
```

Use the same goal commands for either backend. Ollama receives the caller schema
as `format` with temperature 0. The default Nebula endpoint/model was verified to
support native `response_format: {type: json_schema, json_schema: ...}`; it receives
the caller schema with `strict: true`, alongside the snapshot and prompt over HTTPS.
Both responses are always parsed and validated locally against the caller schema.
There is no prompt-only fallback or silent downgrade; a different Nebula model
that rejects native structured output returns a clean action failure. If the server was started with the required environment key,
you can also switch between goals with
`ros2 param set /observe_with_vlm_server backend nebula` (or `ollama`).

| Parameter | Default |
| --- | --- |
| `backend` | `ollama` |
| `camera_topic` | `/tiago/camera/color/image_raw` |
| `ollama_url` | `http://localhost:11434` |
| `ollama_model` | `qwen3-vl:2b` |
| `nebula_url` | `https://nebula.cs.vu.nl/api/chat/completions` |
| `nebula_model` | `SURF.Qwen3.5 122B A10B NVFP4` |
| `nebula_api_key_env` | `NEBULA_API_KEY` |
| `request_timeout_sec` | `120.0` |

Pass overrides using `--ros-args -p name:=value`. Set `camera_topic` at startup;
changing it requires restarting the node. HTTP calls do not stream or retry.
The HTTP timeout bounds network waits; allow more time for slow local inference.

To reproduce failure checks, stop the server and run one of these alternatives,
then send a goal with the commands above. The last three need a camera frame:

```bash
# No camera frame:
ros2 run cmoc_perception observe_with_vlm_server --ros-args -p camera_topic:=/unused_camera
# Ollama unavailable:
ros2 run cmoc_perception observe_with_vlm_server --ros-args -p ollama_url:=http://127.0.0.1:1
# Nebula missing key:
env -u NEBULA_API_KEY ros2 run cmoc_perception observe_with_vlm_server --ros-args -p backend:=nebula
# Nebula unavailable (with the key exported):
ros2 run cmoc_perception observe_with_vlm_server --ros-args -p backend:=nebula -p nebula_url:=http://127.0.0.1:1
```

Test malformed schema handling (no inference is performed):

```bash
ros2 action send_goal /observe_with_vlm cmoc_interfaces/action/ObserveWithVLM \
  "{prompt: 'Do you see a coffee mug?', json_schema: '{'}" --feedback
```

Expect `success: false` with `Invalid requested JSON schema: ...`. Integration
tests also inject malformed VLM JSON and schema violations for deterministic checks,
and exercise different schemas to verify the server stays generic.
Run them after sourcing the built workspace:

```bash
cd ~/ros2_ws/src/cmoc
ROS_DOMAIN_ID=81 python3 -m unittest discover -s ros2/cmoc_perception/test -v
```

Run the same integration suite through colcon (uses local HTTP fixtures, no Webots
or backend credentials required):

```bash
cd ~/ros2_ws
ROS_DOMAIN_ID=81 colcon test --packages-select cmoc_interfaces cmoc_perception simulation_actions
colcon test-result --verbose
```

Confirm the old server is absent (both commands should produce no output):

```bash
ros2 pkg executables simulation_actions
rg -n 'observe_with_vlm|ObserveWithVLM' ~/ros2_ws/src/webots_ros2_simulation/simulation_actions
```


## RoboKGNet knowledge resource

[knowledge/robokgnet](knowledge/robokgnet/README.md) integrates selected WordNet
hierarchies, structured VerbNet semantics, FrameNet descriptions, explicit
SemLink alignments, weighted commonsense fixtures, and references to the existing
Bringing PDDL. It reuses the semantic bridge, VerbNet frame reader, and Unified
Planning parser. External planning RDF can be merged without changing its IRIs;
no existing planning RDF ontology was found in this checkout.

```bash
.venv/bin/python -m knowledge.robokgnet.demo
.venv/bin/python -m unittest discover -s knowledge/robokgnet/tests -v
```

The offline demo writes `knowledge/robokgnet/robokgnet.ttl`. See the linked README
for corpus setup, provenance, API examples, and resource/alignment limitations.

## RoboKGNet semantic memory

```bash
git clone --recurse-submodules https://github.com/Dorteel/cmoc.git
# For an existing clone:
git submodule update --init --recursive
```

`knowledge_interface.KnowledgeInterface` delegates concept/action queries and
location updates to `external.robokgnet.knowledge_interface.KnowledgeInterface`.
The qualified namespace-package import needs no RoboKGNet changes or path edits.
RoboKGNet alone reads/writes its canonical JSON; optional `concepts_path` and
`actions_path` arguments select other data paths. `save()` writes the configured
concepts file, so use a temporary copy for experiments.

Run the real-data integration tests from the CMOC root:

```bash
python3 -m unittest discover -s tests -p 'test_knowledge_interface.py' -v
```

The pinned data has null `bring-11.3` frame roles; FrameNet descriptions are
available through `get_additional_frame_elements()`. The episodic scene-graph
interface remains in `scene_graph_interface.py`.

### Simulator dependency and room navigation

```bash
git clone --recurse-submodules https://github.com/Dorteel/cmoc.git
# Existing clone:
git submodule update --init --recursive
```

Use ROS 2 Jazzy with Webots (`/usr/local/webots/webots` as expected by the upstream
launch), `webots_ros2`, Nav2, and RViz installed. From the ROS workspace root:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --base-paths src/cmoc/external/webots_ros2_simulation/simulation_actions src/cmoc/external/webots_ros2_simulation/navigate_to_position --packages-select simulation_actions navigate_to_position
source install/setup.bash
cd src/cmoc
python demo.py
# Optional movement test before the existing memory/search output:
python demo.py --test-navigation
```

The scoped build selects only the submodule packages, avoiding duplicates from
the original sibling checkout. Python dependencies include PyYAML and Pillow
(listed in `requirements.txt`); use a Python environment that can import sourced
ROS packages. The existing generative-memory fallback requires local Ollama
with `qwen3:1.7b` when stored knowledge has no answer.

`SimulatorLauncher` reuses the simulator's `launch/navigation.launch.py`.
That launch now accepts optional `map`, `params_file`, `map_to_odom` (x,y,yaw),
and `doors_config` arguments; its original defaults remain unchanged. CMOC uses:

- `external/webots_ros2_simulation/maps/apartment_room_aligned/map.yaml` and `map.pgm`;
- `external/webots_ros2_simulation/nav2_params_jazzy.yaml`;
- `alignment.yaml` in the same map directory for the scene→map transform.

The existing Supervisor publishes world-frame odometry. CMOC supplies the
alignment as the static map→odom transform, disabling the old initial-pose
anchor. The existing map-server lifecycle manager and Nav2 bringup are reused;
AMCL is not launched because this stack uses ground-truth localization.
No obsolete absolute paths from alignment metadata are used.

Door preparation reuses the simulator's existing mapping door mode with
`config/navigation_doors.json`: all seven Door PROTOs (`door`, `door(1)` through
`door(6)`, including the entrance) open once and must settle before Nav2 starts.
The existing helper checks hinge handedness and limits and reports errors;
normal `/open` behavior is unchanged. Its log still mentions SLAM because it is
shared with mapping, but this demo launches localization/navigation, not SLAM.

SETUP waits for `/wheel/odom`, then polls `/bt_navigator/get_state`,
`/planner_server/get_state`, and `/controller_server/get_state` until all three
report ACTIVE before initializing memories or sending movement goals. Lifecycle
polling pauses 0.5 seconds between rounds, with a shared 120-second deadline;
timeout reports the last states. Action-server discovery alone is insufficient. `--test-navigation` then visits KITCHEN
and LIVING_ROOM_1 with seed 42. Normal runs print the existing frame/search
output without sending movement goals. ATTEMPT and LOOK_FOR remain knowledge
queries only. Run one apartment per ROS domain.

`navigation.RoomGoals` takes room IDs/centers from alignment metadata and
world-aligned floor bounds from the simulator's repository-local
`scene_graph.json`, using CMOC's existing geometry helper. It samples inside
those bounds, transforms to map coordinates, and rejects unknown/occupied or
out-of-map cells. PGM checks respect resolution, rotated origin, thresholds,
and negate. A conservative clearance square uses the existing global costmap's
inflation radius (0.55 m); this does not guarantee reachability on Nav2's live
costmap. Sampling is bounded and errors clearly if no valid point is found.
`RoomNavigator.go_to_room(room_id, seed=42)` sends one NavigateToPose goal and
returns success/failure without autonomous retries.

The simulator stays open after output. Ctrl+C cancels an active navigation goal
and stops the owned launch process group, with TERM/KILL fallback.
`demo.py` owns the process-wide ROS context: it initializes once before setup
and shuts down after navigator and simulator cleanup. `RoomNavigator.close()`
only releases its action client/node; it does not shut ROS down.
`python demo.py --no-simulator` runs memory queries only.
`--no-simulator --test-navigation` uses an already prepared aligned Nav2 setup;
it does not open doors or reconfigure an external simulator.

Manual integration test (graphical session; no other apartment running):

1. Build and source using the commands above.
2. Run `python demo.py --test-navigation`.
3. Verify all seven doors open and the door helper reports them settled.
4. In another sourced terminal, run `ros2 action list` and confirm
   `/navigate_to_pose`; confirm both readiness messages in the demo terminal.
5. Watch TIAGo navigate to KITCHEN, then LIVING_ROOM_1. Each goal and its result
   is printed; rejection, abort, or timeout fails the movement test.
6. Confirm frame/search output continues, then press Ctrl+C. Verify Webots and
   this run's ROS launch/controller/Nav2 processes terminate (inspect
   `ps -eo pid,pgid,args` if needed).

Automated tests never launch Webots or Nav2:

```bash
python3 -m unittest discover -s tests -p 'test_navigation.py' -v
python3 -m unittest discover -s tests -p 'test_simulator_launcher.py' -v
python3 -m unittest discover -s tests -p 'test_demo_setup.py' -v
```
