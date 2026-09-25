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

`KnowledgeInterface(scene_graph_path)` loads the quantitative scene graph once
and grounds `in` and `on` relations once, combining them with explicit relations
in memory without changing the JSON file. `query_theme_location(theme)` follows
only `in` relations and returns the first matching `Location` ID (or `None`),
ready to use as the frame's `Source`. Theme matching and `query_locations()`
remain available through the same interface.

Run `python knowledge_interface.py scene_graph.json` to inspect the known rooms
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
