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

## Notes

- Keep a `.env` file with `NEBULA_API_KEY` for API access. Do not commit secrets.
- The repository intentionally keeps logic minimal: prompts are plain Markdown files loaded by `PromptLibrary`, and schemas are plain JSON files under `schemas/`.
- FrameNet/VerbNet integration is not implemented yet; the scene-graph schema and prompt separation are intended to make adding such mappings straightforward later.
