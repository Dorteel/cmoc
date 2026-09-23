#!/usr/bin/env python3
"""
Minimal test script to exercise the existing PerceptionModule against test.png.

Requirements satisfied:
- reuses `PerceptionModule` and `PromptLibrary` from `core.py`
- loads `config.yaml` and `.env`
- calls task `perception.create_scene_graph`
- prints raw model response and attempts to parse/pretty-print JSON

Run from repository root:
  python tools/test_perception_api.py
"""
from pathlib import Path
import os
import sys
import json

from dotenv import load_dotenv
import yaml

from core import PerceptionModule, PromptLibrary


ROOT = Path(__file__).resolve().parents[1]


def main():
    load_dotenv(ROOT / ".env")

    # Load config and prompts using existing utilities
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    prompts = PromptLibrary(ROOT / "prompts")

    token = os.getenv("NEBULA_API_KEY")
    if not token:
        print("NEBULA_API_KEY not found in environment (.env).", file=sys.stderr)
        sys.exit(1)

    module = PerceptionModule(
        model=config["perception_module"]["model_name"],
        prompt_library=prompts,
        token=token,
    )

    image_path = ROOT / "test.png"
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Call neutral scene-graph prompt using shared schema
    print("Calling neutral scene-graph prompt (perception.create_scene_graph_neutral)")
    try:
        raw_neutral = module.perceive(
            image_path=image_path,
            prompt_name="perception.create_scene_graph_neutral",
            schema_name="perception.create_scene_graph",
        )
    except Exception as exc:
        print("Error calling perception API (neutral):", type(exc).__name__, str(exc), file=sys.stderr)
        sys.exit(2)

    print("\n--- RAW MODEL RESPONSE (neutral) ---\n")
    print(raw_neutral)
    try:
        parsed = json.loads(raw_neutral)
        print("\n--- PARSED JSON (neutral, pretty) ---\n")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"\nNote: neutral response is not valid JSON: {type(exc).__name__}: {exc}", file=sys.stderr)

    # Call instruction-primed prompt using the same schema
    instruction = "Focus on objects the person is holding and their relations to nearby table surfaces."
    print("\nCalling instruction-primed scene-graph prompt (perception.create_scene_graph_instruction)")
    try:
        raw_inst = module.perceive(
            image_path=image_path,
            prompt_name="perception.create_scene_graph_instruction",
            schema_name="perception.create_scene_graph",
            instruction=instruction,
        )
    except Exception as exc:
        print("Error calling perception API (instruction):", type(exc).__name__, str(exc), file=sys.stderr)
        sys.exit(2)

    print("\n--- RAW MODEL RESPONSE (instruction) ---\n")
    print(raw_inst)
    try:
        parsed = json.loads(raw_inst)
        print("\n--- PARSED JSON (instruction, pretty) ---\n")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"\nNote: instruction response is not valid JSON: {type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
