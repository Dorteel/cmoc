from pathlib import Path
from dotenv import load_dotenv
import os, yaml

from core import PerceptionModule, PromptLibrary


BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

with open(BASE_DIR / "config.yaml") as f:
    config = yaml.safe_load(f)

prompts = PromptLibrary(BASE_DIR / "prompts")

perception = PerceptionModule(
    model=config["perception_module"]["model_name"],
    prompt_library=prompts,
    token=os.getenv("NEBULA_API_KEY"),
)

# Example: neutral scene-graph using the shared schema
print("Running neutral scene-graph (no instruction)...")
result_neutral = perception.perceive(
    image_path=BASE_DIR / "test.png",
    prompt_name="perception.create_scene_graph_neutral",
    schema_name="perception.create_scene_graph",
)
print(result_neutral)

# Example: instruction-primed scene-graph using the same schema
instruction_text = "Focus on objects the person is holding and nearby table surfaces."
print("\nRunning instruction-primed scene-graph...")
result_instruction = perception.perceive(
    image_path=BASE_DIR / "test.png",
    prompt_name="perception.create_scene_graph_instruction",
    schema_name="perception.create_scene_graph",
    instruction=instruction_text,
)
print(result_instruction)