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

result = perception.perceive(BASE_DIR / "test.png")

print(result)