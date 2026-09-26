from listener import NaiveFrameFiller
from knowledge_interface import KnowledgeInterface
from semantic_memory import SemanticMemory
from search_strategy import resolve_search

instruction = "Bring me the coffee mug"

# 1. Extract semantic roles from language.
frame = NaiveFrameFiller(instruction).fill()

# 2. Ask episodic memory whether the Source is already known.
kb = KnowledgeInterface("scene_graph.json")
frame["Source"] = kb.query_theme_location(frame["Theme"])

print("Frame:", frame)

# 3. If Source is unknown, semantic memory ranks possible locations.
semantic_memory = SemanticMemory(model="qwen3:1.7b")
search = resolve_search(frame, kb, semantic_memory)

print("\nTarget:", search["target"])
print("Search order:")

for candidate in search["locations"]:
    if isinstance(candidate, dict):
        print(candidate["location"], candidate["score"])
    else:
        print(candidate)