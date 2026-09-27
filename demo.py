from listener import NaiveFrameFiller
from scene_graph_interface import KnowledgeInterface as EpisodicMemory
from knowledge_interface import KnowledgeInterface as RoboKGNet
from semantic_memory import SemanticMemory
from search_strategy import resolve_search

instruction = "Bring me the coffee mug"

frame = NaiveFrameFiller(instruction).fill()

episodic = EpisodicMemory("scene_graph.json")
robokg = RoboKGNet()
semantic_memory = SemanticMemory(model="qwen3:1.7b")

# 1. First try episodic memory.
frame["Source"] = episodic.query_theme_location(frame["Theme"])

if frame["Source"]:
    search_locations = [frame["Source"]]

else:
    # 2. Try RoboKGNet semantic knowledge.
    theme = frame["Theme"]
    concepts = robokg.resolve_concept(theme)

    # Try WordNet-style lexical spelling too: "coffee mug" -> "coffee_mug".
    if not concepts:
        concepts = robokg.resolve_concept(theme.replace(" ", "_"))

    if len(concepts) == 1:
        concept_id = concepts[0]["id"]
        search_locations = robokg.get_locations(concept_id)
    else:
        search_locations = []

    # 3. Fall back to generative semantic memory if RoboKGNet
    #    cannot resolve the concept OR knows no locations.
    if not search_locations:
        search = resolve_search(frame, episodic, semantic_memory)
        search_locations = search["locations"]

print("Frame:", frame)
print("Search order:")

for candidate in search_locations:
    print(candidate)