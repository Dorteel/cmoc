from listener import NaiveFrameFiller
from knowledge_interface import KnowledgeInterface
instruction = "Bring me the muffin"

# Extract frame from the instruction
frame = NaiveFrameFiller(instruction).fill()

# Maybe some more reasoning to extract actions

# Check whether source is known
uncertainties = [k for k, v in frame.items() if v is None]
kb = KnowledgeInterface('scene_graph.json')
print(frame)
print(uncertainties)
themes = kb.query_theme(frame['Theme'])
source = kb.query_theme_location(frame['Theme'])
print(source)

