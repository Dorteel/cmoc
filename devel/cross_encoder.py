import numpy as np
from sentence_transformers import CrossEncoder

locations = [
    "Kitchen",
    "Bathroom",
    "Living Room",
    "Bedroom",
    "Garage",
    "Garden",
    "Office",
    "Dining Room",
    "Attic",
    "Basement",
    "Pantry",
    "Laundry Room",
]

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
pairs = [("Where is a cup typically found?", location) for location in locations]



scores = model.predict(pairs)

# Convert ranking scores into relative probabilities.
probabilities = np.exp(scores - np.max(scores))
probabilities = probabilities / probabilities.sum()

for location, probability in sorted(zip(locations, probabilities), key=lambda x: x[1], reverse=True):
    print(location, probability)