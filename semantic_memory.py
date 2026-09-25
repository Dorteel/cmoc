"""Minimal semantic-memory ranking for likely object locations.

Semantic memory:
    locatedAt(muffin, KITCHEN)
    = hypothesis / prior

Episodic memory:
    in(cupcake1, KITCHEN)
    = grounded fact
"""

import re

from sentence_transformers import CrossEncoder


class SemanticMemory:
    """Rank known locations by how plausible a theme is there."""

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L6-v2"):
        # Load the cross-encoder once and reuse it across all ranking calls.
        self.model = CrossEncoder(model_name)

    def rank_locations(self, theme, locations):
        """Return ranked location hypotheses for a theme.

        Each item is a simple dictionary with the original location ID, the raw
        score assigned by the model, and the corresponding hypothesis.
        """
        if not locations:
            return []

        # Build one commonsense candidate per known location.
        # Example: "A muffin is typically located in a kitchen."
        # This is not a fact about the current scene; it is a hypothesis / prior.
        pairs = []
        for location in locations:
            location_id = location["id"]

            # KITCHEN -> kitchen
            # LIVING_ROOM_1 -> living room
            readable_name = location_id
            readable_name = readable_name.replace("_", " ")
            readable_name = re.sub(r"\s+\d+$", "", readable_name)
            readable_name = readable_name.lower()

            statement = f"A {theme} is typically located in a {readable_name}."
            context = "This statement is a plausibility judgment for a location hypothesis."
            pairs.append((statement, context))

        # Score each commonsense statement with the cross-encoder. The raw score
        # is kept as-is; it is not interpreted as a calibrated probability.
        scores = self.model.predict(pairs)

        # The model returns a score per pair, but we keep the original location ID
        # and store the ranked hypothesis string explicitly.
        ranked = []
        for location, score in zip(locations, scores):
            location_id = location["id"]
            hypothesis = f"locatedAt({theme}, {location_id})"
            ranked.append({
                "location": location_id,
                "score": float(score),
                "hypothesis": hypothesis,
            })

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked
