"""Resolve a known source or rank places to search for a theme."""

import re


def resolve_search(frame, kb):
    # Always search for the requested theme, whether or not memory contains it.
    theme = frame["Theme"]
    source = frame["Source"]

    # If the source is known, search for the theme at that source.
    if source is not None:
        return {"target": theme, "locations": [source]}

    # If the source is unknown, get all known environment locations.
    locations = kb.query_locations()

    # Rank those locations for the theme, then search them in that order.
    ranked_locations = rank_locations(theme, locations)
    return {"target": theme, "locations": ranked_locations}


def rank_locations(theme, locations):
    # With no locations, return immediately without loading the model.
    if not locations:
        return []

    from sentence_transformers import CrossEncoder

    # Turn each location ID into readable text, keeping its original ID for output.
    question = f"Where is a {theme} likely to be found?"
    pairs = []
    for location in locations:
        label = re.sub(r"_\d+$", "", location["id"])
        label = label.replace("_", " ").lower()
        pairs.append((question, label))

    # Score each question/location pair with the requested cross-encoder.
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    scores = model.predict(pairs)

    # Return the original IDs and numeric scores, highest score first.
    ranked = []
    for location, score in zip(locations, scores):
        ranked.append({"location": location["id"], "score": float(score)})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked
