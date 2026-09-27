"""Resolve a search target and an ordered list of candidate locations.

Semantic memory:
    locatedAt(muffin, KITCHEN)
    = hypothesis / prior

Episodic memory:
    in(cupcake1, KITCHEN)
    = grounded fact
"""


def resolve_search(frame, knowledge_interface, semantic_memory):
    """Branch on whether the Source is already known.

    The task instruction tells us which Theme category to search for. We do not
    check whether a grounded instance already exists; we only decide whether the
    search should be anchored to a known source or ranked across known locations.
    """
    theme = frame["Theme"]
    source = frame["Source"]

    # Source known: the search is already grounded to one location.
    if source is not None:
        return {"target": theme, "locations": [source]}

    # Source unknown: query the known environment locations, then rank them by
    # semantic plausibility that the theme might be there.
    locations = knowledge_interface.query_locations()
    ranked_locations = semantic_memory.rank_locations(theme, locations)
    return {"target": theme, "locations": ranked_locations}


if __name__ == "__main__":
    from scene_graph_interface import KnowledgeInterface
    from semantic_memory import SemanticMemory

    frame = {
        "Agent": "robot",
        "Theme": "muffin",
        "Source": None,
        "Destination": "user",
    }

    kb = KnowledgeInterface("scene_graph.json")
    semantic_memory = SemanticMemory()
    search = resolve_search(frame, kb, semantic_memory)

    print(f"Target: {search['target']}")
    print()
    print("Search order:")
    for index, result in enumerate(search["locations"], start=1):
        print(f"{index}. {result['location']:<15} {result['score']}")
