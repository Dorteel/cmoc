from search_strategy import resolve_search


class FakeKnowledgeInterface:
    def query_locations(self):
        return [
            {"id": "KITCHEN", "type": "Location"},
            {"id": "LIVING_ROOM_1", "type": "Location"},
        ]


class FakeSemanticMemory:
    def rank_locations(self, theme, locations):
        assert theme == "muffin"
        assert [location["id"] for location in locations] == [
            "KITCHEN",
            "LIVING_ROOM_1",
        ]
        return [
            {"location": "KITCHEN", "score": 7.21, "hypothesis": "locatedAt(muffin, KITCHEN)"},
            {"location": "LIVING_ROOM_1", "score": 3.84, "hypothesis": "locatedAt(muffin, LIVING_ROOM_1)"},
        ]


def test_unknown_source_uses_semantic_ranking():
    frame = {
        "Agent": "robot",
        "Theme": "muffin",
        "Source": None,
        "Destination": "user",
    }

    result = resolve_search(frame, FakeKnowledgeInterface(), FakeSemanticMemory())

    assert result["target"] == "muffin"
    assert result["locations"][0]["location"] == "KITCHEN"
    assert result["locations"][0]["score"] == 7.21


def test_known_source_short_circuits_search():
    frame = {
        "Agent": "robot",
        "Theme": "muffin",
        "Source": "KITCHEN",
        "Destination": "user",
    }

    result = resolve_search(frame, FakeKnowledgeInterface(), FakeSemanticMemory())

    assert result == {"target": "muffin", "locations": ["KITCHEN"]}
