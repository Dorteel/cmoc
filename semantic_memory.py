# semantic_memory.py
import json
import re
import requests


class SemanticMemory:
    """Use a local LLM as commonsense semantic memory."""

    def __init__(self, model="qwen3:1.7b", url="http://localhost:11434/api/chat"):
        self.model = model
        self.url = url

    @staticmethod
    def _location_type(location_id):
        """LIVING_ROOM_2 -> living room."""
        return re.sub(r"_\d+$", "", location_id).replace("_", " ").lower()

    def rank_locations(self, theme, locations):
        # Rank semantic types, not individual room instances.
        location_types = sorted({
            self._location_type(location["id"])
            for location in locations
        })

        prompt = f"""
        Object: {theme}
        Possible locations: {location_types}

        Rank these locations by how typically a {theme} would be found there.

        Return ONLY JSON in this format:
        {{
            "locations": [
                {{"location": "kitchen", "score": 1.0}},
                {{"location": "garden", "score": 0.1}}
            ]
        }}
        """

        if not locations:
            prompt += " No symbolic locations are known yet. Suggest plausible location names."

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
            },
        )
        response.raise_for_status()

        result = json.loads(response.json()["message"]["content"])
        ranked_types = result["locations"]
        if not locations:
            return ranked_types
        print(response.json()["message"]["content"])

        # Give every concrete room instance its semantic-type score.
        scores = {
            item["location"]: float(item["score"])
            for item in ranked_types
        }

        ranked = [
            {
                "location": location["id"],
                "score": scores.get(self._location_type(location["id"]), 0.0),
                "hypothesis": f"locatedAt({theme}, {location['id']})",
            }
            for location in locations
        ]

        return sorted(ranked, key=lambda x: x["score"], reverse=True)