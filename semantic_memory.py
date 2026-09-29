# semantic_memory.py
import json
import re
import requests

from semantic_fallback import label


class SemanticMemory:
    """Use a local LLM as commonsense semantic memory."""

    def __init__(self, model="qwen3:1.7b", url="http://localhost:11434/api/chat"):
        self.model = model
        self.url = url

    @staticmethod
    def _location_type(location_id):
        """LIVING_ROOM_2 -> living room."""
        return re.sub(r"_\d+$", "", location_id).replace("_", " ").lower()

    def _anchor_type(self, location):
        return (self._location_type(location['id']) if location.get('type') in (None, 'Location')
                else label(location['type']))

    def rank_locations(self, theme, locations):
        # Rank semantic types, not individual room instances.
        location_types = sorted({
            self._anchor_type(location)
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

        # Give every concrete room instance its semantic-type score.
        scores = {
            item["location"]: float(item["score"])
            for item in ranked_types
        }

        ranked = [
            {
                "location": location["id"],
                "score": scores.get(self._anchor_type(location), 0.0),
                "hypothesis": f"locatedAt({theme}, {location['id']})",
            }
            for location in locations
        ]

        return sorted(ranked, key=lambda x: x["score"], reverse=True)

    def rank_gaze_targets(self, theme, targets, *, checked=()):
        """Prioritize inspection of current entities; assert no object location."""
        if not targets:
            return []
        prompt = (
            f"Theme: {theme}\nCurrently visible targets: "
            + json.dumps([{'id': obj['id'], 'type': obj['type']} for obj in targets])
            + f"\nChecked: {sorted(checked)}\n"
            "Which currently visible target should the robot inspect next to gain evidence "
            "about the Theme? Rank only the supplied IDs. This is inspection priority, "
            "not a claim that the Theme is located there. Return only JSON: "
            '{"locations": [{"location": "<supplied ID>", "score": 1.0}]}'
        )
        response = requests.post(self.url, json={
            'model': self.model, 'messages': [{'role': 'user', 'content': prompt}],
            'stream': False, 'format': 'json'})
        response.raise_for_status()
        ranked = json.loads(response.json()['message']['content'])['locations']
        return ranked

    def choose_gaze_action(self, theme, observation, options, *, checked=()):
        prompt = (
            f"Theme: {theme}\nCurrent observation: {json.dumps(observation)}\n"
            f"Available gaze actions: {json.dumps(options)}\nAlready checked: {json.dumps(sorted(checked))}\n"
            "Choose exactly one supplied action to gain new visual evidence about the Theme. "
            "Do not choose an already checked action. This is a gaze decision, not an object-location claim. "
            'Return only one JSON object: {"action":"look-at","target":"<supplied ID>"} '
            'or {"action":"look-left"} or {"action":"look-right"}.'
        )
        response = requests.post(self.url, json={
            'model': self.model, 'messages': [{'role': 'user', 'content': prompt}],
            'stream': False, 'format': 'json'})
        response.raise_for_status()
        return json.loads(response.json()['message']['content'])
