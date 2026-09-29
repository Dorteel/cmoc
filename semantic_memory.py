# semantic_memory.py
import json
import re
import requests
from time import monotonic

from semantic_fallback import label


GAZE_TIMEOUT_SECONDS = 30


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
            'Return only one JSON object with action, target, and reason: '
            '{"action":"look-at","target":"<supplied ID>","reason":"A table is a plausible place to inspect."} '
            'or use action look-left/look-right with target null. '
            'The reason must be exactly one short explanatory sentence, at most 240 characters, '
            'ending in a period, question mark, or exclamation mark. Do not assert hidden locations.'
        )
        print('[SEARCH] Calling gaze LLM...', flush=True)
        print(f'[SEARCH] Gaze LLM backend: Ollama ({self.url})', flush=True)
        print(f'[SEARCH] Gaze LLM model: {self.model}', flush=True)
        started = monotonic()
        try:
            response = requests.post(self.url, json={
                'model': self.model, 'messages': [{'role': 'user', 'content': prompt}],
                'stream': False, 'format': 'json', 'think': False},
                timeout=GAZE_TIMEOUT_SECONDS)
            response.raise_for_status()
            chosen = json.loads(response.json()['message']['content'])
            if not isinstance(chosen, dict):
                raise ValueError('expected a JSON object')
            reason = chosen.get('reason')
            if (not isinstance(reason, str) or len(reason) > 240
                    or re.fullmatch(r'[^.!?\r\n]+[.!?]', reason.strip()) is None):
                raise ValueError('reason must be one short sentence (maximum 240 characters)')
            chosen['reason'] = reason.strip()
        except requests.Timeout as error:
            message = f'Gaze LLM timed out (timeout {GAZE_TIMEOUT_SECONDS} s)'
            print(f'[SEARCH] Gaze LLM failed after {monotonic() - started:.1f} s: {message}', flush=True)
            raise TimeoutError(message) from error
        except (requests.RequestException, ValueError, KeyError, TypeError) as error:
            message = f'Invalid gaze LLM response: {error}' if not isinstance(error, requests.RequestException) else str(error)
            print(f'[SEARCH] Gaze LLM failed after {monotonic() - started:.1f} s: {message}', flush=True)
            raise ValueError(message) from error
        print(f'[SEARCH] Gaze LLM response received in {monotonic() - started:.1f} s', flush=True)
        return chosen
