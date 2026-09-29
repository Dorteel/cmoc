# semantic_memory.py
import json
import re
import requests
from time import monotonic
from jsonschema import validate, ValidationError

from schemas import build_gaze_choice_schema

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

    def choose_gaze_action(self, theme, observation, options, *, checked=(), sweep_direction=None):
        schema = build_gaze_choice_schema(options)
        sweep_policy = (
            f"Strongly prefer continuing look-{sweep_direction}; reverse direction only with a good reason. "
            if sweep_direction in ('left', 'right') else
            "Choose either direction to start exploring. "
        )
        prompt = (
            f"Theme: {theme}\n"
            f"Allowed gaze actions (choose only from this list): {json.dumps(options)}\n"
            f"{sweep_policy}Choose exactly one supplied action to gain new visual evidence about the Theme. "
            'Continue exploring in the current sweep direction unless a currently visible object is a strong '
            'semantic anchor for the Theme. Prefer inspecting such an object only when it is plausibly more '
            'informative than continuing the sweep. Both directions remain available. '
            'Return only one JSON object with action and reason, plus target only for look-at. '
            'Directional actions must omit target. '
            'Give exactly one short sentence explaining the selected action, at most 240 characters.'
        )
        print('[SEARCH] Gaze LLM context:', flush=True)
        print(f'[SEARCH]   Theme: {theme}', flush=True)
        print(f'[SEARCH]   Current options: {json.dumps(options)}', flush=True)
        print('[SEARCH] Calling gaze LLM...', flush=True)
        print(f'[SEARCH] Gaze LLM backend: Ollama ({self.url})', flush=True)
        print(f'[SEARCH] Gaze LLM model: {self.model}', flush=True)
        started = monotonic()
        try:
            response = requests.post(self.url, json={
                'model': self.model, 'messages': [{'role': 'user', 'content': prompt}],
                'stream': False, 'format': schema, 'think': False},
                timeout=GAZE_TIMEOUT_SECONDS)
            response.raise_for_status()
            raw = response.json()['message']['content']
            print(f'[SEARCH] Raw response: {raw}', flush=True)
            chosen = json.loads(raw)
            if not isinstance(chosen, dict):
                raise ValueError('expected a JSON object')
            try:
                validate(chosen, schema)
            except ValidationError as error:
                raise ValueError(f'Invalid LLM gaze action: {error.message}') from error
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
