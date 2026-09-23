import base64
import json
from pathlib import Path

import requests


class PerceptionModule:
    def __init__(self, model, prompt_library, token, url="https://nebula.cs.vu.nl/api/chat/completions", schema_root=None):
        self.model = model
        self.prompts = prompt_library
        self.token = token
        self.url = url
        self.schema_root = Path(schema_root) if schema_root is not None else Path(__file__).parent / "schemas"

    def _encode_image(self, image_path):
        path = Path(image_path)
        image = base64.b64encode(path.read_bytes()).decode("utf-8")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}[path.suffix[1:].lower()]
        return image, mime
    def perceive(self, image_path=None, prompt_name="perception.create_scene_graph", schema_name=None, instruction=None):
        """
        Call the remote model using a named prompt and optional JSON schema.

        - `image_path`: Path to image to include (optional).
        - `prompt_name`: key used with `PromptLibrary.get()` to load the task prompt.
        - `schema_name`: optional dot-separated schema name (e.g. 'perception.create_scene_graph');
          when provided the schema is loaded from `schema_root` and attached as a json_schema response_format.
        - `instruction`: optional instruction string injected into the user content to guide attention.
        """

        system_msg = {"role": "system", "content": self.prompts.get("perception.system")}

        prompt_text = self.prompts.get(prompt_name)

        user_content = []
        if instruction:
            user_content.append({"type": "text", "text": instruction})
        user_content.append({"type": "text", "text": prompt_text})

        if image_path is not None:
            image, mime = self._encode_image(image_path)
            user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image}"}})

        messages = [system_msg, {"role": "user", "content": user_content}]

        payload = {"model": self.model, "messages": messages}

        if schema_name is not None:
            # schema_name like 'perception.create_scene_graph' -> schemas/perception/create_scene_graph.json
            schema_path = self.schema_root / Path(schema_name.replace('.', '/')).with_suffix('.json')
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "scene_graph", "strict": True, "schema": schema},
            }

        response = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
            json=payload,
            timeout=(15, 120),
        )

        if not response.ok:
            print(response.status_code, response.text)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

class PromptLibrary:
    def __init__(self, root="prompts"):
        self.root = Path(root)
        self.prompts = self._load()

    def _load(self):
        prompts = {}
        for path in self.root.rglob("*.md"):
            key = ".".join(path.relative_to(self.root).with_suffix("").parts)
            prompts[key] = path.read_text(encoding="utf-8").strip()
        return prompts

    def get(self, name):
        return self.prompts[name]

    def list(self):
        return list(self.prompts)
