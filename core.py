from pathlib import Path
import requests


import base64
import requests
from pathlib import Path


class PerceptionModule:
    def __init__(self, model, prompt_library, token, url="https://nebula.cs.vu.nl/api/chat/completions"):
        self.model = model
        self.prompts = prompt_library
        self.token = token
        self.url = url

    def _encode_image(self, image_path):
        path = Path(image_path)
        image = base64.b64encode(path.read_bytes()).decode("utf-8")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}[path.suffix[1:].lower()]
        return image, mime

    def perceive(self, image_path, task="perception.create_scene_graph"):
        image, mime = self._encode_image(image_path)

        messages = [
            {
                "role": "system",
                "content": self.prompts.get("perception.system"),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompts.get(task),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{image}"
                        },
                    },
                ],
            },
        ]

        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
            },
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
    