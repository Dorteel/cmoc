"""Single-graph replay cache for debugging, independent of camera content."""
import json
import os
from pathlib import Path
import tempfile


class VLMResponseCache:
    def __init__(self, directory, log):
        self.path = Path(directory) / 'scene_graph.json'
        self.log = log

    def load(self, validate):
        try:
            text = self.path.read_text()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise ValueError(f'Cannot read VLM cache {self.path}: {error}') from error
        try:
            graph = validate(text)
        except Exception as error:
            raise ValueError(f'Invalid VLM cache {self.path}: {error}') from error
        self.log(f'[VLM CACHE] HIT {self.path}')
        return graph

    def save(self, graph):
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', dir=self.path.parent, delete=False) as output:
                temporary = output.name
                json.dump(graph, output, allow_nan=False)
            os.replace(temporary, self.path)
            temporary = None
        except (OSError, ValueError, TypeError) as error:
            raise ValueError(f'Cannot save VLM cache {self.path}: {error}') from error
        finally:
            if temporary is not None:
                os.unlink(temporary)
        self.log(f'[VLM CACHE] SAVED {self.path}')
