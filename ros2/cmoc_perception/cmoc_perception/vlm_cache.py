"""Content-addressed debug cache; never episodic or semantic memory."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


class VLMResponseCache:
    def __init__(self, directory, log):
        self.directory = Path(directory)
        self.log = log

    @staticmethod
    def key(metadata):
        return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':'),
                                         allow_nan=False).encode()).hexdigest()

    def load(self, metadata, validate):
        key = self.key(metadata)
        try:
            entry = json.loads((self.directory / (key + '.json')).read_text())
            if entry['key'] != key or entry['request'] != metadata:
                raise ValueError('cache metadata mismatch')
            response = json.dumps(entry['response'], allow_nan=False)
            validate(response)
        except (OSError, ValueError, KeyError, TypeError):
            self.log(f'[VLM CACHE] MISS {key[:12]}')
            return None
        self.log(f'[VLM CACHE] HIT {key[:12]}; network VLM call skipped')
        return response

    def save(self, metadata, response):
        key = self.key(metadata)
        temporary = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            entry = {'key': key, 'backend': metadata['backend'], 'model': metadata['model'],
                     'image_sha256': metadata['image_sha256'], 'request': metadata,
                     'response': response}
            with tempfile.NamedTemporaryFile(mode='w', dir=self.directory, delete=False) as output:
                temporary = output.name
                json.dump(entry, output, allow_nan=False)
            os.replace(temporary, self.directory / (key + '.json'))
            temporary = None
            self.log(f'[VLM CACHE] SAVED {key[:12]}')
        except (OSError, ValueError, TypeError):
            self.log(f'[VLM CACHE] WRITE SKIPPED {key[:12]}')
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
