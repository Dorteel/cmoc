"""Repository-local runtime knowledge artifacts, published with atomic replacement."""

import json
import os
from pathlib import Path
import tempfile

ARTIFACT_DIRECTORY = Path(__file__).resolve().parent / 'episodic_memory'
EPISODIC_GRAPH = ARTIFACT_DIRECTORY / 'scene_graph.json'
FILES = {'episodic': 'scene_graph.json', 'g1': 'g1_sense.json',
         'g2': 'g2_plan.json', 'g3': 'g3_action.json'}


def write_graph_snapshot(stage, data):
    destination = ARTIFACT_DIRECTORY / FILES[stage]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
