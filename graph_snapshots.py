"""Repository-local runtime knowledge artifacts, published with atomic replacement."""

from copy import deepcopy
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


def episode_snapshot(state, observation_number=1):
    """Context around raw perception, separate from the VLM object schema."""
    state = deepcopy(state)
    observation = f'observation_{observation_number}'
    state['context_graph'] = {
        'nodes': [{'id': 'episode_1', 'type': 'Episode'},
                  {'id': 'instruction_1', 'type': 'Instruction', 'text': state['instruction']},
                  {'id': observation, 'type': 'Observation'},
                  {'id': 'robot', 'type': 'Robot'}],
        'relations': [
            {'subject': 'episode_1', 'predicate': 'hasInstruction', 'object': 'instruction_1'},
            {'subject': 'episode_1', 'predicate': 'hasObservation', 'object': observation},
            {'subject': observation, 'predicate': 'observedBy', 'object': 'robot'},
            *({'subject': observation, 'predicate': 'observes', 'object': obj['id']}
              for obj in state['scene_graph'].get('objects', []))]}
    return state


def add_frame_graph(state):
    """Expose semantic roles and concrete bindings without conflating them."""
    context = state['context_graph']
    context['nodes'].append({'id': 'BringingFrame_1', 'type': 'Bringing'})
    context['relations'].append({'subject': 'episode_1', 'predicate': 'hasFrame',
                                 'object': 'BringingFrame_1'})
    for role in ('Agent', 'Theme', 'Source', 'Destination'):
        identifier = role + '_FE'
        context['nodes'].append({'id': identifier, 'type': 'FrameElement',
                                 'role': role, 'semanticValue': state['frame'].get(role)})
        context['relations'].append({'subject': 'BringingFrame_1',
                                     'predicate': 'hasFrameElement', 'object': identifier})
        if state['bindings'].get(role) is not None:
            context['relations'].append({'subject': identifier, 'predicate': 'bindsTo',
                                         'object': state['bindings'][role]})
