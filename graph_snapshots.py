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
                  {'id': observation, 'type': 'Observation',
                   **({'generatedByModel': state['perception_provenance']['model'],
                       'perceptionBackend': state['perception_provenance']['backend'],
                       'fallbackUsed': state['perception_provenance']['fallback_used']}
                      if state.get('perception_provenance') else {})},
                  {'id': 'robot', 'type': 'Robot'}],
        'relations': [
            {'subject': 'episode_1', 'predicate': 'hasInstruction', 'object': 'instruction_1'},
            {'subject': 'episode_1', 'predicate': 'hasObservation', 'object': observation},
            {'subject': observation, 'predicate': 'observedBy', 'object': 'robot'},
            *({'subject': observation, 'predicate': 'observes', 'object': obj['id']}
              for obj in state['scene_graph'].get('objects', []))]}
    return state


def add_frame_graph(state):
    """Build only task context; raw perception/memory are not expanded here."""
    observation = deepcopy(next(node for node in state['context_graph']['nodes']
                                if node['type'] == 'Observation'))
    relevant = {'robot', 'user'} | {value for value in state['bindings'].values() if value}
    nodes = [{'id': 'episode_1', 'type': 'Episode', 'instruction': state['instruction']},
             observation, {'id': 'robot', 'type': 'Robot'},
             {'id': 'user', 'type': 'Entity'}, {'id': 'BringingFrame_1', 'type': 'Bringing'}]
    context = state['context_graph'] = {'nodes': nodes, 'relations': [
        {'subject': 'episode_1', 'predicate': 'hasObservation', 'object': observation['id']},
        {'subject': observation['id'], 'predicate': 'observedBy', 'object': 'robot'},
        {'subject': 'episode_1', 'predicate': 'hasFrame', 'object': 'BringingFrame_1'}]}
    for obj in state['scene_graph'].get('objects', []):
        if obj['id'] in relevant:
            relation = {'subject': observation['id'], 'predicate': 'observes', 'object': obj['id']}
            if relation not in context['relations']:
                context['relations'].append(relation)
    for identifier, concept in state.get('entity_links', {}).items():
        if identifier in relevant:
            context['relations'].append({'subject': identifier, 'predicate': 'linkedTo', 'object': concept})
    for role in ('Agent', 'Theme', 'Source', 'Destination'):
        identifier = role + '_FE'
        context['nodes'].append({'id': identifier, 'type': 'FrameElement',
                                 'role': role, 'semanticValue': state['frame'].get(role)})
        context['relations'].append({'subject': 'BringingFrame_1',
                                     'predicate': 'hasFrameElement', 'object': identifier})
        concept = state.get('frame_element_links', {}).get(role)
        if concept is not None:
            context['relations'].append({'subject': identifier, 'predicate': 'linkedTo',
                                         'object': concept})
        if state['bindings'].get(role) is not None:
            context['relations'].append({'subject': identifier, 'predicate': 'bindsTo',
                                         'object': state['bindings'][role]})

    # Declare just endpoints of these task relationships, with no expansion.
    declared = {node['id'] for node in nodes}
    for relation in context['relations']:
        for endpoint in ('subject', 'object'):
            identifier = relation[endpoint]
            if identifier not in declared:
                kind = 'Concept' if endpoint == 'object' and relation['predicate'] == 'linkedTo' else 'Entity'
                nodes.append({'id': identifier, 'type': kind})
                declared.add(identifier)


def interpretation_snapshot(state, sense_graph):
    """Publish the interpretation before Source resolution, separately from execution."""
    snapshot = deepcopy(state)
    for key in ('planning', 'plan', 'executed', 'failed_step', 'action_graph', 'planning_graph', 'entity_positions'):
        snapshot.pop(key, None)
    snapshot['frame']['Source'] = None
    snapshot['bindings']['Source'] = None
    snapshot.get('frame_element_links', {})['Source'] = None
    snapshot['sense_graph'] = deepcopy(sense_graph)
    snapshot['stage'] = 'G2'
    add_frame_graph(snapshot)
    source = next(node for node in snapshot['context_graph']['nodes'] if node['id'] == 'Source_FE')
    source['semanticValue'] = 'Unknown'
    return snapshot


def action_snapshot(result, state=None):
    """Keep the resolved interpretation alongside the actual ordered plan."""
    snapshot = deepcopy(result)
    nodes = [{'id': 'episode_1', 'type': 'Episode', 'status': result['status']},
             {'id': 'Plan_1', 'type': 'Plan'}]
    relations = [{'subject': 'episode_1', 'predicate': 'hasPlan', 'object': 'Plan_1'}]
    attempts = {attempt['step']: attempt['status'] for attempt in result['executed']}
    for index, step in enumerate(result['plan'], 1):
        identifier = f"{step['action']}_{index}"
        nodes.append({'id': identifier, 'type': 'Action', 'action': step['action'],
                      'order': index, 'status': attempts.get(index, 'not_attempted')})
        relations.append({'subject': 'Plan_1', 'predicate': 'hasAction', 'object': identifier})
        for position, value in enumerate(step['args']):
            relations.append({'subject': identifier, 'predicate': f'argument{position + 1}', 'object': value})
    if result.get('reason'):
        nodes[0]['reason'] = result['reason']
    snapshot['episode_id'] = 'episode_1'
    if state is not None and state.get('type') != 'search':
        resolved = deepcopy(state)
        # Only snapshot data, never execution coordinates or nested prior artifacts.
        for key in ('action_graph', 'planning_graph', 'entity_positions'):
            resolved.pop(key, None)
        add_frame_graph(resolved)
        context = resolved['context_graph']
        nodes = context['nodes'] + [node for node in nodes if node['id'] != 'episode_1']
        relations = context['relations'] + relations
        snapshot = {**resolved, **snapshot, 'stage': 'G3'}
    snapshot['context_graph'] = {'nodes': nodes, 'relations': relations}
    return snapshot
