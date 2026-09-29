"""Regenerate the small Bringing examples using the actual legacy PDDL planner.

Run from the repository root: python examples/visualization/generate_bringing.py
Only regeneration needs unified-planning[fast-downward]; viewing needs rdflib.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from graph_snapshots import episode_snapshot, add_frame_graph, interpretation_snapshot, action_snapshot
from knowledge_interface import KnowledgeInterface
from perceived_entity_linking import perceived_entity_linking, bind_task, ground_frame_elements
from procedural_memory.planning.bringing_plan import plan_bring
from utils.view_kg import snapshot_to_graph, graph_to_data, check_visualization_sizes


def examples():
    scene = {'objects': [
        {'id': identifier, 'type': kind, 'qualities': {}}
        for identifier, kind in [('TIAGo', 'robot'), ('pedestrian_1', 'pedestrian'),
                                 ('fork1', 'ForkConnector'), ('table1', 'table'),
                                 ('LIVING_ROOM_1', 'Location')]],
        'relations': [
            {'subject': subject, 'predicate': 'in', 'object': 'LIVING_ROOM_1'}
            for subject in ('TIAGo', 'pedestrian_1', 'table1')]}
    # The initial observation contains no Source fact for the fork. Resolution
    # uses explicit episodic evidence, not simulator coordinates or services.
    world = deepcopy(scene)
    world['objects'].append({'id': 'KITCHEN', 'type': 'Location', 'qualities': {}})
    world['relations'].append({'subject': 'fork1', 'predicate': 'in', 'object': 'KITCHEN'})
    kg = KnowledgeInterface()
    perceived = perceived_entity_linking(scene, kg)
    remembered = perceived_entity_linking(world, kg)
    g1 = episode_snapshot({'instruction': 'Bring me a fork', 'scene_graph': scene})
    g1['entity_links'] = {raw: perceived['entity_links'][canonical]
                          for raw, canonical in perceived['identity_mapping'].items()
                          if canonical in perceived['entity_links']}
    state = deepcopy(g1)
    state.update(bind_task({'Agent': 'robot', 'Theme': 'fork', 'Source': None, 'Destination': 'user'},
                           remembered, kg))
    state['scene_graph'] = perceived['scene_graph']
    state['aliases'] = remembered['aliases']
    state['entity_links'] = remembered['entity_links']
    state['frame_element_links'], state['frame_element_issues'] = ground_frame_elements(state['frame'], kg, remembered)
    add_frame_graph(state)
    state['planning'] = plan_bring(state['bindings'], remembered['scene_graph'])
    if state['planning']['status'] != 'planned':
        raise RuntimeError(state['planning'])
    g2 = interpretation_snapshot(state, g1)
    state['sense_graph'] = deepcopy(g1)
    g3 = action_snapshot({'status': 'planned', 'plan': state['planning']['plan'],
                          'executed': [], 'failed_step': None}, state)
    counts = check_visualization_sizes(*(graph_to_data(snapshot_to_graph(g)) for g in (g1, g2, g3)))
    print('Visible nodes G1/G2/G3:', counts)
    return g1, g2, g3


if __name__ == '__main__':
    for index, graph in enumerate(examples(), 1):
        for path in (ROOT / 'episodic_memory' / {1: 'g1_sense.json', 2: 'g2_plan.json', 3: 'g3_action.json'}[index],
                     ROOT / 'examples/visualization' / f'g{index}.json'):
            path.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + '\n')
