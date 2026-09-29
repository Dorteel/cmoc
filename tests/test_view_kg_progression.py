"""The committed legacy Bringing story stays small and explains every added node."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from rdflib import URIRef

from graph_snapshots import action_snapshot, interpretation_snapshot
from utils import view_kg

ROOT = Path(view_kg.__file__).resolve().parents[1]


@pytest.fixture
def progression():
    # Runtime snapshots may be edited independently; use the coherent example set.
    paths = [ROOT / 'examples/visualization' / f'g{stage}.json' for stage in (1, 2, 3)]
    snapshots = [json.loads(path.read_text()) for path in paths]
    graphs = [view_kg.load_graph(path) for path in paths]
    return snapshots, graphs, [view_kg.graph_to_data(graph) for graph in graphs]


def by_entity(payload):
    return {node['details']['id']: node for node in payload['nodes']}


def relations(payload):
    names = {node['id']: node['details']['id'] for node in payload['nodes']}
    return {(names[edge['source']], edge['label'], names[edge['target']]) for edge in payload['edges']}


def test_progressive_bringing_semantics_and_shared_entities(progression):
    (g1, g2, g3), graphs, payloads = progression
    first, second, third = map(by_entity, payloads)
    assert 'frame' not in g1 and 'BringingFrame_1' not in first
    assert set(first) <= set(second)
    assert set(second) - {'Unknown'} <= set(third)
    for identifier in first:
        assert first[identifier]['id'] == second[identifier]['id'] == third[identifier]['id']
    assert g2['frame']['Source'] is None and g2['bindings']['Source'] is None
    assert g3['bindings']['Source'] == 'KITCHEN'
    assert ('Source_FE', 'bindsTo', 'Unknown') in relations(payloads[1])
    assert ('Source_FE', 'bindsTo', 'KITCHEN') in relations(payloads[2])
    assert 'Unknown' not in third
    for payload in payloads[1:]:
        assert ('Agent_FE', 'bindsTo', 'TIAGo') in relations(payload)
        assert ('Destination_FE', 'bindsTo', 'pedestrian_1') in relations(payload)
        assert ('Theme_FE', 'bindsTo', 'fork1') in relations(payload)
        assert ('Theme_FE', 'linkedTo', 'fork.n.01') in relations(payload)
        assert ('fork1', 'linkedTo', 'fork.n.01') in relations(payload)
    assert second['Theme_FE']['uri'] == view_kg.FRAMENET + 'Bringing%2FFE%2FTheme'
    assert not {'user', 'robot'} & set(third)  # Reuse observed person and TIAGo.
    assert all(node['kind'] != 'literal' or node['details']['type'] == 'Instruction'
               for payload in payloads for node in payload['nodes'])


def test_strict_visible_node_budgets_and_exact_g3_growth(progression):
    (g1, g2, g3), _, payloads = progression
    counts = view_kg.check_visualization_sizes(*payloads)
    assert counts[0] < 20
    assert counts[1] < 2 * counts[0]
    assert counts[2] < 2 * counts[0] + 5
    second, third = map(by_entity, payloads[1:])
    steps = {f"{step['action']}_{index}" for index, step in enumerate(g3['plan'], 1)}
    assert len(steps) == 4
    assert set(third) - set(second) == {'KITCHEN', 'Plan_1'} | steps
    assert set(second) - set(third) == {'Unknown'}
    # No new WordNet, FrameNet or literal neighborhood accompanies resolution.
    assert {key for key, node in third.items() if node['details']['type'] == 'Concept'} == {
        key for key, node in second.items() if node['details']['type'] == 'Concept'}
    new_edges = relations(payloads[2]) - relations(payloads[1])
    assert new_edges == {('Source_FE', 'bindsTo', 'KITCHEN'), ('BringingFrame_1', 'hasPlan', 'Plan_1')} | {
        ('Plan_1', 'hasAction', step) for step in steps} | {
        (f"{step['action']}_{index}", f'argument{position}',
         'pedestrian_1' if argument == 'user' else argument)
        for index, step in enumerate(g3['plan'], 1)
        for position, argument in enumerate(step['args'], 1)}


@pytest.mark.parametrize('stage', [0, 1, 2])
def test_size_guard_rejects_boundary_and_excess(progression, stage):
    payloads = deepcopy(progression[2])
    limits = [20, 2 * len(payloads[0]['nodes']), 2 * len(payloads[0]['nodes']) + 5]
    payloads[stage]['nodes'] += [{}] * (limits[stage] - len(payloads[stage]['nodes']))
    with pytest.raises(ValueError, match=f'G{stage + 1} visualization too large'):
        view_kg.check_visualization_sizes(*payloads)


def test_action_order_and_arguments_are_actual_plan_data(progression):
    (g1, g2, g3), _, payloads = progression
    actions = sorted((node for node in payloads[2]['nodes'] if node['details']['type'] == 'Action'),
                     key=lambda node: node['details']['order'])
    assert [{'action': node['details']['action'], 'args': node['details']['arguments']}
            for node in actions] == g3['plan']
    assert [node['details']['order'] for node in actions] == [1, 2, 3, 4]
    assert [node['label'] for node in actions] == ['1. navigate', '2. pick', '3. navigate', '4. place']
    assert all(node['details']['status'] == 'not_attempted' for node in actions)
    assert 'planning' not in g2


def test_snapshot_builders_do_not_mutate_resolved_state(progression):
    (g1, _, g3), _, _ = progression
    original = deepcopy(g3)
    interpretation = interpretation_snapshot(g3, g1)
    assert interpretation['bindings']['Source'] is None
    assert g3 == original
    result = {'status': 'failed', 'plan': g3['plan'],
              'executed': [{'step': 1, 'status': 'failed'}], 'failed_step': 1}
    action = action_snapshot(result, g3)
    assert g3 == original
    assert action['bindings']['Source'] == 'KITCHEN'
    actions = [node for node in action['context_graph']['nodes'] if node['type'] == 'Action']
    assert len(actions) == 4
    assert actions[0]['status'] == 'failed'
    assert actions[1]['status'] == 'not_attempted'


def test_static_framenet_role_uris_exist(progression):
    from rdflib import Graph
    semantic = Graph().parse(view_kg.SEMANTIC_GRAPH, format='turtle')
    for node in progression[2][1]['nodes']:
        if node['details']['type'] in ('Bringing', 'FrameElement'):
            assert any(semantic.triples((URIRef(node['uri']), None, None)))


def test_arbitrary_copied_progressive_files_need_no_siblings(progression, tmp_path):
    snapshots, _, expected = progression
    for index, snapshot in enumerate(snapshots):
        path = tmp_path / f'custom_{index}.json'
        path.write_text(json.dumps(snapshot))
        actual = view_kg.graph_to_data(view_kg.load_graph(path))
        assert actual == expected[index]


@pytest.mark.parametrize('bad_sense', [None, [], {'scene_graph': {'objects': [None]}}])
def test_malformed_optional_embedded_g1_still_renders(progression, tmp_path, bad_sense):
    g2 = deepcopy(progression[0][1])
    g2['sense_graph'] = bad_sense
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(g2))
    payload = view_kg.graph_to_data(view_kg.load_graph(path))
    assert 'BringingFrame_1' in by_entity(payload)
    assert ('Source_FE', 'bindsTo', 'Unknown') in relations(payload)


def test_g1_preserves_context_grounding_without_expanding_neighborhoods():
    from graph_snapshots import episode_snapshot
    snapshot = episode_snapshot({'instruction': 'Bring me a fork', 'scene_graph': {
        'objects': [{'id': 'fork1', 'type': 'Fork'}], 'relations': []}})
    snapshot['context_graph']['relations'].extend([
        {'subject': 'observation_1', 'predicate': 'linkedTo', 'object': 'fork.n.01'},
        {'subject': 'fork.n.01', 'predicate': 'subClassOf', 'object': 'utensil.n.01'}])
    payload = view_kg.graph_to_data(view_kg.snapshot_to_graph(snapshot))
    assert ('observation_1', 'linkedTo', 'fork.n.01') in relations(payload)
    assert 'utensil.n.01' not in by_entity(payload)
    assert len(payload['nodes']) < 20


def test_instruction_is_one_literal_triple_in_every_stage(progression):
    from rdflib import Literal
    snapshots, _, _ = progression
    instructions = []
    payloads = []
    for original in snapshots:
        snapshot = deepcopy(original)
        snapshot['instruction'] = 'Bring me a book'
        if 'sense_graph' in snapshot:
            snapshot['sense_graph']['instruction'] = 'Bring me a book'
        graph = view_kg.snapshot_to_graph(snapshot)
        triples = list(graph.triples((None, URIRef('urn:cmoc:predicate:instruction'), None)))
        assert triples == [(URIRef('urn:cmoc:entity:observation_1'),
                            URIRef('urn:cmoc:predicate:instruction'), Literal('Bring me a book'))]
        payload = view_kg.graph_to_data(graph)
        payloads.append(payload)
        matches = [node for node in payload['nodes'] if node['details']['type'] == 'Instruction']
        assert len(matches) == 1
        node = matches[0]
        instructions.append(node)
        assert node['kind'] == 'literal'
        assert node['label'] == node['value'] == 'Bring me a book'
        assert set(node['details']) == {'id', 'type'}
        assert 'instruction_1' not in by_entity(payload)
    assert instructions[0] == instructions[1] == instructions[2]
    assert view_kg.check_visualization_sizes(*payloads) == [10, 16, 21]


def test_instruction_context_text_is_used_without_metadata(progression):
    snapshot = deepcopy(progression[0][0])
    snapshot.pop('instruction')
    instruction = next(n for n in snapshot['context_graph']['nodes'] if n['type'] == 'Instruction')
    instruction.update(id='original_request', text='Bring me a book',
                       tokens=['hidden'], embeddings=[0.1], prompt_history=['hidden'])
    payload = view_kg.graph_to_data(view_kg.snapshot_to_graph(snapshot))
    node = by_entity(payload)['Bring me a book']
    assert node['label'] == 'Bring me a book' and node['kind'] == 'literal'
    assert set(node['details']) == {'id', 'type'}
    assert 'hidden' not in json.dumps(payload)


def test_instruction_keeps_dense_g1_under_budget():
    from graph_snapshots import episode_snapshot
    snapshot = episode_snapshot({'instruction': 'Bring me a fork', 'scene_graph': {
        'objects': [{'id': f'object_{i}', 'type': 'Object'} for i in range(12)],
        'relations': [{'subject': 'object_0', 'predicate': 'on', 'object': f'object_{i}'}
                      for i in range(1, 12)]}})
    snapshot['entity_links'] = {f'object_{i}': f'concept_{i}' for i in range(12)}
    snapshot['context_graph']['relations'].append(
        {'subject': 'observation_1', 'predicate': 'linkedTo', 'object': 'observation_concept'})
    payload = view_kg.graph_to_data(view_kg.snapshot_to_graph(snapshot))
    assert len(payload['nodes']) < 20
    assert len([n for n in payload['nodes'] if n['details']['type'] == 'Instruction']) == 1
