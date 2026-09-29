"""G2's read-only overlay follows only explicitly stored grounding links."""
import json
from urllib.parse import quote

import pytest
from rdflib import Graph, URIRef, Literal

from utils import view_kg


def node(kind, value):
    return URIRef('urn:cmoc:' + kind + ':' + quote(value, safe=''))


@pytest.fixture
def snapshots(tmp_path, monkeypatch):
    g1 = {'scene_graph': {'objects': [
        {'id': 'fork1', 'type': 'Fork', 'qualities': {'simulator_metadata': 'excluded'}},
        {'id': 'unrelated', 'type': 'Table'}], 'relations': []},
        'context_graph': {'nodes': [{'id': 'observation_1', 'type': 'Observation', 'generatedByModel': 'excluded'}],
                          'relations': [
                              {'subject': 'observation_1', 'predicate': 'observes', 'object': 'fork1'},
                              {'subject': 'observation_1', 'predicate': 'observes', 'object': 'state_1'},
                              {'subject': 'observation_1', 'predicate': 'observes', 'object': 'unrelated'},
                              {'subject': 'observation_1', 'predicate': 'linkedTo', 'object': 'fork.n.01'}]},
        'entity_links': {'unrelated': 'table.n.02'}}
    g2 = {'frame': {'Theme': 'fork'}, 'context_graph': {
        'nodes': [{'id': 'fork1', 'type': 'Entity'}, {'id': 'state_1', 'type': 'State'}],
        'relations': [{'subject': 'state_1', 'predicate': 'bindsTo', 'object': 'fork1'}]}}
    p1, p2 = tmp_path / 'g1_sense.json', tmp_path / 'g2_plan.json'
    p1.write_text(json.dumps(g1))
    p2.write_text(json.dumps(g2))
    semantic = Graph()
    wn = URIRef(view_kg.WORDNET + 'fork.n.01')
    fn = URIRef(view_kg.FRAMENET + 'LU/123')
    denotes = URIRef('https://example.org/cmoc/robokgnet/denotesConcept')
    semantic.add((fn, denotes, wn))
    semantic.add((wn, URIRef('http://www.w3.org/2000/01/rdf-schema#subClassOf'), URIRef(view_kg.WORDNET + 'utensil.n.01')))
    semantic.add((fn, URIRef('https://example.org/cmoc/robokgnet/definition'), Literal('excluded')))
    semantic_path = tmp_path / 'semantic.ttl'
    semantic.serialize(semantic_path, format='turtle')
    monkeypatch.setattr(view_kg, 'SEMANTIC_GRAPH', semantic_path)
    return p1, p2, g1, g2, (fn, denotes, node('concept', 'fork.n.01'))


def test_g2_adds_exact_chain_without_unrelated_triples_or_file_changes(snapshots):
    p1, p2, g1, g2, semantic_link = snapshots
    before = {path: path.read_bytes() for path in (p1, p2)}
    baseline = view_kg.snapshot_to_graph(g2)
    graph = view_kg.load_graph(p2)
    observation = node('entity', 'observation_1')
    expected = {
        (observation, view_kg.OBSERVES, node('entity', 'fork1')),
        (observation, view_kg.OBSERVES, node('entity', 'state_1')),
        (observation, view_kg.LINKED_TO, node('concept', 'fork.n.01')),
        semantic_link,
    }
    assert set(graph) == set(baseline) | expected
    data = view_kg.graph_to_data(graph)
    assert any(n['label'] == 'G1 observation: observation_1' for n in data['nodes'])
    assert any(n['label'] == 'WordNet concept: fork.n.01' for n in data['nodes'])
    assert any(n['label'].startswith('FrameNet entity:') for n in data['nodes'])
    assert 'unrelated' not in json.dumps(data) and 'excluded' not in json.dumps(data)
    assert {path: path.read_bytes() for path in before} == before
    assert set(view_kg.load_graph(p1)) == set(view_kg.snapshot_to_graph(g1))


@pytest.mark.parametrize('missing', ['g1', 'observation', 'wordnet', 'framenet'])
def test_missing_links_are_not_fabricated(snapshots, missing):
    p1, p2, g1, g2, semantic_link = snapshots
    if missing == 'g1':
        p1.unlink()
    elif missing == 'observation':
        g1['context_graph']['relations'] = [r for r in g1['context_graph']['relations'] if r['predicate'] != 'observes']
        p1.write_text(json.dumps(g1))
    elif missing == 'wordnet':
        g1['context_graph']['relations'] = [r for r in g1['context_graph']['relations'] if r['predicate'] != 'linkedTo']
        p1.write_text(json.dumps(g1))
    else:
        view_kg.SEMANTIC_GRAPH.unlink()
    graph = view_kg.load_graph(p2)
    assert semantic_link not in graph
    if missing in ('g1', 'observation'):
        assert set(graph) == set(view_kg.snapshot_to_graph(g2))
    elif missing == 'wordnet':
        assert not list(graph.triples((None, view_kg.LINKED_TO, None)))
    else:
        assert list(graph.triples((None, view_kg.LINKED_TO, None)))


def test_mismatched_observation_is_not_joined(snapshots):
    p1, p2, g1, g2, _ = snapshots
    g2['context_graph']['nodes'].append({'id': 'observation_2', 'type': 'Observation'})
    p2.write_text(json.dumps(g2))
    assert set(view_kg.load_graph(p2)) == set(view_kg.snapshot_to_graph(g2))


def test_g1_changes_invalidate_g2_watch_stamp(snapshots):
    p1, p2, g1, _, _ = snapshots
    before = view_kg.graph_stamp(p2)
    g1['context_graph']['relations'].clear()
    p1.write_text(json.dumps(g1))
    assert view_kg.graph_stamp(p2) != before


def test_compact_search_g2_loads_without_fabricated_provenance(tmp_path):
    snapshot = {'type': 'search', 'frame': {'Theme': 'fork', 'Location': None, 'GazeAction': 'look-left'},
                'planning': {'status': 'planned', 'plan': []}}
    path = tmp_path / 'g2_plan.json'
    original = json.dumps(snapshot).encode()
    path.write_bytes(original)
    assert set(view_kg.load_graph(path)) == set(view_kg.snapshot_to_graph(snapshot))
    assert path.read_bytes() == original
