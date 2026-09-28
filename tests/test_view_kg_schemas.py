"""Task schemas use the existing read-only tree viewer and live reload."""
import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from utils import view_kg


@pytest.mark.parametrize('watch', [False, True])
def test_schema_shortcut_and_cli(watch):
    directory = Path(view_kg.__file__).resolve().parents[1] / 'schemas/task_frames'
    assert view_kg.resolve_path('schemas') == directory
    with patch('sys.argv', ['view_kg.py', 'schemas'] + (['--watch'] if watch else [])), \
         patch.object(view_kg, 'run_server') as server:
        view_kg.main()
    server.assert_called_once_with(directory, None)


def test_actual_schemas_slots_metadata_and_layout():
    directory = view_kg.resolve_path('schemas')
    originals = {p: p.read_bytes() for p in directory.glob('*.json')}
    documents = view_kg.load_schema_documents(directory)
    assert {p.name for p, _ in documents} >= {'bring-11.3.json', 'search.json', 'search_result.json'}
    assert dict(documents) == {p: json.loads(raw) for p, raw in originals.items()}
    data = view_kg.graph_to_data(view_kg.load_graph(directory))
    assert data['layout'] == 'tree'
    nodes = {n['id']: n for n in data['nodes']}
    for path, schema in documents:
        frame = 'schema:' + path.name
        assert nodes[frame]['label'] == schema['title']
        for name, definition in schema['properties'].items():
            slot = nodes[frame + '/' + name]
            assert slot['label'] == name
            assert slot['parent'] == frame
            assert slot['details'] == {'property': name, 'required': name in schema['required'], **definition}
            expected = definition.get('type', 'unspecified')
            expected = ' | '.join(expected) if isinstance(expected, list) else expected
            assert expected in slot['badges'][0]
            assert ('required' if name in schema['required'] else 'optional') in slot['badges'][0]
            assert any(e['source'] == frame and e['target'] == slot['id'] for e in data['edges'])
            if 'description' in definition:
                assert definition['description'] not in slot['badges']
    for name in ('agent', 'theme', 'source', 'destination'):
        assert 'schema:bring-11.3.json/' + name in nodes
    for name in ('Agent', 'Theme', 'Location', 'Success'):
        assert 'schema:search.json/' + name in nodes
    for identifier in ('schema:search.json/Success', 'schema:search_result.json/success'):
        assert nodes[identifier]['details']['type'] == 'boolean'
        assert nodes[identifier]['family'] == 'Status'
        assert 'status' in nodes[identifier]['badges'][0]
    assert nodes['schema:search.json/verb']['badges'] == ['optional · unspecified', 'const: "search"']
    assert nodes['schema:search_result.json/observed_ids']['badges'] == ['optional · array']
    for i, a in enumerate(data['nodes']):
        for b in data['nodes'][i + 1:]:
            assert abs(a['x'] - b['x']) >= 260 or abs(a['y'] - b['y']) >= 76
    assert {p: p.read_bytes() for p in originals} == originals


def test_new_schema_and_enum_are_loaded_dynamically(tmp_path):
    schema = {'title': 'New frame', 'type': 'object', 'properties': {
        'mode': {'type': 'string', 'enum': ['a', 'b'], 'description': 'Mode detail'},
        'enabled': {'type': 'boolean'}}, 'required': ['mode']}
    path = tmp_path / 'new.json'
    path.write_text(json.dumps(schema))
    data = view_kg.graph_to_data(view_kg.load_graph(tmp_path))
    nodes = {n['label']: n for n in data['nodes']}
    assert nodes['mode']['badges'] == ['required · string', 'enum: ["a", "b"]']
    assert nodes['mode']['details']['description'] == 'Mode detail'
    assert nodes['enabled']['family'] == 'Slot'
    assert nodes['enabled']['badges'] == ['optional · boolean']


def test_schema_watch_edit_add_remove_and_invalid_recovery(tmp_path):
    path = tmp_path / 'frame.json'
    path.write_text(json.dumps({'title': 'First', 'properties': {}}))
    state = {'revision': 0, 'data': {}, 'error': None}
    seen = []

    class Polls:
        count = 0

        def is_set(self):
            return self.count == 6

        def wait(self, seconds):
            seen.append(dict(state))
            self.count += 1
            if self.count == 1:
                path.write_text(json.dumps({'title': 'Edited', 'properties': {}}))
            elif self.count == 2:
                (tmp_path / 'extra.json').write_text(json.dumps({'title': 'Added'}))
            elif self.count == 3:
                path.write_text('{')
            elif self.count == 4:
                path.write_text(json.dumps({'title': 'Recovered'}))
            elif self.count == 5:
                (tmp_path / 'extra.json').unlink()

    view_kg.watch_graph(tmp_path, None, state, threading.Lock(), Polls())
    labels = [{n['label'] for n in item['data']['nodes']} for item in seen]
    assert labels == [
        {'Schemas', 'First'}, {'Schemas', 'Edited'}, {'Schemas', 'Edited', 'Added'},
        {'Schemas', 'Edited', 'Added'}, {'Schemas', 'Recovered', 'Added'}, {'Schemas', 'Recovered'}]
    assert [item['revision'] for item in seen] == [1, 2, 3, 3, 4, 5]
    assert seen[3]['error']
    assert seen[4]['error'] is None


@pytest.mark.parametrize('stage', ['scene_graph', 'g1', 'g2', 'g3'])
def test_existing_shortcuts_keep_graph_mode(stage):
    examples = Path(view_kg.__file__).resolve().parents[1] / 'examples/visualization'
    data = view_kg.graph_to_data(view_kg.load_graph(examples / (stage + '.json')))
    assert 'layout' not in data
    assert data['nodes']
    with patch('sys.argv', ['view_kg.py', stage]), patch.object(view_kg, 'run_server') as server:
        view_kg.main()
    server.assert_called_once_with(view_kg.resolve_path(stage), None)
