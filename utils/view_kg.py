"""Standalone scene graph viewer: python utils/view_kg.py <scene_graph.json>."""

import argparse
import hashlib
import json
import threading
import webbrowser
from pathlib import Path
from urllib.parse import unquote, quote
from wsgiref.simple_server import make_server

from rdflib import BNode, Graph, Literal, RDF, URIRef
from rdflib.util import guess_format


ARTIFACT_DIRECTORY = Path(__file__).resolve().parents[1] / "episodic_memory"
SCHEMA_DIRECTORY = Path(__file__).resolve().parents[1] / "schemas/task_frames"
SNAPSHOTS = {"g1": "g1_sense.json", "g2": "g2_plan.json", "g3": "g3_action.json"}


def resolve_path(value):
    if value == "schemas":
        return SCHEMA_DIRECTORY
    if value == "scene_graph":
        return ARTIFACT_DIRECTORY / "scene_graph.json"
    return (ARTIFACT_DIRECTORY / SNAPSHOTS[value] if value in SNAPSHOTS
            else Path(value).expanduser().resolve())


def missing_message(path):
    return f"Scene graph file not found: {path}"


def is_bringing(data):
    return isinstance(data.get('frame'), dict) and 'Source' in data['frame']


def is_observation(data):
    return 'frame' not in data and 'scene_graph' in data and any(
        item.get('type') == 'Observation' for item in data.get('context_graph', {}).get('nodes', []))


def explanatory_graph(data, sense=None):
    """A bounded, read-only explanation, with properties in details rather than nodes.

    Selection is based only on stored observations, aliases, bindings and plans.
    The raw scene and its omitted entities remain intact in the input JSON.
    """
    stage = 'G3' if 'plan' in data else 'G2' if is_bringing(data) else 'G1'
    sense = sense or data.get('sense_graph') or data
    scene = sense.get('scene_graph', {})
    objects = {item['id']: item for item in scene.get('objects', [])}
    context = sense.get('context_graph', {})
    observation = next((item for item in context.get('nodes', [])
                        if item.get('type') == 'Observation'), None)
    instruction_record = next((item for item in context.get('nodes', [])
                               if item.get('type') == 'Instruction'), {})
    instruction_text = sense.get('instruction', instruction_record.get('text', data.get('instruction')))
    has_instruction = isinstance(instruction_text, str)
    links = sense.get('entity_links', {})
    nodes, relations = {}, set()

    def add(identifier, kind='Entity', **details):
        nodes.setdefault(identifier, {'id': identifier, 'type': kind}).update(details)
        return identifier

    def edge(subject, predicate, target):
        relations.add((subject, predicate, target))

    # Keep people, the observer and a small sample of perceptual objects, then
    # their immediate spatial endpoints. Never select by simulator memory.
    people = {'robot', 'tiago', 'person', 'pedestrian', 'mannequin', 'user'}
    locations = {'location', 'room'}
    ambient = {'cabinet', 'wall', 'floor', 'ceiling', 'door', 'window'}
    def priority(item):
        kind = item.get('type', '').casefold().removesuffix('connector')
        return (0 if kind in people else 3 if kind in ambient else 2 if kind in locations else 1,
                item['id'])
    selected = [item['id'] for item in sorted(objects.values(), key=priority)
                if priority(item)[0] < 3][:6]
    for relation in scene.get('relations', []):
        if (relation['subject'] in selected and relation['object'] in objects
                and relation['object'] not in selected and len(selected) < (7 if has_instruction else 8)):
            selected.append(relation['object'])
    for identifier in selected:
        item = objects[identifier]
        add(identifier, item.get('type', 'Entity'), qualities=item.get('qualities', {}))
    if observation:
        obs = add(observation['id'], 'Observation', **{k: v for k, v in observation.items()
                                                     if k not in ('id', 'type')})
        nodes[obs]['omitted_scene_entities'] = len(objects) - len(selected)
        # Reuse the observed robot for the existing generic observer identity.
        robots = [key for key in selected if objects[key].get('type', '').casefold() in ('robot', 'tiago')]
        observer = next((r['object'] for r in context.get('relations', [])
                         if r['subject'] == obs and r['predicate'] == 'observedBy'), None)
        if observer:
            observer = robots[0] if observer == 'robot' and len(robots) == 1 else observer
            add(observer, 'Robot')
            edge(obs, 'observedBy', observer)
        observed = {r['object'] for r in context.get('relations', [])
                    if r['subject'] == obs and r['predicate'] == 'observes'}
        for identifier in selected:
            if identifier in observed:
                edge(obs, 'observes', identifier)
    for relation in scene.get('relations', []):
        if relation['subject'] in nodes and relation['object'] in nodes:
            edge(relation['subject'], relation['predicate'], relation['object'])
    # At most one stored lexical concept per selected entity; no semantic crawl.
    for identifier in selected:
        if links.get(identifier):
            concept = add(links[identifier], 'Concept')
            edge(identifier, 'linkedTo', concept)
    # Older snapshots can store the same grounding in context relations.
    grounded = {subject for subject, predicate, _ in relations if predicate == 'linkedTo'}
    eligible = set(selected) | ({observation['id']} if observation else set())
    for relation in context.get('relations', []):
        source = relation['subject']
        if relation['predicate'] == 'linkedTo' and source in eligible and source not in grounded:
            edge(source, 'linkedTo', add(relation['object'], 'Concept'))
            grounded.add(source)
    base_ids = set(nodes)

    # Explicit aliases map task identities back onto the same observation nodes.
    aliases = data.get('aliases', {})
    def identity(value):
        if value in nodes:
            return value
        observed_aliases = [alias for alias in aliases.get(value, []) if alias in base_ids]
        if len(observed_aliases) == 1:
            return observed_aliases[0]
        return value

    if stage != 'G1':
        frame = add('BringingFrame_1', 'Bringing', display_label='Bringing')
        if observation:
            edge(observation['id'], 'hasFrame', frame)
        bindings = data.get('bindings', {})
        for role in ('Agent', 'Theme', 'Destination', 'Source'):
            role_id = add(role + '_FE', 'FrameElement', role=role,
                          semanticValue=data['frame'].get(role), display_label=role)
            edge(frame, 'hasFrameElement', role_id)
            target = bindings.get(role)
            concept = data.get('frame_element_links', {}).get(role)
            if role == 'Source' and stage == 'G3':
                concept = None  # Resolution adds a binding, not a semantic neighborhood.
            if concept is None and target is not None and role != 'Source':
                concept = data.get('entity_links', {}).get(target)
            if target is not None:
                target = identity(target)
                add(target)
                edge(role_id, 'bindsTo', target)
            elif role == 'Source':
                edge(role_id, 'bindsTo', add('Unknown', 'Unknown'))
                nodes[role_id]['display_label'] = 'Source = Unknown'
            elif role == 'Theme' and concept is None:
                # A requested value is a concept, not a fabricated perceived item.
                requested = data['frame'].get(role)
                if requested:
                    edge(role_id, 'semanticValue', add('requested:' + requested, 'Concept', display_label=requested))
            if concept:
                add(concept, 'Concept')
                edge(role_id, 'linkedTo', concept)
                if target is not None:
                    stored = data.get('entity_links', {}).get(bindings.get(role))
                    if stored == concept:
                        edge(target, 'linkedTo', concept)
        # Source resolution should not introduce a fresh semantic neighborhood.
        # G3 inherits the G2 lexical links; Source's concrete binding is sufficient.
        if stage == 'G3':
            plan = add('Plan_1', 'Plan', status=data.get('status'))
            edge(frame, 'hasPlan', plan)
            attempts = {item['step']: item['status'] for item in data.get('executed', [])}
            for order, step in enumerate(data.get('plan', []), 1):
                action = add(f"{step['action']}_{order}", 'Action', order=order,
                             action=step['action'], arguments=step['args'],
                             status=attempts.get(order, 'not_attempted'),
                             display_label=f"{order}. {step['action']}")
                edge(plan, 'hasAction', action)
                for position, argument in enumerate(step['args'], 1):
                    target = identity(argument)
                    add(target)
                    edge(action, f'argument{position}', target)

    graph = Graph()
    def term(identifier):
        if nodes[identifier]['type'] == 'FrameElement':
            role = nodes[identifier]['role']
            # CMOC calls FrameNet's Goal role Destination.
            return URIRef(FRAMENET + quote('Bringing/FE/' + ('Goal' if role == 'Destination' else role), safe=''))
        if nodes[identifier]['type'] == 'Bringing':
            return URIRef(FRAMENET + 'Bringing')
        return URIRef('urn:cmoc:' + ('concept' if nodes[identifier]['type'] == 'Concept' else 'entity')
                      + ':' + quote(str(identifier), safe=''))
    graph.node_details = {term(key): value for key, value in nodes.items()}
    for subject, predicate, target in relations:
        graph.add((term(subject), URIRef('urn:cmoc:predicate:' + quote(predicate, safe='')), term(target)))
    if has_instruction:
        subject = observation['id'] if observation else ('BringingFrame_1' if stage != 'G1' else None)
        if subject is not None:
            literal = Literal(instruction_text)
            graph.add((term(subject), URIRef('urn:cmoc:predicate:instruction'), literal))
            graph.node_details[literal] = {'id': instruction_text, 'type': 'Instruction'}
    graph.explanatory = True
    graph.stage = stage
    return graph


def check_visualization_sizes(g1, g2, g3):
    """Strict regression guard on the actual visible payloads, not raw JSON sizes."""
    counts = [len(item['nodes']) for item in (g1, g2, g3)]
    limits = [20, 2 * counts[0], 2 * counts[0] + 5]
    for stage, count, limit in zip(('G1', 'G2', 'G3'), counts, limits):
        if count >= limit:
            raise ValueError(f'{stage} visualization too large: {count} nodes (must be < {limit})')
    return counts


def snapshot_to_graph(data, sense=None, *, compact=True):
    """Visualization-only RDF adapter; never change the stored JSON representation."""
    if compact and (is_bringing(data) or is_observation(data)):
        return explanatory_graph(data, sense)
    graph = Graph()
    def node(kind, value):
        return URIRef("urn:cmoc:" + kind + ":" + quote(str(value), safe=""))
    def edge(subject, label, target):
        graph.add((subject, node("predicate", label), target))
    context = data.get('context_graph')
    if context and 'frame' in data:
        # G2's small context is authoritative; never expand scene_graph here.
        # Types/provenance are node details, not extra hubs or literal nodes.
        concepts = {item['id'] for item in context['nodes'] if item['type'] == 'Concept'}
        def context_node(identifier):
            return node('concept' if identifier in concepts else 'entity', identifier)
        graph.node_details = {context_node(item['id']): item for item in context['nodes']}
        for relation in context['relations']:
            edge(context_node(relation['subject']), relation['predicate'], context_node(relation['object']))
        for item in context['nodes']:
            if item.get('semanticValue') is not None:
                edge(context_node(item['id']), 'semanticValue', Literal(item['semanticValue']))
        return graph
    scene = data.get("scene_graph", data)
    if "objects" in data and "scene_graph" not in data and "context_graph" not in data:
        graph.scene_data = data
    for obj in scene.get("objects", []):
        entity = node("entity", obj["id"])
        graph.add((entity, RDF.type, node("type", obj.get("type", "Entity"))))
        for key, value in obj.get("qualities", {}).items():
            edge(entity, key, Literal(json.dumps(value, ensure_ascii=False)))
    for relation in scene.get("relations", []):
        edge(node("entity", relation["subject"]), relation["predicate"], node("entity", relation["object"]))
    for identifier, concept in data.get("entity_links", {}).items():
        edge(node("entity", identifier), "linkedTo", node("concept", concept))
    context = data.get('context_graph')
    if context:
        for item in context['nodes']:
            entity = node('entity', item['id'])
            graph.add((entity, RDF.type, node('type', item['type'])))
            for key, value in item.items():
                if key not in ('id', 'type') and value is not None:
                    edge(entity, key, Literal(value))
        for relation in context['relations']:
            edge(node('entity', relation['subject']), relation['predicate'],
                 node('concept' if relation['predicate'] == 'linkedTo' else 'entity', relation['object']))
        for canonical, aliases in data.get('aliases', {}).items():
            edge(node('entity', canonical), 'aliases', Literal(', '.join(aliases)))
        for key in ('type', 'issues'):
            if key in data:
                edge(node('entity', 'episode_1'), key, Literal(json.dumps(data[key])))
    else:
        snapshot = node("snapshot", "Snapshot")
        for key in ("instruction", "type", "issues"):
            if key in data:
                edge(snapshot, key, Literal(data[key] if isinstance(data[key], str) else json.dumps(data[key])))
        if "frame" in data:
            frame = node("frame", "Bringing")
            edge(snapshot, "frame", frame)
            for role, value in data["frame"].items():
                edge(frame, role, Literal(value if value is not None else "unbound"))
            for role, value in data.get("bindings", {}).items():
                if value is not None:
                    edge(frame, "bound" + role, node("entity", value))
            if data.get("theme_concept"):
                edge(frame, "themeConcept", node("concept", data["theme_concept"]))
    return graph


SEMANTIC_GRAPH = Path(__file__).resolve().parents[1] / 'knowledge/robokgnet/robokgnet.ttl'
WORDNET = 'https://example.org/cmoc/robokgnet/wordnet/'
FRAMENET = 'https://example.org/cmoc/robokgnet/framenet/'
OBSERVES = URIRef('urn:cmoc:predicate:observes')
LINKED_TO = URIRef('urn:cmoc:predicate:linkedTo')


def g1_path_for_g2(path):
    if path.name == SNAPSHOTS['g2']:
        return path.with_name(SNAPSHOTS['g1'])
    if path.name in ('g2.json', 'g3.json'):
        return path.with_name('g1.json')
    return path.with_name(SNAPSHOTS['g1'])


def add_g1_provenance(graph, g1, semantic_path=None):
    """Copy only explicit, one-hop provenance links; preserve predicate direction."""
    roots = set(graph.subjects()) | set(graph.objects()) | set(getattr(graph, 'node_details', {}))
    expected = {node for node, details in getattr(graph, 'node_details', {}).items()
                if details.get('type') == 'Observation'}
    sources = set()
    stages = {}
    for subject, _, entity in g1.triples((None, OBSERVES, None)):
        if entity in roots and (not expected or subject in expected):
            graph.add((subject, OBSERVES, entity))
            sources.update((subject, entity))
            stages[subject] = 'G1 observation'
    concepts = set()
    for source in sources:
        for origin in (g1, graph):
            for concept in list(origin.objects(source, LINKED_TO)):
                graph.add((source, LINKED_TO, concept))
                concepts.add(concept)
                stages[concept] = 'WordNet concept'
    semantic_path = SEMANTIC_GRAPH if semantic_path is None else semantic_path
    if concepts and semantic_path.exists():
        try:
            semantic = Graph().parse(semantic_path, format='turtle')
        except Exception:
            # The optional static mapping must not prevent scene inspection.
            semantic = Graph()
        # Snapshot concepts use synset IDs; the RDF resource uses the same IDs
        # under its WordNet namespace. Keep the viewer's existing node identity.
        for concept in concepts:
            wn = (URIRef(WORDNET + quote(local_name(concept), safe=''))
                  if str(concept).startswith('urn:cmoc:concept:') else concept)
            if not str(wn).startswith(WORDNET):
                continue
            for subject, predicate, target in set(semantic.triples((wn, None, None))) | set(semantic.triples((None, None, wn))):
                other = target if subject == wn else subject
                if isinstance(other, URIRef) and str(other).startswith(FRAMENET):
                    graph.add((concept if subject == wn else subject, predicate,
                               concept if target == wn else target))
                    stages[other] = 'FrameNet entity'
    graph.provenance_stages = stages
    return graph


def load_schema_documents(path):
    """Load the actual task-frame documents, in deterministic file order."""
    files = sorted(path.glob('*.json'))
    if not files:
        raise ValueError(f'No JSON schemas found in {path}')
    return [(file, json.loads(file.read_text())) for file in files]


# Load a complete snapshot before replacing anything the browser can see.
def read_scene_json(path):
    """Reject invalid JSON/scene shapes without falling through to an RDF parser."""
    try:
        data = json.loads(path.read_bytes())
        if not isinstance(data, dict) or not any(
                key in data for key in ('scene_graph', 'objects', 'context_graph', 'frame')):
            raise ValueError('expected a scene graph object')
        return data
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"Invalid scene graph JSON: {path}") from error


def scene_json_to_graph(data, path, *, compact=True):
    try:
        return snapshot_to_graph(data, compact=compact)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError(f"Invalid scene graph JSON: {path}") from error


def load_graph(path, rdf_format=None, *, g1_path=None):
    path = Path(path).expanduser().resolve()
    if path.is_dir() and rdf_format is None:
        graph = Graph()
        graph.schema_documents = load_schema_documents(path)
        return graph
    if path.suffix.lower() == ".json" and rdf_format is None:
        data = read_scene_json(path)
        # Optional embedded provenance must not prevent rendering the main file.
        graph = scene_json_to_graph({key: value for key, value in data.items()
                                     if key != 'sense_graph'}, path)
        # A frame identifies G2/G3 even when the file has been renamed or moved.
        if 'frame' in data:
            try:
                if g1_path is not None:
                    provenance_path = Path(g1_path).expanduser().resolve()
                    g1_data = read_scene_json(provenance_path)
                elif isinstance(data.get('sense_graph'), dict):
                    provenance_path, g1_data = path, data['sense_graph']
                else:
                    provenance_path = g1_path_for_g2(path)
                    g1_data = read_scene_json(provenance_path)
                g1 = scene_json_to_graph(g1_data, provenance_path, compact=False)
            except (OSError, ValueError):
                pass  # Missing or invalid optional provenance leaves G2 intact.
            else:
                if getattr(graph, 'explanatory', False):
                    graph = scene_json_to_graph({**data, 'sense_graph': g1_data}, path)
                else:
                    add_g1_provenance(graph, g1)
        return graph
    graph = Graph()
    graph.parse(data=path.read_bytes(), format=rdf_format or guess_format(str(path)) or "turtle",
                publicID=path.as_uri())
    return graph


def local_name(term):
    if isinstance(term, BNode):
        return "_:" + str(term)
    value = str(term)
    return unquote(value.rstrip("/#").rsplit("#", 1)[-1].rsplit("/", 1)[-1].rsplit(":", 1)[-1]) or value


def type_color(type_uri):
    # Hash the URI, not its position in the graph, so colors survive reloads.
    hue = int(hashlib.sha256(str(type_uri).encode()).hexdigest()[:8], 16) % 360
    return f"hsl({hue}, 55%, 65%)"


def graph_to_data(graph):
    if hasattr(graph, "schema_documents"):
        return schemas_to_data(graph.schema_documents)
    node_details = getattr(graph, 'node_details', {})
    stages = getattr(graph, 'provenance_stages', {})
    terms = sorted(set(graph.subjects()) | set(graph.objects()) | set(node_details), key=lambda term: term.n3())
    identifiers = {term: term.n3() for term in terms}
    nodes = []
    visible_types = set()
    for term in terms:
        types = sorted(graph.objects(term, RDF.type), key=str) if not isinstance(term, Literal) else []
        visible_types.update(types)
        kind = "literal" if isinstance(term, Literal) else "blank" if isinstance(term, BNode) else "resource"
        label = node_details.get(term, {}).get("display_label") or (str(term) if kind == "literal" else local_name(term))
        nodes.append({
            "details": {**node_details.get(term, {}), **({"provenance_stage": stages[term]} if term in stages else {})},
            "id": identifiers[term], "label": (stages[term] + ": " if term in stages else "") + label[:60] + ("…" if len(label) > 60 else ""),
            "kind": kind, "uri": str(term) if kind == "resource" else None,
            "types": [str(item) for item in types],
            "value": str(term) if kind == "literal" else None,
            "language": term.language if kind == "literal" else None,
            "datatype": str(term.datatype) if kind == "literal" and term.datatype else None,
            "color": "#d9dee5" if kind == "literal" else type_color(types[0]) if types else
                     type_color(node_details[term]["type"]) if term in node_details else "#b4c4d6",
        })
    edges = [{"source": identifiers[subject], "target": identifiers[obj],
              "label": local_name(predicate), "uri": str(predicate)}
             for subject, predicate, obj in sorted(graph, key=lambda triple: tuple(term.n3() for term in triple))]
    legend = [{"label": local_name(item), "uri": str(item), "color": type_color(item)}
              for item in sorted(visible_types, key=str)]
    return {"nodes": nodes, "edges": edges, "legend": legend,
            **({"stage": graph.stage} if hasattr(graph, "stage") else {})}


# Display-only family groups and hierarchy; no ontology or scene facts are changed.
FAMILY_HUES = {'Room / Location': 210, 'Furniture': 30, 'Container': 270,
               'Surface': 165, 'Object': 50, 'Robot': 195, 'Person': 335}
PARENT_PRIORITY = {'in': 0, 'on_top': 1, 'on': 1, 'attached': 2,
                   'attached_to': 2, 'draped': 3, 'draped_over': 3, 'under': 4}


def class_family(kind):
    kind = kind.casefold().removesuffix('connector')
    if kind in ('room', 'location'):
        return 'Room / Location'
    if any(word in kind for word in ('robot', 'tiago')):
        return 'Robot'
    if kind in ('person', 'pedestrian', 'mannequin', 'human'):
        return 'Person'
    if any(word in kind for word in ('cabinet', 'drawer', 'box', 'fridge', 'container', 'bowl')):
        return 'Container'
    if any(word in kind for word in ('table', 'worktop', 'counter', 'shelf', 'desk')):
        return 'Surface'
    if any(word in kind for word in ('chair', 'sofa', 'bed', 'wardrobe', 'dresser')):
        return 'Furniture'
    return 'Object'


def family_color(family, identifier='', category=False):
    tint = 42 if category else 76 + int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16) % 10
    return f'hsl({FAMILY_HUES[family]}, 55%, {tint}%)'


def scene_tree_data(graph):
    """One parent per instance, cycle-safe, using only explicit scene relations.

    Non-room containment precedes support/attachment; room membership is the
    fallback. Ties use IDs, not input order. All unused relations are cross-links.
    """
    scene = getattr(graph, 'scene_data', None)
    if scene is None:
        raise ValueError('Tree mode requires a raw scene graph with objects and relations')
    objects = {obj['id']: obj for obj in scene['objects']}
    relations = sorted(scene.get('relations', []),
                       key=lambda r: (r['subject'], r['predicate'], r['object']))
    # Keep dangling relation endpoints visible without fabricating scene facts.
    for r in relations:
        for endpoint in ('subject', 'object'):
            objects.setdefault(r[endpoint], {'id': r[endpoint]})
    rooms = {key for key, obj in objects.items() if class_family(obj.get('type', '')) == 'Room / Location'}
    parents, chosen = {}, {}
    for identifier in sorted(objects):
        if identifier in rooms:
            continue
        candidates = [i for i, r in enumerate(relations)
                      if r['subject'] == identifier and r['predicate'] in PARENT_PRIORITY
                      and r['object'] != identifier]
        candidates.sort(key=lambda i: (5 if relations[i]['object'] in rooms else
                                       PARENT_PRIORITY[relations[i]['predicate']],
                                       relations[i]['object'], relations[i]['predicate']))
        for i in candidates:
            parent = relations[i]['object']
            ancestor = parent
            while ancestor in parents and ancestor != identifier:
                ancestor = parents[ancestor]
            if ancestor != identifier:
                parents[identifier], chosen[identifier] = parent, i
                break

    def entity_id(identifier):
        return 'entity:' + identifier
    root, unassigned = 'display:World', 'display:Unassigned'
    nodes = [{'id': root, 'label': 'World', 'color': '#42566e', 'details': {}, 'badges': []}]
    orphaned = set(objects) - rooms - parents.keys()
    if orphaned:
        nodes.append({'id': unassigned, 'label': 'Unassigned', 'parent': root,
                      'color': '#64748b', 'details': {}, 'badges': []})
    edges = []
    for identifier, obj in sorted(objects.items()):
        family = class_family(obj.get('type', ''))
        parent = (entity_id(parents[identifier]) if identifier in parents else
                  root if identifier in rooms else unassigned)
        badges = ['type: ' + obj['type']] if 'type' in obj else []
        qualities = obj.get('qualities', {})
        for key in ('position', 'location', 'size', 'orientation', 'graspability'):
            if key in qualities:
                badges.append(('position' if key == 'location' else key) + ': ' + json.dumps(qualities[key], ensure_ascii=False))
                break
        nodes.append({'id': entity_id(identifier), 'label': identifier, 'parent': parent,
                      'family': family, 'color': family_color(family, identifier),
                      'details': obj, 'badges': badges})
        relation = relations[chosen[identifier]] if identifier in chosen else None
        edges.append({'source': parent, 'target': entity_id(identifier), 'kind': 'hierarchy',
                      'label': relation['predicate'] if relation else '',
                      'relation': relation})
    if orphaned:
        edges.append({'source': root, 'target': unassigned, 'kind': 'hierarchy', 'label': ''})
    used = set(chosen.values())
    for i, r in enumerate(relations):
        if i not in used:
            edges.append({'source': entity_id(r['subject']), 'target': entity_id(r['object']),
                          'label': r['predicate'], 'kind': 'secondary', 'relation': r})

    layout_tree(nodes, root)
    families = sorted({node['family'] for node in nodes if 'family' in node})
    return {'layout': 'tree', 'nodes': nodes, 'edges': edges,
            'legend': [{'label': family, 'color': family_color(family, category=True)} for family in families]}


def layout_tree(nodes, root):
    # Fixed-width cards in preorder rows; disjoint subtrees and room spacing.
    by_id = {n['id']: n for n in nodes}
    children = {n['id']: [] for n in nodes}
    for n in nodes:
        if 'parent' in n:
            children[n['parent']].append(n['id'])
    children[root].sort(key=lambda identifier: (identifier == 'display:Unassigned', identifier))
    row = 0
    def place(identifier, depth):
        nonlocal row
        node = by_id[identifier]
        node['x'], node['y'] = depth * 340, row * 100
        row += 1
        for child in children[identifier]:
            place(child, depth + 1)
            if identifier == root:
                row += 0.6
    place(root, 0)
    for node in nodes:
        node.update(kind='resource', uri=None, types=[], value=None, language=None, datatype=None)


def schemas_to_data(documents):
    """Read slot names and constraints from schema files, never copied definitions."""
    root = 'display:Schemas'
    nodes = [{'id': root, 'label': 'Schemas', 'details': {}, 'badges': [], 'color': '#42566e'}]
    edges = []
    for path, schema in documents:
        identifier = 'schema:' + path.name
        nodes.append({'id': identifier, 'parent': root, 'label': schema.get('title', path.stem),
                      'family': 'Schema', 'color': family_color('Object', category=True),
                      'details': {'file': str(path), **schema},
                      'badges': [path.name, 'type: ' + str(schema.get('type', 'unspecified'))]})
        edges.append({'source': root, 'target': identifier, 'kind': 'hierarchy', 'label': ''})
        required = schema.get('required', [])
        for name, definition in schema.get('properties', {}).items():
            slot = identifier + '/' + name
            expected = definition.get('type', 'unspecified')
            expected = ' | '.join(expected) if isinstance(expected, list) else expected
            requirement = 'required' if name in required else 'optional'
            badges = [requirement + ' · ' + expected]
            if 'const' in definition:
                badges.append('const: ' + json.dumps(definition['const']))
            if 'enum' in definition:
                badges.append('enum: ' + json.dumps(definition['enum']))
            # Only explicit status wording or a boolean success field is classified.
            is_status = definition.get('type') == 'boolean' and (
                name.casefold() == 'success' or
                'status' in definition.get('description', '').casefold())
            if is_status:
                badges[0] += ' · status'
            nodes.append({'id': slot, 'parent': identifier, 'label': name,
                          'family': 'Status' if is_status else 'Slot',
                          'color': family_color('Object', name),
                          'badges': badges, 'details': {'property': name,
                          'required': name in required, **definition}})
            edges.append({'source': identifier, 'target': slot, 'kind': 'hierarchy', 'label': requirement})
    layout_tree(nodes, root)
    return {'layout': 'tree', 'nodes': nodes, 'edges': edges, 'legend': [
        {'label': 'Schema', 'color': family_color('Object', category=True)},
        {'label': 'Slot', 'color': family_color('Object', 'slot')}]}


def graph_stamp(path, g1_path=None):
    """Schema directories also reload when an existing JSON file is edited."""
    paths = [path, *sorted(path.glob('*.json'))] if path.is_dir() else [path]
    g1_path = Path(g1_path) if g1_path is not None else g1_path_for_g2(path)
    if g1_path is not None:
        paths.extend(item for item in (g1_path, SEMANTIC_GRAPH) if item.exists())
    return tuple((str(item), stat.st_mtime_ns, stat.st_size, stat.st_ino)
                 for item in paths for stat in [item.stat()])


# Everything the browser needs lives here; RDF text is inserted only as textContent.
def build_html():
    return r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Knowledge graph</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f7f9fc;color:#263349;font:14px system-ui,sans-serif}
header{position:absolute;top:22px;left:26px;pointer-events:none}h1{font-size:18px;margin:0 0 6px}
#status{color:#69778b;font-size:12px;max-width:65vw}svg{width:100vw;height:100vh;display:block;touch-action:none}
#legend{position:absolute;bottom:20px;left:26px;max-width:65vw;max-height:25vh;overflow:auto;display:flex;gap:8px 16px;flex-wrap:wrap;font-size:12px}
.key{display:flex;align-items:center;gap:6px}.swatch{width:11px;height:11px;border-radius:50%;display:inline-block}
#details{position:absolute;right:20px;top:20px;width:290px;max-width:40vw;max-height:80vh;overflow:auto;background:#ffffffed;border:1px solid #e1e7ef;border-radius:12px;padding:16px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;line-height:1.6}
#details:empty{display:none}.edge{fill:none;stroke:#a6b1c0;stroke-width:1.2}.edge-label{font-size:10px;fill:#64748b;paint-order:stroke;stroke:#f7f9fc;stroke-width:4px;stroke-linejoin:round}
.node{cursor:grab}.node text{font-size:11px;fill:#263349;paint-order:stroke;stroke:#f7f9fc;stroke-width:3px;pointer-events:none}.node.selected .shape{stroke:#243b60;stroke-width:3}
.tree .edge{stroke:#7890a8;stroke-width:1.5}.tree .edge.secondary{stroke:#a6b1c0;stroke-dasharray:4 5;opacity:.22;stroke-width:1}
.tree .edge-label.secondary{opacity:0}.tree .edge.active{opacity:1;stroke:#345c8c;stroke-width:2}.tree .edge-label.active{opacity:1}
.tree .node.related .shape{stroke:#345c8c;stroke-width:2}.tree .node text{stroke:none}.tree .node .badge{font-size:10px;fill:#43546a}
.tree .node.synthetic text{fill:white}.tree .node{cursor:pointer}
</style>
<header><h1>Knowledge graph</h1><div id="status">Loading…</div></header>
<svg id="canvas" aria-label="Interactive RDF graph"><defs><marker id="arrow" viewBox="0 -4 8 8" refX="8" refY="0" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,-4L8,0L0,4" fill="#a6b1c0"/></marker></defs><g id="world"><g id="edges"></g><g id="nodes"></g></g></svg>
<div id="legend"></div><aside id="details"></aside>
<script>
const canvas=document.querySelector('#canvas'), world=document.querySelector('#world');
const nodeLayer=document.querySelector('#nodes'), edgeLayer=document.querySelector('#edges');
const details=document.querySelector('#details'), status=document.querySelector('#status');
let nodes=[], edges=[], revision=-1, selected=null, drag=null, energy=0, treeMode=false;
let view={x:innerWidth/2,y:innerHeight/2,scale:1};
function element(tag,attributes,parent){
  const item=document.createElementNS('http://www.w3.org/2000/svg',tag);
  for(const [key,value] of Object.entries(attributes)) item.setAttribute(key,value);
  parent.append(item); return item;
}
function transform(){world.setAttribute('transform',`translate(${view.x},${view.y}) scale(${view.scale})`)}
function describe(node){
  details.textContent=[treeMode?node.label:node.kind==='blank'?'Blank node: '+node.id:node.uri || 'Literal',
    node.types.length?'Types:\n'+node.types.join('\n'):'',
    node.value!==null?'Value: '+node.value:'',node.language?'Language: '+node.language:'',
    node.datatype?'Datatype: '+node.datatype:'',
    node.details && Object.keys(node.details).length ? JSON.stringify(node.details,null,2) : ''].filter(Boolean).join('\n\n');
}
function highlight(){
  if(!treeMode)return;
  const related=new Set();
  for(const edge of edges){
    const active=selected!==null && (edge.source.id===selected || edge.target.id===selected);
    edge.path.classList.toggle('active',active);edge.text.classList.toggle('active',active);
    if(active && edge.kind==='hierarchy'){related.add(edge.source.id);related.add(edge.target.id)}
  }
  for(const node of nodes)node.group.classList.toggle('related',related.has(node.id));
}
function render(data){
  const enteringTree=data.layout==='tree' && !treeMode;
  treeMode=data.layout==='tree';canvas.classList.toggle('tree',treeMode);
  const previous=new Map(nodes.map(node=>[node.id,node]));
  nodeLayer.replaceChildren(); edgeLayer.replaceChildren();
  nodes=data.nodes.map((node,index)=>{
    const old=previous.get(node.id), angle=index*2.39996, radius=40*Math.sqrt(index);
    return {...node,x:treeMode?node.x:old?old.x:Math.cos(angle)*radius,y:treeMode?node.y:old?old.y:Math.sin(angle)*radius,vx:0,vy:0};
  });
  const byId=new Map(nodes.map(node=>[node.id,node])), pairs=new Map();
  edges=data.edges.map(edge=>{
    const key=JSON.stringify([edge.source,edge.target].sort());
    const group=pairs.get(key)||[]; pairs.set(key,group); group.push(edge);
    return edge;
  });
  for(const group of pairs.values()) group.forEach((edge,index)=>{edge.bend=(index-(group.length-1)/2)*30;edge.loopSize=48+index*22});
  edges=edges.map(edge=>({...edge,source:byId.get(edge.source),target:byId.get(edge.target),
    path:element('path',{'class':'edge '+(edge.kind||''),'marker-end':treeMode && edge.kind==='hierarchy'?'':'url(#arrow)'},edgeLayer),
    text:element('text',{'class':'edge-label '+(edge.kind||''),'text-anchor':'middle'},edgeLayer)}));
  for(const edge of edges){edge.text.textContent=edge.label;element('title',{},edge.path).textContent=edge.uri || (edge.relation?`${edge.relation.subject} ${edge.label} ${edge.relation.object}`:edge.label)}
  for(const node of nodes){
    node.group=element('g',{'class':'node'+(selected===node.id?' selected':'')+(treeMode && !node.family?' synthetic':'')},nodeLayer);
    const tag=treeMode?'rect':node.kind==='literal'?'rect':node.kind==='blank'?'path':'circle';
    const shape=treeMode?{x:-130,y:-38,width:260,height:76,rx:9}:node.kind==='literal'?{x:-16,y:-10,width:32,height:20,rx:4}:node.kind==='blank'?{d:'M0,-17L17,0L0,17L-17,0Z'}:{r:15};
    element(tag,{...shape,'class':'shape',fill:node.color,stroke:'#7c8da3','stroke-dasharray':node.kind==='blank'?'3 2':'none'},node.group);
    element('text',treeMode?{x:-118,y:-15}:{y:31,'text-anchor':'middle'},node.group).textContent=treeMode && node.label.length>34?node.label.slice(0,33)+'…':node.label;
    if(treeMode)(node.badges||[]).forEach((badge,i)=>element('text',{x:-118,y:5+i*17,'class':'badge'},node.group).textContent=badge.length>40?badge.slice(0,39)+'…':badge);
    element('title',{},node.group).textContent=node.uri || node.value || node.id;
    node.group.onpointerenter=()=>describe(node);
    node.group.onpointerleave=()=>{const pinned=byId.get(selected);if(pinned)describe(pinned);else details.textContent=''};
    node.group.onpointerdown=event=>{
      event.stopPropagation();selected=node.id;describe(node);
      for(const item of nodes)item.group.classList.toggle('selected',item.id===selected);
      highlight();
      drag={...(treeMode?{}:{node}),lastX:event.clientX,lastY:event.clientY};canvas.setPointerCapture(event.pointerId);
    };
  }
  const legend=document.querySelector('#legend');legend.replaceChildren();
  for(const entry of [...data.legend,...(treeMode?[{label:'— Hierarchy',color:'#7890a8'},{label:'┄ Secondary (select to highlight)',color:'#cbd5e1'}]:[{label:'Untyped',color:'#b4c4d6'},{label:'◇ Blank node',color:'#b4c4d6'},{label:'Literal',color:'#d9dee5'}])]){
    const key=document.createElement('span');key.className='key';key.title=entry.uri||entry.label;
    const swatch=document.createElement('i');swatch.className='swatch';swatch.style.background=entry.color;
    key.append(swatch,document.createTextNode(entry.label));legend.append(key);
  }
  if(byId.has(selected))describe(byId.get(selected));else{selected=null;details.textContent=''}
  if(treeMode && (enteringTree || revision===-1)){view={x:165,y:100,scale:0.8};}
  if(!treeMode && revision===-1)view.scale=Math.min(1,Math.min(innerWidth,innerHeight)/(100*Math.sqrt(nodes.length||1)));
  energy=treeMode?0:240;highlight();transform();draw();
}
function draw(){
  for(const node of nodes)node.group.setAttribute('transform',`translate(${node.x},${node.y})`);
  for(const edge of edges){
    const a=edge.source,b=edge.target;
    if(treeMode && edge.kind==='hierarchy'){
      const start=a.x+130,end=b.x-130,mid=(start+end)/2;
      edge.path.setAttribute('d',`M${start},${a.y}H${mid}V${b.y}H${end}`);
      edge.text.setAttribute('x',mid);edge.text.setAttribute('y',b.y-45);continue;
    }
    if(a===b){
      const size=edge.loopSize;
      edge.path.setAttribute('d',`M${a.x-12},${a.y-10}C${a.x-size},${a.y-size*1.5} ${a.x+size},${a.y-size*1.5} ${a.x+12},${a.y-10}`);
      edge.text.setAttribute('x',a.x);edge.text.setAttribute('y',a.y-size-4);continue;
    }
    // A canonical perpendicular keeps opposite-direction edges in separate lanes.
    const direction=a.id<b.id?1:-1,dx=b.x-a.x,dy=b.y-a.y,distance=Math.hypot(dx,dy)||1;
    const cx=(a.x+b.x)/2-dy/distance*edge.bend*direction,cy=(a.y+b.y)/2+dx/distance*edge.bend*direction;
    const start=Math.hypot(cx-a.x,cy-a.y)||1,end=Math.hypot(b.x-cx,b.y-cy)||1;
    edge.path.setAttribute('d',`M${a.x+(cx-a.x)*18/start},${a.y+(cy-a.y)*18/start}Q${cx},${cy} ${b.x-(b.x-cx)*19/end},${b.y-(b.y-cy)*19/end}`);
    edge.text.setAttribute('x',(a.x+2*cx+b.x)/4);edge.text.setAttribute('y',(a.y+2*cy+b.y)/4-5);
  }
}
function tick(){
  if(!treeMode && energy>0){
    energy--;
    for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
      const a=nodes[i],b=nodes[j],dx=b.x-a.x||0.1,dy=b.y-a.y||0.1,d2=dx*dx+dy*dy+100;
      const force=1100/d2, distance=Math.sqrt(d2);
      a.vx-=dx/distance*force;a.vy-=dy/distance*force;b.vx+=dx/distance*force;b.vy+=dy/distance*force;
    }
    for(const {source:a,target:b} of edges){
      const dx=b.x-a.x,dy=b.y-a.y,distance=Math.hypot(dx,dy)||1,force=(distance-135)*0.006;
      a.vx+=dx/distance*force;a.vy+=dy/distance*force;b.vx-=dx/distance*force;b.vy-=dy/distance*force;
    }
    for(const node of nodes){
      node.vx=(node.vx-node.x*0.0004)*0.8;node.vy=(node.vy-node.y*0.0004)*0.8;
      if(drag?.node!==node){node.x+=Math.max(-12,Math.min(12,node.vx));node.y+=Math.max(-12,Math.min(12,node.vy))}
    }
    draw();
  }
  requestAnimationFrame(tick);
}
canvas.onpointerdown=event=>{drag={lastX:event.clientX,lastY:event.clientY};canvas.setPointerCapture(event.pointerId);selected=null;details.textContent='';for(const node of nodes)node.group.classList.remove('selected');highlight()};
canvas.onpointermove=event=>{
  if(!drag)return;
  const dx=event.clientX-drag.lastX,dy=event.clientY-drag.lastY;
  if(drag.node){drag.node.x+=dx/view.scale;drag.node.y+=dy/view.scale;drag.node.vx=drag.node.vy=0;energy=120;draw()}
  else{view.x+=dx;view.y+=dy;transform()}
  drag.lastX=event.clientX;drag.lastY=event.clientY;
};
canvas.onpointerup=canvas.onpointercancel=canvas.onlostpointercapture=()=>drag=null;
canvas.addEventListener('wheel',event=>{
  event.preventDefault();const scale=Math.max(0.05,Math.min(6,view.scale*Math.exp(-event.deltaY*0.001)));
  const ratio=scale/view.scale;view.x=event.clientX-(event.clientX-view.x)*ratio;view.y=event.clientY-(event.clientY-view.y)*ratio;view.scale=scale;transform();
},{passive:false});
async function refresh(){
  try{
    const response=await fetch('/graph',{cache:'no-store'});if(!response.ok)throw Error(response.status);
    const state=await response.json();
    if(state.revision!==revision){render(state.data);revision=state.revision}
    document.querySelector('h1').textContent=({G1:'G1 · Sense: initial observation',G2:'G2 · Plan: Bringing interpretation',G3:'G3 · Act: resolved frame and ordered plan'})[state.data.stage] || 'Knowledge graph';
    status.textContent=state.error || `${state.data.stage || "Graph"} · ${nodes.length} nodes · ${edges.length} edges · Live · Drag to move, scroll to zoom`;
  }catch(error){status.textContent='Connection lost — retrying…'}
  setTimeout(refresh,1000);
}
transform();tick();refresh();
</script></html>'''


# Publish snapshots with a lock so each request sees matching data and status.
def watch_graph(path, rdf_format, state, lock, stopped, tree=False, g1_path=None):
    last_stamp = None
    while not stopped.is_set():
        try:
            stamp = graph_stamp(path, g1_path)
            if stamp != last_stamp:
                graph = load_graph(path, rdf_format, g1_path=g1_path)
                data = scene_tree_data(graph) if tree else graph_to_data(graph)
                if graph_stamp(path, g1_path) != stamp:
                    raise ValueError("file changed during parsing")
                with lock:
                    state.update(data=data, revision=state["revision"] + 1, error=None)
                last_stamp = stamp
        except Exception as error:
            last_stamp = None
            # Retry failed reads even if the writer preserves the modification time.
            with lock:
                state["error"] = (missing_message(path) if isinstance(error, FileNotFoundError) else
                                  f"Waiting for a valid graph; keeping last graph. {error}")
        stopped.wait(1)


def run_server(path, rdf_format=None, *, tree=False, g1_path=None):
    page = build_html().encode()
    state = {"revision": 0, "data": {"nodes": [], "edges": [], "legend": []},
             "error": "Waiting for the first valid graph…"}
    lock = threading.Lock()
    stopped = threading.Event()

    def application(environ, start_response):
        route = environ.get("PATH_INFO", "/")
        if route == "/":
            body, content_type, status = page, "text/html; charset=utf-8", "200 OK"
        elif route == "/graph":
            with lock:
                body = json.dumps(state).encode()
            content_type, status = "application/json", "200 OK"
        else:
            body, content_type, status = b"Not found", "text/plain", "404 Not Found"
        start_response(status, [("Content-Type", content_type), ("Content-Length", str(len(body))),
                                ("Cache-Control", "no-store")])
        return [body]

    with make_server("127.0.0.1", 0, application) as server:
        watcher = threading.Thread(target=watch_graph, args=(path, rdf_format, state, lock, stopped, tree, g1_path), daemon=True)
        watcher.start()
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Watching {path}\nViewer: {url}\nPress Ctrl+C to stop.", flush=True)
        try:
            webbrowser.open(url)
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            stopped.set()
            watcher.join(timeout=2)


# The module entry point keeps invocation independent of ROS or project setup.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="JSON scene graph path (also: g1/g2/g3 aliases, scene_graph, schemas, or RDF file)")
    parser.add_argument("--g1", type=Path, help="Optional G1 JSON provenance file for G2")
    parser.add_argument("--tree", action="store_true", help="Display scene_graph as a hierarchy with secondary cross-links")
    parser.add_argument("--watch", action="store_true", help="Wait for a missing graph; live reload is always enabled")
    parser.add_argument("--format", choices=("turtle", "xml", "json-ld"), help="Override format detection")
    arguments = parser.parse_args()
    path = resolve_path(arguments.path)
    if arguments.tree and (path.name != "scene_graph.json" or arguments.format):
        parser.error("--tree requires scene_graph (or a scene_graph.json path), without --format")
    try:
        load_graph(path, arguments.format, g1_path=arguments.g1)
    except (OSError, ValueError) as error:
        print(missing_message(path) if isinstance(error, FileNotFoundError) else str(error))
        if not arguments.watch:
            return 1
    options = {}
    if arguments.tree:
        options['tree'] = True
    if arguments.g1 is not None:
        options['g1_path'] = arguments.g1.expanduser().resolve()
    run_server(path, arguments.format, **options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
