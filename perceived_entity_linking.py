"""Deterministic Perceived Entity Linking (PEL) and concrete task grounding."""

from copy import deepcopy
import math
from pathlib import Path
import re

# Explicit demo identities, not a general person/pedestrian type rule.
ENTITY_ALIASES = {'mannequin': 'user', 'mannequin_1': 'user', 'person_1': 'user', 'pedestrian_1': 'user'}



def normalize_type(term):
    """Lexical Connector suffix normalization, independent of grasp support."""
    term = term.strip()
    if term.casefold().endswith('connector'):
        term = term[:-len('Connector')]
        term = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', term)
    return term.casefold()


def is_graspable(entity):
    """Match the local simulator's connectorModel + passive Connector contract.

    Unknown/visual-only types are not evidence of simulator grasp support.
    The pick server still validates the concrete object name at execution time.
    """
    kind = entity.get('type', '')
    if not isinstance(kind, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', kind):
        return False
    proto = Path(__file__).parent / 'external/webots_ros2_simulation/protos' / (kind + '.proto')
    if not proto.is_file():
        return False
    source = re.sub(r'#[^\n]*', '', proto.read_text())
    if not re.search(r'field\s+SFString\s+connectorModel\b', source.split(']', 1)[0]):
        return False
    return any(re.search(r'\bmodel\s+IS\s+connectorModel\b', block)
               and re.search(r'\btype\s+"passive"', block)
               for block in re.findall(r'\bConnector\s*\{([^}]+)\}', source))

def resolve_concept(term, robokg):
    """Exact lexical resolution only; never choose an ambiguous sense."""
    term = normalize_type(term)
    matches = robokg.resolve_concept(term)
    if not matches and ' ' in term:
        matches = robokg.resolve_concept(term.replace(' ', '_'))
    # Demo utensil only: verified RoboKGNet entry, superclass cutlery.n.02.
    # Select only if returned by RoboKGNet; never invent a missing concept.
    if term == 'fork' and len(matches) > 1:
        matches = [match for match in matches if match['id'] == 'fork.n.01']
    return matches[0]['id'] if len(matches) == 1 else None


def perceived_entity_linking(scene_graph, robokg):
    """Perceived Entity Linking (PEL): copy, consolidate demo IDs, link concepts."""
    graph = deepcopy(scene_graph)
    renames, provenance, objects, concepts = {}, {}, {}, {}
    issues = []
    # Demo spatial authority: the positioned simulator pedestrian, not VLM
    # room aliases. Apply only to the planning copy and only with a unique room.
    simulator_user = next((obj for obj in scene_graph.get('objects', [])
                           if obj['id'] == 'pedestrian_1' and _position(obj) is not None), None)
    rooms = {obj['id'] for obj in scene_graph.get('objects', []) if obj['type'] == 'Location'}
    simulator_rooms = {r['object'] for r in scene_graph.get('relations', [])
                       if r['subject'] == 'pedestrian_1' and r['predicate'] == 'in' and r['object'] in rooms}
    prefer_simulator_user = simulator_user is not None and len(simulator_rooms) == 1
    # Prefer an existing canonical record, then stable ID order for alias metadata.
    ordered = sorted(graph.get('objects', []),
                     key=lambda obj: (obj['id'].strip().casefold() in ENTITY_ALIASES, obj['id']))
    for obj in ordered:
        identifier = obj['id']
        canonical = ENTITY_ALIASES.get(identifier.strip().casefold(), identifier)
        # Preserve the earlier mannequin-type rule, without generalizing to people.
        if obj['type'].strip().casefold() == 'mannequin':
            canonical = 'user'
        renames[identifier] = canonical
        if canonical != identifier:
            provenance.setdefault(canonical, []).append(identifier)
        concept = resolve_concept(obj['type'], robokg)
        if concept is not None:
            concepts.setdefault(canonical, set()).add(concept)
        obj['id'] = canonical
        if canonical not in objects:
            objects[canonical] = obj
        else:
            for key, value in obj.get('qualities', {}).items():
                objects[canonical].setdefault('qualities', {}).setdefault(key, value)
    # Keep perceived ordering while emitting each canonical identity only once.
    graph['objects'] = list({renames[obj['id']]: objects[renames[obj['id']]]
                             for obj in scene_graph.get('objects', [])}.values())
    if prefer_simulator_user:
        objects['user'].setdefault('qualities', {})['location'] = deepcopy(simulator_user['qualities']['location'])
    relations = []
    for relation in graph.get('relations', []):
        if (prefer_simulator_user and relation['predicate'] == 'in'
                and renames.get(relation['subject'], relation['subject']) == 'user'
                and (relation['subject'] != 'pedestrian_1' or relation['object'] not in simulator_rooms)):
            continue
        for endpoint in ('subject', 'object'):
            identifier = relation[endpoint]
            relation[endpoint] = renames.get(identifier, ENTITY_ALIASES.get(identifier, identifier))
        if relation not in relations:
            relations.append(relation)
    graph['relations'] = relations
    links = {identifier: next(iter(values)) for identifier, values in concepts.items() if len(values) == 1}
    return {'scene_graph': graph, 'entity_links': links, 'issues': issues,
            'aliases': {key: sorted(set(values)) for key, values in provenance.items()}}


def _position(obj):
    position = obj.get('qualities', {}).get('location')
    if (isinstance(position, list) and len(position) == 3
            and all(type(v) in (int, float) and math.isfinite(v) for v in position)):
        return position[:2]  # Ground-plane distance; object height is irrelevant.
    return None


def bind_task(frame, episodic_pel, robokg):
    """Bind exact episodic IDs; Source always belongs to the selected Theme."""
    frame = deepcopy(frame)
    frame['Source'] = None
    graph = episodic_pel['scene_graph']
    objects = {obj['id']: obj for obj in graph['objects']}
    links = episodic_pel['entity_links']
    issues = list(episodic_pel['issues'])
    bindings = dict.fromkeys(('Agent', 'Theme', 'Source', 'Destination'))
    for role in ('Agent', 'Destination'):
        identifier = frame[role]
        # Bind the semantic robot to the simulator ID without renaming G1/G2
        # entities or changing the semantic frame. Keep other agents unchanged.
        if role == 'Agent' and identifier == 'robot' and 'TIAGo' in objects:
            identifier = 'TIAGo'
        if identifier in objects:
            bindings[role] = identifier
        else:
            issues.append(f'Missing concrete {role}: {identifier}')
    concept = resolve_concept(frame['Theme'], robokg)
    # Exact simulator type labels can identify physical candidates even when
    # WordNet has several senses. Do not invent a concept link in that case.
    candidates = sorted(identifier for identifier, obj in objects.items()
                        if (concept is not None and links.get(identifier) == concept)
                        or (concept is None and normalize_type(obj['type']) == normalize_type(frame['Theme'])))
    semantic_candidates = candidates
    candidates = [identifier for identifier in candidates if is_graspable(objects[identifier])]
    selected = None
    if concept is None and not semantic_candidates:
        issues.append('Theme concept unresolved or ambiguous')
    elif not semantic_candidates:
        issues.append('No episodic instance matches the Theme concept')
    elif not candidates:
        issues.append(f"No graspable concrete Theme found for {frame['Theme']}")
    elif len(candidates) == 1:
        selected = candidates[0]
    else:
        robot = _position(objects.get(bindings['Agent'], {}))
        positions = {identifier: _position(objects[identifier]) for identifier in candidates}
        if robot is None or any(position is None for position in positions.values()):
            issues.append('Cannot select nearest Theme: insufficient position information')
        else:
            selected = min(candidates, key=lambda identifier: (math.dist(robot, positions[identifier]), identifier))
    if selected is not None:
        bindings['Theme'] = selected
        rooms = {identifier for identifier, obj in objects.items() if obj['type'] == 'Location'}
        sources = {r['object'] for r in graph.get('relations', [])
                   if r['subject'] == selected and r['predicate'] == 'in' and r['object'] in rooms}
        if len(sources) == 1:
            frame['Source'] = bindings['Source'] = next(iter(sources))
        else:
            issues.append(f'Source for {selected} is unknown or ambiguous')
    return {'frame': frame, 'bindings': bindings, 'theme_concept': concept,
            'type': 'incomplete' if issues else 'bring', 'issues': issues}


def ground_frame_elements(frame, robokg, episodic_pel=None):
    """Semantic Frame Element grounding, distinct from concrete task bindings."""
    links, issues = {}, []
    for role in ('Agent', 'Theme', 'Source', 'Destination'):
        value = frame.get(role)
        known = episodic_pel['entity_links'] if episodic_pel else {}
        links[role] = known.get(value)
        if links[role] is None and episodic_pel and isinstance(value, str):
            normalized = normalize_type(value)
            candidates = {known[obj['id']] for obj in episodic_pel['scene_graph']['objects']
                          if obj['id'] in known and
                          normalize_type(obj['type']) == normalized}
            if len(candidates) == 1:
                links[role] = next(iter(candidates))
        if links[role] is None:
            links[role] = resolve_concept(value, robokg) if isinstance(value, str) and value.strip() else None
        if links[role] is None:
            issues.append(f'{role} semantic value unresolved or ambiguous: {value}')
    return links, issues
