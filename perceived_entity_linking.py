"""Deterministic Perceived Entity Linking (PEL) and concrete task grounding."""

from copy import deepcopy
import math

# Explicit demo identities, not a general person/pedestrian type rule.
ENTITY_ALIASES = {'mannequin': 'user', 'person_1': 'user', 'pedestrian_1': 'user'}


def resolve_concept(term, robokg):
    """Exact lexical resolution only; never choose an ambiguous sense."""
    term = term.strip().casefold()
    matches = robokg.resolve_concept(term)
    if not matches and ' ' in term:
        matches = robokg.resolve_concept(term.replace(' ', '_'))
    return matches[0]['id'] if len(matches) == 1 else None


def perceived_entity_linking(scene_graph, robokg):
    """Perceived Entity Linking (PEL): copy, consolidate demo IDs, link concepts."""
    graph = deepcopy(scene_graph)
    renames, provenance, objects, concepts = {}, {}, {}, {}
    issues = []
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
    relations = []
    for relation in graph.get('relations', []):
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
        return position
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
        if identifier in objects:
            bindings[role] = identifier
        else:
            issues.append(f'Missing concrete {role}: {identifier}')
    concept = resolve_concept(frame['Theme'], robokg)
    candidates = sorted(identifier for identifier in objects if concept is not None and links.get(identifier) == concept)
    selected = None
    if concept is None:
        issues.append('Theme concept unresolved or ambiguous')
    elif not candidates:
        issues.append('No episodic instance matches the Theme concept')
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
