"""Deterministic Perceived Entity Linking (PEL) and concrete task grounding."""

from copy import deepcopy
import math

ENTITY_ALIASES = {'mannequin': 'user'}


def resolve_concept(term, robokg):
    """Exact lexical resolution only; never choose an ambiguous sense."""
    term = term.strip().casefold()
    matches = robokg.resolve_concept(term)
    if not matches and ' ' in term:
        matches = robokg.resolve_concept(term.replace(' ', '_'))
    return matches[0]['id'] if len(matches) == 1 else None


def perceived_entity_linking(scene_graph, robokg):
    """Return a planning-only graph copy, ID→concept links, and identity conflicts.

    The sole demo identity alias is mannequin→user (matched by ID or type).
    Relation endpoints follow the same renaming; raw perception is never edited.
    """
    graph = deepcopy(scene_graph)
    aliases = {}
    links = {}
    seen = set()
    issues = []
    for obj in graph.get('objects', []):
        identifier = obj['id']
        canonical = ENTITY_ALIASES.get(identifier.strip().casefold(),
                    ENTITY_ALIASES.get(obj['type'].strip().casefold(), identifier))
        aliases[identifier] = canonical
        obj['id'] = canonical
        if canonical in seen:
            issues.append(f'Conflicting episodic identity: {canonical}')
        seen.add(canonical)
        concept = resolve_concept(obj['type'], robokg)
        if concept is not None:
            links[canonical] = concept
    for relation in graph.get('relations', []):
        for endpoint in ('subject', 'object'):
            identifier = relation[endpoint]
            relation[endpoint] = aliases.get(identifier, ENTITY_ALIASES.get(identifier, identifier))
    return {'scene_graph': graph, 'entity_links': links, 'issues': issues}


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
