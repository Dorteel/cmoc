"""Search candidates use semantic priors and interaction evidence only."""

from perceived_entity_linking import normalize_type, resolve_concept


def resolve_search(frame, knowledge_interface, semantic_memory):
    """Branch on whether the Source is already known.

    The task instruction tells us which Theme category to search for. We do not
    check whether a grounded instance already exists; we only decide whether the
    search should be anchored to a known source or ranked across known locations.
    """
    theme = frame["Theme"]
    source = frame["Source"]

    # Source known: the search is already grounded to one location.
    if source is not None:
        return {"target": theme, "locations": [source]}

    # Source unknown: query the known environment locations, then rank them by
    # semantic plausibility that the theme might be there.
    locations = [obj for obj in knowledge_interface.observed_snapshot()['objects']
                 if obj['type'] == 'Location']
    ranked_locations = semantic_memory.rank_locations(theme, locations)
    return {"target": theme, "locations": ranked_locations}


def search_frame(bring, location):
    return {"verb": "search", "Agent": bring["Agent"], "Theme": bring["Theme"],
            "Location": location, "Success": False}


def select_candidate(frame, robokg, semantic_memory, observed, checked=(), suggestions=()):
    """Ground priors against observed symbols only. Never accept simulator memory.

    Return unresolved suggestions explicitly; only grounded IDs enter PDDL.
    Specific candidates precede rooms within the KG tier.
    """
    objects = observed.get('objects', [])
    concept = resolve_concept(frame['Theme'], robokg)
    priors = robokg.get_locations(concept) if concept else []
    unresolved = []

    def ground(items):
        found = []
        for item in items:
            name = (item.get('location', item.get('id')) if isinstance(item, dict)
                    else item[0] if isinstance(item, (tuple, list)) else item)
            if not isinstance(name, str):
                continue
            term = name.split('.n.')[0].replace('_', ' ').casefold()
            matches = [obj for obj in objects if obj['id'].casefold() == name.casefold()]
            if not matches:
                matches = [obj for obj in objects
                           if normalize_type(obj['type']) == term
                           or obj['id'].replace('_', ' ').casefold() == term]
            if not matches:
                unresolved.append(name)
            found.extend(obj for obj in matches if obj['id'] not in checked)
        return sorted(found, key=lambda obj: obj['type'] == 'Location')

    # Trusted observations of the Theme are factual KG evidence, not seed facts.
    theme_ids = {obj['id'] for obj in objects
                 if normalize_type(obj['type']) == normalize_type(frame['Theme'])}
    evidence = [r['object'] for r in observed.get('relations', [])
                if r['subject'] in theme_ids and r['predicate'] in ('on', 'in')]
    candidates = ground(evidence) or ground(priors or [])
    if not candidates:
        candidates = ground(suggestions)
    if not candidates:
        locations = [obj for obj in objects if obj['id'] not in checked
                     and obj['id'] not in theme_ids and obj['type'].casefold() not in ('robot', 'person')]
        candidates = ground(semantic_memory.rank_locations(frame['Theme'], locations))
    return {'location': candidates[0]['id'] if candidates else None,
            'ungrounded': list(dict.fromkeys(unresolved))}


def search_result(frame, fresh_scene, robokg):
    """Only the post-look VLM view can establish success, never merged memory."""
    concept = resolve_concept(frame['Theme'], robokg)
    matches = [obj for obj in fresh_scene.get('objects', [])
               if normalize_type(obj['type']) == normalize_type(frame['Theme'])
               or (concept is not None and resolve_concept(obj['type'], robokg) == concept)]
    return {'theme': frame['Theme'], 'location': frame['Location'],
            'success': bool(matches), 'observed_ids': [obj['id'] for obj in matches]}
