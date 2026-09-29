"""Search candidates use semantic priors and interaction evidence only."""

from perceived_entity_linking import normalize_type, resolve_concept
from semantic_fallback import prune_fallback


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
            "Location": location}


def search_trace(enabled, message):
    """One CLI sink for optional Search diagnostics; never modifies knowledge."""
    if enabled:
        print(f"[SEARCH] {message}", flush=True)


def select_candidate(frame, robokg, semantic_memory, observed, checked=(), suggestions=(), *, debug=False, current=None):
    """Ground priors against observed symbols only. Never accept simulator memory.

    Return unresolved suggestions explicitly; only grounded IDs enter PDDL.
    Specific candidates precede rooms within the KG tier.
    """
    # A single supplied graph is treated as current for standalone callers.
    # SPA always supplies current G1 separately from retained observed evidence.
    current = observed if current is None else current
    remembered_objects = observed.get('objects', [])
    objects = [obj for obj in current.get('objects', [])
               if obj['id'] not in checked
               and normalize_type(obj['type']) != normalize_type(frame['Theme'])
               and normalize_type(obj['type']) not in ('robot', 'tiago', 'person', 'pedestrian', 'mannequin')
               and obj['id'] not in ('robot', 'TIAGo', 'user')]
    concept = resolve_concept(frame['Theme'], robokg)
    priors = robokg.get_locations(concept) if concept else []
    unresolved = []
    def log(message):
        search_trace(debug, message)

    log("Current gaze targets:")
    for obj in objects:
        log(f"  {obj['id']} type={obj['type']}")
    log(f"Theme={frame['Theme']!r}; resolved RoboKGNet concept={concept!r}")
    log(f"Task-local checked anchors: {sorted(checked)!r}")
    log(f"RoboKGNet priors (input order): {priors!r}")
    log(f"VLM suggestions: {suggestions!r}")
    log("Grounding: case-insensitive ID first, then normalized type/ID; no alias lookup.")
    log("Ranking: first nonempty source tier; non-Location anchors before Location; stable input order.")
    active_source = None

    def ground(items, source, pool=None):
        nonlocal active_source
        active_source = source
        pool = objects if pool is None else pool
        found = []
        log(f"{source}: grounding inputs")
        for item in items:
            name = (item.get('location', item.get('id')) if isinstance(item, dict)
                    else item[0] if isinstance(item, (tuple, list)) else item)
            if not isinstance(name, str):
                log(f"{source}: rejected {item!r}: invalid candidate form")
                continue
            term = name.split('.n.')[0].replace('_', ' ').casefold()
            method = 'case-insensitive exact ID'
            matches = [obj for obj in pool if obj['id'].casefold() == name.casefold()]
            if not matches:
                method = 'normalized type/ID'
                matches = [obj for obj in pool
                           if normalize_type(obj['type']) == term
                           or obj['id'].replace('_', ' ').casefold() == term]
            if not matches:
                unresolved.append(name)
                log(f"{source}: rejected {item!r}: could not ground in observed symbols")
            if len(matches) > 1:
                log(f"{source}: {name!r} matched multiple symbols; existing rule retains all")
            for obj in matches:
                if obj['id'] in checked:
                    log(f"{source}: rejected {name!r} -> {obj['id']}: already checked (task-local)")
                else:
                    duplicate = any(o['id'] == obj['id'] for o in found)
                    log(f"{source}: {item!r} -> {obj['id']} type={obj['type']!r}; method={method}; accepted"
                        + (" (duplicate retained by existing rule)" if duplicate else ""))
                    found.append(obj)
            # Keep duplicates and multi-matches exactly as before.
        if not found:
            log(f"{source}: no accepted candidates")
        return sorted(found, key=lambda obj: obj['type'] == 'Location')

    # Trusted observations of the Theme are factual KG evidence, not seed facts.
    theme_ids = {obj['id'] for obj in remembered_objects
                 if normalize_type(obj['type']) == normalize_type(frame['Theme'])}
    relations = [r for r in observed.get('relations', [])
                 if r['subject'] in theme_ids and r['predicate'] in ('on', 'in')]
    for relation in relations:
        log(f"Observed evidence: {relation['subject']} --{relation['predicate']}--> {relation['object']}")
    if not relations:
        log("Observed evidence: none (only Theme-related on/in relations contribute)")
    evidence = [r['object'] for r in relations]
    candidates = ground(evidence, 'observed evidence', remembered_objects)
    if not candidates:
        candidates = ground(priors or [], 'RoboKGNet')
    else:
        log("RoboKGNet grounding: skipped; observed evidence supplied candidates")
    if not candidates:
        candidates = ground(suggestions, 'VLM suggestions')
    else:
        log("VLM suggestion grounding: skipped; higher-priority source supplied candidates")
    if not candidates:
        locations = objects
        ranked = semantic_memory.rank_gaze_targets(frame['Theme'], locations, checked=checked) if locations else []
        retained = prune_fallback(ranked, locations, log)
        active_source = 'semantic fallback'
        by_id = {obj['id']: obj for obj in locations}
        candidates = [by_id[item['location']] for item in retained]
    else:
        log("Semantic fallback: not called; higher-priority source supplied candidates")
    log("LLM inspection ranking:" if active_source == "semantic fallback" else "Final ranked candidates:")
    for rank, obj in enumerate(candidates, 1):
        log(f"  {rank}. {obj['id']} source={active_source}")
    if candidates:
        log(f"Selected gaze target: {candidates[0]['id']}; reason=first ranked anchor from {active_source}")
    else:
        log("Selected: none; no source produced an unchecked grounded anchor")
    return {'location': candidates[0]['id'] if candidates else None,
            'ungrounded': list(dict.fromkeys(unresolved))}


def search_result(frame, fresh_scene, robokg):
    """Only the post-look VLM view can establish success, never merged memory."""
    concept = resolve_concept(frame['Theme'], robokg)
    matches = [obj for obj in fresh_scene.get('objects', [])
               if normalize_type(obj['type']) == normalize_type(frame['Theme'])
               or (concept is not None and resolve_concept(obj['type'], robokg) == concept)]
    return {'Agent': frame['Agent'], 'Theme': frame['Theme'], 'Location': frame['Location'],
            'Success': bool(matches), 'observed_ids': [obj['id'] for obj in matches],
            **({'GazeAction': frame['GazeAction']} if 'GazeAction' in frame else {})}


def gaze_key(action):
    return f"look-at({action['target']})" if action['action'] == 'look-at' else action['action']


def choose_gaze(frame, current, semantic_memory, checked=(), *, debug=False):
    """The LLM chooses one current inspection action, never a Theme location."""
    targets = [obj for obj in current.get('objects', [])
               if normalize_type(obj['type']) not in ('robot', 'tiago', 'person', 'pedestrian', 'mannequin', 'location', 'room')
               and obj['id'] not in ('robot', 'TIAGo', 'user')
               and normalize_type(obj['type']) != normalize_type(frame['Theme'])]
    options = [{'action': 'look-at', 'target': obj['id']} for obj in targets]
    options += [{'action': 'look-left'}, {'action': 'look-right'}]
    search_trace(debug, f"Theme unresolved: {frame['Theme']}")
    search_trace(debug, "Available gaze actions:")
    for option in options:
        search_trace(debug, f"  {gaze_key(option)}" + (" (checked)" if gaze_key(option) in checked else ""))
    if all(gaze_key(option) in checked for option in options):
        raise ValueError('Search exhausted: all current gaze actions already checked')
    chosen = semantic_memory.choose_gaze_action(frame['Theme'], current, options, checked=checked)
    if not isinstance(chosen, dict) or chosen not in options:
        raise ValueError(f'Invalid LLM gaze action: {chosen!r}')
    if gaze_key(chosen) in checked:
        raise ValueError(f'LLM repeated checked gaze action: {gaze_key(chosen)}')
    search_trace(debug, f"LLM selected: {gaze_key(chosen)}")
    return chosen
