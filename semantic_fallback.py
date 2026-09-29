"""Conservative filtering of the last-resort Search source only."""
import json
from pathlib import Path
import math
import re

from perceived_entity_linking import normalize_type

MAX_SEMANTIC_FALLBACK_CANDIDATES = 5
# Use the perception contract's spatial vocabulary, not object-type lists.
SPATIAL_RELATIONS = frozenset(json.loads(
    (Path(__file__).parent / 'schemas/perception/create_scene_graph.json').read_text()
)['properties']['relations']['items']['properties']['predicate']['enum'])


def label(value):
    return re.sub(r"[\W_]+", " ", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", normalize_type(value))).strip()


def identity(value):
    """Spelling normalization only; table_001 and table(1) share an instance suffix."""
    return re.sub(r"\d+", lambda m: str(int(m.group())), label(value))


def fallback_inputs(theme, observed, checked, log):
    """Derive anchors from spatial relation targets, plus explicit rooms."""
    objects = {obj['id']: obj for obj in observed.get('objects', [])}
    anchors = dict.fromkeys(r['object'] for r in observed.get('relations', [])
                           if r['predicate'] in SPATIAL_RELATIONS
                           and r['subject'] in objects and r['object'] in objects)
    anchors.update({obj['id']: None for obj in objects.values()
                    if label(obj['type']) in ('location', 'room')})
    accepted = [objects[identifier] for identifier in anchors
                if identifier not in checked
                and label(objects[identifier]['type']) != label(theme)
                and identity(identifier) != identity(theme)]
    log("LLM fallback anchors derived from episodic graph:")
    for obj in accepted:
        log(f"  {obj['id']}")
    if not accepted:
        log("  none")
    return accepted


def prune_fallback(ranked, objects, log):
    """Ground only to eligible observed IDs; positive support, alias dedup, top K."""
    rows = []
    rejected = 0
    for item in ranked:
        name = item.get('location') if isinstance(item, dict) else None
        score = item.get('score') if isinstance(item, dict) else None
        if type(score) not in (int, float) or not math.isfinite(score) or score <= 0:
            log(f"Semantic fallback rejected: {name} score={score}; reason=no positive semantic support")
            rejected += 1
            continue
        matches = [o for o in objects if isinstance(name, str) and o['id'].casefold() == name.casefold()]
        if not matches and isinstance(name, str):
            matches = [o for o in objects if label(o['type']) == label(name) or identity(o['id']) == identity(name)]
        if not matches:
            log(f"Semantic fallback rejected: {name}; reason=not an eligible observed anchor")
            rejected += 1
        rows.extend((o, score) for o in matches)

    # Explicit aliases and identical normalized IDs establish families. Never
    # collapse different numbered objects just because their concepts match.
    parents = {o['id']: o['id'] for o in objects}
    def root(key):
        while parents[key] != key:
            key = parents[key]
        return key
    owners = {}
    for obj in objects:
        for alias in [obj['id'], *obj.get('aliases', [])]:
            key = identity(alias)
            if key in owners:
                parents[root(obj['id'])] = root(owners[key])
            else:
                owners[key] = obj['id']
    # A bare label joins a numbered family only when exactly one family exists
    # and its perceived type agrees.
    for obj in objects:
        key = identity(obj['id'])
        if re.search(r" \d+$", key):
            continue
        numbered = {root(o['id']) for o in objects
                    if re.sub(r" \d+$", "", identity(o['id'])) == key
                    and re.search(r" \d+$", identity(o['id']))
                    and label(o['type']) == label(obj['type'])}
        if len(numbered) == 1:
            parents[root(obj['id'])] = numbered.pop()
    groups = {}
    for obj, score in rows:
        groups.setdefault(root(obj['id']), []).append((obj, score))
    retained = []
    for group in groups.values():
        # Prefer explicitly numbered instances; preserve observed order on ties.
        obj, _ = max(group, key=lambda pair: bool(re.search(r" \d+$", identity(pair[0]['id']))))
        retained.append({'location': obj['id'], 'score': max(score for _, score in group)})
    retained.sort(key=lambda item: item['score'], reverse=True)
    log(f"Semantic fallback score/grounding rejected: {rejected}")
    log(f"Semantic fallback deduplicated: {len(rows) - len(retained)}")
    log(f"Semantic fallback capped: {max(0, len(retained) - MAX_SEMANTIC_FALLBACK_CANDIDATES)}")
    retained = retained[:MAX_SEMANTIC_FALLBACK_CANDIDATES]
    log(f"Semantic fallback retained: {len(retained)}")
    log("LLM ranking:")
    for rank, item in enumerate(retained, 1):
        log(f"  {rank}. {item['location']} score={item['score']}")
    return retained
