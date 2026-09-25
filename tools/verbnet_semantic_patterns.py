"""Inspect recurring predicates in the direct frames of an NLTK VerbNet class."""

import argparse
from collections import Counter
import json
import re

from nltk.corpus import verbnet as vn


def load_class(class_id):
    # Resolve the class and read its direct roles and frames, without silently
    # adding subclass frames or inferring inherited semantics.
    class_id = vn.vnclass(class_id).get("ID")
    return class_id, vn.themroles(class_id), vn.frames(class_id)


def normalize_predicate(predicate):
    """Normalize event spelling locally, preserving arguments, phases and polarity."""
    events = {}

    def rename_event(match):
        # First distinct event becomes E0, next becomes E1. Thus equals(E0,E1)
        # stays different from equals(E0,E0), regardless of original spelling.
        name = match.group()
        if name not in events:
            events[name] = f"E{len(events)}"
        return events[name]

    arguments = []
    argument_types = []
    phases = []
    for argument in predicate["arguments"]:
        value = argument["value"]
        if argument["type"] == "Event":
            value = re.sub(r"\bE\d*\b", rename_event, value)
            phase = re.fullmatch(r"(start|during|end)\(E\d+\)", value)
            if phase:
                phases.append(phase.group(1))
        arguments.append(value)
        argument_types.append(argument["type"])

    # Keep event arguments too: dropping them would lose argument order,
    # event identity within a predicate, and mixed temporal-phase distinctions.
    return {
        "predicate": predicate["predicate_value"],
        "phase": phases[0] if len(set(phases)) == 1 else "none",
        "arguments": arguments,
        "argument_types": argument_types,
        "negated": predicate["negated"],
    }


def extract_frames(raw_frames, roles):
    # Preserve examples and syntax so each summary can be checked against data.
    role_names = {role["type"] for role in roles}
    frames = []
    for frame in raw_frames:
        present = set()
        for element in frame["syntax"]:
            value = element["modifiers"].get("value")
            if value in role_names:
                present.add(value)
        frames.append({
            "example": frame["example"],
            "description": frame["description"],
            "syntax": frame["syntax"],
            "roles_present": sorted(present),
            "semantics": [normalize_predicate(p) for p in frame["semantics"]],
            # Local normalization does not retain event links BETWEEN predicates.
            # Keep the original records for inspecting those links explicitly.
            "raw_semantics": frame["semantics"],
        })
    return frames


def detect_patterns(frames):
    # Count presence per frame, not repeated occurrences within the same frame.
    frame_keys = []
    for frame in frames:
        frame_keys.append({json.dumps(p, sort_keys=True) for p in frame["semantics"]})
    counts = Counter(key for keys in frame_keys for key in keys)
    records = []
    for key in sorted(counts):
        records.append({
            "semantics": json.loads(key),
            "count": counts[key],
            "frames": [i + 1 for i, keys in enumerate(frame_keys) if key in keys],
        })
    total = len(frames)

    # Every frame = invariant. Most = strictly more than half, but not all.
    invariant = [record for record in records if record["count"] == total]
    majority = [record for record in records if total / 2 < record["count"] < total]
    specific = [record for record in records if record["count"] == 1 and total > 1]

    # Compare predicate frequency with and without each syntactically present
    # role. Both groups must exist; ubiquitous roles cannot explain variation.
    conditioned = []
    role_names = sorted({role for frame in frames for role in frame["roles_present"]})
    for role in role_names:
        present = [i for i, frame in enumerate(frames) if role in frame["roles_present"]]
        absent = [i for i, frame in enumerate(frames) if role not in frame["roles_present"]]
        if not present or not absent:
            continue
        for key in sorted(counts):
            with_role = sum(key in frame_keys[i] for i in present)
            without_role = sum(key in frame_keys[i] for i in absent)
            # A positive frequency association is descriptive, not causal or a
            # logical implication. Print both fractions, including exceptions.
            if with_role / len(present) > without_role / len(absent):
                conditioned.append({
                    "role": role, "semantics": json.loads(key),
                    "with_role": [with_role, len(present)],
                    "without_role": [without_role, len(absent)],
                })
    return {"invariant": invariant, "majority": majority,
            "role_conditioned": conditioned, "frame_specific": specific,
            "frequencies": records}


def analyze_class(class_id):
    # Load -> normalize each frame -> compare predicate presence across frames.
    class_id, roles, raw_frames = load_class(class_id)
    frames = extract_frames(raw_frames, roles)
    return {"class_id": class_id, "roles": roles, "frames": frames,
            "patterns": detect_patterns(frames)}


def format_predicate(predicate):
    # Display exactly the normalized predicate, including temporal arguments.
    text = f"{predicate['predicate']}({', '.join(predicate['arguments'])})"
    if predicate["negated"]:
        text = f"not({text})"
    return text


def print_summary(analysis):
    total = len(analysis["frames"])
    print(f"VERBNET CLASS: {analysis['class_id']} ({total} direct frames)")
    print("\nROLES")
    for role in analysis["roles"]:
        print(f"  {role['type']}")
    for title, category in [("CORE PATTERN / INVARIANT (every frame)", "invariant"),
                            ("RECURRING SEMANTICS (strict majority, excluding invariants)", "majority")]:
        print(f"\n{title}")
        records = analysis["patterns"][category]
        if not records:
            print("  (none)")
        for record in records:
            print(f"  {format_predicate(record['semantics'])} [{record['count']}/{total}]")

    print("\nROLE-CONDITIONED PATTERNS (observed associations)")
    conditioned = analysis["patterns"]["role_conditioned"]
    if not conditioned:
        print("  (none)")
    for record in conditioned:
        yes, yes_total = record["with_role"]
        no, no_total = record["without_role"]
        print(f"  {record['role']}: {format_predicate(record['semantics'])} "
              f"[present {yes}/{yes_total}; absent {no}/{no_total}]")

    print("\nFRAME-SPECIFIC PATTERNS (one frame only)")
    specific = analysis["patterns"]["frame_specific"]
    if not specific:
        print("  (none)")
    for record in specific:
        index = record["frames"][0]
        print(f"  Frame {index}: {analysis['frames'][index - 1]['example']}")
        print(f"    {format_predicate(record['semantics'])}")
    print("\nEvent labels are local to each predicate; no cross-predicate equivalence is claimed.")


def test_bring_patterns():
    # Corpus-backed sanity check, kept in this file rather than another module.
    analysis = analyze_class("bring-11.3")
    assert len(analysis["frames"]) > 1
    core = analysis["patterns"]["invariant"]
    assert any(r["semantics"]["predicate"] == "cause" for r in core)
    assert any(r["semantics"]["predicate"] == "motion"
               and "Theme" in r["semantics"]["arguments"] for r in core)
    assert any(r["role"] == "Destination" and r["semantics"]["phase"] == "end"
               for r in analysis["patterns"]["role_conditioned"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("class_id")
    args = parser.parse_args()
    print_summary(analyze_class(args.class_id))


if __name__ == "__main__":
    main()
