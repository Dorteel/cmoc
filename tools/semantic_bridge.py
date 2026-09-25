"""Inspect explicit lexical alignments through SemLink2 and NLTK."""

import json
import re
from urllib.error import URLError
from urllib.request import urlopen

from nltk.corpus import framenet, verbnet, wordnet
from nltk.corpus.reader.framenet import FramenetError
from nltk.corpus.reader.wordnet import WordNetError


SEMLINK_URL = (
    "https://raw.githubusercontent.com/cu-clear/semlink/master/instances/vn-fn2.json"
)


def _find_frame(identifier):
    """Accept an exact FrameNet name or numeric frame ID."""
    text = str(identifier)
    if text.isdecimal():
        return framenet.frame(int(text))
    return framenet.frame_by_name(text)


def _find_class(identifier):
    """Accept a full VerbNet class ID or its numeric short form."""
    return verbnet.vnclass(str(identifier))


def _find_synset(identifier):
    """Accept a synset name or an explicit WordNet sense key."""
    text = str(identifier)
    if "%" in text:
        # VerbNet abbreviates verb sense keys by omitting the empty head fields.
        if re.fullmatch(r"[^%]+%2:\d{2}:\d{2}", text):
            text += "::"
        return wordnet.lemma_from_key(text).synset()
    return wordnet.synset(text)


def _load_semlink():
    """Read SemLink2's (class number, member lemma) -> frame names table."""
    try:
        with urlopen(SEMLINK_URL, timeout=30) as response:
            data = json.load(response)
    except (URLError, OSError, ValueError) as error:
        raise RuntimeError(f"Could not read SemLink2 from {SEMLINK_URL}: {error}") from error

    mappings = {}
    for key, frames in data.items():
        match = re.fullmatch(r"([0-9.]+(?:-[0-9]+)*)-(.+)", key)
        if not match or not isinstance(frames, list) or not all(
            isinstance(frame, str) for frame in frames
        ):
            raise ValueError(f"Unexpected SemLink2 entry: {key!r}")
        mappings[match.groups()] = frames
    return mappings


def _members_for_synset(synset, missing):
    """Follow exact sense keys back to the members that declare them."""
    keys = set()
    for lemma in synset.lemmas():
        keys.add(lemma.key())
        keys.add(lemma.key().removesuffix("::"))

    class_ids = set()
    for key in sorted(keys):
        class_ids.update(verbnet.classids(wordnetid=key))
        for class_id in verbnet.classids(wordnetid="?" + key):
            missing.append(f"{class_id}: excluded uncertain WordNet link ?{key}.")

    members = set()
    for class_id in sorted(class_ids):
        for member in _find_class(class_id).findall("MEMBERS/MEMBER"):
            if keys.intersection(member.get("wn", "").split()):
                members.add((verbnet.shortid(class_id), member.get("name")))
    return members


def _synsets_for_member(member, class_id, missing):
    """Resolve declared sense keys without substituting other senses of a verb."""
    names = set()
    keys = member.get("wn", "").split()
    context = f"{class_id}/{member.get('name')}"
    if not keys:
        missing.append(f"{context}: no WordNet sense keys in NLTK VerbNet.")
    for key in keys:
        if key.startswith("?"):
            missing.append(f"{context}: excluded uncertain WordNet link {key}.")
            continue
        try:
            names.add(_find_synset(key).name())
        except (WordNetError, ValueError):
            missing.append(f"{context}: WordNet sense key {key} could not be resolved.")
    return sorted(names)


def show_link(resource, identifier):
    """Print and return explicit alignments, their member-level links, and gaps.

    The source stays fixed: this does not recursively expand related frames,
    classes, or senses. Missing corpora raise NLTK's installation instructions;
    unavailable SemLink2 data raises RuntimeError rather than implying no links.
    """
    resource = resource.strip().lower()
    if resource not in {"framenet", "verbnet", "wordnet"}:
        raise ValueError("resource must be 'framenet', 'verbnet', or 'wordnet'")

    found = {"framenet": set(), "verbnet": set(), "wordnet": set()}
    missing = []
    links = []
    if resource == "framenet":
        source = _find_frame(identifier).name
    elif resource == "verbnet":
        source_class = _find_class(identifier)
        source = source_class.get("ID")
    else:
        source_synset = _find_synset(identifier)
        source = source_synset.name()
    found[resource].add(source)

    mappings = _load_semlink()
    if resource == "framenet":
        members = {pair for pair, frames in mappings.items() if source in frames}
    elif resource == "verbnet":
        members = {
            (verbnet.shortid(source), member.get("name"))
            for member in source_class.findall("MEMBERS/MEMBER")
        }
    else:
        members = _members_for_synset(source_synset, missing)

    for class_number, lemma in sorted(members):
        try:
            vn_class = _find_class(class_number)
        except ValueError:
            missing.append(f"SemLink2 {class_number}/{lemma}: class absent from NLTK VerbNet.")
            continue
        class_id = vn_class.get("ID")
        found["verbnet"].add(class_id)
        member = next(
            (item for item in vn_class.findall("MEMBERS/MEMBER")
             if item.get("name") == lemma),
            None,
        )
        if member is None:
            missing.append(f"{class_id}/{lemma}: member absent from NLTK VerbNet.")
            continue

        frame_names = []
        for name in mappings.get((class_number, lemma), []):
            if resource == "framenet" and name != source:
                continue
            try:
                frame_names.append(_find_frame(name).name)
            except FramenetError:
                missing.append(f"SemLink2 {class_number}/{lemma}: frame {name} absent from NLTK.")
        if not mappings.get((class_number, lemma)):
            missing.append(f"{class_id}/{lemma}: no SemLink2 FrameNet mapping.")

        if resource == "wordnet":
            synset_names = [source]
        else:
            synset_names = _synsets_for_member(member, class_id, missing)
        found["framenet"].update(frame_names)
        found["wordnet"].update(synset_names)
        links.append({
            "verbnet": class_id,
            "member": lemma,
            "framenet": sorted(set(frame_names)),
            "wordnet": synset_names,
        })

    for name, identifiers in found.items():
        if not identifiers:
            missing.append(f"No explicit {name} alignment found for {resource} {source}.")
    result = {name: sorted(identifiers) for name, identifiers in found.items()}
    result["links"] = links
    result["missing"] = sorted(set(missing))

    print(f"{resource}: {source}")
    for name in found:
        print(f"  {name}: {', '.join(result[name]) or '(no explicit alignment)'}")
    for message in result["missing"]:
        print(f"  missing: {message}")
    return result


def main():
    """Start the demonstration with a FrameNet frame."""
    links = show_link("verbnet", "bring-11.3")
    print(links)

if __name__ == "__main__":
    main()
