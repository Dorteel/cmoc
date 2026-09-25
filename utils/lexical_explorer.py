"""Offline browser for WordNet, FrameNet, and VerbNet.

Run:
    streamlit run lexical_browser.py

Required NLTK corpora:
    wordnet
    omw-1.4
    framenet_v17
    verbnet
"""

from difflib import get_close_matches

import streamlit as st
from nltk.corpus import framenet as fn
from nltk.corpus import verbnet as vn
from nltk.corpus import wordnet as wn


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def normalize(text):
    """Normalize text for simple case-insensitive matching."""
    return text.strip().lower().replace("_", " ")


def fuzzy_matches(query, choices, limit=10, cutoff=0.45):
    """Return approximate matches when an exact match is unavailable."""
    normalized_to_original = {normalize(choice): choice for choice in choices}
    matches = get_close_matches(
        normalize(query),
        normalized_to_original.keys(),
        n=limit,
        cutoff=cutoff,
    )
    return [normalized_to_original[match] for match in matches]


# ---------------------------------------------------------------------------
# WordNet
# ---------------------------------------------------------------------------

@st.cache_data
def wordnet_vocabulary():
    """Return all WordNet lemma names."""
    return sorted({lemma.replace("_", " ") for lemma in wn.all_lemma_names()})


def search_wordnet(query):
    """Find exact WordNet synsets, otherwise suggest nearby lemma names."""
    synsets = wn.synsets(query.replace(" ", "_"))

    if synsets:
        return synsets, []

    suggestions = fuzzy_matches(query, wordnet_vocabulary())
    return [], suggestions


def show_wordnet_synset(synset):
    """Render one WordNet synset."""
    st.subheader(synset.name())
    st.write(synset.definition())

    if synset.examples():
        st.caption("Examples: " + " | ".join(synset.examples()))

    st.write("**Lemmas:**", ", ".join(lemma.name() for lemma in synset.lemmas()))

    hypernyms = [item.name() for item in synset.hypernyms()]
    hyponyms = [item.name() for item in synset.hyponyms()[:15]]

    if hypernyms:
        st.write("**Hypernyms:**", ", ".join(hypernyms))
    if hyponyms:
        st.write("**Hyponyms:**", ", ".join(hyponyms))


# ---------------------------------------------------------------------------
# FrameNet
# ---------------------------------------------------------------------------

@st.cache_data
def framenet_index():
    """Build searchable labels for FrameNet frames and lexical units."""
    index = {}

    for frame in fn.frames():
        labels = {frame.name}

        for lexical_unit in frame.lexUnit:
            # "put.v" -> "put"
            labels.add(lexical_unit.rsplit(".", 1)[0])

        for label in labels:
            index.setdefault(normalize(label), set()).add(frame.ID)

    return {key: sorted(value) for key, value in index.items()}


def search_framenet(query):
    """Return matching FrameNet frames or fuzzy alternatives."""
    index = framenet_index()
    normalized_query = normalize(query)

    frame_ids = set()

    # Prefer exact and substring matches.
    for label, ids in index.items():
        if normalized_query == label or normalized_query in label:
            frame_ids.update(ids)

    if frame_ids:
        return [fn.frame(frame_id) for frame_id in sorted(frame_ids)], []

    suggestions = fuzzy_matches(query, list(index.keys()))
    return [], suggestions


def show_framenet_frame(frame):
    """Render one FrameNet frame including its frame elements."""
    st.subheader(f"{frame.name}  ·  Frame {frame.ID}")
    st.write(frame.definition)

    st.markdown("**Frame elements**")
    for name, element in frame.FE.items():
        core_type = element.get("coreType", "Unknown")
        definition = element.get("definition", "").replace("\n", " ")
        st.write(f"- **{name}** `{core_type}`: {definition}")

    lexical_units = sorted(frame.lexUnit.keys())
    if lexical_units:
        st.markdown("**Lexical units**")
        st.write(", ".join(lexical_units))


# ---------------------------------------------------------------------------
# VerbNet
# ---------------------------------------------------------------------------

@st.cache_data
def verbnet_index():
    """Build searchable labels for VerbNet lemmas and class IDs."""
    index = {}

    for class_id in vn.classids():
        index.setdefault(normalize(class_id), set()).add(class_id)

        for lemma in vn.lemmas(class_id):
            index.setdefault(normalize(lemma), set()).add(class_id)

    return {key: sorted(value) for key, value in index.items()}


def search_verbnet(query):
    """Return matching VerbNet classes or fuzzy alternatives."""
    index = verbnet_index()
    normalized_query = normalize(query)

    class_ids = set()

    for label, ids in index.items():
        if normalized_query == label or normalized_query in label:
            class_ids.update(ids)

    if class_ids:
        return sorted(class_ids), []

    suggestions = fuzzy_matches(query, list(index.keys()))
    return [], suggestions


def show_verbnet_class(class_id):
    """Render one VerbNet class with roles, syntax, and semantics."""
    st.subheader(class_id)

    lemmas = vn.lemmas(class_id)
    if lemmas:
        st.write("**Members:**", ", ".join(sorted(lemmas)))

    st.markdown("**Thematic roles**")
    for role in vn.themroles(class_id):
        role_type = role.get("type", "")
        modifiers = role.get("modifiers", [])
        st.write(f"- **{role_type}** {modifiers}")

    st.markdown("**Frames**")
    for frame in vn.frames(class_id):
        example = frame.get("example", "")
        syntax = frame.get("syntax", [])
        semantics = frame.get("semantics", [])

        if example:
            st.write(f"**Example:** {example}")
        st.write("**Syntax:**", syntax)
        st.write("**Semantics:**", semantics)
        st.divider()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    """Run the local lexical-resource browser."""
    st.set_page_config(page_title="Lexical Resource Browser", layout="wide")

    st.title("WordNet · FrameNet · VerbNet Browser")
    st.caption("Offline browser for exploring structures useful for JSON-schema design.")

    resource = st.selectbox(
        "Resource",
        ["FrameNet", "VerbNet", "WordNet"],
    )

    query = st.text_input(
        "Search",
        placeholder="Try: put, placing, container, grasp...",
    )

    if not query:
        return

    if resource == "WordNet":
        results, suggestions = search_wordnet(query)

        if results:
            for synset in results:
                show_wordnet_synset(synset)
                st.divider()
        else:
            show_suggestions("WordNet", suggestions)

    elif resource == "FrameNet":
        results, suggestions = search_framenet(query)

        if results:
            for frame in results[:20]:
                show_framenet_frame(frame)
                st.divider()
        else:
            show_suggestions("FrameNet", suggestions)

    else:
        results, suggestions = search_verbnet(query)

        if results:
            for class_id in results[:20]:
                show_verbnet_class(class_id)
        else:
            show_suggestions("VerbNet", suggestions)


def show_suggestions(resource, suggestions):
    """Show fuzzy alternatives when no direct match exists."""
    st.info(f"No direct {resource} match found.")

    if suggestions:
        st.write("Closest entries:")
        for suggestion in suggestions:
            st.write(f"- {suggestion}")
    else:
        st.write("No close matches found.")


if __name__ == "__main__":
    main()
