"""Offline FrameNet candidate extraction for natural-language instructions."""

from __future__ import annotations

import re
from typing import Any, Iterable, List

import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import framenet
from nltk.stem import WordNetLemmatizer


_STOPWORDS = {
    "the",
    "a",
    "an",
    "on",
    "in",
    "at",
    "to",
    "of",
    "for",
    "from",
    "with",
    "by",
    "into",
    "out",
    "up",
    "down",
    "over",
    "under",
    "and",
    "or",
    "but",
    "if",
    "then",
    "as",
    "be",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
}


def _download_if_needed(resource_name: str) -> None:
    try:
        nltk.data.find(f"corpora/{resource_name}")
    except LookupError:
        nltk.download(resource_name)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _first_present(obj: Any, *names: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        return None
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _flatten_text(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        items = []
        for item in value:
            text = _flatten_text(item)
            if text:
                items.append(text)
        return " | ".join(items)
    if isinstance(value, dict):
        pieces = []
        for key in ("name", "label", "text", "value", "definition"):
            if key in value:
                text = _flatten_text(value[key])
                if text:
                    pieces.append(text)
        return " | ".join(pieces)
    return _normalize_text(value)


def _wordnet_pos(tag: str) -> str | None:
    if not tag:
        return None
    if tag.startswith("J"):
        return "a"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("N"):
        return "n"
    if tag.startswith("R"):
        return "r"
    return None


def extract_instruction_terms(instruction: str) -> List[str]:
    """Return a deterministic list of noun/verb lemmas from the instruction."""
    if not instruction:
        return []

    tokens = word_tokenize(instruction)
    tagged = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    terms: list[str] = []

    for word, tag in tagged:
        cleaned = re.sub(r"[^A-Za-z]", "", word)
        if not cleaned:
            continue
        lemma = lemmatizer.lemmatize(cleaned.lower(), pos=_wordnet_pos(tag) or "n")
        if len(lemma) <= 2 or lemma in _STOPWORDS:
            continue
        terms.append(lemma.lower())

    return sorted(set(terms))


def _frame_keywords(frame: Any) -> List[str]:
    values = [
        _first_present(frame, "name", "frame_name", "label"),
        _first_present(frame, "definition", "def"),
        _first_present(frame, "lexical_units", "lexUnit", "LU"),
        _first_present(frame, "FE", "frame_elements", "frameElements"),
    ]

    text_parts = []
    for value in values:
        text_parts.append(_flatten_text(value))
    text = " ".join(part for part in text_parts if part)
    return extract_instruction_terms(text)


def _frame_lexical_units(frame: Any) -> List[str]:
    raw_lus = _first_present(frame, "lexical_units", "lexUnit", "LU")
    units: list[str] = []
    for lu in _as_list(raw_lus):
        unit_name = _first_present(lu, "name", "lemma", "lexical_unit")
        if unit_name:
            units.append(_normalize_text(unit_name))
    return units


def _frame_elements(frame: Any) -> List[str]:
    raw_fes = _first_present(frame, "FE", "frame_elements", "frameElements")
    items: list[str] = []
    for fe in _as_list(raw_fes):
        value = _first_present(fe, "name", "label", "FEName")
        if value:
            items.append(_normalize_text(value))
    return items


def _frame_record(frame: Any) -> dict[str, Any]:
    frame_id = _first_present(frame, "ID", "frame_id", "id")
    name = _first_present(frame, "name", "label")
    definition = _first_present(frame, "definition", "def")
    lexical_units = _frame_lexical_units(frame)
    frame_elements = _frame_elements(frame)

    return {
        "frame_id": _normalize_text(frame_id),
        "name": _normalize_text(name),
        "definition": _normalize_text(definition),
        "lexical_units": lexical_units,
        "frame_elements": frame_elements,
    }


def extract_candidate_frames(instruction: str) -> List[dict[str, Any]]:
    """Return FrameNet frame candidates relevant to the instruction."""
    _download_if_needed("framenet_v17")

    candidate_terms = extract_instruction_terms(instruction)
    if not candidate_terms:
        return []

    matches: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for frame in framenet.frames():
        frame_record = _frame_record(frame)
        frame_id = frame_record["frame_id"]
        frame_name = frame_record["name"]
        frame_keywords = _frame_keywords(frame)
        lexical_units = [unit.lower() for unit in frame_record["lexical_units"]]

        if not frame_id and not frame_name:
            continue

        relevant = False
        for term in candidate_terms:
            if term in frame_keywords:
                relevant = True
                break
            if term in {unit.lower() for unit in lexical_units}:
                relevant = True
                break
            if term in frame_name.lower():
                relevant = True
                break

        if not relevant:
            continue

        key = (frame_id, frame_name)
        if key in seen:
            continue
        seen.add(key)
        matches.append(frame_record)

    matches.sort(key=lambda item: (item["frame_id"], item["name"]))
    return matches
