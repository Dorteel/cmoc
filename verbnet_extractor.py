"""Offline VerbNet candidate extraction for natural-language instructions."""

from __future__ import annotations

import re
from typing import Any, List

import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import verbnet
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


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _normalize_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


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


def _class_text(class_obj: Any) -> str:
    parts = [
        _first_present(class_obj, "ID", "class_id", "id"),
        _first_present(class_obj, "members", "member"),
        _first_present(class_obj, "thematic_roles", "thematicRoles", "thematicroles", "roles"),
        _first_present(class_obj, "frames", "syntax_frames", "syntax"),
        _first_present(class_obj, "semantics", "semantic_predicates", "predicates"),
        _first_present(class_obj, "class_name", "name"),
    ]
    return " ".join(_normalize_string(part) for part in parts if _normalize_string(part))


def _flatten_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        pieces = []
        for part in value:
            text = _flatten_value(part)
            if text:
                pieces.append(text)
        return " ".join(pieces)
    if isinstance(value, dict):
        values = []
        for key in ("value", "text", "name", "label", "type", "id"):
            if key in value:
                text = _flatten_value(value[key])
                if text:
                    values.append(text)
        return " ".join(values)
    return _normalize_string(value)


def _extract_roles(class_obj: Any) -> list[str]:
    roles = _first_present(class_obj, "thematic_roles", "thematicRoles", "thematicroles", "roles")
    values: list[str] = []
    for entry in _as_list(roles):
        text = _first_present(entry, "type", "name", "label", "role")
        if text:
            values.append(_normalize_string(text))
    return values


def _extract_syntax_frames(class_obj: Any) -> list[str]:
    syntax = _first_present(class_obj, "syntax_frames", "syntax", "frames")
    values: list[str] = []
    for entry in _as_list(syntax):
        text = _flatten_value(entry)
        if text:
            values.append(text)
    return values


def _extract_semantic_predicates(class_obj: Any) -> list[str]:
    semantics = _first_present(class_obj, "semantic_predicates", "semantics", "predicates")
    values: list[str] = []
    for entry in _as_list(semantics):
        text = _flatten_value(entry)
        if text:
            values.append(text)
    return values


def _class_record(class_obj: Any) -> dict[str, Any]:
    class_id = _first_present(class_obj, "ID", "class_id", "id")
    return {
        "class_id": _normalize_string(class_id),
        "thematic_roles": _extract_roles(class_obj),
        "syntax_frames": _extract_syntax_frames(class_obj),
        "semantic_predicates": _extract_semantic_predicates(class_obj),
    }


def extract_candidate_classes(instruction: str) -> List[dict[str, Any]]:
    """Return VerbNet class candidates relevant to the instruction."""
    _download_if_needed("verbnet")

    candidate_terms = extract_instruction_terms(instruction)
    if not candidate_terms:
        return []

    matches: list[dict[str, Any]] = []
    seen: set[str] = set()

    for class_id in verbnet.classids():
        try:
            class_obj = verbnet.vnclass(class_id)
        except Exception:
            continue

        class_text = _class_text(class_obj).lower()
        relevant = False
        for term in candidate_terms:
            if term in class_text:
                relevant = True
                break
            if term in _normalize_string(class_id).lower():
                relevant = True
                break

        if not relevant:
            continue

        record = _class_record(class_obj)
        class_id_value = record["class_id"]
        if not class_id_value or class_id_value in seen:
            continue
        seen.add(class_id_value)
        matches.append(record)

    matches.sort(key=lambda item: item["class_id"])
    return matches
