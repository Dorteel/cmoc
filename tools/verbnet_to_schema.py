"""Turn a VerbNet class's thematic roles into a small task-frame schema."""

import argparse
import json
from pathlib import Path

from nltk.corpus import verbnet as vn


OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "schemas" / "task_frames"


def _describe_restrictions(element):
    """Keep selectional restrictions and their logical grouping readable."""
    if element is None:
        return ""
    if element.tag == "SELRESTR":
        return element.get("Value", "") + element.get("type", "")
    parts = [_describe_restrictions(child) for child in element]
    if not parts:
        return ""
    operator = f" {element.get('logic', 'and')} "
    return "(" + operator.join(parts) + ")"


def verbnet_to_schema(class_id):
    """Load direct thematic roles, save their JSON Schema, and return it."""
    vn_class = vn.vnclass(class_id)
    class_id = vn_class.get("ID")
    roles = vn.themroles(class_id)

    # The role API omits logical groups, so retain those from the class XML.
    restrictions = {
        role.get("type"): role.find("SELRESTRS")
        for role in vn_class.findall("THEMROLES/THEMROLE")
    }
    properties = {}
    for role in roles:
        name = role["type"]
        description = f"Entity ID for the {name} role, or null if unknown."
        selection = _describe_restrictions(restrictions.get(name))
        if selection:
            description += f" VerbNet selectional restrictions: {selection}."
        properties[name.lower()] = {
            "type": ["string", "null"],
            "description": description,
        }

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"VerbNet task frame: {class_id}",
        "verbnet_class": class_id,
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / f"{class_id}.json"
    output_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("class_id", help="VerbNet class ID, for example bring-11.3")
    args = parser.parse_args()
    schema = verbnet_to_schema(args.class_id)
    print(f"Saved {OUTPUT_DIRECTORY / (schema['verbnet_class'] + '.json')}")


if __name__ == "__main__":
    main()
