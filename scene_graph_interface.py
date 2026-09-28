"""Query the objects and known locations in a saved episodic scene graph."""

import json
from pathlib import Path
from copy import deepcopy

from spatial_grounder import ground_in_relations, ground_on_relations


class KnowledgeInterface:
    def __init__(self, scene_graph_path=None):
        # None starts empty; updates stay in memory and never overwrite the file.
        scene_graph = {"objects": [], "relations": []}
        if scene_graph_path is not None:
            with open(scene_graph_path, encoding="utf-8") as source:
                scene_graph = json.load(source)
        self._observed = deepcopy(scene_graph.get("observed_evidence", {"objects": [], "relations": []}))
        self._objects = scene_graph.get("objects", [])
        self._explicit_relations = scene_graph.get("relations", [])
        self._ground()

    def _ground(self):
        graph = {"objects": self._objects, "relations": self._explicit_relations}
        self.relations = deepcopy(self._explicit_relations)
        for relation in ground_in_relations(graph) + ground_on_relations(graph):
            if relation not in self.relations:
                self.relations.append(relation)

    def observed_snapshot(self):
        """Only interaction evidence; simulator seeds are never evidence."""
        return deepcopy(self._observed)

    def merge_observation(self, observation):
        evidence = KnowledgeInterface()
        evidence._objects = deepcopy(self._observed["objects"])
        evidence._explicit_relations = deepcopy(self._observed["relations"])
        evidence._merge_observation(observation)
        self._observed = evidence.snapshot()
        self._merge_observation(observation)

    def _merge_observation(self, observation):
        """Upsert exact IDs, retaining unseen entities and unspecified qualities.

        New spatial evidence supersedes old in/on relations for that subject.
        Missing visual geometry is not invented. Identity tracking is left to
        the observer; this method never guesses that two different IDs match.
        """
        incoming = deepcopy(observation.get("objects", []))
        relations = deepcopy(observation.get("relations", []))
        positioned = {obj["id"] for obj in incoming
                      if "location" in obj.get("qualities", {})}
        spatial = {r["subject"] for r in relations if r["predicate"] in ("in", "on")}
        by_id = {obj["id"]: obj for obj in self._objects}
        for obj in incoming:
            identifier = obj["id"]
            if identifier not in by_id:
                by_id[identifier] = {}
                self._objects.append(by_id[identifier])
            stored = by_id[identifier]
            qualities = {**stored.get("qualities", {}), **obj.get("qualities", {})}
            stored.update(obj)
            stored["qualities"] = qualities
        # A new symbolic location without new geometry invalidates old coordinates.
        for identifier in spatial - positioned:
            if identifier in by_id:
                by_id[identifier].get("qualities", {}).pop("location", None)
        replaced = {(r["subject"], r["predicate"]) for r in relations}
        self._explicit_relations = [r for r in self._explicit_relations
                                    if (r["subject"], r["predicate"]) not in replaced
                                    and not (r["subject"] in positioned | spatial
                                             and r["predicate"] in ("in", "on"))]
        self._explicit_relations.extend(relations)
        self._ground()

    def snapshot(self):
        """Independent episodic graph for planning, including grounded relations."""
        return deepcopy({"objects": self._objects, "relations": self.relations})

    def query_theme(self, theme):
        """Return exact ID/type matches first, followed by substring matches."""
        # Normalize the theme so capitalization and surrounding spaces do not matter.
        theme = theme.strip().casefold()
        if not theme:
            return []

        # Keep each group in scene order, adding each object at most once.
        exact_matches = []
        substring_matches = []
        for obj in self._objects:
            identifier = obj.get("id", "").casefold()
            object_type = obj.get("type", "").casefold()
            if theme == identifier or theme == object_type:
                exact_matches.append(obj)
            elif theme in identifier or theme in object_type:
                substring_matches.append(obj)

        # Exact names and types take priority over broader text matches.
        return exact_matches + substring_matches

    def query_locations(self):
        """Return all objects explicitly classified as environment locations."""
        # Scan the loaded snapshot and keep only objects with type "Location".
        locations = []
        for obj in self._objects:
            if obj.get("type") == "Location":
                locations.append(obj)
        return locations

    def query_theme_location(self, theme):
        """Return the first matching room's ID through an 'in' relation, or None."""
        # Collect valid room IDs so dangling or non-Location targets are skipped.
        location_ids = {location["id"] for location in self.query_locations()}

        # Keep the existing theme matching and first-known-instance behavior.
        for obj in self.query_theme(theme):
            for relation in self.relations:
                # Follow only object --in--> Location. Never follow 'on' here.
                if relation.get("subject") != obj["id"]:
                    continue
                if relation.get("predicate") != "in":
                    continue
                location_id = relation.get("object")
                if location_id in location_ids:
                    return location_id

        # No matching object has a known room-containment relation.
        return None


if __name__ == "__main__":
    import argparse

    # Example: python scene_graph_interface.py path/to/scene_graph.json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_graph_path", nargs="?", default=str(Path(__file__).resolve().parent / "episodic_memory/scene_graph.json"))
    args = parser.parse_args()

    kb = KnowledgeInterface(args.scene_graph_path)
    print(kb.query_theme_location("plate"))
    print(kb.query_theme_location("wine glass"))
    print(kb.query_theme_location("nonexistent object"))
