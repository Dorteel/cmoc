"""Query the objects and known locations in a saved episodic scene graph."""

import json

from spatial_grounder import ground_in_relations, ground_on_relations


class KnowledgeInterface:
    def __init__(self, scene_graph_path):
        # Load the scene graph once; later queries use this in-memory snapshot.
        with open(scene_graph_path, encoding="utf-8") as source:
            scene_graph = json.load(source)
        self._objects = scene_graph.get("objects", [])

        # Ground room containment and physical support once, from that snapshot.
        in_relations = ground_in_relations(scene_graph)
        on_relations = ground_on_relations(scene_graph)

        # Keep explicit relations first, then add grounded relations in memory.
        # Do not write the enriched knowledge back to the source JSON file.
        self.relations = list(scene_graph.get("relations", []))
        self.relations.extend(in_relations)
        self.relations.extend(on_relations)

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

    # Example: python knowledge_interface.py path/to/scene_graph.json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_graph_path", nargs="?", default="scene_graph.json")
    args = parser.parse_args()

    kb = KnowledgeInterface(args.scene_graph_path)
    print(kb.query_theme_location("plate"))
    print(kb.query_theme_location("wine glass"))
    print(kb.query_theme_location("nonexistent object"))
