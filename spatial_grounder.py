"""Ground room containment and physical support from world-frame geometry."""

import argparse
import json
import math


# Match declared types, never object names. The generated Webots fridge uses
# lowercase "fridge" as its type; keep that explicit alongside common variants.
CONTAINER_TYPES = {"Cabinet", "Refrigerator", "Fridge", "fridge"}


def get_locations(scene_graph):
    # Find all objects explicitly marked as environment locations.
    return [obj for obj in scene_graph.get("objects", [])
            if obj.get("type") == "Location"]


def get_location_bounds(location):
    """Return world XY bounds, or None when room geometry is unavailable."""
    # Read the room center and its already world-aligned width and length.
    qualities = location.get("qualities", {})
    center = qualities.get("location")
    size = qualities.get("size")
    if not isinstance(center, list) or len(center) != 3:
        return None
    if not isinstance(size, list) or len(size) not in (2, 3):
        return None

    # Only finite XY coordinates and positive floor dimensions define a room.
    x, y = center[:2]
    width, length = size[:2]
    for value in (x, y, width, length):
        if type(value) not in (int, float) or not math.isfinite(value):
            return None
    if width <= 0 or length <= 0:
        return None

    # Extend half the room's width and length on either side of its center.
    return {
        "min_x": x - width / 2,
        "max_x": x + width / 2,
        "min_y": y - length / 2,
        "max_y": y + length / 2,
    }


def is_inside(object_position, bounds):
    """Test a representative point against inclusive world XY bounds."""
    # Skip unknown positions; height and object dimensions do not affect this test.
    if bounds is None or not isinstance(object_position, list):
        return False
    if len(object_position) != 3:
        return False
    x, y = object_position[:2]
    for value in (x, y):
        if type(value) not in (int, float) or not math.isfinite(value):
            return False

    # Include boundary points. Overlapping rooms may both contain the same point.
    return (bounds["min_x"] <= x <= bounds["max_x"]
            and bounds["min_y"] <= y <= bounds["max_y"])


def ground_room_in_relations(scene_graph):
    """Return inferred 'in' relations without changing the supplied graph."""
    # Separate rooms from candidate contained objects.
    locations = get_locations(scene_graph)
    objects = [obj for obj in scene_graph.get("objects", [])
               if obj.get("type") != "Location"]
    relations = []

    # For each room, establish its world-aligned floor bounds.
    for location in locations:
        bounds = get_location_bounds(location)
        if bounds is None:
            continue

        # For each object with a known point inside those bounds, link its ID.
        for obj in objects:
            position = obj.get("qualities", {}).get("location")
            if is_inside(position, bounds):
                relations.append({
                    "subject": obj["id"],
                    "predicate": "in",
                    "object": location["id"],
                })
    return relations


def get_object_bounds(obj):
    """Use a centered XY footprint and a base-to-top Z interval for support."""
    # A support needs its world origin and all three dimensions.
    qualities = obj.get("qualities", {})
    origin = qualities.get("location")
    size = qualities.get("size")
    if not isinstance(origin, list) or len(origin) != 3:
        return None
    if not isinstance(size, list) or len(size) != 3:
        return None
    for value in origin + size:
        if type(value) not in (int, float) or not math.isfinite(value):
            return None
    if any(dimension <= 0 for dimension in size):
        return None

    # The demo treats x/y as the footprint center and z as the base height.
    # Therefore support_top = support.location.z + support.size.z.
    x, y, z = origin
    width, length, height = size
    return {
        "min_x": x - width / 2,
        "max_x": x + width / 2,
        "min_y": y - length / 2,
        "max_y": y + length / 2,
        "min_z": z,
        "max_z": z + height,
    }


def get_container_bounds(container):
    """Return a known container's world volume, or None without complete size."""
    # First require an explicitly supported container type, excluding Locations.
    if container.get("type") not in CONTAINER_TYPES:
        return None

    # Reuse the demo's geometry convention: centered XY footprint, base-to-top Z.
    # This validates all three position/size numbers; partial sizes are skipped.
    return get_object_bounds(container)


def is_inside_3d(position, bounds):
    """Test a representative point strictly inside a world-aligned volume."""
    # An object's position is sufficient; its dimensions are not required.
    if bounds is None or not isinstance(position, list) or len(position) != 3:
        return False
    for value in position:
        if type(value) not in (int, float) or not math.isfinite(value):
            return False

    # Check all three axes. Do not enlarge bounds or include surface points:
    # a point on the cabinet top should not become a containment relation.
    for axis, coordinate in zip(("x", "y", "z"), position):
        if not bounds[f"min_{axis}"] < coordinate < bounds[f"max_{axis}"]:
            return False
    return True


def ground_container_in_relations(scene_graph):
    """Infer point-in-volume containment separately from room containment."""
    objects = scene_graph.get("objects", [])
    relations = []
    seen = set()

    # For each candidate container, require its declared type and complete bounds.
    for container in objects:
        bounds = get_container_bounds(container)
        if bounds is None:
            continue

        # For each non-Location object, test its known point against that volume.
        for obj in objects:
            if obj.get("type") == "Location" or obj["id"] == container["id"]:
                continue
            position = obj.get("qualities", {}).get("location")
            pair = (obj["id"], container["id"])
            if pair not in seen and is_inside_3d(position, bounds):
                relations.append({
                    "subject": obj["id"], "predicate": "in", "object": container["id"],
                })
                seen.add(pair)
    return relations


def ground_in_relations(scene_graph):
    """Combine unchanged room tests with the separate container-volume tests."""
    # Keep room relations first so existing room lookup ordering stays unchanged.
    room_relations = ground_room_in_relations(scene_graph)
    container_relations = ground_container_in_relations(scene_graph)
    return room_relations + container_relations


def is_on(subject, support, tolerance=0.04):
    """Compare the subject's origin with the support top, in world metres."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Vertical tolerance must be finite and nonnegative")

    # Exclude self-relations and semantic locations on either side.
    if subject.get("id") == support.get("id"):
        return False
    if subject.get("type") == "Location" or support.get("type") == "Location":
        return False

    # The subject only needs a known world position; never require its size.
    position = subject.get("qualities", {}).get("location")
    if not isinstance(position, list) or len(position) != 3:
        return False
    for value in position:
        if type(value) not in (int, float) or not math.isfinite(value):
            return False

    # The support must have a full 3D size to establish its footprint and top.
    bounds = get_object_bounds(support)
    if bounds is None:
        return False

    # If the subject's XY point is outside the support footprint, reject it.
    if not is_inside(position, bounds):
        return False

    # Otherwise infer on when the subject origin is close to the support top.
    support_top = bounds["max_z"]
    return abs(position[2] - support_top) <= tolerance


def ground_on_relations(scene_graph, tolerance=0.04):
    """Return only inferred 'on' relations; leave the original graph unchanged."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Vertical tolerance must be finite and nonnegative")

    # Every non-Location object can be a subject; is_on checks its position.
    # Only objects with complete geometry can be supports.
    objects = []
    supports = []
    for obj in scene_graph.get("objects", []):
        if obj.get("type") == "Location":
            continue
        objects.append(obj)
        if get_object_bounds(obj) is not None:
            supports.append(obj)

    # Try each directed subject/support pair, retaining each ID pair only once.
    relations = []
    seen = set()
    for subject in objects:
        for support in supports:
            pair = (subject["id"], support["id"])
            if pair not in seen and is_on(subject, support, tolerance):
                relations.append({
                    "subject": subject["id"], "predicate": "on", "object": support["id"],
                })
                seen.add(pair)
    return relations


def print_objects_by_location(scene_graph):
    # Ground once, then group object IDs under each room for inspection.
    relations = ground_in_relations(scene_graph)
    for location in get_locations(scene_graph):
        print(location["id"])
        for relation in relations:
            if relation["object"] == location["id"]:
                print(f"  - {relation['subject']}")
        print()


def print_container_contents(scene_graph):
    # Ground container containment once, then show object IDs under each container.
    print("CONTAINER CONTENTS:")
    relations = ground_container_in_relations(scene_graph)
    for container in scene_graph.get("objects", []):
        if container.get("type") not in CONTAINER_TYPES:
            continue
        print(f"\n{container['id']}")
        if get_container_bounds(container) is None:
            print("  (complete geometry unavailable)")
            continue
        contents = [relation for relation in relations if relation["object"] == container["id"]]
        if not contents:
            print("  (no grounded contents)")
        for relation in contents:
            print(f"  - {relation['subject']}")
    print()


def print_on_relations(scene_graph, tolerance=0.04):
    # Show physical support separately from the existing room-containment report.
    print("ON relations:")
    relations = ground_on_relations(scene_graph, tolerance)
    for relation in relations:
        print(f"  {relation['subject']} -> {relation['object']}")
    if not relations:
        print("  (none)")


if __name__ == "__main__":
    # Load the requested scene graph and immediately print the room assignments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_graph_path")
    parser.add_argument("--tolerance", type=float, default=0.04,
                        help="Vertical support tolerance in metres (default: 0.04)")
    args = parser.parse_args()
    with open(args.scene_graph_path, encoding="utf-8") as source:
        scene_graph = json.load(source)
    print_objects_by_location(scene_graph)
    print_container_contents(scene_graph)
    print_on_relations(scene_graph, args.tolerance)
