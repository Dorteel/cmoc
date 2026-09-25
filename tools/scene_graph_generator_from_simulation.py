"""Extract explicit named scene objects from a saved Webots world."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from itertools import product
import math
from pathlib import Path
import re
import warnings

from jsonschema import Draft202012Validator


SCHEMAS = Path(__file__).resolve().parents[1] / "schemas"
TOKENS = re.compile(r'"(?:\\.|[^"\\])*"|\#[^\n]*|[{}\[\]]|[^\s{}\[\],]+')
NUMERIC_FIELDS = {
    "translation", "rotation", "scale", "size", "color", "baseColor", "diffuseColor",
    "width", "height", "depth", "radius", "bottomRadius",
}
SPATIAL_FIELDS = {"children", "endPoint", "device"}
PRIMITIVES = {"Box", "Sphere", "Cylinder", "Cone", "Capsule", "Plane"}
JOINTS = {"HingeJoint", "Hinge2Joint", "SliderJoint", "BallJoint"}
NON_OBJECT_TYPES = PRIMITIVES | JOINTS | {
    "PBRAppearance", "Appearance", "Material", "ImageTexture", "TextureTransform",
    "IndexedFaceSet", "Mesh", "Coordinate", "TextureCoordinate", "Normal", "Color",
    "Physics", "WorldInfo", "Viewpoint", "ContactProperties",
}


# Known physical node/PROTO types in this world may omit both name and DEF.
# Keep this explicit: anonymous Pose, Shape, appearance, and helper nodes are
# structure, not additional semantic objects. Extend the list for new PROTOs.
SEMANTIC_OBJECT_TYPES = {
    "Solid", "Robot", "Pedestrian", "Worktop", "StraightStairs", "WallPlug",
    "SquareManhole", "PipeSection", "Barbecue", "Cabinet", "RoundTable",
    "Bed", "Radiator", "Door", "FloorLight", "Table", "Sofa",
    "BunchOfSunFlowers", "PottedTree", "PortraitPainting", "LandscapePainting",
    "Television", "Clock", "WashingMachine", "Bathtube", "BathroomSink",
    "CeilingSpotLight", "Muffin", "SolidBox",
    "BiscuitBoxConnector", "CerealBoxConnector", "LaptopConnector",
    "CardboardBoxConnector", "RubberDuckConnector", "CookwareConnector",
    "LidConnector", "WoodenSpoonConnector", "DesktopComputerConnector",
    "KeyboardConnector", "MonitorConnector", "ComputerMouseConnector",
    "CarafeConnector", "PlateConnector", "WaterBottleConnector",
    "WineglassConnector", "BookConnector", "FruitBowlConnector",
    "AppleConnector", "OrangeConnector", "BeerBottleConnector", "CanConnector",
    "FireExtinguisherConnector", "OfficeTelephoneConnector",
}


def load_world(world_file):
    path = Path(world_file).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    if not text.startswith("#VRML_SIM"):
        raise ValueError(f"Expected a Webots world header in {path}")
    return path, text


def extract_nodes(text):
    """Scan node boundaries and a small set of direct fields, not PROTO bodies."""
    tokens = [match.group() for match in TOKENS.finditer(text)
              if not match.group().startswith("#")]
    nodes, roots, scopes, parents = [], [], [], []
    for index, token in enumerate(tokens):
        if token == "PROTO":
            raise ValueError("Inline PROTO definitions are not supported; use a saved world.")
        if token == "{":
            start = index - 1
            definition = None
            if index >= 3 and tokens[index - 3] == "DEF":
                definition = tokens[index - 2]
                start = index - 3
            field = None
            if scopes:
                field = scopes[-1][1] if scopes[-1][0] == "[" else tokens[start - 1]
            parent = parents[-1] if parents else None
            siblings = parent["children"] if parent else roots
            path = (parent["path"] if parent else "world") + f"/{tokens[index - 1]}[{len(siblings)}]"
            node = {
                "type": tokens[index - 1], "def": definition, "field": field,
                "fields": {}, "parent": parent, "children": [], "path": path,
            }
            nodes.append(node)
            siblings.append(node)
            parents.append(node)
            scopes.append(("{", None))
        elif token == "[":
            scopes.append(("[", tokens[index - 1]))
        elif token in {"}", "]"}:
            expected = "{" if token == "}" else "["
            if not scopes or scopes.pop()[0] != expected:
                raise ValueError("Unbalanced Webots node or list delimiters")
            if token == "}":
                parents.pop()
        elif token == "USE":
            field = scopes[-1][1] if scopes and scopes[-1][0] == "[" else tokens[index - 1]
            if field in SPATIAL_FIELDS or field.endswith("Slot"):
                warnings.warn("Spatial USE instances are not expanded by this extractor.", stacklevel=2)
        elif parents and scopes[-1][0] == "{":
            fields = parents[-1]["fields"]
            if token in {"name", "model"} and index + 1 < len(tokens):
                if tokens[index + 1].startswith('"'):
                    fields[token] = json.loads(tokens[index + 1])
            elif token in NUMERIC_FIELDS:
                values = []
                for value in tokens[index + 1:index + 5]:
                    try:
                        values.append(float(value))
                    except ValueError:
                        break
                if values:
                    fields[token] = values
    if scopes:
        raise ValueError("Unclosed Webots node or list")
    return nodes


def find_named_objects(nodes):
    """Keep named objects and assign IDs to known unnamed physical objects."""
    objects = []
    generated_counts = Counter()
    # Reserve explicit IDs before generating any, even if they occur later.
    reserved_names = set()
    for node in nodes:
        identifier = node["fields"].get("name") or node["def"]
        if identifier:
            reserved_names.add(identifier)
    for node in nodes:
        # Prefer the explicit name, then DEF; never promote rendering internals.
        name = node["fields"].get("name") or node["def"]
        if node["type"] in NON_OBJECT_TYPES:
            continue
        if not name and node["type"] not in SEMANTIC_OBJECT_TYPES:
            continue
        if node["type"] in {"Group", "Pose", "Transform"} and not node["children"]:
            parent = node["parent"]
            is_floor_child = (
                parent is not None
                and parent["fields"].get("name") == "floor"
                and node["field"] == "children"
            )
            if not is_floor_child:
                continue
        ancestor = node
        while ancestor["parent"] is not None:
            field = ancestor["field"] or ""
            if field not in SPATIAL_FIELDS and not field.endswith("Slot"):
                break
            ancestor = ancestor["parent"]
        else:
            # Only after checking the spatial ancestry, generate type_N for a
            # meaningful unnamed object. Counting in file order is repeatable.
            if not name:
                prefix = node["type"].lower()
                generated_counts[prefix] += 1
                name = f"{prefix}_{generated_counts[prefix]}"
                while name in reserved_names:
                    generated_counts[prefix] += 1
                    name = f"{prefix}_{generated_counts[prefix]}"
                reserved_names.add(name)
            node["id"] = name
            objects.append(node)

    counts = Counter(node["id"] for node in objects)
    reserved = set(counts)
    for node in objects:
        if counts[node["id"]] > 1:
            identifier = f"{node['id']}@{node['path']}"
            while identifier in reserved:
                identifier += "@"
            node["id"] = identifier
            reserved.add(identifier)
    return objects


# Interpret the world's floor convention without reclassifying other Pose nodes.
def identify_locations(objects):
    """Identify named direct spatial children of the object named 'floor'."""
    locations = {}
    for node in objects:
        parent = node["parent"]
        if parent is None or node["field"] != "children":
            continue
        if parent["fields"].get("name") == "floor":
            locations[node["id"]] = node
    return locations


def compute_area(node):
    """Use one direct Shape's Plane dimensions; do not search descendants."""
    visuals = []
    for child in node["children"]:
        if child["field"] == "children":
            visuals.append(child)
    if len(visuals) != 1 or visuals[0]["type"] != "Shape":
        return None
    for geometry in visuals[0]["children"]:
        if geometry["field"] != "geometry" or geometry["type"] != "Plane":
            continue
        dimensions = geometry["fields"].get("size", [])
        if len(dimensions) != 2:
            return None
        width, height = dimensions
        if not math.isfinite(width) or not math.isfinite(height):
            return None
        if width <= 0 or height <= 0:
            return None
        area = width * height
        if math.isfinite(area):
            return area
    return None


def _rotate_position(position, rotation):
    """Rotate a vector with Rodrigues' axis-angle formula."""
    if len(rotation) != 4 or not all(math.isfinite(value) for value in rotation):
        return None
    axis_x, axis_y, axis_z, angle = rotation
    if angle == 0:
        return list(position)
    length = math.hypot(axis_x, axis_y, axis_z)
    if length == 0:
        return None
    axis = [axis_x / length, axis_y / length, axis_z / length]
    x, y, z = position
    cross = [axis[1] * z - axis[2] * y,
             axis[2] * x - axis[0] * z,
             axis[0] * y - axis[1] * x]
    dot = sum(axis[index] * position[index] for index in range(3))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotated = []
    for index in range(3):
        rotated.append(position[index] * cosine + cross[index] * sine
                       + axis[index] * dot * (1 - cosine))
    return rotated


def _rotation_quaternion(rotation):
    """Represent a valid axis-angle as a unit quaternion [w, x, y, z]."""
    if rotation is None or len(rotation) != 4:
        return None
    if not all(math.isfinite(value) for value in rotation):
        return None
    x, y, z, angle = rotation
    if angle == 0:
        return [1.0, 0.0, 0.0, 0.0]
    length = math.hypot(x, y, z)
    if length == 0:
        return None
    factor = math.sin(angle / 2) / length
    return [math.cos(angle / 2), x * factor, y * factor, z * factor]


def _compose_rotations(parent, child):
    """Multiply parent * child: apply the child's rotation first."""
    w, x, y, z = parent
    a, b, c, d = child
    return [w*a - x*b - y*c - z*d,
            w*b + x*a + y*d - z*c,
            w*c - x*d + y*a + z*b,
            w*d + x*c - y*b + z*a]


def get_world_transform(node):
    """Return world position, rotation quaternion, and transformed basis vectors.

    The basis also retains Transform scaling for geometry bounds. Quaternion
    rotation stays separate because nonuniform scale can introduce shear.
    Unknown components remain None rather than borrowing hidden PROTO defaults.
    """
    builtins = {"Pose", "Transform", "Solid", "Robot", "Group"}
    fields = node["fields"]
    identity = [0.0, 0.0, 1.0, 0.0]
    rotation = fields.get("rotation")
    if rotation is None and node["type"] in builtins:
        rotation = identity
    quaternion = _rotation_quaternion(rotation)
    position = fields.get("translation")
    if position is not None:
        position = list(position) if len(position) == 3 else None

    # Start with the node's local rotation and scale. Its own rotation changes
    # its geometry axes, but does not rotate its translation (the local origin).
    basis = None
    if quaternion is not None:
        basis = []
        scale = fields.get("scale", [1.0, 1.0, 1.0])
        if node["type"] != "Transform":
            scale = [1.0, 1.0, 1.0]
        if len(scale) != 3:
            return None, None, None
        for axis in range(3):
            vector = [0.0, 0.0, 0.0]
            vector[axis] = scale[axis]
            basis.append(_rotate_position(vector, rotation))

    # Walk the complete hierarchy, including unnamed spatial ancestors.
    parent = node["parent"]
    while parent is not None:
        kind = parent["type"]
        if kind in JOINTS or kind == "Group":
            parent = parent["parent"]
            continue
        if kind not in builtins:
            # PROTO slots can hide offsets. The saved file cannot resolve them.
            return None, None, None
        fields = parent["fields"]
        rotation = fields.get("rotation", identity)
        parent_rotation = _rotation_quaternion(rotation)
        translation = fields.get("translation", [0.0, 0.0, 0.0])
        scale = [1.0, 1.0, 1.0]
        if kind == "Transform":
            scale = fields.get("scale", scale)
        if parent_rotation is None or len(translation) != 3 or len(scale) != 3:
            return None, None, None

        # First scale and rotate the current origin; then add parent translation.
        if position is not None:
            position = [position[i] * scale[i] for i in range(3)]
            position = _rotate_position(position, rotation)
            position = [position[i] + translation[i] for i in range(3)]

        # Compose parent rotation on the left. Transform geometry basis vectors
        # too, but never translate vectors: translation cancels when finding size.
        if quaternion is not None:
            quaternion = _compose_rotations(parent_rotation, quaternion)
        if basis is not None:
            for axis, vector in enumerate(basis):
                vector = [vector[i] * scale[i] for i in range(3)]
                basis[axis] = _rotate_position(vector, rotation)
        parent = parent["parent"]

    if position is not None and not all(math.isfinite(v) for v in position):
        position = None
    if basis is not None and not all(math.isfinite(v) for axis in basis for v in axis):
        basis = None
    return position, quaternion, basis


def get_world_position(node):
    """Return only the world origin [x, y, z], in metres."""
    position, _, _ = get_world_transform(node)
    return position


def get_world_orientation(node):
    """Convert the composed quaternion back to world-frame axis-angle."""
    _, quaternion, _ = get_world_transform(node)
    if quaternion is None:
        return None
    length = math.hypot(*quaternion)
    quaternion = [value / length for value in quaternion]
    if quaternion[0] < 0:
        quaternion = [-value for value in quaternion]
    w, x, y, z = quaternion
    sine = math.hypot(x, y, z)
    if sine < 1e-12:
        return [0.0, 0.0, 1.0, 0.0]
    return [x / sine, y / sine, z / sine, 2 * math.atan2(sine, w)]


def get_world_dimensions(node):
    """Bound direct geometry or an explicit 3D size in world coordinates."""
    # First look for one direct visual. Never borrow dimensions from child
    # objects: a component does not establish the dimensions of its parent.
    geometry = None
    visuals = [child for child in node["children"] if child["field"] == "children"]
    if len(visuals) == 1 and visuals[0]["type"] == "Shape":
        for child in visuals[0]["children"]:
            if child["field"] == "geometry":
                geometry = child
    if geometry is None and node["type"] == "SolidBox":
        geometry = node
    # Prefer supported direct geometry. Otherwise use only a complete size
    # explicitly written on the node (including PROTO instances such as Table).
    # Partial width/depth/height fields cannot establish a full bounding box.
    if geometry is not None and geometry["type"] in {"Box", "SolidBox", "Plane", "Sphere"}:
        dimensions = geometry["fields"].get("size", [])
        expected = 2 if geometry["type"] == "Plane" else 3
    else:
        geometry = None
        dimensions = node["fields"].get("size", [])
        expected = 3
    _, _, basis = get_world_transform(node)
    if basis is None:
        return None

    # A transformed sphere's exact axis bounds come from the affine basis.
    if geometry is not None and geometry["type"] == "Sphere":
        radius = geometry["fields"].get("radius", [])
        if len(radius) != 1 or not math.isfinite(radius[0]) or radius[0] <= 0:
            return None
        return [2 * radius[0] * math.hypot(*(axis[i] for axis in basis)) for i in range(3)]

    # Both sources enter the same validation and corner transformation below.
    # Require positive finite numbers; no fabricated dimensions or defaults.
    if not isinstance(dimensions, (list, tuple)) or len(dimensions) != expected:
        return None
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in dimensions):
        return None
    dimensions = list(dimensions)
    if expected == 2:
        dimensions.append(0.0)  # Webots Plane geometry lies in local XY.

    # Transform every local corner using the composed world basis. No 90-degree
    # special case: arbitrary rotations and ancestor scaling follow this path.
    corners = []
    for signs in product((-1, 1), repeat=3):
        local = [signs[i] * dimensions[i] / 2 for i in range(3)]
        corner = []
        for world_axis in range(3):
            corner.append(sum(local[i] * basis[i][world_axis] for i in range(3)))
        corners.append(corner)

    # World-aligned size is maximum minus minimum on each world axis.
    bounds = []
    for axis in range(3):
        coordinates = [corner[axis] for corner in corners]
        bounds.append(max(coordinates) - min(coordinates))
    parent = node["parent"]
    is_location = parent is not None and parent["fields"].get("name") == "floor" and node["field"] == "children"
    if is_location and expected == 2 and bounds[2] < 1e-10:
        return bounds[:2]
    return bounds


# Unknown qualities are omitted, but the object itself always remains.
def remove_unknown_qualities(qualities):
    known = {}
    for name, value in qualities.items():
        if value is not None:
            known[name] = value
    return known


def _instantiate_object(node, locations, object_schema):
    description = {
        "id": node["id"],
        "type": node["fields"].get("model") or node["type"],
    }
    qualities = {name: None for name in object_schema["properties"]["qualities"]["properties"]}
    fields = node["fields"]
    known = {}
    known["location"] = get_world_position(node)
    known["orientation"] = get_world_orientation(node)
    known["size"] = get_world_dimensions(node)
    if len(fields.get("color", [])) == 3:
        known["color"] = f"RGB [0, 1]: {json.dumps(fields['color'])}"

    # A single direct visual can describe the whole object; a child object cannot.
    visuals = [child for child in node["children"] if child["field"] == "children"]
    geometry = None
    if len(visuals) == 1 and visuals[0]["type"] == "Shape":
        for child in visuals[0]["children"]:
            if child["field"] == "geometry" and child["type"] in PRIMITIVES:
                geometry = child
                known["shape"] = child["type"]
            if child["field"] == "appearance" and child["type"] == "PBRAppearance":
                # A base color with a texture is a tint, not the object's full color.
                textured = any(item["field"] == "baseColorMap" for item in child["children"])
                color = child["fields"].get("baseColor")
                if not textured and color and len(color) == 3:
                    known.setdefault("color", f"RGB [0, 1]: {json.dumps(color)}")

    for name in qualities:
        qualities[name] = known.get(name)
    if node["id"] in locations:
        description["type"] = "Location"
        qualities["area"] = compute_area(node)
    description["qualities"] = remove_unknown_qualities(qualities)
    return description


def instantiate_objects(objects, locations, object_schema):
    """Describe every named object, adding the explicit location interpretation."""
    descriptions = []
    for node in objects:
        descriptions.append(_instantiate_object(node, locations, object_schema))
    return descriptions


def derive_relations(objects, allowed_predicates):
    """Only explicit joint endpoints establish attachments in this first version."""
    relations = []
    if "attached_to" not in allowed_predicates:
        return relations
    for node in objects:
        joint = node["parent"]
        if node["field"] != "endPoint" or not joint or joint["type"] not in JOINTS:
            continue
        parent = joint["parent"]
        while parent and parent["type"] in {"Group", "Pose", "Transform"}:
            parent = parent["parent"]
        if parent and "id" in parent and parent["type"] in {"Solid", "Robot"}:
            relations.append({
                "subject": node["id"], "predicate": "attached_to", "object": parent["id"],
            })
    return relations


def build_graph(descriptions, relations):
    """Keep object descriptions separate from relations between their IDs."""
    return {"objects": descriptions, "relations": relations}


def validate_graph(graph, object_schema, graph_schema):
    # Keep the existing relation contract while using objects.json for objects.
    schema = deepcopy(graph_schema)
    schema["properties"]["objects"]["items"] = object_schema
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(graph)
    identifiers = [item["id"] for item in graph["objects"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Scene object IDs must be unique")
    for relation in graph["relations"]:
        if relation["subject"] not in identifiers or relation["object"] not in identifiers:
            raise ValueError("Relation references an unknown object ID")


def save_graph(graph, world_path):
    """Save the generated reference beside its source world."""
    output = world_path.with_suffix(".scene_graph.json")
    output.write_text(json.dumps(graph, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def generate_scene_graph_from_simulation(world_file):
    """Save <world>.scene_graph.json beside the source world and return its graph."""
    # Load the world and the existing object/relation contracts.
    path, text = load_world(world_file)
    object_schema = json.loads((SCHEMAS / "objects.json").read_text(encoding="utf-8"))
    graph_schema = json.loads(
        (SCHEMAS / "perception" / "create_scene_graph.json").read_text(encoding="utf-8")
    )
    allowed = graph_schema["properties"]["relations"]["items"]["properties"]["predicate"]["enum"]

    # Extract named objects, then interpret only the floor's direct children.
    nodes = extract_nodes(text)
    objects = find_named_objects(nodes)
    locations = identify_locations(objects)

    # Populate known qualities and retain only reliably derived relations.
    descriptions = instantiate_objects(objects, locations, object_schema)
    relations = derive_relations(objects, allowed)
    graph = build_graph(descriptions, relations)

    # Validate the semantic result before replacing the generated file.
    validate_graph(graph, object_schema, graph_schema)
    save_graph(graph, path)
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world_file", type=Path)
    args = parser.parse_args()
    graph = generate_scene_graph_from_simulation(args.world_file)
    print(f"Saved {len(graph['objects'])} objects and {len(graph['relations'])} relations to "
          f"{args.world_file.resolve().with_suffix('.scene_graph.json')}")


if __name__ == "__main__":
    main()
