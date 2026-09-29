"""Transient instance correspondence for execution, never a knowledge source."""
from copy import deepcopy
import re

from external.webots_ros2_simulation.controllers.fallback_action_supervisor.action_cli import send_action
from perceived_entity_linking import resolve_concept
from search_strategy import search_trace
from ros_camera_geometry import acquire_camera_geometry


class ExecutionGroundingOracle:
    # This Oracle substitutes for a real perception/localization stack. Its
    # simulator output is execution-only, never perceptual/Search evidence or
    # semantic knowledge. Perceived IDs and simulator IDs remain distinct.
    def __init__(self, observed, robokg, *, debug=False, current_observed_ids=(), current_identity_mapping=None):
        self._objects = {o['id']: deepcopy(o) for o in observed.get('objects', [])}
        self._robokg = robokg
        self.debug = debug
        raw_ids = frozenset(current_observed_ids)
        # Only this Sense/PEL call establishes current correspondence. Stored
        # aliases, old evidence, and simulator bindings cannot extend visibility.
        self._current_identity_mapping = {
            raw: canonical for raw, canonical in (current_identity_mapping or {}).items()
            if raw in raw_ids}
        self._current_observed_ids = raw_ids | frozenset(self._current_identity_mapping.values())
        self.execution_bindings = {}
        self.optical_targets = {}

    def resolve(self, perceived_id, *, require_current=False):
        def log(message):
            search_trace(self.debug, message)

        log(f"Execution grounding: perceived instance={perceived_id}")
        obj = self._objects.get(perceived_id)
        if obj is None:
            raise RuntimeError(f"Cannot execution-ground {perceived_id}: no observed instance")
        if require_current and perceived_id not in self._current_observed_ids:
            raise RuntimeError(f"Cannot execution-ground {perceived_id}: not currently observed")
        log(f"Gaze target: {perceived_id}")
        log("Same-name simulator lookup: ignored (perceived identity)")
        # Reuse PEL concept resolution; split Webots-style CamelCase labels
        # before resolution. Taxonomic compatibility is not instance identity.
        label = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", obj['type']).replace('_', ' ')
        concept = resolve_concept(label, self._robokg)
        compatible_types = []
        # Preserve legacy concept/parent compatibility. Camera grounding also
        # accepts stored subtypes of the perceived concept, never sibling types
        # reached through a broad common superclass.
        if concept is not None:
            parents = self._robokg.get_superclasses(concept)
            for identifier in [concept, *(parents if isinstance(parents, list) else [])]:
                entry = self._robokg.get_concept(identifier)
                if isinstance(entry, dict) and isinstance(entry.get('type'), str):
                    compatible_types.append(entry['type'])
        entries = self._robokg.resolve_concept(label.casefold())
        if not entries and ' ' in label:
            entries = self._robokg.resolve_concept(label.casefold().replace(' ', '_'))
        pending = ([(entry['id'], [entry['id']]) for entry in entries]
                   if perceived_id in self._current_observed_ids else [])
        seen = set()
        while pending:
            identifier, path = pending.pop()
            if identifier in seen:
                continue
            seen.add(identifier)
            entry = self._robokg.get_concept(identifier)
            if isinstance(entry, dict) and isinstance(entry.get('type'), str):
                if entry['type'] not in compatible_types:
                    compatible_types.append(entry['type'])
                for alias in entry.get('alternative_names', []):
                    if resolve_concept(alias, self._robokg) == identifier:
                        compatible_types.append(alias)
                log(f"[GROUNDING] Compatible semantic type {entry['type']}: "
                    f"RoboKG subtype/equivalence evidence {' -> '.join(path)}")
            children = self._robokg.get_subclasses(identifier)
            if isinstance(children, list):
                pending.extend((child, [*path, child]) for child in children)
        log(f"  raw concept={obj['type']!r}; canonical concept={concept!r}")
        log(f"[GROUNDING] Perceived entity: {perceived_id}")
        log(f"[GROUNDING] Semantic type: {obj['type']}; PEL concept={concept!r}")
        log(f"  compatible canonical types (RoboKG lexical concepts and subtypes)={compatible_types!r}")
        log("Current-observation provenance:")
        for raw, canonical in self._current_identity_mapping.items():
            if canonical == perceived_id or raw == perceived_id:
                log(f"  raw entity {raw} -> retained {canonical}")
        current = perceived_id in self._current_observed_ids
        log(f"  selected anchor {perceived_id} observed_this_cycle={'yes' if current else 'no'}")
        log(f"  currently observed anchor: {'yes' if current else 'no'}")
        log(f"  camera grounding enabled: {'yes' if current else 'no'}")
        log(f"Identity source: {'current VLM observation' if current else 'retained observation'}")
        if current:
            log("  attempting camera-relative grounding")
        if not current:
            log("  camera grounding skipped: target absent from current Sense IDs; retained evidence alone is insufficient")
        try:
            camera_context = None
            if current:
                robot = send_action('get_object_pose', {'target': 'TIAGo'})
                if not robot.get('ok'):
                    raise RuntimeError(f"Cannot check simulator/map alignment: {robot.get('error')}")
                camera_context = acquire_camera_geometry(robot['result']['position'], log)
            response = send_action('resolve_execution_instance', {
                'perceived_id': perceived_id, 'perceived_type': obj['type'],
                'concept': concept, 'qualities': obj.get('qualities', {}),
                'compatible_types': compatible_types,
                'use_camera': current,
                **({'camera_context': camera_context} if current else {}),
            })
            if not response.get('ok'):
                raise RuntimeError(response.get('error', 'unknown supervisor error'))
            result = response['result']
            target = result['target']
            if not isinstance(target, str) or not target:
                raise ValueError('invalid execution instance')
            log(f"  compatible simulator instances={result['candidates']!r}")
            for detail in result.get("camera_diagnostics", []):
                log(f"  {detail}")
            log(f"  resolved execution instance={target}")
            if current:
                self.optical_targets[perceived_id] = result["optical_target"]
            self.execution_bindings[perceived_id] = target
            return target
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            log(f"  execution grounding failed: {error}")
            raise RuntimeError(f"Cannot execution-ground {perceived_id}: {error}") from error
