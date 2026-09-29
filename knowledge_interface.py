"""CMOC semantic-memory facade; all knowledge access is owned by RoboKGNet."""

from external.robokgnet.knowledge_interface import (
    DATA,
    KnowledgeInterface as RoboKGNetKnowledgeInterface,
)


class KnowledgeInterface:
    """Delegate to RoboKGNet without changing its resolution or storage semantics.

    Defaults use the submodule's canonical data. To persist local updates without
    changing that checkout, supply a writable copy as concepts_path.
    """

    def __init__(self, concepts_path=DATA / "robokgconceptnet.json",
                 actions_path=DATA / "robokgverbnet_framenet.json"):
        self._backend = RoboKGNetKnowledgeInterface(concepts_path, actions_path)

    def resolve_concept(self, term):
        return self._backend.resolve_concept(term)

    def get_concept(self, concept_id):
        return self._backend.get_concept(concept_id)

    def get_alternative_names(self, concept_id):
        return self._backend.get_alternative_names(concept_id)

    def get_superclasses(self, concept_id):
        return self._backend.get_superclasses(concept_id)

    def get_subclasses(self, concept_id):
        return self._backend.get_subclasses(concept_id)

    def get_locations(self, concept_id):
        return self._backend.get_locations(concept_id)

    def update_location(self, concept_id, location_synset, increment=1):
        return self._backend.update_location(concept_id, location_synset, increment)

    def save(self):
        return self._backend.save()

    def resolve_action(self, term):
        return self._backend.resolve_action(term)

    def get_action(self, action_id):
        return self._backend.get_action(action_id)

    def get_action_frame(self, action_id):
        return self._backend.get_action_frame(action_id)

    def get_frame_roles(self, action_id):
        return self._backend.get_frame_roles(action_id)

    def get_role(self, action_id, role):
        return self._backend.get_role(action_id, role)

    def get_additional_frame_elements(self, action_id):
        return self._backend.get_additional_frame_elements(action_id)
