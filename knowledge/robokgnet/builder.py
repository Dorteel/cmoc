"""A small RDF graph builder; no network downloads or application execution."""
import json
from pathlib import Path
from rdflib import Graph, URIRef, Literal
from . import wordnet, verbnet, framenet, commonsense, planning
from .namespaces import KG, SOURCE, SKOS, PROV, DCTERMS, entity, identifier, schema


class RoboKGNetBuilder:
    def __init__(self):
        self.graph = Graph()
        schema(self.graph)
        self.concepts = {}
        self.warnings = []

    def add_wordnet_concept(self, concept, synset=None):
        node = wordnet.add_concept(self.graph, concept, synset)
        self.concepts[concept.replace(' ', '_')] = node
        return node

    def add_verbnet_class(self, class_id):
        return verbnet.add_class(self.graph, class_id)

    def add_member_wordnet_senses(self, class_id, member):
        """Follow only exact sense keys declared by a selected VerbNet member."""
        from tools.semantic_bridge import _find_class, _find_synset, _synsets_for_member
        from .namespaces import VN
        xml = _find_class(class_id)
        class_id = xml.get('ID')
        record = next((m for m in xml.findall('MEMBERS/MEMBER') if m.get('name') == member), None)
        if record is None:
            raise ValueError(f'{member} is not a direct member of {class_id}')
        self.add_verbnet_class(class_id)
        senses = _synsets_for_member(record, class_id, self.warnings)
        unit = identifier(VN, f'{class_id}/member/{member}')
        for sense in senses:
            concept = wordnet.add_synset(self.graph, _find_synset(sense))
            self.graph.add((unit, KG.denotesConcept, concept))
        return senses

    def add_framenet_frame(self, name):
        return framenet.add_frame(self.graph, name)

    def add_alignments(self, path):
        """Import an explicit SemLink subset produced by refresh_semlink.py."""
        data = json.loads(Path(path).read_text())
        snapshot = identifier(SOURCE, 'SemLink-' + data['parsed_table_sha256'])
        entity(self.graph, snapshot, PROV.Entity, URIRef(data['source']), 'SemLink2 subset')
        self.graph.add((snapshot, DCTERMS.identifier, Literal(data['parsed_table_sha256'])))
        self.graph.add((snapshot, DCTERMS.date, Literal(data['retrieved'])))
        results = []
        for record in data['alignments']:
            self.add_verbnet_class(record['verbnet'])
            self.add_framenet_frame(record['framenet'])
            results.append(framenet.add_alignment(self.graph, record['verbnet'], record['member'],
                                                  record['framenet'], snapshot))
        return results

    def add_commonsense(self, subject, relation, object, weight, *, source):
        def concept(value):
            if isinstance(value, URIRef):
                return value
            key = value.replace(' ', '_')
            if key in self.concepts:
                return self.concepts[key]
            return entity(self.graph, identifier(KG, 'concept/' + key), SKOS.Concept,
                          URIRef(source), value.replace('_', ' '))
        return commonsense.add_assertion(self.graph, concept(subject), relation, concept(object), weight, source)

    def add_planning_knowledge(self, knowledge, *, source):
        return planning.import_rdf(self.graph, knowledge, source)

    def add_pddl_references(self, domain_path, problem_path, *, source_base):
        return planning.add_pddl_references(self.graph, domain_path, problem_path, source_base)

    def link_planning(self, semantic, planning_entity, *, source):
        return planning.link(self.graph, semantic, planning_entity, source)

    def save(self, path):
        self.graph.serialize(destination=str(path), format='turtle')
        return Path(path)
