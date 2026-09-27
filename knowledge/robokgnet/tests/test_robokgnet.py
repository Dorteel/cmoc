"""Small graph queries against installed corpora and the offline demo snapshot."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef
from rdflib.compare import isomorphic
from nltk.corpus import framenet as fn
from knowledge.robokgnet import RoboKGNetBuilder
from knowledge.robokgnet.demo import build_demo, HERE
from knowledge.robokgnet.namespaces import KG, WN, VN, FN, SOURCE, PROV, identifier

PREFIX = f'PREFIX kg: <{KG}> PREFIX rdfs: <{RDFS}> PREFIX rdf: <{RDF}> '


class RoboKGNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # No SemLink requests, corpus downloads, or model calls during building.
        with patch('tools.semantic_bridge._load_semlink', side_effect=AssertionError('offline build')), \
             patch('nltk.download', side_effect=AssertionError('offline build')):
            cls.kg = build_demo()
        cls.graph = cls.kg.graph

    def test_wordnet_actual_superclasses_and_explicit_senses(self):
        rows = list(self.graph.query(PREFIX + '''
            SELECT ?parent WHERE { ?mug rdfs:subClassOf+ ?parent }
        ''', initBindings={'mug': WN['coffee_mug.n.01']}))
        self.assertIn((WN['physical_entity.n.01'],), rows)
        self.assertEqual(self.kg.concepts['robot'], WN['automaton.n.02'])
        with self.assertRaises(ValueError):
            RoboKGNetBuilder().add_wordnet_concept('person')

    def test_explicit_member_sense_keys_connect_wordnet_to_verbnet(self):
        unit = identifier(VN, 'bring-11.3-1/member/bring')
        self.assertIn((unit, KG.denotesConcept, WN['bring.v.01']), self.graph)
        self.assertEqual(self.kg.warnings, [])

    def test_bring_roles_and_subclass_inheritance(self):
        rows = self.graph.query(PREFIX + '''SELECT ?name WHERE {
            ?class rdfs:subClassOf*/kg:hasRole/rdfs:label ?name
        }''', initBindings={'class': VN['bring-11.3-1']})
        self.assertTrue({'Agent', 'Theme', 'Source', 'Destination'} <= {str(r.name) for r in rows})

    def test_end_location_is_structured_and_refers_to_theme(self):
        rows = list(self.graph.query(PREFIX + '''SELECT ?predicate WHERE {
            ?class kg:hasFrame/kg:hasPredicate ?predicate .
            ?predicate kg:predicateName "location" ; kg:argument ?time, ?theme .
            ?time kg:refersTo/kg:phase "end" .
            ?theme kg:refersTo/rdfs:label "Theme" .
        }''', initBindings={'class': VN['bring-11.3-1']}))
        self.assertTrue(rows)
        for predicate in self.graph.subjects(RDF.type, KG.SemanticPredicate):
            args = list(self.graph.objects(predicate, KG.argument))
            self.assertEqual(sorted(int(self.graph.value(a, KG.position)) for a in args),
                             list(range(len(args))))

    def test_frame_elements_have_exact_resource_definitions(self):
        frame = fn.frame_by_name('Bringing')
        for name in ('Agent', 'Theme', 'Source', 'Goal'):
            element = identifier(FN, 'Bringing/FE/' + name)
            self.assertIn((FN.Bringing, KG.hasFrameElement, element), self.graph)
            self.assertEqual(str(self.graph.value(element, KG.definition)), frame.FE[name].definition)

    def test_member_qualified_alignment_and_no_guessed_search_frame(self):
        rows = list(self.graph.query(PREFIX + '''SELECT ?member WHERE {
            ?a a kg:Alignment ; rdf:subject ?class ; rdf:object ?frame ; kg:member ?member .
        }''', initBindings={'class': VN['bring-11.3-1'], 'frame': FN.Bringing}))
        self.assertEqual(len(rows), 1)
        self.assertIn((VN['bring-11.3-1'], KG.member, rows[0].member), self.graph)
        snapshot = json.loads((HERE / 'data/semlink_subset.json').read_text())
        self.assertEqual(len(list(self.graph.subjects(RDF.type, KG.Alignment))), len(snapshot['alignments']))
        self.assertFalse(any(self.graph.triples((FN.Search, None, None))))
        # The available SemLink table has class/member links, no role links.
        self.assertFalse(any(self.graph.triples((None, OWL.equivalentClass, None))))

    def test_weighted_assertions_are_typed_and_have_fixture_provenance(self):
        rows = list(self.graph.query(PREFIX + '''SELECT ?weight WHERE {
            ?assertion a kg:CommonsenseAssertion ; rdf:subject ?mug ;
                       rdf:predicate kg:LocatedAt ; kg:weight ?weight .
        }''', initBindings={'mug': WN['coffee_mug.n.01']}))
        self.assertEqual(sorted(float(r.weight) for r in rows), [0.6, 0.95])
        for assertion in self.graph.subjects(RDF.type, KG.CommonsenseAssertion):
            self.assertIn((assertion, PROV.wasDerivedFrom, SOURCE.DemoFixtures), self.graph)
        self.assertFalse(any(self.graph.triples((KG.LocatedAt, KG.weight, None))))
        for value in (-0.1, 1.1, math.nan, math.inf, True, '0.9'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RoboKGNetBuilder().add_commonsense('mug', 'LocatedAt', 'kitchen', value,
                                                  source=SOURCE.DemoFixtures)

    def test_external_planning_ontology_is_preserved_not_redeclared(self):
        external = Namespace('https://example.test/existing-planning#')
        source = URIRef('https://example.test/planning.ttl')
        existing = Graph()
        existing.add((external.PlanningAction, RDF.type, OWL.Class))
        existing.add((external.bring, RDF.type, external.PlanningAction))
        kg = RoboKGNetBuilder()
        kg.add_planning_knowledge(existing, source=source)
        semantic = kg.add_verbnet_class('bring-11.3')
        kg.link_planning(semantic, external.bring, source=SOURCE.DemoFixtures)
        self.assertTrue(all(t in kg.graph for t in existing))
        self.assertIn((semantic, KG.planningEntity, external.bring), kg.graph)
        self.assertNotIn((KG.PlanningAction, RDF.type, OWL.Class), kg.graph)
        self.assertIn((external.bring, PROV.wasDerivedFrom, source), kg.graph)
        self.assertEqual(len(existing), 2)

    def test_pddl_links_reference_real_actions_without_new_planning_classes(self):
        targets = list(self.graph.objects(VN['search-35.2'], KG.planningEntity))
        self.assertEqual(len(targets), 1)
        self.assertEqual(str(self.graph.value(targets[0], RDFS.label)), 'look_for')
        for kind in ('PlanningDomain', 'PlanningAction', 'PlanningPredicate', 'PlanningProblem',
                     'InitialState', 'GoalState'):
            self.assertTrue(list(self.graph.subjects(KG.planningKind, Literal(kind))))
            self.assertNotIn((KG[kind], RDF.type, OWL.Class), self.graph)

    def test_imported_records_have_provenance(self):
        for subject in set(self.graph.subjects()):
            if str(subject).startswith((str(WN), str(VN), str(FN))):
                self.assertTrue(list(self.graph.objects(subject, PROV.wasDerivedFrom)), subject)

    def test_turtle_roundtrip_and_checked_in_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'graph.ttl'
            self.kg.save(path)
            self.assertTrue(isomorphic(self.graph, Graph().parse(path)))
        self.assertTrue(isomorphic(self.graph, Graph().parse(HERE / 'robokgnet.ttl')))

    def test_reimports_are_idempotent(self):
        kg = RoboKGNetBuilder()
        kg.add_verbnet_class('bring-11.3-1')
        before = len(kg.graph)
        kg.add_verbnet_class('bring-11.3-1')
        self.assertEqual(len(kg.graph), before)


if __name__ == '__main__':
    unittest.main()
