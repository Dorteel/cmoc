"""Project-local identifiers, with source identities recorded separately."""
from urllib.parse import quote
from rdflib import Namespace, RDF, RDFS, OWL, Literal
from rdflib.namespace import DCTERMS, PROV, SKOS

KG = Namespace('https://example.org/cmoc/robokgnet/')
WN = Namespace(str(KG) + 'wordnet/')
VN = Namespace(str(KG) + 'verbnet/')
FN = Namespace(str(KG) + 'framenet/')
SOURCE = Namespace(str(KG) + 'source/')


def identifier(namespace, text):
    return namespace[quote(str(text), safe='')]


def entity(graph, node, kind, source, label=None):
    graph.add((node, RDF.type, kind))
    graph.add((node, PROV.wasDerivedFrom, source))
    if label is not None:
        graph.add((node, RDFS.label, Literal(label)))
    return node


def schema(graph):
    for prefix, ns in [('kg', KG), ('wn', WN), ('vn', VN), ('fn', FN),
                       ('source', SOURCE), ('prov', PROV), ('dcterms', DCTERMS),
                       ('skos', SKOS), ('owl', OWL)]:
        graph.bind(prefix, ns)
    entity(graph, KG.schema, OWL.Ontology, SOURCE.CMOC, 'RoboKGNet integration vocabulary')
    for name in ('VerbClass', 'ThematicRole', 'VerbFrame', 'SemanticPredicate',
                 'Argument', 'Event', 'EventPhase', 'Frame', 'FrameElement',
                 'LexicalUnit', 'Alignment', 'CommonsenseAssertion'):
        entity(graph, KG[name], OWL.Class, SOURCE.CMOC, name)
    for name in ('hasRole', 'member', 'hasFrame', 'hasPredicate', 'argument',
                 'refersTo', 'event', 'hasFrameElement', 'lexicalUnit',
                 'denotesConcept', 'planningEntity'):
        entity(graph, KG[name], OWL.ObjectProperty, SOURCE.CMOC, name)
    for name in ('weight', 'position', 'rawValue', 'argumentType', 'phase',
                 'predicateName', 'negated', 'coreType', 'definition', 'planningKind'):
        entity(graph, KG[name], OWL.DatatypeProperty, SOURCE.CMOC, name)
    for name in ('LocatedAt', 'UsedFor'):
        entity(graph, KG[name], RDF.Property, SOURCE.CMOC, name)
    for name, label in [('WordNet', 'NLTK WordNet 3.0'), ('VerbNet', 'NLTK VerbNet 2.1'),
                        ('FrameNet', 'NLTK FrameNet 1.7'), ('CMOC', 'CMOC integration'),
                        ('DemoFixtures', 'Hand-authored demo fixtures; not empirical confidence')]:
        graph.add((SOURCE[name], RDF.type, PROV.Entity))
        graph.add((SOURCE[name], RDFS.label, Literal(label)))
