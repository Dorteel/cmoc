"""Weighted assertions; weights qualify assertions, never RDF predicates."""
import hashlib
import json
import math
from rdflib import Literal, RDF, URIRef
from .namespaces import KG, entity


def add_assertion(graph, subject, relation, object, weight, source):
    if relation not in ('LocatedAt', 'UsedFor'):
        raise ValueError('relation must be LocatedAt or UsedFor')
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('weight must be a finite number in [0, 1]')
    if not source:
        raise ValueError('Commonsense assertions require provenance')
    weight = float(weight)
    digest = hashlib.sha256(json.dumps([str(subject), relation, str(object), weight, str(source)]).encode()).hexdigest()
    node = entity(graph, KG['assertion/' + digest], KG.CommonsenseAssertion, URIRef(source))
    graph.add((node, RDF.subject, subject))
    graph.add((node, RDF.predicate, KG[relation]))
    graph.add((node, RDF.object, object))
    graph.add((node, KG.weight, Literal(weight)))
    return node
