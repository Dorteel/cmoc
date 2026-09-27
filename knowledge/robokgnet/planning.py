"""Preserve external planning RDF; expose existing PDDL without a new ontology.

No RDF planning ontology/converter is present in this CMOC checkout. PDDL
entities are therefore source references (prov:Entity), not substitute OWL
planning classes. An existing RDF ontology can be merged without renaming it.
"""
from pathlib import Path
from urllib.parse import quote
from rdflib import Graph, Literal, RDF, URIRef
from procedural_memory.planning.planner import PDDLReader
from .namespaces import KG, SOURCE, PROV, DCTERMS, entity


def import_rdf(graph, knowledge, source):
    incoming = knowledge if isinstance(knowledge, Graph) else Graph().parse(knowledge)
    # Keep all existing terms, classes, relationships and namespaces unchanged.
    for prefix, ns in incoming.namespaces():
        graph.bind(prefix, ns, replace=False)
    for triple in incoming:
        graph.add(triple)
    for subject in set(incoming.subjects()):
        graph.add((subject, PROV.wasDerivedFrom, URIRef(source)))
    return graph


def add_pddl_references(graph, domain_path, problem_path, source_base):
    """Reuse CMOC's UP parser; return queryable references, not a PDDL converter."""
    problem = PDDLReader().parse_problem(str(domain_path), str(problem_path))
    source_base = source_base.rstrip('/')
    domain_source = URIRef(source_base + '/domain.pddl')
    problem_source = URIRef(source_base + '/problem.pddl')
    for source, path in [(domain_source, domain_path), (problem_source, problem_path)]:
        entity(graph, source, PROV.Entity, SOURCE.CMOC, Path(path).name)
        graph.add((source, DCTERMS.format, Literal('application/pddl')))

    def reference(key, kind, label, source):
        node = entity(graph, URIRef(str(source) + '#' + quote(key, safe='')),
                      PROV.Entity, source, label)
        graph.add((node, KG.planningKind, Literal(kind)))
        return node

    domain = reference('domain', 'PlanningDomain', Path(domain_path).name, domain_source)
    problem_node = reference('problem', 'PlanningProblem', problem.name, problem_source)
    graph.add((problem_node, DCTERMS.isPartOf, domain))
    actions = {}
    for action in problem.actions:
        node = reference(action.name, 'PlanningAction', action.name, domain_source)
        actions[action.name] = node
        graph.add((domain, DCTERMS.hasPart, node))
        graph.add((node, DCTERMS.description, Literal(str(action))))
    for fluent in problem.fluents:
        node = reference('predicate/' + fluent.name, 'PlanningPredicate', fluent.name, domain_source)
        graph.add((domain, DCTERMS.hasPart, node))
        graph.add((node, DCTERMS.description, Literal(str(fluent))))
    initial = reference('initial', 'InitialState', 'Parsed initial state (including UP defaults)', problem_source)
    goal = reference('goal', 'GoalState', 'Conjunctive goals', problem_source)
    for state in (initial, goal):
        graph.add((problem_node, DCTERMS.hasPart, state))
    for fluent, value in problem.initial_values.items():
        graph.add((initial, DCTERMS.description, Literal(f'{fluent} = {value}')))
    for value in problem.goals:
        graph.add((goal, DCTERMS.description, Literal(str(value))))
    return {'domain': domain, 'problem': problem_node, 'actions': actions}


def link(graph, semantic, planning, source):
    if not any(graph.triples((semantic, None, None))) or not any(graph.triples((planning, None, None))):
        raise ValueError('Import both semantic and planning endpoints first')
    # Reify a curated link so its provenance is distinct from corpus assertions.
    import hashlib
    digest = hashlib.sha256(f'{semantic}|{planning}|{source}'.encode()).hexdigest()
    node = entity(graph, KG['planning-link/' + digest], RDF.Statement, URIRef(source))
    graph.add((node, RDF.subject, semantic))
    graph.add((node, RDF.predicate, KG.planningEntity))
    graph.add((node, RDF.object, planning))
    graph.add((semantic, KG.planningEntity, planning))
    return node
