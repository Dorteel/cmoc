"""Preserve direct VerbNet frames, argument order, events and role references."""
import re
from nltk.corpus import verbnet as vn
from rdflib import Literal, RDFS
from tools.verbnet_semantic_patterns import load_class
from .namespaces import KG, VN, SOURCE, DCTERMS, entity, identifier


def add_class(graph, class_id):
    class_id, roles, frames = load_class(class_id)
    node = identifier(VN, class_id)
    entity(graph, node, KG.VerbClass, SOURCE.VerbNet, class_id)
    graph.add((node, DCTERMS.identifier, Literal(class_id)))
    # Resolve inheritance from XML nesting, retaining the declaring class's URI.
    root = vn.vnclass(vn.fileids(class_id)[0])
    def ancestry(element):
        if element.get('ID') == class_id:
            return []
        for child in element.findall('SUBCLASSES/VNSUBCLASS'):
            path = ancestry(child)
            if path is not None:
                return [element.get('ID')] + path
        return None
    role_nodes = {}
    for parent_id in ancestry(root) or []:
        parent = add_class(graph, parent_id)
        for role_node in graph.objects(parent, KG.hasRole):
            role_nodes[str(graph.value(role_node, RDFS.label))] = role_node
    for role in roles:
        name = role['type']
        role_nodes[name] = entity(graph, identifier(VN, f'{class_id}/role/{name}'),
                                  KG.ThematicRole, SOURCE.VerbNet, name)
        graph.add((node, KG.hasRole, role_nodes[name]))
    xml = vn.vnclass(class_id)
    for member in xml.findall('MEMBERS/MEMBER'):
        unit = entity(graph, identifier(VN, f'{class_id}/member/{member.get("name")}'),
                      KG.LexicalUnit, SOURCE.VerbNet, member.get('name'))
        graph.add((node, KG.member, unit))
    # Only explicit direct subclass membership in the XML, not ID heuristics.
    for child in xml.findall('SUBCLASSES/VNSUBCLASS'):
        child_node = entity(graph, identifier(VN, child.get('ID')), KG.VerbClass,
                            SOURCE.VerbNet, child.get('ID'))
        graph.add((child_node, RDFS.subClassOf, node))
    for i, frame in enumerate(frames):
        base = f'{class_id}/frame/{i}'
        f = entity(graph, identifier(VN, base), KG.VerbFrame, SOURCE.VerbNet,
                   frame['description']['primary'])
        graph.add((node, KG.hasFrame, f))
        graph.add((f, DCTERMS.description, Literal(frame['example'], lang='en')))
        for j, predicate in enumerate(frame['semantics']):
            p = entity(graph, identifier(VN, f'{base}/predicate/{j}'),
                       KG.SemanticPredicate, SOURCE.VerbNet)
            graph.add((f, KG.hasPredicate, p))
            graph.add((p, KG.predicateName, Literal(predicate['predicate_value'])))
            graph.add((p, KG.negated, Literal(predicate['negated'])))
            for k, argument in enumerate(predicate['arguments']):
                a = entity(graph, identifier(VN, f'{base}/predicate/{j}/arg/{k}'),
                           KG.Argument, SOURCE.VerbNet)
                graph.add((p, KG.argument, a))
                graph.add((a, KG.position, Literal(k)))
                graph.add((a, KG.argumentType, Literal(argument['type'])))
                value = argument['value']
                graph.add((a, KG.rawValue, Literal(value)))
                if argument['type'] == 'ThemRole' and value in role_nodes:
                    graph.add((a, KG.refersTo, role_nodes[value]))
                if argument['type'] == 'Event':
                    match = re.fullmatch(r'(?:(start|during|end)\()?([Ee]\d*)\)?', value)
                    if match:
                        phase, event = match.groups()
                        e = entity(graph, identifier(VN, f'{base}/event/{event}'),
                                   KG.Event, SOURCE.VerbNet, event)
                        target = e
                        if phase:
                            target = entity(graph, identifier(VN, f'{base}/event/{event}/{phase}'),
                                            KG.EventPhase, SOURCE.VerbNet)
                            graph.add((target, KG.event, e))
                            graph.add((target, KG.phase, Literal(phase)))
                        graph.add((a, KG.refersTo, target))
    return node
