"""Read exact FrameNet definitions and the existing SemLink mapping format."""
from rdflib import Literal, RDF, URIRef
from tools.semantic_bridge import _find_frame
from .namespaces import KG, FN, VN, SOURCE, DCTERMS, entity, identifier


def add_frame(graph, name):
    frame = _find_frame(name)
    node = entity(graph, identifier(FN, frame.name), KG.Frame, SOURCE.FrameNet, frame.name)
    graph.add((node, DCTERMS.identifier, Literal(frame.ID)))
    graph.add((node, KG.definition, Literal(frame.definition, lang='en')))
    for name, fe in frame.FE.items():
        element = entity(graph, identifier(FN, f'{frame.name}/FE/{name}'),
                         KG.FrameElement, SOURCE.FrameNet, name)
        graph.add((node, KG.hasFrameElement, element))
        graph.add((element, DCTERMS.identifier, Literal(fe.ID)))
        graph.add((element, KG.definition, Literal(fe.definition, lang='en')))
        graph.add((element, KG.coreType, Literal(fe.coreType)))
    for name, lu in frame.lexUnit.items():
        unit = entity(graph, identifier(FN, f'LU/{lu.ID}'), KG.LexicalUnit, SOURCE.FrameNet, name)
        graph.add((node, KG.lexicalUnit, unit))
    return node


def add_alignment(graph, class_id, member, frame_name, source):
    """Member-qualified alignment, not class equivalence or an invented role map."""
    cls, frame = identifier(VN, class_id), identifier(FN, frame_name)
    if (cls, RDF.type, KG.VerbClass) not in graph or (frame, RDF.type, KG.Frame) not in graph:
        raise ValueError('Import both alignment endpoints first')
    unit = identifier(VN, f'{class_id}/member/{member}')
    if (cls, KG.member, unit) not in graph:
        raise ValueError(f'{member} is not a direct member of {class_id}')
    a = entity(graph, identifier(KG, f'alignment/{class_id}/{member}/{frame_name}'),
               KG.Alignment, URIRef(source))
    graph.add((a, RDF.subject, cls))
    graph.add((a, KG.member, unit))
    graph.add((a, RDF.object, frame))
    return a
