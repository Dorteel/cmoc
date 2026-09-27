"""Import one disambiguated noun sense and its real WordNet ancestors."""
from nltk.corpus import wordnet as wn
from tools.semantic_bridge import _find_synset
from rdflib import Literal, OWL, RDFS
from .namespaces import WN, SOURCE, DCTERMS, SKOS, entity, identifier


def add_concept(graph, concept, synset=None):
    if synset is not None or '%' in concept or '.n.' in concept:
        selected = _find_synset(synset or concept)
    else:
        senses = wn.synsets(concept.replace(' ', '_'), pos='n')
        if len(senses) != 1:
            raise ValueError(f'{concept!r}: choose an explicit synset from {[s.name() for s in senses]}')
        selected = senses[0]
    if selected.pos() != 'n':
        raise ValueError('Concept hierarchy requires a noun synset')
    return add_synset(graph, selected)


def add_synset(graph, selected):
    if selected.instance_hypernyms():
        raise ValueError('Named WordNet instances are not supported as concept classes')
    seen = set()

    def visit(sense):
        node = identifier(WN, sense.name())
        if node in seen:
            return node
        seen.add(node)
        entity(graph, node, OWL.Class, SOURCE.WordNet, sense.name())
        graph.add((node, DCTERMS.identifier, Literal(sense.name())))
        for lemma in sense.lemma_names():
            graph.add((node, SKOS.altLabel, Literal(lemma.replace('_', ' '), lang='en')))
        graph.add((node, DCTERMS.description, Literal(sense.definition(), lang='en')))
        for parent in sense.hypernyms():
            graph.add((node, RDFS.subClassOf, visit(parent)))
        # Instance hypernyms are NOT subclass edges; avoid turning individuals
        # into classes. Demo imports common nouns, not named WordNet instances.
        return node

    return visit(selected)
