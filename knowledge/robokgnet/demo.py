"""Generate the Bringing knowledge resource from installed, offline resources."""
import argparse
import json
from pathlib import Path
from .builder import RoboKGNetBuilder
from .namespaces import SOURCE, FN, VN, identifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def build_demo():
    kg = RoboKGNetBuilder()
    for name, sense in [('coffee_mug', 'coffee_mug.n.01'), ('muffin', 'muffin.n.01'),
                        ('kitchen', 'kitchen.n.01'), ('living_room', 'living_room.n.01'),
                        ('robot', 'automaton.n.02'), ('person', 'person.n.01')]:
        kg.add_wordnet_concept(name, synset=sense)
    kg.add_verbnet_class('bring-11.3')
    kg.add_verbnet_class('bring-11.3-1')
    kg.add_verbnet_class('search-35.2')
    for cls, member in [('bring-11.3', 'take'), ('bring-11.3-1', 'bring'), ('search-35.2', 'search')]:
        kg.add_member_wordnet_senses(cls, member)
    kg.add_framenet_frame('Bringing')
    kg.add_framenet_frame('Searching_scenario')
    kg.add_alignments(HERE / 'data/semlink_subset.json')
    data = json.loads((HERE / 'data/commonsense_demo.json').read_text())
    for assertion in data['assertions']:
        kg.add_commonsense(**assertion, source=data['source'])
    pddl = ROOT / 'procedural_memory/planning/bringing/test'
    refs = kg.add_pddl_references(pddl / 'domain.pddl', pddl / 'problem.pddl',
                                 source_base='https://example.org/cmoc/procedural_memory/planning/bringing/test')
    # These are explicit CMOC demo associations, not SemLink or equivalence.
    # Bringing is implemented by the whole domain, not by a nonexistent bring action.
    kg.link_planning(identifier(FN, 'Bringing'), refs['domain'], source=SOURCE.DemoFixtures)
    kg.link_planning(identifier(VN, 'bring-11.3'), refs['domain'], source=SOURCE.DemoFixtures)
    kg.link_planning(identifier(VN, 'search-35.2'), refs['actions']['look_for'], source=SOURCE.DemoFixtures)
    kg.link_planning(identifier(FN, 'Scrutiny'), refs['actions']['look_for'], source=SOURCE.DemoFixtures)
    return kg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'robokgnet.ttl')
    args = parser.parse_args()
    kg = build_demo()
    kg.save(args.output)
    print(f'{args.output}: {len(kg.graph)} triples')
    for warning in kg.warnings:
        print('Unresolved:', warning)


if __name__ == '__main__':
    main()
