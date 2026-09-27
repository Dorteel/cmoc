# RoboKGNet

A small, offline RDF/OWL knowledge resource for CMOC. It imports selected senses,
classes and frames, not entire lexical databases. It contains no ROS, perception,
LLM calls, task execution, or navigation.

| Source | Contribution |
| --- | --- |
| WordNet 3.0 | Selected noun senses and actual hypernym chains (`rdfs:subClassOf`); selected verb senses linked through exact VerbNet sense keys |
| VerbNet 2.1 | Classes, members, XML subclass links, thematic roles, direct frames, structured predicates and ordered arguments |
| FrameNet 1.7 | Frames, Frame Elements, original definitions, core types and lexical units |
| SemLink2 | Explicit **class/member → frame** alignment records |
| CMOC commonsense | Reified LocatedAt/UsedFor assertions with numeric weights and source attribution |
| CMOC planning | References to existing Bringing PDDL parsed with Unified Planning; identity-preserving import of external planning RDF |

```text
WordNet ← exact sense keys → VerbNet ← SemLink records → FrameNet
   ↑                           ↓ curated links             ↓
weighted assertions       existing PDDL / imported planning RDF
```

## Layout

```text
knowledge/robokgnet/
  builder.py, namespaces.py
  wordnet.py, verbnet.py, framenet.py, commonsense.py, planning.py
  demo.py, refresh_semlink.py
  data/commonsense_demo.json, data/semlink_subset.json
  robokgnet.ttl
  tests/test_robokgnet.py
  README.md, requirements.txt
```

## Run

From the CMOC repository root, using its existing Python environment:

```bash
.venv/bin/python -m knowledge.robokgnet.demo
.venv/bin/python -m unittest discover -s knowledge/robokgnet/tests -v
```

Output: `knowledge/robokgnet/robokgnet.ttl`. Specify another destination with
`--output /tmp/robokgnet.ttl`. Regeneration replaces only that output file.
Tests verify SPARQL queries, exact definitions, alignment scope, weights,
provenance, external ontology preservation, idempotence and Turtle round trips.
They require installed corpora and perform no network downloads. The checked-in
artifact is compared to the locally regenerated graph; other corpus versions
can legitimately change that comparison.

For a fresh environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r knowledge/robokgnet/requirements.txt
.venv/bin/python utils/download_nltk_resources.py
```

Tested with NLTK 3.10.3, rdflib 7.6.0 and Unified Planning 1.3.0. No planning
engine is needed to build this graph. The existing corpus downloader is the
explicit online setup step; missing corpora otherwise raise NLTK's error.

SemLink regeneration is optional and explicitly online:

```bash
.venv/bin/python -m knowledge.robokgnet.refresh_semlink
.venv/bin/python -m knowledge.robokgnet.demo
```

The bundled subset records the upstream URL, retrieval time, a SHA-256 of the
parsed source table, exact member mappings and unmapped members. Normal builds
use that snapshot. Updating upstream data may change the demo and tests.

## API

```python
from knowledge.robokgnet import RoboKGNetBuilder
from knowledge.robokgnet.namespaces import SOURCE

kg = RoboKGNetBuilder()
kg.add_wordnet_concept("coffee_mug")
kg.add_wordnet_concept("kitchen")
kg.add_wordnet_concept("person", synset="person.n.01")
kg.add_verbnet_class("bring-11.3")
kg.add_framenet_frame("Bringing")
kg.add_alignments("knowledge/robokgnet/data/semlink_subset.json")
kg.add_commonsense("coffee_mug", "LocatedAt", "kitchen", 0.95,
                  source=SOURCE.DemoFixtures)
kg.save("/tmp/robokgnet.ttl")
```

Ambiguous bare nouns require an explicit sense; there is no automatic first-sense
selection. Demo `robot` resolves to `automaton.n.02`; labels retain WordNet lemmas.
Named WordNet instances are rejected rather than modeled as subclasses.
Commonsense terms use imported synsets when registered in the builder; otherwise
they get local SKOS concept IDs with no invented WordNet alignment. URIRef inputs
allow callers to select an existing concept explicitly.

Weights must be finite numbers in [0, 1]. They qualify `kg:CommonsenseAssertion`
records via `rdf:subject`, `rdf:predicate`, `rdf:object`, and `kg:weight`; no
unqualified LocatedAt/UsedFor fact is asserted. The four demo scores are
**hand-authored fixtures**, not measurements, probabilities, or ConceptNet data.

## Queries

```python
from rdflib import Graph

g = Graph().parse("knowledge/robokgnet/robokgnet.ttl")
print(list(g.query('''
    PREFIX kg: <https://example.org/cmoc/robokgnet/>
    PREFIX vn: <https://example.org/cmoc/robokgnet/verbnet/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?role WHERE {
        vn:bring-11.3-1 rdfs:subClassOf*/kg:hasRole/rdfs:label ?role
    }
''')))
```

`class → kg:hasFrame → kg:hasPredicate → kg:argument` exposes semantics.
Each argument has a zero-based `kg:position`, `kg:argumentType`, `kg:rawValue`,
and, where resolvable, `kg:refersTo` a role or event. `start`, `during`, and `end`
are explicit event-phase nodes linked to the same frame-scoped event. Negation
is a Boolean, and distinct frames never share event variables. Inherited roles
retain the URI of their declaring class; direct frames stay direct.

For example, query a `location` predicate with an argument referring to a phase
`end` and another referring to role `Theme`; the tests contain this exact query.
Definitions are copied verbatim from NLTK FrameNet, including any source markup.

## Reuse and planning boundary

Reused components:

- `tools/verbnet_semantic_patterns.load_class`: direct structured frames and roles.
- `tools/semantic_bridge`: exact frame/synset/class lookup, member sense-key
  resolution, and the SemLink loader used for the explicit snapshot refresh.
- `procedural_memory/planning/planner.py`'s `PDDLReader` and the unchanged
  `procedural_memory/planning/bringing/test/{domain,problem}.pddl`.
- The installed NLTK corpora and existing `utils/download_nltk_resources.py`.

The flat candidate extractors and Streamlit lexical browser were inspected;
their presentation records are not used because they lose structure or introduce
UI dependencies. `utils/view_kg.py` already reads rdflib files. CMOC's
`semantic_memory.py` generates location rankings at runtime, but no saved ranking
dataset was found; it is not invoked by this resource builder.

**No planning RDF ontology or UP-to-RDF converter was found in this checkout,
its Git history, or the nearby workspace repositories.** The sibling RobotNet
prototype has a small generic action ontology, not the requested planning
ontology; it was not copied or expanded into a competing planning ontology.

Consequently `add_pddl_references` represents source references as `prov:Entity`,
with a descriptive `kg:planningKind` literal: PlanningDomain, PlanningAction,
PlanningPredicate, PlanningProblem, InitialState or GoalState. It stores UP's
readable action/state descriptions, not a complete logical RDF translation.
There are **no new OWL planning classes**. Bringing links to the whole domain;
search/Scrutiny link to its actual `look_for` action. These are explicitly curated
demo associations, not SemLink assertions, executable bindings, or equivalences.
The existing `look_for` implementation is unchanged.

To integrate the actual planning ontology when available:

```python
kg.add_planning_knowledge("/path/to/existing_planning.ttl",
                          source="https://your-project.example/planning/source")
# Use an existing action's URIRef, unchanged:
kg.link_planning(semantic_node, existing_action_uri, source=SOURCE.CMOC)
```

The importer also accepts an rdflib Graph. It preserves all triples and term IRIs,
including domain/action/problem/state/plan classes and instances. The demo does
not fabricate a plan or claim a full Unified Planning ontology conversion.

## Provenance and resource limits

Imported records carry `prov:wasDerivedFrom`; alignment and commonsense records
have their own provenance. Planning links are reified with their curated source.
The SemLink snapshot node retains its upstream URL, fingerprint and retrieval
time. Local `https://example.org/cmoc/robokgnet/` IRIs are provisional project
identifiers, not claimed official corpus URLs or published resolvable endpoints.

NLTK VerbNet is 2.1 while SemLink2 targets 3.3. Only exact available class/member
matches are retained. `bring` occurs in `bring-11.3-1`; parent `bring-11.3` directly
contains `take`. `search` maps to **Scrutiny**. `Searching_scenario` is also
imported, but no unsupported link to it is invented. There is no exact FrameNet
frame named `Search` in the installed resource. This SemLink table provides no
role-level mappings: Destination is not asserted equivalent to FrameNet Goal.
Uncertain/missing WordNet sense keys are reported in `kg.warnings`, never replaced
by another sense. The selected demo member senses resolve without warnings.

Source attribution: [WordNet](https://wordnet.princeton.edu/),
[VerbNet](https://verbs.colorado.edu/verbnet/),
[FrameNet](https://framenet.icsi.berkeley.edu/), and
[SemLink](https://github.com/cu-clear/semlink). Corpus excerpts and definitions
retain the terms of their respective resources; this integration does not
relicense their data.
