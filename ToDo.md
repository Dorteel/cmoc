# Bringing Demo TODO

- [ ] Instruction intake
  - [ ] Read natural-language instruction
  - [ ] Split into sentences
  - [ ] POS-tag and lemmatize
  - [ ] Detect main task verb
  - [ ] Map verb to allowed `Bringing` frame

- [ ] Task-frame construction
  - [ ] Load/generate `Bringing` JSON Schema
  - [ ] Set `Agent = robot`
  - [ ] Extract `Theme`
  - [ ] Set `Destination = user`
  - [ ] Leave `Source = null`
  - [ ] Validate frame instance

- [ ] Mock knowledge graph
  - [ ] Define `Object`, `Location`, `Robot`, `Person`
  - [ ] Define `LocatedAt`
  - [ ] Add several locations
  - [ ] Add known objects
  - [ ] Leave some object locations unknown
  - [ ] Verify live KG viewer

- [ ] Knowledge lookup
  - [ ] Query `LocatedAt(Theme, ?location)`
  - [ ] Fill `Source` if found
  - [ ] Detect unresolved `Source`
  - [ ] Wrap lookup in a small knowledge-interface function

- [ ] Semantic search heuristic
  - [ ] Get candidate locations from KG
  - [ ] Rank locations for the Theme
  - [ ] Start with simple/manual heuristics
  - [ ] Later plug in cross-encoder scoring

- [ ] Search controller
  - [ ] Select highest-ranked unexplored location
  - [ ] Navigate there
  - [ ] Mark location as searched
  - [ ] Repeat until Theme is found

- [ ] Perception
  - [ ] Capture current image
  - [ ] Provide image + Theme + scene-graph schema to VLM
  - [ ] Detect whether Theme is present
  - [ ] Ground Theme to an object instance
  - [ ] Extract qualitative relations

- [ ] Memory update
  - [ ] Add/update Theme instance
  - [ ] Add `LocatedAt(Theme, current_location)`
  - [ ] Add scene-graph relations if useful
  - [ ] Resolve `Source`

- [ ] Manipulation
  - [ ] Navigate close to object
  - [ ] Trigger pick action
  - [ ] Handle success/failure
  - [ ] Mock if needed initially

- [ ] Delivery
  - [ ] Navigate to user
  - [ ] Trigger place/hand-over
  - [ ] Mark task complete

- [ ] End-to-end orchestration
  - [ ] `listen`
  - [ ] `frame`
  - [ ] `query KG`
  - [ ] `search if needed`
  - [ ] `perceive`
  - [ ] `update KG`
  - [ ] `pick`
  - [ ] `deliver`

- [ ] Demo hardening
  - [ ] Test object with known location
  - [ ] Test object requiring search
  - [ ] Test several different objects
  - [ ] Add clear stage logging
  - [ ] Make failure cases graceful