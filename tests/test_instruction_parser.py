from instruction_parser import parse_instruction


def test_demo_instructions():
    examples = [
        ("Bring me the cup.", ["Bring", "me", "the", "cup", "."], "cup"),
        ("Please bring me the red mug.",
         ["Please", "bring", "me", "the", "red", "mug", "."], "red mug"),
        ("Could you bring me the bottle?",
         ["Could", "you", "bring", "me", "the", "bottle", "?"], "bottle"),
    ]
    for sentence, tokens, object_phrase in examples:
        result = parse_instruction(sentence)
        assert len(result["sentences"]) == 1
        parsed = result["sentences"][0]
        assert parsed["sentence"] == sentence
        assert parsed["tokens"] == tokens
        assert len(parsed["pos_tags"]) == len(parsed["lemmas"]) == len(tokens)
        assert parsed["main_verb"] == "bring"
        assert parsed["object_phrase"] == object_phrase


def test_multiple_sentences():
    result = parse_instruction("Bring me the cup. Bring me the bottle.")
    assert [sentence["object_phrase"] for sentence in result["sentences"]] == [
        "cup", "bottle",
    ]


def test_lemmatization():
    parsed = parse_instruction("You brought the cups.")["sentences"][0]
    assert parsed["main_verb"] == "bring"
    assert parsed["lemmas"] == ["you", "bring", "the", "cup", "."]


def test_object_phrase_stops_at_preposition():
    parsed = parse_instruction("Bring me the cup on the table.")["sentences"][0]
    assert parsed["object_phrase"] == "cup"


def test_no_verb():
    parsed = parse_instruction("The red cup.")["sentences"][0]
    assert parsed["main_verb"] is None
    assert parsed["object_phrase"] == ""


def test_empty_instruction():
    assert parse_instruction("  ") == {"sentences": []}
