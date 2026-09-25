"""Expose simple linguistic candidates before robot-instruction frame lookup."""

from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("punkt_tab", quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('framenet_v17', quiet=True)
nltk.download('verbnet', quiet=True)

def lemmatize_tokens(tagged_tokens: list[tuple[str, str]]) -> list[str]:
    """Translate Penn POS tags into the categories WordNet understands."""
    lemmatizer = WordNetLemmatizer()
    wordnet_tags = {"J": "a", "V": "v", "N": "n", "R": "r"}
    lemmas = []
    for word, tag in tagged_tokens:
        category = wordnet_tags.get(tag[0], "n")
        lemmas.append(lemmatizer.lemmatize(word.lower(), pos=category))
    return lemmas


def extract_object_phrase(tagged_tokens: list[tuple[str, str]]) -> str:
    """Collect the first determiner/adjective/noun run after the verb."""
    words = []
    started = False
    for word, tag in tagged_tokens:
        if not started and tag.startswith("PRP"):
            continue
        if tag == "DT" or tag.startswith(("JJ", "NN")):
            started = True
            # Determiners help locate the phrase but add little to object lookup.
            if tag != "DT":
                words.append(word.lower())
        else:
            break
    return " ".join(words)


def parse_sentence(sentence: str) -> dict:
    """Return tokens, tags, lemmas, and two deliberately simple candidates."""
    tokens = word_tokenize(sentence, preserve_line=True)
    # Sentence-initial capitals can make NLTK tag polite "Please" as a verb.
    normalized_tokens = [word.lower() for word in tokens]
    tagged_tokens = pos_tag(normalized_tokens)
    lemmas = lemmatize_tokens(tagged_tokens)
    main_verb = None
    object_phrase = ""
    for index, (_, tag) in enumerate(tagged_tokens):
        if tag.startswith("VB"):
            main_verb = lemmas[index]
            object_phrase = extract_object_phrase(tagged_tokens[index + 1:])
            break
    return {
        "sentence": sentence,
        "tokens": tokens,
        "pos_tags": [tag for _, tag in tagged_tokens],
        "lemmas": lemmas,
        "main_verb": main_verb,
        "object_phrase": object_phrase,
    }


def parse_instruction(instruction: str) -> dict:
    """Return a sentence record for each sentence; empty input yields no records."""
    sentences = sent_tokenize(instruction)
    return {"sentences": [parse_sentence(sentence) for sentence in sentences]}


def main():
    """Show the structures used by the first step of the demo."""
    instructions = [
        "Bring me the cup.",
        "Please bring me the red mug.",
        "Could you bring me the bottle?",
    ]
    for instruction in instructions:
        print(parse_instruction(instruction))


if __name__ == "__main__":
    main()
