import nltk
from nltk.corpus import framenet as fn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import verbnet as vn
from nltk.corpus import framenet as fn

lemmatizer = WordNetLemmatizer()

def detect_command(tagged_tokens):
    """Detect simple direct or polite robot commands."""
    words = [word.lower() for word, _ in tagged_tokens]

    if tagged_tokens[0][1] == "VB":
        return True

    return words[0] in {"please", "can", "could", "would"}


def detect_main_verb(tagged_tokens):
    """Return the first verb in the command."""
    for word, tag in tagged_tokens:
        if tag.startswith("VB"):
            return lemmatizer.lemmatize(word.lower(), "v")

    return None


def lookup_frame(verb):
    """Return the first FrameNet frame evoked by the verb."""
    frames = fn.frames_by_lemma(verb)
    return frames[0].name if frames else None

def get_allowed_frames(path="prompts/perception/system.md"):
    text = open(path, encoding="utf-8").read()
    block = text.split("allowed_frames:")[1].split("```")[0]
    return [line.strip()[2:] for line in block.splitlines() if line.strip().startswith("- ")]

def detect_frame(instruction):
    """Instruction -> command -> verb -> FrameNet frame."""
    for sentence in nltk.sent_tokenize(instruction):
        tagged = nltk.pos_tag(nltk.word_tokenize(sentence))

        if detect_command(tagged):
            verb = detect_main_verb(tagged)

            return {
                "frame": lookup_frame(verb),
                "trigger_verb": verb,
            }

    return {"frame": None, "trigger_verb": None}



def get_trigger_verbs(frame_name="Bringing"):
    """Return verb lexical units that evoke a FrameNet frame."""
    frame = fn.frame_by_name(frame_name)
    return sorted(lu.rsplit(".", 1)[0] for lu in frame.lexUnit if lu.endswith(".v"))

def build_task_dictionary():
    return {frame: get_trigger_verbs(frame) for frame in get_allowed_frames()}

def fill_frame(instruction, frame_info):
    pass

def listen(instruction="Bring me the cup on the table."):
    
    # instruction = input("Instruction: ")
    
    frame = detect_frame(instruction)
    schema = fill_frame(instruction, frame)
#


if __name__ == "__main__":
    listen()