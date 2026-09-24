"""Download the NLTK resources needed by lexical_explorer.py.

Run this once while you still have internet access:
    python download_nltk_resources.py
"""

import nltk


RESOURCES = [
    "wordnet",
    "omw-1.4",
    "framenet_v17",
    "verbnet",
]


def main():
    """Download all lexical resources used by the offline explorer."""
    for resource in RESOURCES:
        print(f"Downloading {resource}...")
        nltk.download(resource)

    print("\nDone. WordNet, FrameNet, and VerbNet are ready for offline use.")


if __name__ == "__main__":
    main()
