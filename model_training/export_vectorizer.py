import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "vectorizers"


def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', " ", text)
    return re.sub("[^0-9a-z]", " ", text)


class Lemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, sentence):
        return [
            self.lemmatizer.lemmatize(word)
            for word in sentence.split()
            if len(word) > 2
        ]


def load_corpus(dataset):
    if dataset == "essays":
        dataframe = pd.read_csv(
            REPOSITORY_ROOT / "dataset/raw/essays.csv",
            encoding="iso-8859-1",
        )
        return dataframe["TEXT"].astype(str)
    if dataset == "mbti":
        dataframe = pd.read_csv(REPOSITORY_ROOT / "dataset/raw/mbti.csv")
        return dataframe["posts"].astype(str)
    raise ValueError(f"Unsupported dataset: {dataset}")


def main():
    parser = argparse.ArgumentParser(
        description="Export the fitted TF-IDF state as a portable NPZ file."
    )
    parser.add_argument("dataset", choices=("essays", "mbti"))
    args = parser.parse_args()

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        tokenizer=Lemmatizer(),
        token_pattern=None,
    )
    vectorizer.fit([clean_text(text) for text in load_corpus(args.dataset)])

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / f"{args.dataset}_tfidf.npz"
    np.savez_compressed(
        output_path,
        terms=np.asarray(vectorizer.get_feature_names_out(), dtype=str),
        idf=vectorizer.idf_.astype(np.float32),
    )
    print(f"Saved {len(vectorizer.vocabulary_)} TF-IDF features to {output_path}")


if __name__ == "__main__":
    main()
