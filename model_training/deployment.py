import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VECTORIZER_DIRECTORY = Path(__file__).resolve().parent / "vectorizers"
DIMENSIONS = {
    "essays": ("O", "C", "E", "A", "N"),
    "mbti": ("O", "C", "E", "A"),
}


class CustomNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, inputs):
        inputs = torch.relu(self.fc1(inputs))
        inputs = torch.relu(self.fc2(inputs))
        return torch.sigmoid(self.fc3(inputs))


def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+', " ", text)
    return re.sub("[^0-9a-z]", " ", text)


def _lemmatize(text):
    try:
        from nltk.stem import WordNetLemmatizer
    except ImportError as error:
        raise RuntimeError(
            "NLTK is required for text prediction. Install it with "
            "`pip install nltk==3.8.1`."
        ) from error

    lemmatizer = WordNetLemmatizer()
    try:
        return [
            lemmatizer.lemmatize(word)
            for word in text.split()
            if len(word) > 2
        ]
    except LookupError as error:
        raise RuntimeError(
            "NLTK WordNet data is missing. Run "
            "`python -m nltk.downloader wordnet omw-1.4`."
        ) from error


def raw_corpus(dataset):
    if dataset == "essays":
        dataframe = pd.read_csv(
            REPOSITORY_ROOT / "dataset/raw/essays.csv",
            encoding="iso-8859-1",
        )
        return dataframe["TEXT"].astype(str).tolist()
    if dataset == "mbti":
        dataframe = pd.read_csv(REPOSITORY_ROOT / "dataset/raw/mbti.csv")
        return dataframe["posts"].astype(str).tolist()
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_vectorizer(dataset):
    path = VECTORIZER_DIRECTORY / f"{dataset}_tfidf.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing vectorizer artifact: {path}. Run "
            "`/usr/bin/python3 model_training/export_vectorizer.py "
            f"{dataset}` using the preprocessing environment."
        )

    with np.load(path) as artifact:
        terms = artifact["terms"].tolist()
        idf = artifact["idf"].astype(np.float32)

    return {
        "terms": terms,
        "vocabulary": {term: index for index, term in enumerate(terms)},
        "idf": idf,
    }


def verify_vectorizer(vectorizer, dataframe, samples=5):
    raw_texts = raw_corpus_from_rows(dataframe)
    vectorizer_bundle = {
        "input_size": len(vectorizer["terms"]),
        "vocabulary": vectorizer["vocabulary"],
        "idf": vectorizer["idf"],
    }
    actual = np.stack(
        [vectorize_text(text, vectorizer_bundle) for text in raw_texts]
    )
    expected = np.stack(dataframe["text"].iloc[:samples].to_numpy())

    if not np.allclose(actual, expected, rtol=1e-5, atol=1e-7):
        difference = float(np.max(np.abs(actual - expected)))
        raise RuntimeError(
            "Rebuilt TF-IDF vectors do not match the stored training data "
            f"(maximum absolute difference: {difference:.6g}). Refusing to "
            "save an incompatible deployment artifact."
        )


def raw_corpus_from_rows(dataframe, samples=5):
    dataset = "essays" if "N" in dataframe.columns else "mbti"
    corpus = raw_corpus(dataset)
    return [
        corpus[int(user_id)]
        for user_id in dataframe["user"].iloc[:samples]
    ]


def save_bundle(path, models, vectorizer, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "format_version": 1,
        "dataset": config["dataset"],
        "feature": config["feature"],
        "loss": config["loss"],
        "threshold": 0.5,
        "input_size": len(vectorizer["terms"]),
        "dimensions": list(models),
        "vocabulary": vectorizer["vocabulary"],
        "idf": vectorizer["idf"],
        "models": {
            dimension: {
                key: value.detach().cpu()
                for key, value in model.network.state_dict().items()
            }
            for dimension, model in models.items()
        },
        "metrics": {
            dimension: {
                "epoch": model.epoch,
                "balanced_accuracy": model.ba,
                "regular_accuracy": model.ra,
            }
            for dimension, model in models.items()
        },
    }
    torch.save(bundle, path)
    return path


def load_bundle(path):
    bundle = torch.load(Path(path), map_location="cpu")
    required = {
        "format_version",
        "input_size",
        "dimensions",
        "vocabulary",
        "idf",
        "models",
    }
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Invalid model bundle; missing: {sorted(missing)}")
    return bundle


def vectorize_text(text, bundle):
    vocabulary = bundle["vocabulary"]
    # The notebook fitted vocabulary on cleaned text, but transformed the
    # already-created splits from raw text. Preserve that training behavior.
    counts = Counter(_lemmatize(text.lower()))
    features = np.zeros(bundle["input_size"], dtype=np.float32)

    for token, count in counts.items():
        index = vocabulary.get(token)
        if index is not None:
            features[index] = count

    features *= np.asarray(bundle["idf"], dtype=np.float32)
    norm = np.linalg.norm(features)
    if norm:
        features /= norm
    return features


def predict_text(text, bundle):
    features = torch.from_numpy(vectorize_text(text, bundle)).unsqueeze(0)
    threshold = float(bundle.get("threshold", 0.5))
    predictions = {}

    with torch.no_grad():
        for dimension in bundle["dimensions"]:
            network = CustomNetwork(bundle["input_size"])
            network.load_state_dict(bundle["models"][dimension])
            network.eval()
            probability = float(network(features).item())
            predictions[dimension] = {
                "probability": probability,
                "prediction": int(probability >= threshold),
            }

    return predictions
