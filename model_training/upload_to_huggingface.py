import argparse
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = (
    REPOSITORY_ROOT / "model_training/artifacts/essays_bow_cfbce.pt"
)
DEFAULT_VECTORIZER_PATH = (
    REPOSITORY_ROOT / "model_training/vectorizers/essays_tfidf.npz"
)


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required upload file does not exist: {path}")


def model_card(repo_id):
    return f"""---
library_name: pytorch
pipeline_tag: text-classification
tags:
- personality-prediction
- big-five
- essays
---

# Essays Big Five Personality Predictor

This repository contains a PyTorch deployment bundle trained in
`AFLPS-ML-Psycholinguistics` to predict Big Five personality dimensions from
essay text.

## Files

- `pytorch_model.bin`: PyTorch bundle containing model weights, TF-IDF
  vocabulary, IDF weights, threshold, and metrics.
- `vectorizers/essays_tfidf.npz`: portable TF-IDF artifact exported from the
  preprocessing environment.
- `deployment.py`: model architecture and preprocessing helpers.
- `predict.py`: command-line batch and single-text inference script.
- `requirements.txt`: minimal inference dependencies.

## Local Inference

```bash
pip install -r requirements.txt
python predict.py pytorch_model.bin --text "Paste a new essay here."
```

For a CSV with a `text` column:

```bash
python predict.py pytorch_model.bin --input essays.csv --output predictions.csv
```

## Output

The model returns probabilities and binary predictions for:

- `O`: Openness
- `C`: Conscientiousness
- `E`: Extraversion
- `A`: Agreeableness
- `N`: Neuroticism

Original local source repo upload target: `{repo_id}`.
"""


def upload_file(api, *, local_path, path_in_repo, repo_id, token):
    print(f"Uploading {local_path} -> {path_in_repo}")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Upload this repo's trained personality model to Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face model repo, e.g. username/essays-big5-personality.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hugging Face repo as private.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN or cached login.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Local model bundle path (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--vectorizer-path",
        type=Path,
        default=DEFAULT_VECTORIZER_PATH,
        help=f"Local TF-IDF artifact path (default: {DEFAULT_VECTORIZER_PATH}).",
    )
    args = parser.parse_args()

    model_path = args.model_path.expanduser().resolve()
    vectorizer_path = args.vectorizer_path.expanduser().resolve()
    deployment_path = REPOSITORY_ROOT / "model_training/deployment.py"
    predict_path = REPOSITORY_ROOT / "model_training/predict.py"
    requirements_path = REPOSITORY_ROOT / "model_training/hf_requirements.txt"

    for path in (
        model_path,
        vectorizer_path,
        deployment_path,
        predict_path,
        requirements_path,
    ):
        require_file(path)

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
        token=args.token,
    )

    uploads = [
        (model_path, "pytorch_model.bin"),
        (vectorizer_path, "vectorizers/essays_tfidf.npz"),
        (deployment_path, "deployment.py"),
        (predict_path, "predict.py"),
        (requirements_path, "requirements.txt"),
    ]
    for local_path, path_in_repo in uploads:
        upload_file(
            api,
            local_path=local_path,
            path_in_repo=path_in_repo,
            repo_id=args.repo_id,
            token=args.token,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        readme_path = Path(tmpdir) / "README.md"
        readme_path.write_text(model_card(args.repo_id), encoding="utf-8")
        upload_file(
            api,
            local_path=readme_path,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            token=args.token,
        )

    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
