import argparse
import json
import sys

from deployment import load_bundle, predict_text


def main():
    parser = argparse.ArgumentParser(
        description="Predict personality dimensions from new text."
    )
    parser.add_argument("bundle", help="Path to a saved .pt model bundle.")
    parser.add_argument(
        "--text",
        help="Text to classify. If omitted, text is read from standard input.",
    )
    args = parser.parse_args()

    text = args.text if args.text is not None else sys.stdin.read()
    if not text.strip():
        parser.error("prediction text cannot be empty")

    bundle = load_bundle(args.bundle)
    result = {
        "dataset": bundle["dataset"],
        "predictions": predict_text(text, bundle),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
