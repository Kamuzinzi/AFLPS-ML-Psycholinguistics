import argparse
import csv
import json
import sys
from pathlib import Path

from deployment import load_bundle, load_networks, predict_text


def read_records(path, text_column):
    suffix = path.suffix.lower()

    if suffix == ".txt":
        records = []
        with path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                text = line.strip()
                if text:
                    records.append({"id": line_number, text_column: text})
        return records

    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as input_file:
            records = list(csv.DictReader(input_file))
        if records and text_column not in records[0]:
            raise ValueError(
                f"CSV does not contain a '{text_column}' column. "
                f"Available columns: {list(records[0])}"
            )
        return records

    if suffix in {".jsonl", ".ndjson"}:
        records = []
        with path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(
                        f"JSONL line {line_number} must contain an object."
                    )
                if text_column not in record:
                    raise ValueError(
                        f"JSONL line {line_number} does not contain "
                        f"'{text_column}'."
                    )
                records.append(record)
        return records

    raise ValueError("Input file must be .txt, .csv, .jsonl, or .ndjson.")


def add_predictions(record, predictions):
    result = dict(record)
    for dimension, values in predictions.items():
        result[f"{dimension}_probability"] = values["probability"]
        result[f"{dimension}_prediction"] = values["prediction"]
    return result


def write_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        if not records:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        return

    if suffix in {".jsonl", ".ndjson"}:
        with path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(json.dumps(record) + "\n")
        return

    if suffix == ".json":
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return

    raise ValueError("Output file must be .csv, .json, .jsonl, or .ndjson.")


def main():
    parser = argparse.ArgumentParser(
        description="Predict personality dimensions from new text."
    )
    parser.add_argument("bundle", help="Path to a saved .pt model bundle.")
    parser.add_argument(
        "--text",
        help="A single text to classify.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Batch input file: .txt, .csv, .jsonl, or .ndjson.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Batch output file: .csv, .json, .jsonl, or .ndjson.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Text field for CSV/JSONL input (default: text).",
    )
    args = parser.parse_args()

    bundle = load_bundle(args.bundle)
    networks = load_networks(bundle)

    if args.input:
        if args.text is not None:
            parser.error("--text and --input cannot be used together")
        if not args.output:
            parser.error("--output is required with --input")
        if not args.input.is_file():
            parser.error(f"input file does not exist: {args.input}")

        try:
            records = read_records(args.input, args.text_column)
            results = []
            for row_number, record in enumerate(records, start=1):
                text = record.get(args.text_column)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"Row {row_number} has an empty or invalid "
                        f"'{args.text_column}' value."
                    )
                results.append(
                    add_predictions(
                        record,
                        predict_text(text, bundle, networks=networks),
                    )
                )
            write_records(args.output, results)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))

        print(f"Wrote {len(results)} prediction(s) to {args.output}")
        return

    if args.output:
        parser.error("--output can only be used with --input")

    text = args.text if args.text is not None else sys.stdin.read()
    if not text.strip():
        parser.error("prediction text cannot be empty")

    result = {
        "dataset": bundle["dataset"],
        "predictions": predict_text(text, bundle, networks=networks),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
