import argparse
from datetime import datetime
from pathlib import Path

import torch


CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}


def find_checkpoints(root):
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
    )


def describe_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as error:
        return f"INVALID ({type(error).__name__}: {error})"

    if isinstance(checkpoint, dict):
        keys = list(checkpoint)
        tensor_count = sum(torch.is_tensor(value) for value in checkpoint.values())

        if tensor_count == len(checkpoint) and checkpoint:
            return f"valid model state_dict ({tensor_count} tensors)"

        key_preview = ", ".join(str(key) for key in keys[:6])
        if len(keys) > 6:
            key_preview += ", ..."
        return f"valid checkpoint dictionary (keys: {key_preview})"

    return f"loadable object ({type(checkpoint).__name__})"


def main():
    repository_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Find and validate saved PyTorch model checkpoints."
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=repository_root,
        help="Directory to scan (default: repository root).",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"scan root is not a directory: {root}")

    checkpoints = find_checkpoints(root)
    if not checkpoints:
        print(f"No model checkpoints found under {root}")
        print("Searched for: .pt, .pth, .ckpt")
        return

    print(f"Found {len(checkpoints)} checkpoint(s) under {root}:")
    for path in checkpoints:
        stat = path.stat()
        saved_at = datetime.fromtimestamp(stat.st_mtime).isoformat(
            sep=" ", timespec="seconds"
        )
        size_mb = stat.st_size / (1024 * 1024)
        status = describe_checkpoint(path)
        print(f"- {path.relative_to(root)}")
        print(f"  saved: {saved_at}, size: {size_mb:.2f} MB, status: {status}")


if __name__ == "__main__":
    main()
