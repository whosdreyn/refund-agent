"""Freeze a version of the LangSmith dataset so the gate always compares on the same examples.

Puts every example in the given split, then tags the resulting dataset version.
eval.toml then points to that tag ([dataset] tag), and the baseline and every
candidate are evaluated on exactly these examples (eval-kit design, principle P7).

    uv run python -m scripts.freeze_dataset v1
    uv run python -m scripts.freeze_dataset v1 --split holdout

Run it again with a new tag (v2, v3...) after adding or changing scenarios; then
recompute the baseline, since the old one was measured on other examples.
"""

import argparse

from dotenv import load_dotenv
from langsmith import Client

DATASET = "refund-scenarios"


def freeze(tag: str, split: str) -> None:
    """Move every example of the dataset to `split`, then tag the latest version.

    Args:
        tag: Name of the version, e.g. "v1".
        split: Split given to every example, e.g. "holdout".
    """
    client = Client()
    dataset = client.read_dataset(dataset_name=DATASET)
    examples = list(client.list_examples(dataset_id=dataset.id))
    for example in examples:
        client.update_example(example.id, split=split, dataset_id=dataset.id)

    latest = next(iter(client.list_dataset_versions(dataset_id=dataset.id, limit=1)))
    client.update_dataset_tag(dataset_id=dataset.id, as_of=latest.as_of, tag=tag)
    print(f"{DATASET}: {len(examples)} examples in split {split!r}, "
          f"version of {latest.as_of.isoformat()} tagged {tag!r}")


def main() -> None:
    """Read the command line and freeze the dataset."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Tag a frozen version of the dataset.")
    parser.add_argument("tag", help="version name, e.g. v1")
    parser.add_argument("--split", default="holdout", help="split for every example")
    args = parser.parse_args()
    freeze(args.tag, args.split)


if __name__ == "__main__":
    main()
