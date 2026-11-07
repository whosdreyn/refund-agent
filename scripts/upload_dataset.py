"""Create the LangSmith dataset `refund-scenarios` from data/scenarios.csv.

Run once, from the project root:
    uv run python -m scripts.upload_dataset

The upload fails if the dataset already exists: delete it in LangSmith first
to recreate it.
"""

from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

CSV_FILE = Path(__file__).resolve().parents[1] / "data" / "scenarios.csv"


def main() -> None:
    """Upload the scenarios CSV as a LangSmith dataset."""
    load_dotenv()
    dataset = Client().upload_csv(
        str(CSV_FILE),
        input_keys=["input"],
        output_keys=["decision", "refund_amount", "expected_tools"],
        name="refund-scenarios",
        description="Scénarios de remboursement avec références vérifiables",
    )
    print("Dataset créé :", dataset.name, dataset.id)


if __name__ == "__main__":
    main()
