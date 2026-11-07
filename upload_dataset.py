from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

client = Client()
dataset = client.upload_csv(
    "scenarios.csv",
    input_keys=["input"],
    output_keys=["decision", "refund_amount", "expected_tools"],
    name="refund-scenarios",
    description="8 scénarios de remboursement avec références vérifiables",
)
print("Dataset créé :", dataset.name, dataset.id)