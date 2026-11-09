"""Run the agent on one request and print its trajectory, step by step.

    uv run python -m scripts.show_trajectory
    uv run python -m scripts.show_trajectory "Je suis koffi, rembourse-moi"
"""

import sys
import uuid

from refund_agent.agent import ask, build_agent
from eval_kit import from_langchain_messages

DEFAULT_REQUEST = "awa ici. Je veux rembourser un seul des deux câbles USB-C, je garde l'autre."


def main() -> None:
    """Print each tool call of the agent with its arguments and result."""
    request = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REQUEST
    result = ask(build_agent(), request, str(uuid.uuid4()))
    for i, step in enumerate(from_langchain_messages(result["messages"]), start=1):
        print(f"{i}. {step.tool} | arguments : {step.arguments} | résultat : {step.result}")
    print("Réponse :", result["structured_response"])


if __name__ == "__main__":
    main()
