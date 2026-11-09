"""Code guardrail: no refund ticket unless the customer explicitly asked for a refund.

The prompt already says it, but a prompt only lowers the error rate: the v6 agent
still created a ticket on 1 run out of 30 for "Mon casque est-il remboursable ?".
This middleware enforces the rule in code, whatever the model decides.

The check is deliberately rule-based (no LLM), so it does not share the blind spots
of the LLM judge `aucun_ticket_non_demande` that evaluates the same rule.
"""

import json
import re

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import BaseMessage, ToolMessage

GUARDED_TOOL = "create_refund_ticket"

# An explicit request: a form of the verb "rembourser" ("rembourse-moi", "remboursez",
# "me faire rembourser"), or a wish verb followed by "remboursement" ("je demande
# le remboursement"). "remboursable" and "le remboursement est-il possible ?" do not match.
_REQUEST = re.compile(
    r"\brembours(e|ez|er)\b"
    r"|\b(veux|voudrais|souhaite|souhaiterais|demande|exige)\b[^.?!]{0,60}\bremboursement\b",
    re.IGNORECASE,
)

# A short yes to a question the agent asked about the refund ("Voulez-vous que je
# lance le remboursement ?" -> "oui").
_CONFIRMATION = re.compile(
    r"^\s*(oui|ok|d'accord|daccord|vas-y|allez-y|je confirme|confirmé|c'est bon)\b",
    re.IGNORECASE,
)

REFUSAL = {
    "refused": True,
    "error": "Ticket refusé par le garde-fou : le client n'a pas demandé explicitement "
             "de remboursement. Réponds à sa question sans créer de ticket et demande-lui "
             "s'il souhaite lancer le remboursement.",
}


def is_explicit_refund_request(text: str) -> bool:
    """Tell whether a customer message explicitly asks for a refund.

    Args:
        text: One customer message.

    Returns:
        True for "rembourse-moi", "je veux me faire rembourser", "je demande le
        remboursement"; False for questions such as "est-ce remboursable ?".
    """
    return bool(_REQUEST.search(text))


def refund_was_requested(messages: list[BaseMessage]) -> bool:
    """Tell whether the conversation contains an explicit refund request.

    A request is either a customer message that asks for a refund, or a short
    confirmation ("oui") answering an agent message that offered a refund.

    Args:
        messages: The conversation so far (customer, agent and tool messages).

    Returns:
        True if a ticket may be created, False otherwise.
    """
    previous_agent_text = ""
    for message in messages:
        if message.type == "ai" and isinstance(message.content, str) and message.content:
            previous_agent_text = message.content
        elif message.type == "human":
            text = message.content if isinstance(message.content, str) else str(message.content)
            if is_explicit_refund_request(text):
                return True
            if _CONFIRMATION.search(text) and "rembours" in previous_agent_text.lower():
                return True
    return False


@wrap_tool_call
def refund_request_guard(request, handler):
    """Refuse create_refund_ticket when the customer never asked for a refund."""
    if request.tool_call["name"] != GUARDED_TOOL:
        return handler(request)
    if refund_was_requested(request.state["messages"]):
        return handler(request)
    return ToolMessage(
        content=json.dumps(REFUSAL, ensure_ascii=False),
        tool_call_id=request.tool_call["id"],
        name=GUARDED_TOOL,
        status="error",
    )
