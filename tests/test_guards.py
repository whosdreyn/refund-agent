"""Unit tests of the refund guardrail: free, no model call.

The guard must let through every explicit refund request of the dataset, block
questions about eligibility, and accept a short "oui" only after the agent offered
a refund.
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from refund_agent.guards import (
    GUARDED_TOOL,
    is_explicit_refund_request,
    refund_request_guard,
    refund_was_requested,
)

REQUESTS = [
    "Bonjour, je suis awa. Je veux me faire rembourser mon casque audio.",
    "Je suis koffi, je voudrais rembourser mes chaussures de course.",
    "awa ici. Rembourse-moi seulement les câbles USB-C, pas le casque.",
    "Je suis lina. Je veux rembourser mon clavier et ma souris.",
    "Bonjour, je veux me faire rembourser ma dernière commande.",
    "Remboursez-moi s'il vous plaît.",
    "Je demande le remboursement de ma commande A100.",
]

QUESTIONS = [
    "Je suis awa. Mon casque audio est-il encore remboursable ? ",
    "Le remboursement est-il possible pour mon clavier ?",
    "Quelle est votre politique de remboursement ?",
    "Bonjour, je suis awa.",
]


@pytest.mark.parametrize("text", REQUESTS)
def test_explicit_requests_are_recognised(text):
    assert is_explicit_refund_request(text)


@pytest.mark.parametrize("text", QUESTIONS)
def test_questions_are_not_requests(text):
    assert not is_explicit_refund_request(text)


def test_request_in_an_earlier_turn_counts():
    messages = [HumanMessage("Bonjour, je voudrais me faire rembourser."),
                AIMessage("Quel est votre nom d'utilisateur ?"),
                HumanMessage("Je suis awa, c'est pour mon casque audio.")]
    assert refund_was_requested(messages)


def test_yes_after_a_refund_offer_counts():
    messages = [HumanMessage("Mon casque est-il remboursable ?"),
                AIMessage("Oui, il est éligible. Voulez-vous que je lance le remboursement ?"),
                HumanMessage("Oui")]
    assert refund_was_requested(messages)


def test_yes_to_another_question_does_not_count():
    messages = [HumanMessage("Mon casque est-il remboursable ?"),
                AIMessage("Êtes-vous bien awa ?"),
                HumanMessage("Oui")]
    assert not refund_was_requested(messages)


class _Request:
    """Minimal stand-in for ToolCallRequest: only the fields the guard reads."""

    def __init__(self, tool_name, messages):
        self.tool_call = {"name": tool_name, "args": {}, "id": "call_1"}
        self.state = {"messages": messages}


def _guard(request, handler):
    return refund_request_guard.wrap_tool_call(request, handler)


def _executed(request):
    return ToolMessage(content='{"ticket_id": "RF-0001"}', tool_call_id="call_1")


def test_guard_blocks_ticket_on_a_question():
    request = _Request(GUARDED_TOOL, [HumanMessage("Mon casque est-il remboursable ?")])
    result = _guard(request, _executed)
    assert json.loads(result.content)["refused"] is True
    assert result.status == "error"


def test_guard_lets_ticket_through_on_a_request():
    request = _Request(GUARDED_TOOL, [HumanMessage("Rembourse-moi mon casque.")])
    assert "ticket_id" in _guard(request, _executed).content


def test_guard_ignores_other_tools():
    request = _Request("lookup_customer_purchases", [HumanMessage("Bonjour")])
    assert "ticket_id" in _guard(request, _executed).content
