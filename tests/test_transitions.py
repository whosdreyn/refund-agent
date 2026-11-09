"""Tests of the refund agent's DAG: legitimate paths pass, dangerous ones are rejected."""

import pytest

from evaluation.config_trajectoire import TRANSITIONS
from eval_kit import Step, check_transitions


def trajectory(*tools):
    """Build a trajectory from tool names only (arguments and results are not checked here)."""
    return [Step(tool=t, arguments={}, result=None) for t in tools]


def faults(steps):
    """Return the reasons of every forbidden transition."""
    return [v["reason"] for v in check_transitions(steps, TRANSITIONS) if not v["valid"]]


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param(trajectory(), id="no name: answer without any tool"),
        pytest.param(trajectory("lookup_customer_purchases"), id="unknown customer: stop after lookup"),
        pytest.param(trajectory("lookup_customer_purchases", "check_refund_eligibility"), id="koffi: refused"),
        pytest.param(
            trajectory("lookup_customer_purchases", "check_refund_eligibility", "check_refund_eligibility",
                       "calculate_refund_amount", "calculate_refund_amount",
                       "create_refund_ticket", "create_refund_ticket"),
            id="lina: two items grouped by operation",
        ),
        pytest.param(
            trajectory("lookup_customer_purchases", "check_refund_eligibility", "calculate_refund_amount",
                       "create_refund_ticket", "check_refund_eligibility", "calculate_refund_amount",
                       "create_refund_ticket"),
            id="two items handled one by one",
        ),
    ],
)
def test_legitimate_paths_are_allowed(steps):
    assert faults(steps) == []


@pytest.mark.parametrize(
    "steps, expected",
    [
        pytest.param(trajectory("lookup_customer_purchases", "create_refund_ticket"),
                     "lookup_customer_purchases → create_refund_ticket", id="ticket without any check"),
        pytest.param(trajectory("create_refund_ticket"), "START → create_refund_ticket", id="ticket first"),
        pytest.param(trajectory("lookup_customer_purchases", "rembourser_tout"),
                     "lookup_customer_purchases → rembourser_tout", id="invented tool"),
        pytest.param(trajectory("lookup_customer_purchases", "check_refund_eligibility", "calculate_refund_amount"),
                     "calculate_refund_amount → END", id="stops after a calculation"),
    ],
)
def test_dangerous_paths_are_rejected(steps, expected):
    assert any(expected in reason for reason in faults(steps))
