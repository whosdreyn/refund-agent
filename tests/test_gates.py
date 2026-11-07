"""Tests of the eligibility gate: no refund ticket without a confirmed eligibility check."""

import pytest

from evaluation.config_trajectoire import GATES
from trajectory_engine import Step, check_gates


def lookup(*purchases):
    return Step("lookup_customer_purchases", {"username": "x"},
                {"found": True, "purchases": [{"order_id": o, "purchase_date": d} for o, d in purchases]})


def check(date, answer):
    return Step("check_refund_eligibility", {"purchase_date": date}, answer)


CALC = Step("calculate_refund_amount", {}, {"refund_amount": 10.0})


def ticket(order_id):
    return Step("create_refund_ticket", {"order_id": order_id, "amount": 10.0}, {"ticket_id": "RF-1"})


def faults(steps):
    """Return the reasons of every failed gate."""
    return [v["reason"] for v in check_gates(steps, GATES) if not v["valid"]]


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param([lookup(("A100", "2026-10-20"), ("A101", "2026-10-25")),
                      check("2026-10-25", {"eligible": True}), CALC, ticket("A101")], id="cable, eligible"),
        pytest.param([lookup(("L300", "2026-10-28"), ("L301", "2026-10-28")),
                      check("2026-10-28", {"eligible": True}), check("2026-10-28", {"eligible": True}),
                      CALC, CALC, ticket("L300"), ticket("L301")], id="lina, two items same date"),
    ],
)
def test_confirmed_eligibility_opens_the_gate(steps):
    assert faults(steps) == []


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param([lookup(("K200", "2026-08-15")), check("2026-08-15", {"eligible": False}), CALC, ticket("K200")],
                     id="refund despite a refusal"),
        pytest.param([lookup(("A100", "2026-10-20"), ("A101", "2026-10-25")),
                      check("2026-10-20", {"eligible": True}), CALC, ticket("A101")], id="checked the wrong item"),
        pytest.param([lookup(("A101", "2026-10-25")), check("2026-10-25", {"error": "timeout"}), CALC, ticket("A101")],
                     id="eligibility check in error"),
        pytest.param([Step("lookup_customer_purchases", {}, "server error"),
                      check("2026-10-25", {"eligible": True}), CALC, ticket("A101")], id="lookup returned text"),
    ],
)
def test_unconfirmed_eligibility_closes_the_gate(steps):
    assert faults(steps) != []
