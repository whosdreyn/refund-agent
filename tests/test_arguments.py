"""Tests of argument provenance: every controlled argument must come from an earlier tool result."""

import pytest

from evaluation.config_trajectoire import ARGUMENTS
from eval_kit import Step, check_arguments

AWA_PURCHASES = {"found": True, "purchases": [
    {"order_id": "A100", "item": "casque audio", "unit_price": 89.99, "quantity": 1, "purchase_date": "2026-10-20"},
    {"order_id": "A101", "item": "câble USB-C", "unit_price": 12.5, "quantity": 2, "purchase_date": "2026-10-25"},
]}
LOOKUP = Step("lookup_customer_purchases", {"username": "awa"}, AWA_PURCHASES)


def check(date):
    return Step("check_refund_eligibility", {"purchase_date": date}, {"eligible": True})


def calc(price, quantity):
    return Step("calculate_refund_amount", {"unit_price": price, "quantity": quantity},
                {"refund_amount": round(price * quantity, 2)})


def ticket(order_id, amount):
    return Step("create_refund_ticket", {"order_id": order_id, "amount": amount}, {"ticket_id": "RF-0001"})


def faults(steps):
    """Return the reasons of every invented argument."""
    return [v["reason"] for v in check_arguments(steps, ARGUMENTS) if not v["valid"]]


def test_single_cable_refund_is_valid():
    assert faults([LOOKUP, check("2026-10-25"), calc(12.5, 1), ticket("A101", 12.5)]) == []


@pytest.mark.parametrize(
    "steps, invented",
    [
        pytest.param([LOOKUP, check("2026-10-25"), calc(12.5, 1), ticket("A105", 12.5)],
                     "order_id='A105'", id="invented order id"),
        pytest.param([LOOKUP, check("2026-10-25"), calc(12.5, 1), ticket("A101", 25.0)],
                     "amount=25.0", id="amount different from the calculation"),
        pytest.param([LOOKUP, check("2026-10-30"), calc(12.5, 1), ticket("A101", 12.5)],
                     "purchase_date='2026-10-30'", id="invented date"),
        pytest.param([LOOKUP, check("2026-10-25"), calc(10.0, 1), ticket("A101", 10.0)],
                     "unit_price=10.0", id="invented price, consistent ticket"),
        pytest.param([LOOKUP, check("2026-10-25"), ticket("A101", 12.5)],
                     "amount=12.5", id="ticket before any calculation"),
    ],
)
def test_invented_arguments_are_detected(steps, invented):
    assert any(invented in reason for reason in faults(steps))
