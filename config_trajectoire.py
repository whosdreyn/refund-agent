"""Trajectory engine configuration for the refund agent.

Three parts, read by trajectory_engine.evaluate_trajectory:
  - TRANSITIONS: which tool may follow which (the DAG);
  - ARGUMENTS: where each controlled tool argument must come from;
  - GATES: the business checks required before a risky tool call.
"""

from trajectory_engine import END, START

# Allowed successors of each node. Self-loops let the agent handle several items
# (lina: check → check, calculate → calculate, ticket → ticket), and ticket → check
# lets it handle items one by one instead of grouped by operation.
TRANSITIONS = {
    START: ["lookup_customer_purchases", END],
    "lookup_customer_purchases": ["check_refund_eligibility", END],
    "check_refund_eligibility": ["check_refund_eligibility", "calculate_refund_amount", END],
    "calculate_refund_amount": ["calculate_refund_amount", "create_refund_ticket"],
    "create_refund_ticket": ["check_refund_eligibility", "create_refund_ticket", END],
}

# Called tool: {argument: (source tool, field in the source tool's result)}.
# `quantity` and `username` are not controlled: their right value comes from the
# customer's message, not from a tool result.
ARGUMENTS = {
    "check_refund_eligibility": {
        "purchase_date": ("lookup_customer_purchases", "purchase_date"),
    },
    "calculate_refund_amount": {
        "unit_price": ("lookup_customer_purchases", "unit_price"),
    },
    "create_refund_ticket": {
        "order_id": ("lookup_customer_purchases", "order_id"),
        "amount": ("calculate_refund_amount", "refund_amount"),
    },
}


def _order_date(order_id, previous_steps):
    """Find the purchase date of an order in earlier lookup results.

    Args:
        order_id: The order to look for, e.g. "A101".
        previous_steps: The steps made before the current one.

    Returns:
        The order's purchase date, or None if no lookup result contains the order.
    """
    for step in previous_steps:
        if step.tool != "lookup_customer_purchases" or not isinstance(step.result, dict):
            continue
        for purchase in step.result.get("purchases", []):
            if purchase.get("order_id") == order_id:
                return purchase.get("purchase_date")
    return None


def _eligibility_confirmed(date, previous_steps):
    """Tell whether an earlier eligibility check on this date answered eligible: True.

    Only an explicit True counts: a missing value or a tool error does not
    open the gate.

    Args:
        date: The purchase date that must have been checked.
        previous_steps: The steps made before the current one.

    Returns:
        True if a confirmed eligibility check exists for this date, else False.
    """
    for step in previous_steps:
        if (
            step.tool == "check_refund_eligibility"
            and isinstance(step.result, dict)
            and step.arguments.get("purchase_date") == date
            and step.result.get("eligible") is True
        ):
            return True
    return False


def eligibility_gate(step, previous_steps):
    """Allow a refund ticket only if its order was checked AND found eligible.

    The ticket carries an order_id while the eligibility check only receives a
    date, so the gate links them through the lookup results. Two items bought
    the same day (lina) share one check, which is correct since eligibility
    depends only on the purchase date.

    Args:
        step: The create_refund_ticket step being checked.
        previous_steps: The steps made before it.

    Returns:
        A (valid, reason) tuple.
    """
    order_id = step.arguments.get("order_id")

    date = _order_date(order_id, previous_steps)
    if date is None:
        return False, f"order {order_id!r} not found in the customer's purchases"

    if not _eligibility_confirmed(date, previous_steps):
        return False, f"ticket created without a confirmed eligibility check for order {order_id!r}"

    return True, "eligibility confirmed"


GATES = {
    "create_refund_ticket": [eligibility_gate],
}
