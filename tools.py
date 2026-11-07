"""Les 4 outils de l'agent : recherche client, date, calcul, ticket simulé."""

from datetime import date
from itertools import count

from langchain_core.tools import tool

from data import CUSTOMERS, REFERENCE_DATE, REFUND_WINDOW_DAYS

_ticket_numbers = count(1)


@tool
def lookup_customer_purchases(username: str) -> dict:
    """Retrouve les achats d'un client à partir de son nom d'utilisateur.

    À appeler en premier dès que le nom d'utilisateur est connu.

    Args:
        username: nom d'utilisateur du client, en minuscules (ex. "awa")
    """
    purchases = CUSTOMERS.get(username.strip().lower())
    if purchases is None:
        return {"found": False, "error": f"Client '{username}' introuvable."}
    return {"found": True, "username": username.lower(), "purchases": purchases}


@tool
def check_refund_eligibility(purchase_date: str) -> dict:
    """Vérifie si un achat est encore dans le délai de remboursement.

    Args:
        purchase_date: date d'achat au format AAAA-MM-JJ
    """
    days = (REFERENCE_DATE - date.fromisoformat(purchase_date)).days
    return {
        "days_since_purchase": days,
        "refund_window_days": REFUND_WINDOW_DAYS,
        "eligible": 0 <= days <= REFUND_WINDOW_DAYS,
    }


@tool
def calculate_refund_amount(unit_price: float, quantity: int) -> dict:
    """Calcule le montant à rembourser pour une ligne de commande.

    Args:
        unit_price: prix unitaire de l'article, en euros
        quantity: nombre d'articles à rembourser
    """
    return {"refund_amount": round(unit_price * quantity, 2)}


@tool
def create_refund_ticket(username: str, order_id: str, item: str, amount: float) -> dict:
    """Crée un ticket de remboursement (simulé). Action IRRÉVERSIBLE.

    À appeler seulement si le client DEMANDE un remboursement, que l'achat est
    éligible et que le montant a été calculé. Jamais pour une simple question.

    Args:
        username: nom d'utilisateur du client
        order_id: identifiant de la commande (ex. "A100")
        item: nom de l'article remboursé
        amount: montant calculé par calculate_refund_amount
    """
    ticket_id = f"RF-{next(_ticket_numbers):04d}"
    return {"ticket_id": ticket_id, "username": username, "order_id": order_id,
            "item": item, "amount": amount, "status": "créé"}


TOOLS = [lookup_customer_purchases, check_refund_eligibility,
         calculate_refund_amount, create_refund_ticket]
