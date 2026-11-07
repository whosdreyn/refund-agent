from trajectory_engine import Step
from trajectory_engine import check_arguments
from config_trajectoire import ARGUMENTS

ACHATS_AWA = {"found": True, "purchases": [
    {"order_id": "A100", "item": "casque audio", "unit_price": 89.99, "quantity": 1, "purchase_date": "2026-10-20"},
    {"order_id": "A101", "item": "câble USB-C", "unit_price": 12.5, "quantity": 2, "purchase_date": "2026-10-25"},
]}
lookup = Step("lookup_customer_purchases", {"username": "awa"}, ACHATS_AWA)


def check(date):
    return Step("check_refund_eligibility", {"purchase_date": date}, {"eligible": True})


def calc(prix, quantite):
    return Step("calculate_refund_amount", {"unit_price": prix, "quantity": quantite},
                 {"refund_amount": round(prix * quantite, 2)})


def ticket(commande, montant):
    return Step("create_refund_ticket", {"order_id": commande, "amount": montant}, {"ticket_id": "RF-0001"})


cas = {
    "câble unique (correct)":   [lookup, check("2026-10-25"), calc(12.5, 1), ticket("A101", 12.5)],
    "order_id inventé":         [lookup, check("2026-10-25"), calc(12.5, 1), ticket("A105", 12.5)],
    "montant ≠ calcul":         [lookup, check("2026-10-25"), calc(12.5, 1), ticket("A101", 25.0)],
    "date inventée":            [lookup, check("2026-10-30"), calc(12.5, 1), ticket("A101", 12.5)],
    "prix inventé":             [lookup, check("2026-10-25"), calc(10.0, 1), ticket("A101", 10.0)],
    "ticket avant tout calcul": [lookup, check("2026-10-25"), ticket("A101", 12.5)],
}

for nom, etapes in cas.items():
    verdicts = check_arguments(etapes, ARGUMENTS)
    fautes = [v["reason"] for v in verdicts if not v["valid"]]
    print(f"{nom:26} → {'✔ valide' if not fautes else '❌ ' + ' ; '.join(fautes)}")



        