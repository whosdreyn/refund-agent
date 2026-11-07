"""Données fictives de la boutique et politique de remboursement.

La date de référence est fixe pour que les tests donnent toujours le même
résultat, quel que soit le jour où on les lance.
"""

from datetime import date

REFERENCE_DATE = date(2026, 11, 2)
REFUND_WINDOW_DAYS = 30

CUSTOMERS = {
    "awa": [
        {"order_id": "A100", "item": "casque audio", "unit_price": 89.99,
         "quantity": 1, "purchase_date": "2026-10-20"},
        {"order_id": "A101", "item": "câble USB-C", "unit_price": 12.50,
         "quantity": 2, "purchase_date": "2026-10-25"},
    ],
    "koffi": [
        {"order_id": "K200", "item": "chaussures de course", "unit_price": 120.00,
         "quantity": 1, "purchase_date": "2026-08-15"},
    ],
    "lina": [
        {"order_id": "L300", "item": "clavier", "unit_price": 59.90,
         "quantity": 1, "purchase_date": "2026-10-28"},
        {"order_id": "L301", "item": "souris", "unit_price": 24.90,
         "quantity": 1, "purchase_date": "2026-10-28"},
    ],
}
