"""Les 8 scénarios de test : réponse attendue + outils attendus.

Chaque scénario couvre un cas que la démo "tout va bien" ne montre pas.
"""

from deepeval.dataset import Golden
from deepeval.test_case import ToolCall

LOOKUP = "lookup_customer_purchases"
CHECK = "check_refund_eligibility"
CALC = "calculate_refund_amount"
TICKET = "create_refund_ticket"

SCENARIOS = [
    {
        "name": "1-eligible",
        "input": "Bonjour, je suis awa. Je veux me faire rembourser mon casque audio.",
        "expected_output": "Remboursement accordé : un ticket est créé pour le casque audio "
                           "(commande A100), montant 89,99 €.",
        "expected_tools": [LOOKUP, CHECK, CALC, TICKET],
    },
    {
        "name": "2-hors-delai",
        "input": "Je suis koffi, je voudrais rembourser mes chaussures de course.",
        "expected_output": "Remboursement refusé : l'achat date de plus de 30 jours. "
                           "Aucun ticket n'est créé.",
        "expected_tools": [LOOKUP, CHECK],
    },
    {
        "name": "3-article-precis",
        "input": "awa ici. Rembourse-moi seulement les câbles USB-C, pas le casque.",
        "expected_output": "Remboursement accordé uniquement pour les 2 câbles USB-C "
                           "(commande A101), montant 25,00 €. Le casque n'est pas remboursé.",
        "expected_tools": [LOOKUP, CHECK, CALC, TICKET],
    },
    {
        "name": "4-plusieurs-articles",
        "input": "Je suis lina. Je veux rembourser mon clavier et ma souris.",
        "expected_output": "Remboursement accordé pour les deux articles : deux tickets "
                           "(clavier 59,90 € et souris 24,90 €), total 84,80 €.",
        "expected_tools": [LOOKUP, CHECK, CHECK, CALC, CALC, TICKET, TICKET],
    },
    {
        "name": "5-simple-question",
        "input": "Je suis awa. Mon casque audio est-il encore remboursable ? "
                 "Je veux juste savoir, je ne demande pas le remboursement.",
        "expected_output": "Oui, le casque audio est encore remboursable (acheté il y a "
                           "13 jours, délai de 30 jours). Aucun ticket n'est créé.",
        "expected_tools": [LOOKUP, CHECK],
    },
    {
        "name": "6-sans-nom",
        "input": "Bonjour, je veux me faire rembourser ma dernière commande.",
        "expected_output": "L'agent demande le nom d'utilisateur et n'appelle aucun outil.",
        "expected_tools": [],
    },
    {
        "name": "7-client-inconnu",
        "input": "Je suis paul, je veux rembourser ma montre.",
        "expected_output": "Le client paul est introuvable. Aucun remboursement, aucun ticket.",
        "expected_tools": [LOOKUP],
    },
    {
        "name": "8-jamais-achete",
        "input": "Je suis awa, je veux rembourser ma tablette.",
        "expected_output": "Awa n'a jamais acheté de tablette. Aucun remboursement, "
                           "aucun ticket.",
        "expected_tools": [LOOKUP],
    },
]


def goldens() -> list[Golden]:
    """Convertit les scénarios en Goldens DeepEval (le "dataset")."""
    return [
        Golden(
            name=s["name"],
            input=s["input"],
            expected_output=s["expected_output"],
            expected_tools=[ToolCall(name=t) for t in s["expected_tools"]],
        )
        for s in SCENARIOS
    ]
