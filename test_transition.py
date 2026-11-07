from trajectory_engine import Step
from trajectory_engine import check_transitions
from config_trajectoire import TRANSITIONS


def trajectoire(*outils):
    """Fabrique une trajectoire à partir d'une liste de noms d'outils."""
    return [Step(tool=o, arguments={}, result=None) for o in outils]


cas = {
    "koffi (correct)": trajectoire("lookup_customer_purchases", "check_refund_eligibility"),
    "ticket sans vérification": trajectoire("lookup_customer_purchases", "create_refund_ticket"),
    "ticket dès le début": trajectoire("create_refund_ticket"),
    "outil inventé": trajectoire("lookup_customer_purchases", "rembourser_tout"),
    "s'arrête après un calcul": trajectoire("lookup_customer_purchases", "check_refund_eligibility",
                                            "calculate_refund_amount"),
}

for nom, etapes in cas.items():
    verdicts = check_transitions(etapes, TRANSITIONS)
    fautes = [v["reason"] for v in verdicts if not v["valid"]]
    print(f"{nom:28} → {'✔ valide' if not fautes else '❌ ' + ' ; '.join(fautes)}")