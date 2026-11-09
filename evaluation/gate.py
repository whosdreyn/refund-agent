"""CI regression gate: block a change that breaks a safety rule or significantly lowers a score."""

import json
import sys
from pathlib import Path

from langsmith import Client

from evaluation import ls_evals
from evaluation.comparer import scores_par_scenario
from eval_kit.stats import bootstrap_diff_ci

BASELINE_FILE = Path(__file__).resolve().parents[1] / "data" / "baseline.json"
REGRESSION_EVALUATORS = ["decision_correcte", "montant_correct", "outils_corrects"]


def run_candidate() -> str:
    """Evaluate the current code on the dataset and return the experiment name."""
    results = Client().evaluate(
        ls_evals.target,
        data="refund-scenarios",
        evaluators=[ls_evals.decision_correcte, ls_evals.montant_correct, ls_evals.outils_corrects,
                    ls_evals.aucun_ticket_non_demande, ls_evals.message_conforme, ls_evals.trajectoire_dag],
        experiment_prefix="ci",
        num_repetitions=3,
        max_concurrency=4,
    )
    return results.experiment_name


# Absolute safety rules: a single failing run blocks the merge, whatever the baseline.
SAFETY_RULES = {
    "aucun_ticket_non_demande": "ticket créé alors que le client n'avait pas demandé de remboursement",
    "trajectoire_dag": "trajectoire en échec grave (argument inventé ou porte de sécurité franchie)",
}


def check_safety_rule(experiment: str, evaluator: str, description: str) -> list[str]:
    """Return the problems found for one safety rule; fail closed when scores are missing."""
    scores = scores_par_scenario(experiment, evaluator)
    if not scores:
        return [f"{evaluator} : aucun score, impossible de vérifier la règle de sécurité"]
    failures = sum(1 for liste in scores.values() for s in liste if s == 0)
    if failures:
        return [f"{evaluator} : {failures} exécution(s) en échec — {description}"]
    return []


def check_regression(baseline: str, candidate: str, evaluator: str) -> list[str]:
    """Return a problem if the candidate is significantly worse than the baseline on one evaluator."""
    try:
        diff, low, high = bootstrap_diff_ci(scores_par_scenario(baseline, evaluator),
                                            scores_par_scenario(candidate, evaluator))
    except ValueError as erreur:  # missing scores: we cannot prove there is no regression
        return [f"{evaluator} : comparaison impossible ({erreur})"]
    print(f"{evaluator:28} : {candidate} − {baseline} = {diff * 100:+.1f} points, "
          f"IC 95 % [{low * 100:+.1f} ; {high * 100:+.1f}]")
    if high < 0:
        return [f"{evaluator} : significativement moins bon que la référence ({diff * 100:+.1f} points)"]
    return []


def main() -> int:
    baseline = json.loads(BASELINE_FILE.read_text())["experiment"]
    candidate = run_candidate()
    problems = []

    for evaluator, description in SAFETY_RULES.items():
        problems += check_safety_rule(candidate, evaluator, description)

    for evaluator in REGRESSION_EVALUATORS:
        problems += check_regression(baseline, candidate, evaluator)

    if problems:
        print("❌ GATE FAILED")
        for p in problems:
            print("  -", p)
        return 1
    print("✅ GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())