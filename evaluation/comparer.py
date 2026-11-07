from collections import defaultdict

from dotenv import load_dotenv
from langsmith import Client

from trajectory_engine.stats import bootstrap_diff_ci

load_dotenv()
client = Client()


def scores_par_scenario(experiment: str, evaluator: str) -> dict:
    """Map each scenario (example id) to the list of its scores in an experiment."""
    projet = client.read_project(project_name=experiment)
    runs = list(client.list_runs(project_id=projet.id, is_root=True))
    scenario_du_run = {str(r.id): str(r.reference_example_id) for r in runs}

    scores = defaultdict(list)
    for f in client.list_feedback(run_ids=[r.id for r in runs], feedback_key=[evaluator]):
        scores[scenario_du_run[str(f.run_id)]].append(float(f.score))
    return dict(scores)


def comparer(exp_a: str, exp_b: str, evaluator: str) -> None:
    """Print whether experiment B significantly beats experiment A on one evaluator."""
    a = scores_par_scenario(exp_a, evaluator)
    b = scores_par_scenario(exp_b, evaluator)

    # Un evaluator absent d'une experiment (ex. ajouté après coup) : on le signale et on passe.
    if not a or not b:
        manque = exp_a if not a else exp_b
        print(f"{evaluator:28} : pas de scores dans {manque} → comparaison impossible")
        return

    try:
        diff, low, high = bootstrap_diff_ci(a, b)
    except ValueError as erreur:  # aucun scénario en commun entre les deux experiments
        print(f"{evaluator:28} : {erreur} → comparaison impossible")
        return

    if low > 0:
        verdict = f"significatif : {exp_b} est meilleur"
    elif high < 0:
        verdict = f"significatif : {exp_b} est moins bon"
    else:
        verdict = "pas significatif : on ne peut pas conclure"
    print(f"{evaluator:28} : {exp_b} − {exp_a} = {diff * 100:+.1f} points, "
          f"IC 95 % [{low * 100:+.1f} ; {high * 100:+.1f}] → {verdict}")


if __name__ == "__main__":
    for ev in ["decision_correcte", "outils_corrects", "trajectoire_dag"]:
        comparer("prompt-v4-cecc48db", "prompt-v6-7521d4ba", ev)