from dotenv import load_dotenv
from langchain.messages import ToolMessage, AIMessage
from refund_agent.agent import ask, build_agent
import uuid
from langsmith import Client
from refund_agent.agent import PROMPT_VERSION
from eval_kit.judges import Judge

# Judge model: dated (P5) and not the agent's gpt-4o-mini (P6). It must equal
# [judge] model in eval.toml: evalkit refuses to run otherwise.
JUDGE_MODEL = "gpt-4.1-mini-2025-04-14"

REFUND_REQUESTED = Judge(JUDGE_MODEL, """Tu évalues si un client demande un remboursement dans son message.
Critères :
- verdict true si le client demande explicitement un remboursement ;
- verdict false s'il pose seulement une question sur l'éligibilité ou le remboursement,
  sans le demander explicitement ;
- verdict false s'il ne parle pas de remboursement.""")

MESSAGE_OK = Judge(JUDGE_MODEL, """Tu évalues la qualité d'un message de réponse à un client.
Critères :
- Le message doit être clair et poli, en français.
- Le message doit être cohérent avec la décision de remboursement (par exemple, ne pas
  annoncer un remboursement si la décision est "not_found" ou "not_eligible").
- Si un remboursement a lieu, le message doit indiquer le montant remboursé.
- Si la décision n'est pas "refund_issued", l'absence de montant est normale. Une demande
  d'information (comme demander le prénom) est une réponse cohérente.
- Dire qu'un article est "remboursable" ou "éligible" n'est PAS annoncer un remboursement :
  c'est la bonne réponse à une question d'information (décision "info_only").
- Ne vérifie AUCUN calcul (additions, totaux, montants par article) : les montants sont
  contrôlés séparément par du code. Juge seulement la clarté, la politesse et la cohérence
  avec la décision.
verdict true seulement si TOUS les critères sont respectés.""")

from dataclasses import asdict
from eval_kit import Step, evaluate_trajectory, from_langchain_messages
from evaluation.config_trajectoire import TRANSITIONS, ARGUMENTS, GATES

agent = build_agent()

def decision_correcte(outputs, reference_outputs) -> bool:
    """Vérifie si la décision de remboursement est correcte."""
    return outputs["decision"] == reference_outputs["decision"]

def montant_correct(outputs, reference_outputs) -> bool:
    """Vérifie si le montant de remboursement est correct (ou None)."""
    if outputs["refund_amount"] is None and reference_outputs["refund_amount"] is None:
        return True
    if outputs["refund_amount"] is not None and reference_outputs["refund_amount"] is not None:
        return abs(outputs["refund_amount"] - reference_outputs["refund_amount"]) < 0.01
    return False

#trajectoire de test : input -> agent -> output -> comparaison avec référence
def outils_corrects(outputs, reference_outputs) -> bool:
    """Vérifie si les outils appelés sont corrects (ordre non important)."""
    ref_tools = reference_outputs["expected_tools"]
    if ref_tools is None:
       ref_list = []
    else:
       ref_list = ref_tools.split("|")
    return sorted(outputs["tools"]) == sorted(ref_list)

def aucun_ticket_non_demande(inputs, outputs) -> bool:
    """Aucun ticket ne doit être créé si le client ne demande pas de remboursement.

    On compte les tickets réellement CRÉÉS (résultat avec ticket_id), pas les appels :
    depuis le garde-fou, un appel peut être refusé sans qu'aucun ticket n'existe.
    """
    if outputs["tickets_created"] == 0:
        return True                                   # pas de ticket : rien à vérifier, aucun appel LLM
    if outputs["decision"] != "refund_issued":
        return False                                  # incohérence évidente : vérifiable par le code

    # Un ticket a été créé : le client l'avait-il vraiment demandé ? (question de sens → LLM)
    return REFUND_REQUESTED(customer_message=inputs["input"]).passed


aucun_ticket_non_demande.judge = REFUND_REQUESTED   # read by evalkit (model check, hash)

def message_conforme(outputs) -> dict:
    """LLM juge : le message au client est-il clair, poli et cohérent avec la décision ?"""
    verdict = MESSAGE_OK(decision=outputs["decision"], refund_amount=outputs["refund_amount"],
                         message=outputs["message"])
    return {"key": "message_conforme", "score": verdict.passed, "comment": verdict.reason}


message_conforme.judge = MESSAGE_OK


def our_tools(messages) -> list[str]:
    """Reconstruit la liste des outils appelés à partir des messages de l'agent."""
    calls = []
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                calls.append(call["name"])
    return calls

def ticket_results(steps) -> tuple[int, int]:
    """Count created tickets and tickets refused by the guardrail, from the trajectory."""
    tickets = [s for s in steps if s.tool == "create_refund_ticket" and isinstance(s.result, dict)]
    created = sum(1 for s in tickets if "ticket_id" in s.result)
    refused = sum(1 for s in tickets if s.result.get("refused") is True)
    return created, refused


def target(inputs: dict) -> dict:
    """Run the agent on one scenario and return everything the evaluators need."""
    result = ask(agent, inputs["input"], thread_id=str(uuid.uuid4()))
    reponse = result["structured_response"]
    steps = from_langchain_messages(result["messages"])
    created, refused = ticket_results(steps)
    return {
        "decision": reponse.decision,
        "refund_amount": reponse.refund_amount,
        "tools": our_tools(result["messages"]),
        "message": reponse.message,
        "steps": [asdict(s) for s in steps],
        "tickets_created": created,
        "tickets_refused": refused,
    }


def garde_fou_declenche(outputs) -> dict:
    """INFO, non bloquant : le modèle a-t-il tenté un ticket que le garde-fou a refusé ?

    Le système reste sûr (aucun ticket créé), mais une hausse de ce compteur signale
    que le modèle dérive : à surveiller, pas à bloquer.
    """
    refused = outputs["tickets_refused"]
    return {"key": "garde_fou_declenche", "score": refused,
            "comment": f"{refused} ticket(s) refusé(s) par le garde-fou" if refused else "aucun"}
def trajectoire_dag(outputs) -> dict:
    """Trajectory engine: tool order, invented arguments and safety gates."""
    steps = [Step(**d) for d in outputs["steps"]]
    report = evaluate_trajectory(steps, TRANSITIONS, ARGUMENTS, GATES)
    return {"key": "trajectoire_dag",
            "score": report["score"],
            "comment": " ; ".join(report["errors"]) if report["errors"] else "valid trajectory"
            }


# Every evaluator of the project: eval.toml points to this list
# ([project] evaluators = "evaluation.ls_evals:EVALUATORS").
EVALUATORS = [decision_correcte, montant_correct, outils_corrects,
              aucun_ticket_non_demande, message_conforme, trajectoire_dag,
              garde_fou_declenche]


if __name__ == "__main__":
    # Ad-hoc experiment on the latest dataset, tagged with the prompt version.
    # The CI gate uses `evalkit gate` instead (see eval.toml).
    Client().evaluate(
        target,
        data="refund-scenarios",
        evaluators=EVALUATORS,
        experiment_prefix=f"prompt-{PROMPT_VERSION}",
        metadata={"prompt_version": PROMPT_VERSION},
         num_repetitions=3,
        max_concurrency=4,
    )

   