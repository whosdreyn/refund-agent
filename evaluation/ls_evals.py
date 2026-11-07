from dotenv import load_dotenv
from langchain.messages import ToolMessage, AIMessage
from refund_agent.agent import ask, build_agent
import uuid
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import Client
from refund_agent.agent import PROMPT_VERSION
juge = wrap_openai(OpenAI())

from dataclasses import asdict
from trajectory_engine import Step, evaluate_trajectory, from_langchain_messages
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
    """Aucun ticket ne doit être créé si le client ne demande pas de remboursement."""
    if "create_refund_ticket" not in outputs["tools"]:
        return True                                   # pas de ticket : rien à vérifier, aucun appel LLM
    if outputs["decision"] != "refund_issued":
        return False                                  # incohérence évidente : vérifiable par le code

    # Un ticket a été créé : le client l'avait-il vraiment demandé ? (question de sens → LLM)
    consigne = """Tu es un juge qui évalue si un client a demandé un remboursement dans son message.
    Critères :
    - Si le client demande explicitement un remboursement, réponds OUI.
    - Si le client pose une question sur l'éligibilité ou le remboursement, mais ne demande pas explicitement un remboursement, réponds NON.
    - Si le client ne mentionne pas le remboursement, réponds NON.
    Réponds uniquement OUI ou NON."""
    reponse = juge.chat.completions.create(
        model="gpt-4o-mini", temperature=0, max_tokens=3,
        messages=[{"role": "system", "content": consigne},
                  {"role": "user", "content": inputs["input"]}],
    )
    return reponse.choices[0].message.content.strip().upper().startswith("OUI")

def message_conforme(outputs) -> bool:
    """LLM juge : le message au client est-il clair, poli et cohérent avec la décision ?"""
    consigne = """Tu es un juge qui évalue la qualité d'un message de réponse à un client.

Critères :
- Le message doit être clair et poli, en français.
- Le message doit être cohérent avec la décision de remboursement (par exemple, ne pas annoncer un remboursement si la décision est "not_found" ou "not_eligible").
- Si un remboursement a lieu, le message doit indiquer le montant remboursé.
- Si la décision n'est pas "refund_issued", l'absence de montant est normale. Une demande d'information (comme demander le prénom) est une réponse cohérente.
- Dire qu'un article est "remboursable" ou "éligible" n'est PAS annoncer un remboursement : c'est la bonne réponse à une question d'information (décision "info_only").
- Ne vérifie AUCUN calcul (additions, totaux, montants par article) : les montants sont contrôlés séparément par du code. Juge seulement la clarté, la politesse et la cohérence avec la décision.

Si TOUS les critères sont respectés, réponds OUI. Sinon, réponds NON.
Commence ta réponse par OUI ou NON, puis explique en une phrase."""
    question = f"Décision : {outputs['decision']}\nMontant : {outputs['refund_amount']}\nMessage : {outputs['message']}"
    reponse = juge.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=150,
        messages=[
            {"role": "system", "content": consigne},
            {"role": "user", "content": question},
        ],
    )
    texte = reponse.choices[0].message.content.strip()
    return {"key": "message_conforme",
            "score": texte.upper().startswith("OUI"),
            "comment": texte} 

def our_tools(messages) -> list[str]:
    """Reconstruit la liste des outils appelés à partir des messages de l'agent."""
    calls = []
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                calls.append(call["name"])
    return calls

def target(inputs: dict) -> dict:
    """Run the agent on one scenario and return everything the evaluators need."""
    result = ask(agent, inputs["input"], thread_id=str(uuid.uuid4()))
    reponse = result["structured_response"]
    return {
        "decision": reponse.decision,
        "refund_amount": reponse.refund_amount,
        "tools": our_tools(result["messages"]),
        "message": reponse.message,
        "steps": [asdict(s) for s in from_langchain_messages(result["messages"])],
    }
def trajectoire_dag(outputs) -> dict:
    """Trajectory engine: tool order, invented arguments and safety gates."""
    steps = [Step(**d) for d in outputs["steps"]]
    report = evaluate_trajectory(steps, TRANSITIONS, ARGUMENTS, GATES)
    return {"key": "trajectoire_dag",
            "score": report["score"],
            "comment": " ; ".join(report["errors"]) if report["errors"] else "valid trajectory"
            }


if __name__ == "__main__":
    #print(target({"input": "Je suis koffi, je voudrais rembourser mes chaussures de course."}).keys())
    Client().evaluate(
        target,
        data="refund-scenarios",
        evaluators=[decision_correcte, montant_correct, outils_corrects,
                    aucun_ticket_non_demande, message_conforme, trajectoire_dag],
        experiment_prefix=f"prompt-{PROMPT_VERSION}",
        metadata={"prompt_version": PROMPT_VERSION},
         num_repetitions=3,
        max_concurrency=4,
    )

   