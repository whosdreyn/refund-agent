"""Évaluation de l'agent avec 7 métriques DeepEval.

Deux familles de métriques, deux façons de les brancher :
- métriques de TRAJECTOIRE (task completion, step efficiency, plan quality,
  plan adherence) : elles lisent la trace complète de l'agent, capturée par
  le CallbackHandler DeepEval pendant l'exécution ;
- métriques de CAS DE TEST (tool correctness, argument correctness, GEval) :
  elles comparent ce que l'agent a fait à ce qui était attendu.

Usage :
    uv run evaluate.py                    # les 7 métriques
    uv run evaluate.py tool_correctness   # une seule (pour comprendre chacune)
    uv run evaluate.py --list             # noms disponibles
"""

import json
import os
import sys

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import AsyncConfig, DisplayConfig
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import (ArgumentCorrectnessMetric, GEval, PlanAdherenceMetric,
                              PlanQualityMetric, StepEfficiencyMetric,
                              TaskCompletionMetric, ToolCorrectnessMetric)
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall
from langchain_core.messages import AIMessage, ToolMessage

from agent import PROMPT_VERSION, ask, build_agent
from scenarios import goldens

load_dotenv()
# gpt-4o-mini est trop faible pour juger une trajectoire (il a jugé "inutile"
# l'étape de calcul exigée par la politique) : on prend un juge plus solide.
JUDGE = os.getenv("JUDGE_MODEL", "gpt-4.1")

# Chaque fabrique reçoit le golden : task_completion a besoin de connaître
# l'issue correcte, sinon le juge suppose que "la tâche" est toujours de
# rembourser et compte un refus légitime (hors délai, client inconnu) comme un échec.
TRACE_METRICS = {
    "task_completion": lambda g: TaskCompletionMetric(
        model=JUDGE, threshold=0.7,
        task=("Traiter la demande du client en appliquant la politique de remboursement. "
              f"Issue correcte attendue : {g.expected_output}")),
    "step_efficiency": lambda g: StepEfficiencyMetric(model=JUDGE, threshold=0.7),
    "plan_quality": lambda g: PlanQualityMetric(model=JUDGE, threshold=0.7),
    "plan_adherence": lambda g: PlanAdherenceMetric(model=JUDGE, threshold=0.7),
}

CASE_METRICS = {
    # Pur code : compare les noms des outils appelés aux outils attendus.
    "tool_correctness": lambda: ToolCorrectnessMetric(threshold=0.7),
    # LLM juge : les arguments passés aux outils sont-ils cohérents avec la demande ?
    "argument_correctness": lambda: ArgumentCorrectnessMetric(model=JUDGE, threshold=0.7),
    # LLM juge configurable : la réponse arrive-t-elle à la conclusion attendue ?
    "answer_correctness": lambda: GEval(
        name="Answer Correctness",
        model=JUDGE,
        threshold=0.7,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                           SingleTurnParams.EXPECTED_OUTPUT],
        evaluation_steps=[
            "Compare la décision de la réponse (remboursé, refusé, simple information, "
            "information manquante, introuvable) à celle de la réponse attendue.",
            "Vérifie que le montant remboursé correspond, s'il y en a un.",
            "Pénalise fortement un ticket créé alors que la réponse attendue n'en prévoit pas.",
            "Ignore les différences de formulation et de politesse.",
        ],
    ),
}


def tools_called(messages) -> list[ToolCall]:
    """Reconstruit la liste des outils appelés à partir des messages de l'agent."""
    outputs = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    calls = []
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                calls.append(ToolCall(name=call["name"], input_parameters=call["args"],
                                      output=outputs.get(call["id"])))
    return calls


def main(selected: list[str]) -> None:
    trace_factories = [TRACE_METRICS[n] for n in selected if n in TRACE_METRICS]
    case_metrics = [CASE_METRICS[n]() for n in selected if n in CASE_METRICS]

    print(f"Prompt {PROMPT_VERSION} | juge {JUDGE}")
    agent = build_agent()
    dataset = EvaluationDataset(goldens=goldens())
    test_cases = []

    # 1) On exécute l'agent sur chaque scénario. Si des métriques de trajectoire
    #    sont demandées, le CallbackHandler capture la trace et les y attache.
    iterator = (dataset.evals_iterator(async_config=AsyncConfig(max_concurrent=4))
                if trace_factories else dataset.goldens)
    for golden in iterator:
        # Métriques NEUVES pour chaque trace : une métrique garde son score et son
        # explication en mémoire ; partagée entre traces évaluées en parallèle,
        # les résultats s'écraseraient.
        callbacks = ([CallbackHandler(metrics=[make(golden) for make in trace_factories])]
                     if trace_factories else None)
        result = ask(agent, golden.input, thread_id=golden.name, callbacks=callbacks)
        response = result["structured_response"]
        test_cases.append(LLMTestCase(
            name=golden.name,
            input=golden.input,
            actual_output=json.dumps(response.model_dump(), ensure_ascii=False),
            expected_output=golden.expected_output,
            tools_called=tools_called(result["messages"]),
            expected_tools=golden.expected_tools,
        ))
        print(f"[{golden.name}] {response.decision} | outils : "
              f"{[t.name for t in test_cases[-1].tools_called]}")

    # 2) Les métriques de cas de test s'évaluent sur la liste construite ci-dessus.
    if case_metrics:
        evaluate(test_cases=test_cases, metrics=case_metrics,
                 async_config=AsyncConfig(max_concurrent=4),
                 display_config=DisplayConfig(truncate_passing_cases=False))


if __name__ == "__main__":
    args = sys.argv[1:]
    available = list(TRACE_METRICS) + list(CASE_METRICS)
    if args == ["--list"]:
        print("\n".join(available))
        sys.exit(0)
    unknown = [a for a in args if a not in available]
    if unknown:
        sys.exit(f"Métrique inconnue : {unknown}. Disponibles : {available}")
    main(args or available)
