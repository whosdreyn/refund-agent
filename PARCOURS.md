# Parcours d'une journée — savoir évaluer un agent

**Objectif du soir** : savoir appliquer, sur n'importe quel agent, la méthode
traces → dataset → evaluators → experiments → production. Et savoir où DeepEval s'insère.

**Règle du jour** : un seul agent, `refund-agent`. Il a des outils, une politique métier,
des cas pièges, et tu connais déjà ses bugs. Tout se fait dessus, dans l'ordre.

| Bloc | Durée | Ce que tu sauras faire à la fin |
|---|---|---|
| 0. Le modèle mental | 15 min | expliquer les 6 notions et la boucle |
| 1. Les traces | 1 h | lire une trace d'agent et y trouver un bug |
| 2. Le dataset | 1 h | construire un bon dataset (code, CSV, traces) |
| 3. Les evaluators | 2 h | évaluer la réponse ET la trajectoire, comparer des versions |
| 4. DeepEval | 45 min | traduire LangSmith ↔ DeepEval, savoir quand l'utiliser |
| 5. La production | 1 h | surveiller un agent en production et nourrir le dataset |
| 6. La fiche méthode | 30 min | ta checklist pour tous tes futurs agents |

Ne passe au bloc suivant que si tu réponds « oui » aux questions de contrôle.

---

## Bloc 0 — Le modèle mental (15 min)

Les 6 notions, valables dans tous les outils :

| Notion | Définition | Dans DeepEval |
|---|---|---|
| **Trace** | tout ce que l'agent a fait pour une demande (appels LLM, outils, arguments) | trace |
| **Dataset** | des exemples : une entrée + ce qu'on attend | `EvaluationDataset` |
| **Example** | une ligne du dataset | `Golden` |
| **Evaluator** | une fonction qui note une sortie | métrique (`BaseMetric`) |
| **Experiment** | une version de l'agent passée sur tout le dataset, avec les scores | un passage de `evaluate()` |
| **Feedback** | un score attaché à une trace | score + `reason` |

La boucle :
```
construire → TRACER → dataset → EVALUATORS → experiment v1, v2… → PRODUCTION
     ▲                                                                │
     └───────── les échecs de production reviennent dans le dataset ──┘
```

Deux choses à évaluer sur un agent :
- **la réponse finale** : la bonne décision, le bon montant ;
- **la trajectoire** : les bons outils, les bons arguments, aucune action interdite.

**Contrôle** : peux-tu expliquer la différence entre un example et un experiment ?

---

## Bloc 1 — Les traces (1 h)

**Mise en place** : dans `refund-agent/.env`, ajoute
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=refund-agent
```
`create_agent` est tracé automatiquement : rien à changer dans le code.

**Exercices**
1. Lance `uv run agent.py`. Ouvre le projet `refund-agent` sur smith.langchain.com.
   Dans une trace, repère : les appels au LLM, chaque outil avec ses arguments et sa
   sortie, les tokens, le coût, la durée.
2. Va dans l'onglet **Threads** : les 2 tours de la démo y sont regroupés, parce qu'ils
   partagent le même `thread_id`. C'est ça, la mémoire, vue dans les traces.
3. Ajoute la version du prompt en métadonnée, pour pouvoir filtrer par version.
   Dans `ask()`, ajoute au `config` :
   `"metadata": {"prompt_version": PROMPT_VERSION}`.
4. Relance avec `PROMPT_VERSION=v2` le scénario « tablette » :
   « Je suis awa, je veux rembourser ma tablette. »
   Puis, dans la barre de recherche, trouve les traces qui ont appelé
   `create_refund_ticket` (portée **Any run**) :
   `run_type:tool AND name:create_refund_ticket`.
   Tu viens de trouver la régression de la v2 **par les traces**, sans aucun evaluator.

**Contrôle**
- Sais-tu dire, en lisant une trace, quel outil a été appelé, avec quels arguments ?
- Sais-tu filtrer les traces par outil appelé et par version de prompt ?

---

## Bloc 2 — Le dataset (1 h)

**L'idée clé** : une référence doit être **vérifiable par du code**. « Remboursement
accordé pour le casque » est une phrase, difficile à vérifier. Préfère des champs :
```json
{"decision": "refund_issued", "refund_amount": 89.99,
 "expected_tools": ["lookup_customer_purchases", "check_refund_eligibility",
                    "calculate_refund_amount", "create_refund_ticket"]}
```

**Exercices**
1. **Par CSV** : crée `scenarios.csv` avec les colonnes `input, decision,
   refund_amount, expected_tools` (outils séparés par `|`), une ligne par scénario
   de `scenarios.py`. Importe-le :
   ```python
   from langsmith import Client
   Client().upload_csv("scenarios.csv", input_keys=["input"],
                       output_keys=["decision", "refund_amount", "expected_tools"],
                       name="refund-scenarios")
   ```
   Vérifie dans l'interface que les 8 exemples sont là.
2. **Depuis une trace** : ouvre la trace de la démo (2 tours) et ajoute-la au dataset
   avec le bouton **Add to dataset**. C'est ainsi qu'un cas réel devient un test.
3. **Ajoute 2 scénarios pièges de ton invention**, par exemple « je veux rembourser la
   moitié de mes câbles » ou un client qui change d'avis en cours de route.

**Contrôle**
- Ton dataset couvre-t-il surtout des cas difficiles ?
- Chaque référence est-elle vérifiable par du code ?

---

## Bloc 3 — Les evaluators (2 h, le cœur de la journée)

Crée `ls_evals.py`. La fonction cible renvoie la réponse **et** la liste des outils
appelés, pour pouvoir évaluer la trajectoire :

```python
def target(inputs: dict) -> dict:
    result = ask(agent, inputs["input"], thread_id=str(uuid.uuid4()))
    tools = [c["name"] for m in result["messages"] if isinstance(m, AIMessage)
             for c in m.tool_calls]
    r = result["structured_response"]
    return {"decision": r.decision, "refund_amount": r.refund_amount, "tools": tools}
```

**Écris 5 evaluators**, du plus simple au plus fin :

| # | Evaluator | Type | Vérifie |
|---|---|---|---|
| 1 | `decision_correcte` | code | `decision` == référence |
| 2 | `montant_correct` | code | montant à 0,01 près (ou les deux sont vides) |
| 3 | `outils_corrects` | code | les outils appelés == les outils attendus (même liste, ordre ignoré) |
| 4 | `aucun_ticket_non_demande` | code, **sans référence** | pas de `create_refund_ticket` si la décision n'est pas `refund_issued` |
| 5 | `message_conforme` | LLM juge | le message au client est clair, poli et cohérent avec la décision |

Pour le n° 5, applique tout ce que tu as appris : question binaire, `temperature=0`,
critères explicites, et vérification du juge sur quelques traces.

**Lance 3 experiments**, une par version :
```python
client.evaluate(target, data="refund-scenarios",
                evaluators=[decision_correcte, montant_correct, outils_corrects,
                            aucun_ticket_non_demande, message_conforme],
                experiment_prefix=f"prompt-{PROMPT_VERSION}",
                metadata={"prompt_version": PROMPT_VERSION})
```
(avec `PROMPT_VERSION=v1`, puis `v2`, puis `v3`, au lancement)

**Analyse dans l'interface** (vue **Compare** des 3 experiments) :
1. Retrouve la régression de la v2 : quel evaluator la détecte ? Pourquoi
   `aucun_ticket_non_demande` est-il ton evaluator le plus important ?
2. Compare tes scores aux résultats DeepEval du `GUIDE.md`. Lesquels de tes evaluators
   sont plus fiables que les métriques génériques, et pourquoi ?
3. Ajoute un **summary evaluator** `taux_decisions_correctes` (paramètre
   `summary_evaluators=`).

**Contrôle**
- Sais-tu évaluer une trajectoire, pas seulement une réponse ?
- Sais-tu dire si un échec vient de l'agent, du dataset ou de l'evaluator ?
- Sais-tu prouver, chiffres à l'appui, qu'une version est meilleure sans régression ?

---

## Bloc 4 — DeepEval (45 min)

Tu ne vas pas tout apprendre de DeepEval. Seulement ce qui sert.

**La traduction**
| LangSmith | DeepEval |
|---|---|
| example | `Golden` |
| run + référence | `LLMTestCase` (`actual_output`, `expected_output`, `tools_called`, `expected_tools`) |
| evaluator | métrique |
| `client.evaluate()` | `evaluate(test_cases, metrics)` |
| trace dans un experiment | `CallbackHandler` + `dataset.evals_iterator()` |

**Exercices**
1. Relis `evaluate.py` avec ce tableau en tête : chaque ligne doit avoir du sens.
2. Lance `uv run evaluate.py tool_correctness` (gratuit), puis
   `uv run evaluate.py answer_correctness`. Lis les explications (`reason`).
3. Retiens les 3 pièges que tu as rencontrés : créer des métriques neuves pour chaque
   trace, donner sa tâche précise au juge, se méfier d'un juge trop faible.

**Quand utiliser DeepEval** : quand tu veux une métrique d'agent prête à l'emploi
(`ToolCorrectness`, `GEval`) sans l'écrire toi-même. Pour une règle métier précise, ton
evaluator sur mesure du bloc 3 est plus fiable.

**Contrôle**
- Peux-tu traduire un evaluator LangSmith en métrique DeepEval, et inversement ?

---

## Bloc 5 — La production (1 h)

Simule la production : lance l'agent sur 10 demandes variées (bonnes et mauvaises) dans
le projet `refund-agent`.

1. **Évaluation automatique** : transforme `aucun_ticket_non_demande` en evaluator de
   code dans l'interface (fonction `perform_eval(run)` à un seul argument), sur les runs
   racine, à 100 %.
2. **Règle d'automation** : score 0 → ajout au dataset `refund-scenarios`, et → file
   de relecture `revue-refund`.
3. **Alerte** : dans **Monitoring**, regarde le taux d'erreurs et le coût par trace.
4. Provoque un échec (relance la v2 sur la tablette) et suis-le de bout en bout :
   trace → score 0 → dataset → file de relecture.

**Contrôle**
- Sais-tu quels evaluators sont utilisables en production, c'est-à-dire sans référence ?
- Sais-tu expliquer comment un échec de production devient un test ?

---

## Bloc 6 — La fiche méthode (30 min)

Écris, avec tes mots, ta checklist pour **tout futur agent**. Le squelette :

1. **Tracer dès le premier jour** (`LANGSMITH_TRACING=true`, métadonnées de version).
2. **Écrire la politique** : ce que l'agent doit faire et ne jamais faire.
3. **Dataset de 10 à 20 cas**, surtout des pièges, avec des références vérifiables.
4. **Evaluators** :
   - par code d'abord (décision, valeurs, outils, actions interdites) ;
   - LLM juge seulement pour le sens, avec des critères précis ;
   - au moins un evaluator sans référence, pour la production.
5. **Une experiment par changement**, et relancer **tous** les cas à chaque fois.
6. **Lire les traces et les explications avant de corriger**, et se demander : agent,
   dataset ou evaluator ?
7. **En production** : évaluation automatique + règle « échec → dataset » + relecture
   humaine d'un échantillon.

Applique-la tout de suite, en 10 minutes, à ton projet de mémoire : quelle politique ?
quels 10 cas pièges ? quels evaluators par code ?

---

## Si tu bloques

Tout ce qu'il faut est déjà dans tes projets :
- `langsmith-reliable-agents/` : dataset, evaluators, A/B, règles de production ;
- `refund-agent/evaluate.py` : extraction des outils appelés, DeepEval ;
- `refund-agent/GUIDE.md` : les résultats v1/v2/v3 et les 4 causes d'échec.
