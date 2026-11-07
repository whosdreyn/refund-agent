# Agent de remboursement évalué avec DeepEval — guide (20 min)

Projet construit d'après la description du tutoriel (LangChain + OpenAI + DeepEval).
Le code est original, testé, et les résultats ci-dessous sont **réels** (2026-11-02).

## Lancer

```bash
cp .env.example .env              # mets ta clé OPENAI_API_KEY
uv sync
uv run agent.py                   # démo : 2 tours de conversation (mémoire)
uv run evaluate.py --list         # les 7 métriques
uv run evaluate.py tool_correctness           # une seule métrique (gratuit, pur code)
uv run evaluate.py                             # les 7 (~1,85 $ avec le juge gpt-4.1)
PROMPT_VERSION=v1 uv run evaluate.py           # rejouer une ancienne version
```

**Coût** : les 4 métriques de trajectoire avec `gpt-4.1` coûtent ~1,80 $ par passage
complet, les 3 autres ~0,04 $. Pendant que tu itères, lance une métrique à la fois.
`JUDGE_MODEL=gpt-4o-mini` est 15× moins cher mais donne des scores trompeurs (voir §5).

---

## 1. L'agent (`data.py`, `tools.py`, `schemas.py`, `agent.py`)

| Fichier | Rôle |
|---|---|
| `data.py` | 3 clients fictifs (awa, koffi, lina), délai de 30 jours, date fixe |
| `tools.py` | 4 outils `@tool` : recherche client, éligibilité (date), calcul, ticket simulé |
| `schemas.py` | réponse structurée Pydantic : `decision`, `refund_amount`, `ticket_ids`, `message` |
| `agent.py` | `create_agent` + `MemorySaver` (mémoire par `thread_id`) + prompt v1/v2/v3 |

Points clés :
- **La description d'un outil est une instruction** : c'est elle que lit le LLM pour décider
  quand l'appeler. `create_refund_ticket` dit explicitement « IRRÉVERSIBLE, jamais pour une
  simple question ».
- **La politique métier va dans le prompt** (30 jours, quand créer un ticket) ET dans les
  scénarios de test : c'est la même règle, vue côté agent et côté évaluation.
- **`response_format=RefundResponse`** force une sortie exploitable par du code
  (`result["structured_response"].decision`).
- **La date est fixe** (`REFERENCE_DATE`) sinon les tests changeraient chaque jour.

## 2. Les 8 scénarios (`scenarios.py`)

Chaque scénario = une demande + la **réponse attendue** + les **outils attendus** :

| # | Cas | Outils attendus | Ce qu'il teste |
|---|---|---|---|
| 1 | casque, éligible | lookup, check, calc, ticket | le chemin normal |
| 2 | hors délai | lookup, check | savoir refuser |
| 3 | un article précis | lookup, check, calc, ticket | ne pas rembourser le reste |
| 4 | deux articles | lookup, check×2, calc×2, ticket×2 | traiter chaque article |
| 5 | simple question | lookup, check | **ne pas créer de ticket** |
| 6 | sans nom | aucun | demander avant d'agir |
| 7 | client inconnu | lookup | s'arrêter proprement |
| 8 | jamais acheté | lookup | ne pas inventer |

Dans DeepEval, ce sont des `Golden` (l'équivalent d'un exemple de dataset LangSmith).

## 3. Les 7 métriques (`evaluate.py`)

Deux familles, branchées différemment :

**A. Trajectoire** — lisent la trace complète, capturée par `CallbackHandler` pendant
l'exécution, dans une boucle `dataset.evals_iterator()` :

| Métrique | Question | Type |
|---|---|---|
| Task completion | la tâche est-elle accomplie ? | LLM juge |
| Step efficiency | des étapes inutiles ? | LLM juge |
| Plan quality | le plan était-il bon ? | LLM juge |
| Plan adherence | le plan a-t-il été suivi ? | LLM juge |

**B. Cas de test** — comparent ce qui a été fait à ce qui était attendu, via `evaluate()` :

| Métrique | Question | Type |
|---|---|---|
| Tool correctness | les bons outils ? | **pur code** (noms comparés) |
| Argument correctness | les bons arguments ? | LLM juge |
| Answer correctness (GEval) | la bonne conclusion ? | LLM juge **avec tes critères** |

Pour la famille B, `tools_called()` reconstruit la liste des outils appelés à partir des
messages de l'agent (`AIMessage.tool_calls` + `ToolMessage`).

## 4. Résultats réels : 3 itérations

| Métrique | v1 | v2 | v3 |
|---|---|---|---|
| Task completion | 0,89 | 0,89 | **0,99** |
| Step efficiency | 0,94 | 0,72 | 0,88 |
| Plan quality | 0,28 | 0,25 | 0,25 |
| Plan adherence | 0,94 | 0,81 | 0,88 |
| Tool correctness | 0,75 | 0,92 | **0,92** |
| Argument correctness | 0,97 | 0,87 | 0,88 |
| Answer correctness | 0,82 | 0,82 | **0,94** |
| **Bonnes décisions** | 6/8 | 7/8 | **8/8** |

- **v1** : 2 erreurs. Lina dit « Je suis lina » → l'agent redemande le nom (scénario 4).
  Question sur le casque → l'agent répond sans rien vérifier (scénario 5).
- **v2** : corrige les deux… et **casse le scénario 8** : pour une tablette jamais achetée,
  l'agent crée 2 tickets pour le casque et les câbles. La règle « traite chaque article »
  l'a poussé à rembourser ce qu'on ne demandait pas — une action irréversible.
- **v3** : ajoute « ne traite que les articles nommés, sinon not_found » → 8/8.

**Leçon n°1 : on relance TOUS les scénarios à chaque changement.** Sans le scénario 8,
la v2 aurait été déployée avec un bug qui rembourse de l'argent par erreur.

## 5. Lire un échec : 4 causes possibles

Un score bas n'est pas forcément une erreur de l'agent. Tout ce qui suit est arrivé ici :

| Cause | Ce qui s'est passé | Correction |
|---|---|---|
| **Erreur de l'agent** | v1 ignore « Je suis lina » ; v2 rembourse des articles non demandés | prompt v2, v3 |
| **Bug de mesure** | les 4 métriques de trajectoire étaient créées une fois et partagées entre les 8 traces évaluées en parallèle : les explications du scénario 1 parlaient des câbles du scénario 3 | une métrique neuve par trace |
| **Info manquante pour le juge** | task completion supposait que « la tâche » = rembourser, et comptait le refus légitime de koffi comme un échec | passer `task=` avec l'issue attendue → 0,53 → 0,99 |
| **Juge trop faible ou grille inadaptée** | `gpt-4o-mini` jugeait « inutile » l'étape de calcul exigée (efficacité 0,25 → 0,94 avec `gpt-4.1`) ; plan quality attend un plan de service client générique (« demander la raison du retour ») et reste à 0,25 ; argument correctness reproche à la recherche client de ne pas filtrer sur « tablette » alors que l'outil ne prend que le nom | juge plus fort ; ignorer ou remplacer plan quality |

**Leçon n°2 : lis l'explication (`reason`) avant de corriger quoi que ce soit.**

## 6. Ce qui reste (pour aller plus loin)

1. **Scénario 5** : l'agent calcule le délai lui-même au lieu d'appeler
   `check_refund_eligibility`. Réponse juste mais non vérifiée par l'outil.
2. **Plan quality** n'accepte pas de critères : remplace-la par un `GEval` qui connaît la
   politique (« le plan suit-il : recherche → éligibilité → calcul → ticket ? »).
3. Ajouter des scénarios multi-tours (le nom donné au 2ᵉ message, comme dans la démo).
4. Brancher aussi LangSmith (`LANGSMITH_TRACING=true`) pour voir les traces dans l'interface.

## DeepEval vs LangSmith, en une phrase

DeepEval apporte des **métriques d'agent toutes faites** (trajectoire, outils) qui tournent
en local comme des tests ; LangSmith apporte la **plateforme** (traces, datasets,
experiments, production). Ils se combinent.
