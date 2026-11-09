# Quels moteurs d'évaluation construire avec LangSmith

Synthèse de la documentation officielle (docs.langchain.com/langsmith), lue le 2026-11-09.
Les chemins entre parenthèses sont les pages sources (préfixe `docs.langchain.com/langsmith/`).

**Ce qui a été lu.** La doc LangSmith compte 1 269 pages. Environ 1 000 portent sur le déploiement,
l'auto-hébergement, la passerelle LLM et la référence de l'API REST. Elles n'ont rien à voir avec
l'évaluation et ont été écartées. Ont été lues en entier les ~130 pages qui touchent à l'évaluation :
concepts, évaluateurs, tutoriels par type d'application, évaluation en ligne, retours humains,
datasets, expériences, tests, CI, surveillance, Engine. Le dépôt GitHub `agentevals` a aussi été
consulté, pour les points que la doc renvoie vers lui.

---

## 1. La carte en une image

```
           AVANT LA MISE EN PRODUCTION (offline)          EN PRODUCTION (online)
           ─────────────────────────────────────          ──────────────────────
 Dataset (exemples : inputs + reference outputs)         Traces réelles (runs, threads)
        │                                                        │
        ▼                                                        ▼
 Cible (fonction, graphe, nœud, expérience existante)    Règles d'automatisation (filtre + échantillonnage)
        │                                                        │
        ▼                                                        ▼
 Évaluateurs  ──────────────►  FEEDBACK  ◄──────────────  Évaluateurs en ligne (sans référence)
 (code, juge, paires,         (key, score|value,          (code, juge, composite, thread)
  résumé, composite)           comment, correction)                │
        │                         │   ▲                            ▼
        ▼                         │   └── humains (files d'annotation, utilisateurs finaux)
 Expérience  →  comparaison       ▼
 (baseline, rouge/vert)    Tableaux de bord · Alertes · Engine (détection de problèmes)
                                  │
                                  └──► les traces en échec redeviennent des exemples du dataset
```

L'idée centrale : **tout est du feedback**. Un évaluateur de code, un juge LLM, un humain dans une
file d'annotation ou un client qui clique 👎 écrivent tous le même objet `{key, score|value, comment}`.
Les alertes, les tableaux de bord, les règles et Engine lisent ce même objet. Un moteur d'évaluation
LangSmith, c'est donc : **qui produit le feedback, sur quoi, quand, et qui le consomme.**

---

## 2. Les briques disponibles

### 2.1 Les types d'évaluateurs

| Type | Reçoit | Renvoie | Où il tourne |
|---|---|---|---|
| **Code (SDK)** | n'importe quel sous-ensemble de `inputs, outputs, reference_outputs, run, example` | `bool`, `int/float`, `str` (catégorie), `dict {key, score/value, comment}`, `list[dict]` (plusieurs scores) | `client.evaluate` (`code-evaluator-sdk`) |
| **Code (UI)** | `perform_eval(run, example)` | `dict` métrique → score | lié à un dataset ou en ligne ; bac à sable sans réseau, avec numpy, pandas, scipy, scikit-learn, jsonschema (`code-evaluator-ui`) |
| **Juge LLM** | variables du prompt reliées à l'entrée, la sortie et la référence | sortie structurée : chaque clé = un feedback | SDK, UI, en ligne (`llm-as-judge`) |
| **Modèles préconstruits** | selon le modèle | booléen + justification | UI : sécurité (fuite de PII, injection de prompt), sûreté, qualité (hallucination, pertinence, concision), conversation (satisfaction, tâche accomplie, mémoire), trajectoire (sélection d'outils, respect du plan), image, voix (`evaluators`) |
| **openevals / agentevals** | `inputs, outputs, reference_outputs` | `{key, score, comment}` | bibliothèques open source, dans `evaluate` ou pytest (`openevals`, `trajectory-evals`) |
| **Par paires** | 2 sorties pour la même entrée | `{key, scores: {run_id: score}}` | sur 2 expériences existantes, avec ordre aléatoire contre le biais de position (`evaluate-pairwise`) |
| **Résumé** | des **listes** sur toute l'expérience | un score global (ex. F1) | `summary_evaluators=[...]` (`summary`) |
| **Composite** | plusieurs clés de feedback + poids | moyenne ou somme pondérée | UI, ou script après coup (`composite-evaluators-ui/sdk`) |
| **Modèle de décision** | un « état » + des questions typées | probabilité, choix ou niveau | UI et REST seulement, offre Plus ou Enterprise (`decision-model-evaluator`) |
| **Humain** | file d'annotation (une trace, ou deux côte à côte) | grille de notation | UI et SDK (`annotation-queues`) |
| **Assertions** | affirmations libres écrites par un humain sur un exemple | un score par assertion | stockées dans `reference_outputs["assertions"]` (`assertions`) |

⚠️ **Piège de polarité** : pour les modèles de détection (sécurité, sûreté, hallucination, erreur
perçue, image, qualité audio), `true` veut dire **problème trouvé**. Pour les autres, `true` veut dire
réussi.

### 2.2 Les techniques documentées

- **Avec ou sans référence.** Les évaluateurs sans référence (sécurité, format, pertinence,
  ancrage dans les sources) servent avant **et** après la mise en production. Ceux qui ont besoin
  d'une référence (exactitude, correspondance exacte) ne servent qu'avant.
- **Répétitions** : `num_repetitions=N` relance la cible **et** les évaluateurs. L'interface affiche
  la moyenne et l'écart-type (`repetition`).
- **Évaluer une expérience existante** : `evaluate("nom-expérience", evaluators=[...])` ajoute un
  évaluateur sans relancer l'agent (`evaluate-existing-experiment`).
- **Calibrer un juge** : « Align Evaluator » calcule le % d'accord entre le juge et des étiquettes
  humaines ; la doc recommande au moins 20 exemples équilibrés. Les corrections humaines peuvent aussi
  servir d'exemples few-shot au juge (`improve-judge-evaluator-feedback`, `create-few-shot-evaluators`).
- **Évaluateurs en chaîne** : un juge coûteux ne se lance que sur les traces qu'un premier
  évaluateur a mal notées (`evaluators#chain-evaluators`).
- **Plafond de dépenses** hebdomadaire par évaluateur, remis à zéro le lundi à 00:00 UTC
  (`evaluator-spend`).
- **Relance des erreurs** : `error_handling="ignore"`, puis on relance les exemples en échec dans la
  même expérience (`evaluate-with-retry`).

---

## 3. Les moteurs que tu peux construire, par type d'application

### Moteur 1 — Chatbot (un appel ou une conversation)

**Pour un seul tour** (`evaluate-chatbot-tutorial`)
- Dataset : `inputs={"question"}`, `outputs={"answer"}`. Commencer par 10 à 20 exemples écrits à la
  main.
- Évaluateurs :
  - un juge d'exactitude par rapport à la référence ;
  - une heuristique de concision (`len(réponse) < 2 × len(référence)`) ;
  - les modèles prêts à l'emploi : pertinence, ton, toxicité, fuite de PII.

**Pour plusieurs tours, sans vrais utilisateurs** (`multi-turn-simulation`)
- `openevals.simulators` fournit `create_llm_simulated_user(system="<persona>", fixed_responses=[...])`
  et `run_multiturn_simulation(app, user, max_turns=5, trajectory_evaluators=[...])`.
- Une colonne `simulated_user_prompt` dans le dataset permet de faire varier les personas : client
  pressé, client confus, client qui essaie de tricher.
- L'application reçoit **un seul message par tour** et doit garder elle-même l'historique, par
  `thread_id`.

**Pour plusieurs tours, en production** (`online-evaluations-multi-turn`)
- Les évaluateurs « thread » se déclenchent une fois la conversation inactive (10 minutes par défaut,
  2 minutes au minimum).
- Variables disponibles : `all_messages`, `human_ai_pairs`, `first_human_last_ai`, `trajectory`
  (avec les appels d'outils, seulement sur Cloud GCP US).
- Condition : chaque trace a une liste `messages` en entrée et en sortie, et `thread_id` est posé sur
  **toutes** les traces enfants.
- Modèles prêts à l'emploi : satisfaction, tâche accomplie, mémoire de la conversation, ton.

**Ce que tu peux en faire** : un moteur « conversation » qui génère des personas, simule 5 tours,
puis juge trois choses. La tâche est-elle accomplie ? L'agent a-t-il oublié une information donnée
plus tôt ? Est-il resté dans son rôle ?

### Moteur 2 — RAG

La doc définit **4 familles** d'évaluateurs (`evaluate-rag-tutorial`) :

| Famille | Compare | Référence ? |
|---|---|---|
| exactitude | réponse ↔ réponse de référence | oui |
| pertinence | réponse ↔ question | non |
| ancrage (*groundedness*) | réponse ↔ documents récupérés : détecte l'hallucination | non |
| pertinence de la récupération | question ↔ documents récupérés | non |

Deux façons de montrer les documents récupérés aux évaluateurs :
1. **Dans la sortie** : la cible renvoie `{"answer", "documents"}`.
2. **Dans la trace** : on marque le récupérateur avec `@traceable(run_type="retriever")`, puis
   l'évaluateur reçoit `run` et parcourt `run.child_runs` (`evaluate-on-intermediate-steps`).
   La sortie publique reste ainsi propre.

Les trois familles sans référence tournent aussi **en production** sur le trafic réel. Prompts
prêts dans le Hub : `rag-document-relevance`, `rag-answer-hallucination`, `rag-answer-helpfulness`,
`rag-answer-vs-reference` (`evaluation-approaches`).

**Ce que tu peux en faire** : un moteur RAG qui mesure séparément **la récupération** (les bons
documents sont-ils là ?) et **la génération** (la réponse s'appuie-t-elle dessus ?). C'est ce qui
permet de savoir quoi corriger. Ton idée d'agent d'instruction de dossiers (OCR + LLM) est
exactement ce cas.

### Moteur 3 — Graphe LangGraph

Il y a quatre niveaux d'évaluation, plus deux points techniques (`evaluate-graph`,
`evaluate-complex-agent`) :

1. **Bout en bout.** `target = example_to_state | app`, puis on évalue `outputs["messages"][-1]`.
   Si un nœud est asynchrone, il faut `aevaluate`.
2. **Un nœud isolé.** `example_to_state | app.nodes["agent"]` teste un seul nœud, sans lancer tout
   le graphe. C'est idéal pour un **nœud de routage** : on compare `command.goto` à la route attendue.
3. **Trajectoire de nœuds, à la main.**
   - On parcourt le graphe avec `graph.astream(..., subgraphs=True, stream_mode="debug")` et on
     garde le nom de chaque tâche, plus le nom des outils appelés.
   - On note la correspondance partielle avec la trajectoire attendue : un crédit partiel, pas un
     simple oui/non.
   - Sans `subgraphs=True`, les nœuds des sous-graphes sont invisibles.
4. **Trajectoire de graphe avec agentevals** (dans le dépôt GitHub, pas dans la doc LangSmith).
   - `extract_langgraph_trajectory_from_thread(graph, config)` renvoie `{"inputs", "outputs":
     {"results", "steps"}}`. `steps` est la liste des nœuds pour chaque tour. Cette fonction capture
     **les interruptions et les reprises** (`__interrupt__`, `__resuming__`).
   - Deux évaluateurs utilisent ce format : `graph_trajectory_strict_match` (correspondance exacte)
     et `create_graph_trajectory_llm_as_judge` (juge LLM).
   - C'est la seule manière documentée d'évaluer un graphe avec validation humaine (*human-in-the-loop*).
5. **Neutraliser les effets de bord.** Un nœud destructeur (remboursement, suppression) lit
   `config["configurable"]["env"] == "test"` et ne fait rien en test.
   ⚠️ Le tutoriel passe `config={"env": "test"}` alors que le nœud lit `configurable` : tel qu'il est
   écrit dans la doc, le neutraliseur ne se déclenche pas.
6. **Lire l'état directement.** Si l'état du graphe contient les étapes, on peut les vérifier sans
   parcourir la trace.

### Moteur 4 — Agents complexes

La doc définit **trois niveaux** (`evaluation-approaches`) :

| Niveau | Question posée | Forces et faiblesses |
|---|---|---|
| Réponse finale | Le résultat est-il bon ? | simple ; mais lent et ne dit pas *où* l'agent a échoué |
| Une étape | Le bon outil a-t-il été choisi à ce moment ? | rapide et localise l'erreur ; mais dataset difficile à écrire pour les étapes tardives |
| Trajectoire | Le chemin complet est-il acceptable ? | couvre tout ; mais il existe plusieurs bons chemins |

**agentevals** (`trajectory-evals` et dépôt GitHub) :
- `create_trajectory_match_evaluator(trajectory_match_mode=...)` compare les outils appelés :
  - `strict` : mêmes outils, même ordre ;
  - `unordered` : mêmes outils, dans n'importe quel ordre ;
  - `subset` : aucun outil en dehors de la référence ;
  - `superset` : au moins les outils de la référence, des outils en plus sont permis.
- `tool_args_match_mode` choisit comment comparer les arguments : `exact` (par défaut), `ignore`,
  `subset` ou `superset`.
- `tool_args_match_overrides` règle la comparaison **outil par outil** : `exact`, `ignore`, une liste
  de champs à vérifier, ou ta propre fonction de comparaison.
- `create_trajectory_llm_as_judge(prompt=TRAJECTORY_ACCURACY_PROMPT[_WITH_REFERENCE])` fait juger la
  trajectoire par un LLM. Options : `continuous`, `choices`, `few_shot_examples`.

**En production** : les modèles de trajectoire prêts à l'emploi (sélection d'outils, respect du plan,
justesse de la trajectoire) s'appliquent aux conversations complètes.

**Backtesting** (`run-backtests-new-agent`)
1. `convert_runs_to_test(prod_runs, dataset_name=..., load_child_runs=True)` transforme de vraies
   traces en dataset et en expérience de référence.
2. On lance la nouvelle version sur ce dataset, puis on compare avec des évaluateurs sans référence.
3. Résultat documenté : le nouveau modèle formatait mieux mais **s'ancrait moins** dans les données.

**Agents à environnement (code, terminal)** (`harbor-integrations`)
- Harbor exécute chaque tâche dans un conteneur ou un bac à sable LangSmith.
- Un vérificateur `tests/test.sh` écrit une récompense. Le plugin `--plugin langsmith` enregistre
  chaque lot de tâches comme une expérience, avec le coût par essai.

**Où se place ton moteur.** Ton `eval-kit` va **plus loin** qu'agentevals sur deux points.
1. **Le graphe des transitions autorisées.** agentevals ne connaît que 4 modes de correspondance.
   Toi, tu acceptes *tous* les chemins valides, sans écrire une référence par scénario.
2. **La provenance des arguments.** agentevals compare les arguments à une référence. Toi, tu
   vérifies qu'ils **viennent d'un résultat précédent**, ce qui détecte une hallucination même quand
   il n'y a pas de référence.

Les deux se combinent : agentevals pour « les bons outils », ton moteur pour « le chemin est permis et
rien n'est inventé ».

### Moteur 5 — Production (surveillance continue)

- **Évaluateurs en ligne** : code, juge, composite ou conversation, avec un filtre, un taux
  d'échantillonnage et un rattrapage sur le passé (`online-evaluations-*`). Pas de référence, donc
  seulement des évaluateurs sans référence.
- **Règles d'automatisation** (`rules`) : un filtre et un échantillonnage déclenchent une ou plusieurs
  actions.
  - Actions possibles : ajouter à une file d'annotation, ajouter au dataset, appeler un webhook,
    lancer un évaluateur, prolonger la conservation des traces.
  - Exemple typique : chaque 👎 d'un client part en file d'annotation, puis entre dans le dataset.
- **Retours des utilisateurs finaux** :
  - `client.create_feedback(...)`, où `session_id` est **désormais obligatoire** ;
  - ou des jetons présignés, que ton front-end peut utiliser **sans clé d'API**
    (`presigned-feedback-tokens`).
- **Alertes** (`alerts`) : sur le volume, le coût, les erreurs, un **score de feedback moyen** ou la
  latence. Fenêtre de 5 ou 15 minutes ; envoi vers Slack, PagerDuty ou un webhook.
- **Tableaux de bord** (`dashboards`) : graphiques sur n'importe quelle clé de feedback, ventilés par
  métadonnée (par exemple `prompt_version`).
- **Engine** (`engine*`), un produit récent :
  - Il analyse les traces, regroupe les échecs en **problèmes** (gravité, diagnostic, traces
    témoins, correctif proposé), crée des exemples de dataset avec assertions, et peut ouvrir une PR.
  - Il facture à l'usage, en LSU.
  - Deux modules en bêta privée, sur Cloud seulement : la vérification qu'un correctif marche en
    rejouant les traces, et le *red teaming* (10 classes d'attaques).
- **Insights** (`insights`, offre Plus) : regroupement automatique de 1 000 traces par thème, pour
  savoir *ce que font vraiment* les utilisateurs.

### Moteur 6 — Humain et calibration des juges

- **Files d'annotation** (`annotation-queues`) :
  - une grille de notation (continue ou par catégories) ;
  - plusieurs relecteurs par trace, qui ne voient pas les notes des autres ;
  - des files **par paires** (A meilleur, B meilleur, égalité) pour comparer deux versions à l'œil.
- **Boucle de calibration** :
  1. Des humains étiquettent 20 exemples ou plus.
  2. « Align Evaluator » mesure le % d'accord avec le juge.
  3. On corrige le prompt du juge.
  4. Les corrections deviennent des exemples few-shot pour le juge.
- **Assertions** : un relecteur écrit en français « doit citer le numéro de commande ». Un évaluateur
  vérifie ensuite chaque assertion. Engine peut aussi les proposer à partir des problèmes détectés.

### Moteur 7 — Barrière de régression en CI

- **pytest** (`pytest`) :
  - `@pytest.mark.langsmith`, avec `t.log_inputs`, `t.log_outputs`, `t.log_feedback` et `expect(...)`
    (distance d'édition, distance d'embedding) ;
  - chaque fichier de test devient un dataset, chaque lancement une expérience ;
  - `LANGSMITH_TEST_CACHE` met en cache les appels au LLM pour une CI gratuite et déterministe.
- **Versions du dataset** : `update_dataset_tag(tag="prod")`, puis `list_examples(as_of="prod")` pour
  évaluer toujours sur la même version gelée (`manage-datasets`).
- **Comparaison dans l'interface** : on définit une expérience de référence (*baseline*) ; les
  régressions apparaissent en rouge, les améliorations en vert (`compare-experiment-results`).
- **Ce que la doc propose comme barrière** : une moyenne sous 0,85 → `sys.exit(1)`
  (`read-local-experiment-results`). C'est tout. La page `cicd-pipeline-example` parle de déploiement,
  pas de seuils.

### Bonus — Multimodal

- Les exemples peuvent porter des **pièces jointes** : image, PDF, audio, 20 Mo au plus.
- La cible doit avoir la signature `target(inputs, attachments)`. Chaque pièce jointe arrive avec une
  URL présignée et un lecteur binaire.
- Le juge doit accepter la modalité **et** la sortie structurée. Pour l'audio, la doc ne cite que
  Gemini (`evaluate-with-attachments`).

---

## 4. Ce que LangSmith ne fait pas : là où ta bibliothèque a de la valeur

| Manque dans la doc | Ce que tu as déjà, ou pourrais ajouter |
|---|---|
| Aucun intervalle de confiance ni test de significativité : la doc affiche des moyennes, des écarts-types et du rouge/vert par exemple | ✅ `stats.py` : bootstrap apparié ; ta barrière bloque seulement si l'intervalle est entièrement sous 0 |
| La barrière de la doc est un simple seuil sur une moyenne | ✅ `gate.py` : règles de sécurité sans tolérance + régression significative + échec si un score manque |
| Trajectoire = correspondance à une référence (4 modes) | ✅ graphe des transitions autorisées : plusieurs chemins valides sans référence |
| Arguments comparés à une référence | ✅ provenance : un argument doit venir d'un résultat précédent |
| Portes de sécurité métier (éligibilité avant remboursement) | ✅ `gates.py` |
| Accord juge/humain réduit à un % | ➕ kappa de Cohen + intervalle de confiance bootstrap sur l'accord |
| Rien sur la taille de dataset nécessaire | ➕ calcul du nombre de scénarios pour détecter une baisse de X points |
| Pas d'évaluateur documenté pour les interruptions | ➕ adaptateur `extract_langgraph_trajectory_from_thread` vers ton `Step` |

---

## 5. Recommandation

Ton moteur fait aujourd'hui une chose : **juger une trajectoire d'agent LangChain**. La doc montre
que la même architecture (adaptateur → étapes → vérifications → score) couvre les autres cas. Je
propose de l'étendre dans cet ordre, une brique à la fois :

1. **Adaptateur LangGraph** : une seule fonction, de `extract_langgraph_trajectory_from_thread`
   vers une liste de `Step`. Ton graphe des transitions s'applique alors aux **nœuds** d'un graphe
   avec interruptions. Effort faible, gain direct.
2. **Module RAG** : les 4 familles, avec les documents lus dans `run.child_runs`. L'ancrage réutilise
   ton idée de provenance : chaque affirmation doit venir d'un document.
3. **Module conversation** : simulation de personas avec openevals et un évaluateur de mémoire. Ton
   agent de remboursement a déjà un `thread_id` et de la mémoire.
4. **Calibration** : kappa et intervalle de confiance entre ton juge `message_conforme` et 20
   étiquettes humaines.
5. **Production** : brancher les évaluateurs sans référence (`trajectoire_dag`,
   `aucun_ticket_non_demande`) **en ligne** sur l'agent déployé sur Cloudflare, avec une alerte
   Slack si le score moyen baisse.

## 6. À corriger dans ton code avant le 31 janvier 2027

- **`client.list_runs()` est obsolète.** La CI l'a d'ailleurs signalé dans `evaluation/comparer.py`.
  - Il faut le remplacer par `await client.runs.query(project_ids=[...], min_start_time=..., selects=[...])`.
  - Attention : sans `min_start_time`, la requête ne couvre que **les dernières 24 heures**.
  - Les noms de champs changent aussi (`project_name` disparaît au profit de `project_ids`) et la
    pagination se fait par curseur. Version minimale : `langsmith>=0.10.15`.
- **`create_feedback`** exige maintenant `session_id`.
- Les méthodes `*_feedback_formula` sont supprimées ; les remplacer par des évaluateurs composites.
