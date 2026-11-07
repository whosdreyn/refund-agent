# Ma méthode pour évaluer un agent

À appliquer à chaque nouvel agent. Construite sur l'agent de remboursement (`refund-agent`)
et l'agent du marché (`langsmith-reliable-agents`).

**Le principe** : un dataset de scénarios, une fonction cible qui fait tourner l'agent sur chaque
entrée et renvoie sa sortie réelle, des evaluators qui comparent cette sortie à la sortie attendue
(ou vérifient une règle), et une experiment par version de l'agent.

---

## 1. Tracer dès le premier jour
**Règle** : J'active le tracing avant même d'évaluer, pour voir chaque étape de l'agent
(outils, arguments, tokens, coût). Avec LangChain, les variables d'environnement suffisent ;
avec du code maison, il faut un wrapper et `@traceable`.
**Exemple vécu** : Dans la trace du casque, j'ai vu la boucle de l'agent :
4 outils + 1 réponse = 5 appels au LLM.

## 2. Écrire la politique métier
**Règle** : J'écris les règles que l'agent doit respecter, puis j'évalue s'il les respecte.
Sans règle écrite, impossible d'écrire la réponse attendue d'un test.
**Exemple vécu** : koffi (« rembourse-moi » sans article) : aucune règle ne couvrait ce cas,
j'ai dû choisir la règle B avant de pouvoir écrire le test, et l'agent l'ignorait.

## 3. Construire le dataset
**Règle** : Ce n'est pas la quantité qui compte, mais les cas qui testent la politique :
surtout des cas pièges qui pourraient la casser, et quelques cas normaux qui la respectent.
Chaque réponse attendue est écrite en champs vérifiables par du code (décision, montant, outils),
pas en phrase.
**Exemple vécu** : 10 scénarios (casque, hors délai, tablette jamais achetée, simple question…).
Dans DeepEval, une référence écrite en phrase a été mal lue par le juge (scénario 3).

## 4. Écrire les evaluators
- **La réponse** : ce que l'agent dit (décision, montant, message).
- **La trajectoire** : les étapes suivies pour y arriver (outils, nombre d'appels, arguments,
  actions interdites).
- **Code ou LLM juge** : du code pour tester des éventualités strictes, connues d'avance ;
  un LLM juge pour évaluer le sens. Toujours le code d'abord, le LLM seulement si nécessaire.
  Un juge se teste et se corrige comme le reste, et il justifie ses verdicts (`comment`).
- **Chaque clé lue par un evaluator doit être produite par la fonction cible.**

**Exemple vécu** : `sorted` et non `set` pour compter les tickets de lina ; le mot-clé
« rembourser » ratait « rembourse-moi » et les négations ; mon juge confondait « remboursable »
et « remboursé ».

## 5. Comparer les versions
**Règle** : À chaque modification du prompt, je rejoue TOUS les scénarios avec la nouvelle
version, pour vérifier qu'elle ne casse rien (régression). Je répète chaque scénario plusieurs
fois (`num_repetitions=3`), je compare exemple par exemple, je fais un seul changement par
version, et je garde les anciennes versions pour comparer et revenir en arrière.
**Exemple vécu** : la v2 corrigeait lina et la simple question, mais remboursait 114,99 € pour
une tablette jamais achetée. La v3 créait un ticket abusif 1 fois sur 3 pour la simple question :
invisible sur un seul passage.

## 6. Diagnostiquer un échec
**Règle** : Avant de corriger, je lis la trace et la justification du juge, puis je cherche
la cause parmi 4 :

| Cause | La question à se poser |
|---|---|
| L'agent | s'est-il vraiment trompé ? |
| L'evaluator | le juge ou le code note-t-il correctement ? |
| La mesure | le chiffre que je lis est-il complet et à jour ? |
| La politique ou le dataset | la règle et la réponse attendue sont-elles claires ? |

**Exemple vécu** : le juge rejetait « remboursable » (evaluator) ; le graphique montrait 100 %
pour la v2 sur 6 exemples sur 10 (mesure) ; koffi n'avait aucune règle (politique) ;
la tablette remboursée en v2 (agent).

## 7. Passer en production
**Règle** : En production, il n'y a pas de réponse attendue : j'utilise un evaluator sans
référence, qui vérifie une règle sur la sortie seule. Les traces qui échouent sont ajoutées
automatiquement au dataset, relues par un humain, puis servent à tester les versions suivantes.
Pas besoin de déploiement : LangSmith évalue toutes les traces envoyées dans le projet.
**Exemple vécu** : sur l'agent du marché, `coherence` (quantité × prix = total) a noté 12 traces,
détecté 2 échecs (les prix négociés), ajoutés au dataset en ~6 minutes.

---

## Mes pièges à ne plus refaire

**Pratiques**
- Lancer une commande depuis le mauvais dossier : je vérifie que le chemin du terminal finit
  par le nom du projet avant chaque `uv run`.
- Oublier d'enregistrer le fichier (Cmd + S) avant de relancer : le script lit la version
  du disque, pas celle de l'écran.
- Laisser des tests à la main dans `if __name__ == "__main__":` au lieu de l'évaluation.

**Techniques**
- Une clé lue par un evaluator mais absente de la sortie de la fonction cible (`KeyError: 'message'`).
- Les clés de la fonction cible qui ne correspondent pas à celles du dataset (`"message"` au lieu de `"input"`).
- Une consigne de juge mal formulée (« le message doit se terminer par “Réponds OUI ou NON” »),
  ou contradictoire (« uniquement OUI ou NON, puis une explication »).
- `max_tokens=3` alors que je demande une explication au juge.
- Utiliser des mots-clés pour comprendre une intention.

**Méthode**
- Valider un juge sur 2 cas inventés seulement : il faut le tester sur des réponses réelles.
- Croire un score à 100 % sans vérifier qu'il mesure vraiment quelque chose
  (la règle de sécurité était aveugle aux erreurs cohérentes).
- Conclure à partir d'un seul passage, ou d'un graphique sans vérifier sur combien d'exemples il porte.
- Ajouter une trace au dataset sans relire sa sortie : une référence fausse pollue toutes les mesures.
- Écrire une règle de prompt trop large (« toujours », « chaque ») : elle crée des régressions.

## Quand utiliser DeepEval
Pour ajouter vite une métrique que je n'ai pas (exactitude des arguments, complétion de la tâche).
Pas pour remplacer un evaluator sur mesure : sur mes règles métier, les miens sont plus justes et
moins chers. Toujours lire la `reason`, même pour un PASS.
