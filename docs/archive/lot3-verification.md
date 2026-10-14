# Vérification Lot 3

> Archive historique : résultats et limites au moment de cette vérification.
> Pour l’état actuel, consulter la [roadmap](../market-data-roadmap.md#9-travail-restant-et-méthode)
> et les documents d’API qui y sont liés.

Date : 12 octobre 2026. Environnement Python 3.12 existant réutilisé, caches
existants, sans installation de modèle, lecteur PDF ou données de marché.

| Vérification | Résultat |
|---|---|
| Suite complète `pytest -q` | **859 passed**, 44,61 s |
| Tests nouveaux Lot 3 | **92** : documents, extraction, graphes/events, Ollama, CLI |
| `ruff check .` | OK |
| `ruff format --check .` | 147 fichiers conformes |
| `git diff --check` | OK |
| `examples/document_features.py` | OK, sans réseau/modèle |
| Build `uv build --offline` | Wheel + sdist OK, 337 309 + 699 295 octets |
| Import indépendant du wheel | API/export/catalogue OK ; aucune dépendance facultative importée |
| Contenu des distributions | Modules Lot 3 dans wheel ; corpus et CLI dans sdist |

Couverture vérifiée :

- Corps/texte/hash, premières réceptions et versions immuables, droits/métadonnées,
  imports TXT/HTML et chemin PDF avec lecteur injecté, collecte URL/SEC bornée.
- A–D complets, enveloppes JSON, plages/null, rôles, classes financières, statuts,
  dates ISO/naturelles explicites, marqueurs de devises/%, perte entre parenthèses.
- Source untrusted dans un message séparé, aucune exécution d'instructions source,
  citations/document/entités/nombres invalides, JSON dupliqué/non fini, rejets archivés.
- Cache document/modèle/configuration/prompts/chunk, reprise sans réinférence,
  plafond de réponse/requêtes/disque, sorties incomplètes distinctes d'absence de fait.
- Revue nommée, invariance au futur, alias tardifs/ambigus, flux fournisseur→client,
  fin documentée/retrait de preuve, scopes/périodes/poids contradictoires, historiques.
- Séparation des horloges de source et de revue : revue tardive d'un ancien
  document ne remplace pas une terminaison/un statement source plus récent.
- Référentiels industrie/facteur/pays/matière, propagation explicite de facteurs,
  joins voisins à unités/périodes compatibles, âges/masques/qualification.
- Events distincts dans un article, doublons, identités explicites entre documents,
  comptes/recency/statuts et magnitudes séparées ; outputs modèles compatibles et disponibles.
- CLI complète import→JSON→revue→Parquet, budgets et index sans écrasement de versions.

Le corpus annoté **synthétique** contient 18 cas et 11 faits attendus. Le replay
avec les réponses JSON annotées obtient précision/rappel 1,0 ; il teste validation
et calcul des métriques. Il ne mesure pas les performances d'un LLM réel.

La vérification utilise des réponses injectées pour le transport, digest, contexte,
GGUF/localité et sorties Ollama. Collecte et inférence réelles restent à tester. `pypdf`
était absent du runtime ; la branche PDF utilise un lecteur injecté. Qualification de
droits/publications/identités et évaluation d'un modèle figé sur corpus réel restent
nécessaires avant une utilisation DL de ces données.

API, limites et commandes : [document-features.md](../document-features.md).
