# Vérification du Lot 2 — 12 octobre 2026

> Archive historique : résultats et limites au moment de cette vérification.
> Pour l’état actuel, consulter la [roadmap](../market-data-roadmap.md#9-travail-restant-et-méthode)
> et les documents d’API qui y sont liés.

Les deux correctifs ci-dessous valident le code du [Lot
2](../market-data-roadmap.md#lot-2--p1-références-macro-et-fondamentaux). La couverture
et les horaires de livraison demandent une qualification sur données réelles.

## Périmètre et preuves

| Critère | Implémentation | Vérification |
|---|---|---|
| Références de marché | `data/reference_assets.py`, `features/reference.py` | Identités/unités/devise, fenêtres par période, returns, beta/corr et force relative sur clôtures exactement appariées ; sens FX et coefficients énergie explicites |
| FRED/ALFRED | `data/fred.py`, `data/alfred.py`, `download_fred.py` | Observations/vintages/releases, pagination bornée, révisions, cache/reprise sans réseau, unités/fréquences/bases saisonnières compatibles |
| EIA/Cboe/Pink Sheet | Trois adaptateurs et `download_context.py` | Fenêtres/facets/units exactes, CSV/XLSX, versions de fichiers, limites, reprise et expurgation de clés |
| SEC | `data/sec.py`, `download_sec.py` | Client identifié, accès borné, submissions récentes/historiques, companyfacts, accessions, acceptation distincte de publication, merge immuable |
| Fondamentaux | `features/fundamentals.py` | Croissance CA/EPS, marges, levier, cash-flow/CapEx, Q4 annuel−YTD, EPS non additif, bases et périodes séparées, qualité/âges/masques |
| Intégration | API publique, catalogue, extras, lockfile, CI, exemple et docs | Imports, sélection, Parquet, exemple offline et distribution |

Les tests vérifient également les valeurs avant/à disponibilité, les révisions
futures sans modification du passé, les anciennes périodes révisées, les batches
simultanés et l'absence de remplissage par zéro.

## Deux défauts trouvés et corrigés

1. **Ancienneté du benchmark apparié.** Lorsqu'un benchmark avait un cours plus
   récent que la dernière observation de la cible, `max_age` utilisait la réception
   de ce cours non apparié. Une ancienne observation benchmark pouvait ainsi
   produire un beta/une force relative malgré son expiration. Le contrôle utilise
   maintenant la réception du benchmark à la clôture effectivement appariée,
   ainsi que celle de la cible. Les horizons et les valeurs historiques sont
   conservés. Regression : `test_benchmark_expiry_uses_the_matched_endpoint_not_a_newer_unmatched_price`.
2. **Conventions FRED lors d'un append.** Une même révision numérique reçue de
   nouveau avec une fréquence ou une base saisonnière différente était dédupliquée
   sans signaler le conflit. L'append vérifie désormais ces conventions et refuse
   l'opération avant de modifier le master existant. La première disponibilité
   d'une révision identique reste conservée. Regression paramétrée :
   `test_repeated_value_revision_cannot_silently_hide_changed_conventions`.

Les deux comportements ont été comparés au code original sauvegardé localement ;
les corrections sont couvertes par trois nouveaux cas de test.

## Validation

- **767 tests passent** sur la suite complète, avec les trois nouveaux cas.
- Ruff, format et contrôle des espaces dans le diff passent.
- Exemple `examples/context_features.py` : offline, calculs manuels et invariance
  aux révisions futures ; aucune collecte réelle.
- Wheel et sdist reconstruits ; les modules du Lot 2 sont présents et correspondent
  au source courant. Les imports du wheel sont contrôlés sans extras fournisseur,
  calendrier, Excel ou plots.
- Dépendances du lockfile inchangées. Un venv temporaire avec ce même lockfile a
  permis les contrôles complémentaires malgré les caches iCloud du venv local.

Commandes usuelles :

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync python examples/context_features.py
uv build
```

## Limites conservées

- Profondeur réelle, droits par dataset, quotas de compte, ajustements, heures de
  publication et latence doivent être qualifiés par captures réelles bornées.
- Une reconstruction ALFRED à la journée ou SEC depuis acceptation ne certifie
  pas un historique PIT intraday. `point_in_time=False` reste explicite.
- Le bridge macro refuse une révision numérique→null : elle est conservée dans
  les observations mais le moteur ne sait pas encore retirer un point. Ce refus
  empêche d'utiliser silencieusement la valeur retirée.
- Company Facts ne constitue pas une couverture exhaustive des concepts custom,
  segments ou taxonomies étrangères. Les périodes fiscales atypiques et mappings complémentaires demandent qualification ; une dette totale exige
  des composants justifiés.
- Guidance/LLM, graphes et événements restent au Lot 3. Consensus historiques
  et autres sources premium restent dans les lots et inventaires correspondants.

[Conventions, API, collecteurs et commandes détaillées](../context-features.md).
