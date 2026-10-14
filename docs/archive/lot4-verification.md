# Vérification Lot 4 — 12 octobre 2026

> Archive historique : résultats et limites au moment de cette vérification.
> Pour l’état actuel, consulter la [roadmap](../market-data-roadmap.md#9-travail-restant-et-méthode)
> et les documents d’API qui y sont liés.

Implémentation et interfaces : [international-features.md](../international-features.md).

| Vérification | Résultat |
|---|---|
| Suite complète | **917 tests passés**, 45,93 s |
| Tests nouveaux Lot 4 | **58** : sources, features, CLI |
| Ruff lint et format | OK, 158 fichiers |
| `git diff --check` | OK |
| Exemple international offline | OK ; absence de valeur avant réception, niveau/change après réception |
| CLI `--help` et exports offline | OK ; Parquet, manifestes, partial, immutabilité, caps |
| Wheel et sdist | Build offline réussi depuis les dépendances déjà en cache |
| Import depuis le wheel | API Lot 4 accessible, imports facultatifs bloqués, checkout editable exclu du contrôle |

## Cas vérifiés

- Les onze adaptateurs/captures acceptent les formes synthétiques attendues et
  rejouent les mêmes sorties/réceptions ; preuves et sept champs communs conservés.
- TSMC : millions TWD, pourcentages, mois futurs vides et total exclus ; publication
  fournie distincte de la période et disponibilité toujours conservatrice.
- MOPS : chinois JSON/CSV, sélection d'émetteur, ROC, milliers TWD ; date de table
  conservée sans créer une heure de publication.
- TWSE : action/mois exacts, ROC, OHLC cohérent, volume en actions, calendrier
  explicite et absence de calendrier signalée par clôture inconnue.
- OpenDART : état/basis/compte/devise/amount exacts, YTD séparé, intervalle fiscal
  et fin de terme contrôlés ; absence de compte ne devient pas zéro.
- Eurostat : dimensions singleton et positions de cube, cellule sparse manquante
  null, status conservé ; autre combinaison dimensionnelle refusée.
- BoJ/e-Stat : code/table/unité/fréquence et carte temporelle explicites ; réponses
  avec continuation restent partielles.
- NBS : sélection précise de tableau, unité et libellés de mois ; pas de date
  artificielle de publication ni extraction libre de toute page.
- USGS : CSV unique, pays/année/unité, suffixe estimé préservé, sélection fichier
  CSV via `name` ; ZIP refusé.
- GDELT : cap explicite, détection séparée de publication, codes CAMEO avec zéros
  initiaux, sortie fournisseur marquée modèle ; TSV importé partiel au flux monde.
- Features : gate sur décisions aware, absence/fraîcheur/null et fréquence/unités,
  exact mois précédent/année précédente, invariance passée aux révisions tardives.
- Une arête disponible demain n'autorise pas une feature société aujourd'hui ;
  preuve de lien conservée. Aucun lien créé depuis les listes de secteurs.
- Recapture GDELT : pas de double compte ni réinitialisation de première
  disponibilité ; masque de source disponible et couverture du corpus explicites.
- GET injectés sur tous les endpoints réseau : une requête, cap transmis ; clés
  redacted/exclues du raw, reprise et replay sans clé après échec d'écriture.
- Hôtes/URLs et config bornés, cap source/archive, sorties immuables et caps CLI.

## Limites de cette vérification

Les contrats de parsing suivent les documentations officielles. Les tests utilisent des
réponses synthétiques et un transport injecté ; le build reprend les dépendances en
cache, offline. Cette vérification se limite au code, sans collecte live ni
téléchargement de modèle.

Pour un usage réel, vérifier accès, quotas, droits, codes de séries, archives, horaires
de publication, calendriers et précision du LLM. La documentation Lot 4 décrit les
imports prix/FX et les adaptateurs disponibles ; J-Quants/KRX natifs et un flux news
exhaustif restent hors périmètre.

Avant constitution d'un dataset réel : choisir pays/entreprises/séries/période,
tester une petite capture par source, établir identités/unités/calendriers/droits,
archiver les vintages et conserver les valeurs/relations manquantes. Pour les
textes, appliquer le protocole de qualification/revue Lot 3. Lot 5 reste à réaliser.
