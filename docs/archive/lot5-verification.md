# Vérification Lot 5 — 14 octobre 2026

> Archive historique : résultats et limites au moment de cette vérification.
> Pour l’état actuel, consulter la [roadmap](../market-data-roadmap.md#9-travail-restant-et-méthode)
> et les documents d’API qui y sont liés.

Interfaces : [sources P3](../p3-sources.md), [datasets modèles](../model-datasets.md).

| Vérification | Résultat |
|---|---|
| Suite complète | **970 tests passés**, 48,27 s |
| Tests nouveaux Lot 5 | **53**, sources/datasets/splits/scaler/graphes/events/CLI |
| Ruff lint et format | OK |
| `git diff --check` | OK |
| Exemple offline | `(42, 4, 1)`, train 10 / validation 12 / test 8 / purgés 12 |
| Build | Wheel + sdist offline, dépendances de build déjà en cache |
| Import isolé du wheel | OK ; APIs publiques et catalogue, checkout editable exclu, SDK optionnels et frameworks DL bloqués |
| Artefacts de build | Wheel 387 079 octets, sdist 779 583 octets |

## Contrats et scénarios couverts

- COT : six rapports Legacy/Disaggregated/TFF futures-only/combined sélectionnés
  explicitement, code marché/période/colonnes, publication distincte du mardi.
  Fixture de publication décalée au lundi en heure New York. Gate réception,
  unités, ratios, source cap partiel, cache et requête bornée injectée.
- Consensus : fichiers structurés synthétiques et client injecté, sans licence
  réelle ni accès payant. Fiscal quarter/période, devise/échelle, base EPS,
  GAAP/non-GAAP et classe d'énoncé contrôlées. Guidance rejetée comme consensus.
- Import non qualifié : disponibilité historique inconnue. Import PIT
  qualifié : déclaration de package/référence et temps source par version exigés.
- Surprise : snapshot et disponibilité strictement avant annonce ; révision
  post-annonce exclue. Actual retiré ou annonce inconnue → masque. EPS basic et
  diluted ne se joignent pas. Dernier snapshot prime sur correction tardive d'un
  ancien snapshot. Détails analystes dédupliqués, retraits et couverture partielle.
- CSV : booléens/nulls et catégories mappés explicitement. Source/archives bornées,
  reprise après échec d'écriture sans client, captures chevauchantes sans doublon.
- Options : archives immuables et replay, capture de chaîne native bornée,
  snapshot économique séparé de réception et timestamp UTC vault normalisé.
  OI/quotes absents restent NaN ; échantillon plafonné reste identifié comme tel.
- Panel : disponibilité et période ≤ décision, preuve/révision par cellule,
  manquants préservés. La passerelle de features calculées exige ses horaires.
- Fenêtres : layout `[sample,time,feature]`, ordre stable, labels séparés,
  queue future masquée ; prix stale répété ne crée pas un nouveau label.
- Chunk/batch : mêmes valeurs, masques, labels et indices pour batches 1/3/17/100.
- Split : bornes chronologiques, purge des labels qui touchent la frontière et
  disponibilité des labels contrôlée. Aucune séparation aléatoire des lignes.
- Scaler : cellules train uniques, overlaps non surpondérés. Outlier futur en
  validation ne modifie ni mean ni scale train ; feature jamais observée en train
  reste masquée. Split vide/étranger ou indices train/validation permutés refusés.
- GNN : direction source→cible, snapshot seulement après revue, poids partiel
  0,2 sans renormalisation, nature/propriétaire et preuves d'arêtes conservés.
- Event Encoder : padding/magnitudes/embedding masqués, liens société connus,
  revue et première disponibilité ; modèle futur ou inférence tardive ne peuvent
  remplir les vecteurs passés. Export des identités et versions d'événements.
- Artefacts : replay identique, registry/provenance/axes et config hashés,
  inputs/labels/splits séparés, checksum corrompu rejeté, `allow_pickle=False`.
  Déclaration NPY d'une allocation énorme rejetée avant chargement. Caps axes,
  lignes Parquet et mémoire, CLI entièrement locale et captures P3 explicites.
- Les tests options déjà présents continuent de passer après extraction du
  normaliseur de payload et ajout de la route chaîne au transport borné.

## Limites opérationnelles

La vérification porte sur les fixtures locales et les dépendances existantes. Les
sources payantes utilisent les exports autorisés ou un client fourni par l'utilisateur ;
leurs endpoints et droits demandent une configuration dédiée. Datasets live, modèles et
frameworks DL restent hors de cette vérification, réalisée sans achat ni test réseau.

Une qualification reste nécessaire pour les sources et packages réellement choisis :
droits, codes, profondeur, univers/delistings, horloges, vintages, conventions
comptables/ajustements, couverture options/news et modèle textuel. L'export conserve les
garanties déclarées et les masques de qualité des entrées.

La bibliothèque fournit les inputs et labels pour l'entraînement. Stratégie, backtest
d'exécution et collecte à grande échelle sont à configurer dans le projet consommateur.