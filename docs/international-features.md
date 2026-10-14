# Sources internationales

Les adaptateurs collectent des sources ciblées, conservent leurs versions et alignent
les features sur les décisions horodatées. La collecte se lance séparément de l'import
Python. Les tests utilisent des sources synthétiques ; couverture, droits et accès
fournisseur doivent être vérifiés sur données réelles.

## Couverture

| Source | Adaptateur | Sélection et limites |
|---|---|---|
| TSMC IR | `parse_tsmc_revenue` | Page anglaise d'une année, CA consolidé mensuel en millions TWD et YoY en points de pourcentage ; totaux et mois futurs vides exclus |
| MOPS/TWSE open data | `parse_mops_revenue` | `t187ap05_L` JSON ou CSV, codes sociétés explicites ; CA mensuel en milliers TWD, YoY ; snapshot courant à archiver |
| TWSE | `parse_twse_daily` | Une action et un mois, OHLCV, volume en actions, turnover en TWD ; dates ROC ; ajustement inconnu tant que non qualifié |
| OpenDART | `parse_opendart_accounts` | Comptes majeurs anglais d'une société et d'un rapport ; comptes, états CFS/OFS, champs courants/cumulés et intervalles fiscaux explicites |
| Eurostat | `parse_eurostat` | JSON-stat, un code pour chaque dimension hors temps ; période bornée, cellules nulles et flags conservés ; dernière version |
| BoJ | `parse_boj` | API 2026 `getDataCode`, un code exact, unité/fréquence et bornes ; Tankan, taux, prix ou FX selon série |
| e-Stat | `parse_estat` | Une table et combinaison dimensionnelle, codes temps associés explicitement aux périodes ; inflation/industrie selon table |
| Chine NBS | `parse_nbs_release` | Un communiqué HTML officiel, table/colonnes/unités et libellés de périodes explicites ; PMI et industrie mensuelle/cumulée restent distincts |
| USGS | `parse_usgs_csv` | Un CSV MCS sélectionné, pays/matière/colonnes annuelles/unité issus des métadonnées ; flags estimé/révisé préservés |
| GDELT DOC | `parse_gdelt_articles` | Recherche ciblée 15 min–7 jours, 1–250 références d'articles ; titre/langue/pays du média et détection, sans texte intégral |
| GDELT Events 2.0 | `parse_gdelt_events` | Import d'un petit TSV local à 61 colonnes, pays/CAMEO sélectionnés ; aucune collecte ZIP du flux mondial |

`collect_international(source, config, store, snapshot_id=...)` encapsule ces parsers.
`InternationalCollection` contient `observations`, `complete`, `issues`, `metadata` et
l'identité de capture. Toutes les fonctions sont exportées par `pfe`.
`default_source_registry()` contient les nouvelles déclarations. Vérifier les droits de
redistribution et d'entraînement propres à chaque source.

OpenDART exige `OPENDART_API_KEY` ; e-Stat exige `ESTAT_APP_ID`. Les autres
adaptateurs n'exigent pas de clé ici. Clés séparées de la configuration, absentes
des identités de captures et réponses archivées ; replay sans clé.

Les prix Corée/Japon/Europe et les paires FX utilisent aussi les APIs existantes LSE,
`load_reference_assets`, `reference_assets_from_daily` et
`reference_assets_from_observations`. J-Quants/KRX passent par un import CSV/Parquet
canonique, avec symbole, devise, calendrier et clôture explicites. La qualité et les
droits des données doivent être qualifiés auprès du fournisseur.

## Taille, cache et captures successives

Par défaut : 512 Kio par réponse, 5 000 lignes/cellules/observations, 4 Mio cumulés pour
raw + Parquet normalisé + résumé, hors petits manifestes du `RawStore`. Une invocation
effectue au plus un GET, sans retries ni redirections automatiques. Le transport limite
aussi les octets pendant la lecture. Un fichier dépassant la limite est refusé.

Le snapshot MOPS courant est fourni en une réponse de marché ; seuls les codes
demandés sont normalisés. Le cap s'applique aussi à cette réponse et à ses lignes.
Si elle dépasse le budget, fournir un petit export autorisé local ou augmenter
explicitement le budget après vérification de l'espace disque.

Les pages raw, reçus et sorties sont immuables. `snapshot_id` distingue les
captures prospectives ; rejouer la même capture préserve sa première réception.
Les pages déjà reçues permettent la reprise sans réseau après échec du parsing
ou de l'export. `merge_international_observations` cumule des captures, garde la
première disponibilité d'une même révision et refuse les valeurs contradictoires.
Les valeurs numériques et les données structurées GDELT s'exportent séparément.

`complete=True` signifie uniquement que la réponse sélectionnée n'indique pas de
continuation. BoJ `NEXTPOSITION`, e-Stat `NEXT_KEY` ou un plafond GDELT atteint
produisent une sortie partielle. Réduire la période ou les dimensions pour une nouvelle
capture : chaque invocation traite une seule réponse. Un TSV GDELT sélectionné reste
explicitement partiel par rapport au flux mondial.

## Chronologie et conventions

Chaque observation conserve `entity_id`, `observation_date`, `published_at`,
`available_at`, `value`, `source_id`, `revision_id`, métrique/unité/version, début
et fin de période, réception, pays et preuve/identité de ligne source.

Disponibilité par défaut : `max(received_at, published_at)` si la publication
exacte est fournie, sinon réception. Les dates de mise à jour Eurostat/BoJ/e-Stat,
`出表日期` MOPS et le numéro de dépôt DART ne deviennent pas des heures de
publication. Une publication connue uniquement au jour reste `publication_date`,
avec `published_at=null`. `point_in_time=False` signale l'historique non qualifié.

Une table importée aujourd'hui devient disponible à sa réception. Pour un historique des
sources latest-only, utiliser des captures successives ou des archives de vintages et
publications qualifiées.

- ROC : `115/09/01` → `2026-09-01`, mois MOPS `11509` → `2026-09`.
- TWSE : `close_schedule` facultatif associe chaque date à sa clôture aware ;
  journée locale `Asia/Taipei` contrôlée. Sans calendrier, l'heure de clôture reste inconnue. Les dates TWSE
  suivent les séances locales, sans interpolation des jours fériés.
- TSMC millions TWD et MOPS milliers TWD gardent leurs unités distinctes.
  Conversion FX et rapprochement des sources sont à configurer séparément.
- OpenDART : quarter/YTD/annual/instant séparés, devise vérifiée à la réponse.
  Les dates de terme structurées disponibles doivent correspondre à la fin
  déclarée ; début cumulé/fiscal fourni par l'utilisateur reste signalé déclaré.
  Ni numéro de rapport ni date de dépôt ne déterminent un trimestre calendaire.
- BoJ : séries fiscales `ANNUAL(MAR)` et hebdomadaires ne sont pas assimilées aux
  séries annuelles calendaires/daily ; le parser les refuse dans ce périmètre.
- e-Stat : tous les codes dimensionnels sont sélectionnés, les codes de temps
  ont une carte explicite. Cette carte fournit les périodes utilisées par le parser.
- NBS : chiffres mensuels, cumul janvier–mois, PMI et variations YoY exigent des
  métriques distinctes et unités adaptées au tableau choisi.
- USGS : période de production et publication du fichier distinctes. Un CSV
  publié après le PDF ne reçoit pas automatiquement la date du PDF.
- GDELT : `seendate`/`DATEADDED` = détection, pas première publication prouvée.
  CAMEO/FIPS restent leurs propres systèmes ; pays du média ≠ pays affecté.

Les parsers TXT/HTML/CSV/JSON utilisent la bibliothèque standard. UTF-8 par défaut ;
encodage déclaré possible pour NBS/MOPS/USGS. Scans, traduction et tableaux PDF
demandent un prétraitement séparé.

## Features et liens aux sociétés

```python
import pfe

features = pfe.compute_international_features(
    decisions,
    observations,
    series=[
        pfe.InternationalSeries(
            "tsmc_revenue",
            "company:tsmc",
            "revenue",
            "million_TWD",
            "tsmc",
            "monthly",
            max_age="45D",
            max_observation_age="90D",
        )
    ],
)
```

Chaque série exige entité/métrique/unité/source/fréquence exactes. Sorties :
`level`, `change`, `pct_change`, `yoy`, masques de disponibilité/valeur/fraîcheur,
qualité de fréquence/PIT, âges de disponibilité/publication/période et provenance.
Révision d'un mois ancien ne remplace pas le mois économique plus récent. Une
valeur actuelle nulle reste manquante. Changes et YoY demandent les périodes
exactes, pas simplement la ligne précédente ; périodes fiscales exposent le
niveau uniquement. Daily compare au jour civil précédent ; pour rendements de
séances étrangères, réutiliser `compute_reference_features` et ses horizons.

`compute_linked_international_features(..., graph_history, entity_ids=[...])` attache
ces facteurs aux sociétés seulement après connaissance d'une arête documentée, active et
du type sélectionné. `mask.link` distingue absence de lien et absence de valeur. La
sortie conserve la preuve de l'arête et les poids documentés.

Relations/expositions : utiliser les rapports locaux dans le pipeline Lot 3, revue des
preuves, `EntityRegistry`, `build_graph_history`. Les facteurs/pays/ matières déclarés
utilisent `merge_graph_declarations`. Le moteur existant `compute_graph_features`
propage des features de voisins compatibles avec des poids connus, conservés sans
renormaliser le réseau partiel.

FX TWD/KRW/JPY : sélectionner DEXTAUS/DEXKOUS/DEXJPUS via FRED/ALFRED ou des
prix autorisés. Garder devise étrangère par USD, fréquence et disponibilité H.10,
puis joindre avec les APIs références du Lot 2. Une série FRED n'est pas une
clôture de marché FX tradable. [Conventions et commandes](context-features.md).

`SECTOR_CONTEXTS` reprend les sept secteurs demandés comme liste de candidats, sans
preuve d'exposition société. `compute_gdelt_features` calcule des counts 7D/30D et
récence sur le corpus fourni ; captures répétées ne réinitialisent pas l'âge.
`quality.complete_feed=0`, `quality.verified_events=0`. Le tone GDELT est la
classification fournie par GDELT ; son usage comme signal demande une évaluation
séparée. Pour un événement financier vérifié, importer le texte autorisé de l'article
dans le Lot 3, extraire D, revoir et utiliser les features events existantes.

## CLI

Configuration TSMC `tsmc.json` :

```json
{"year": 2026, "entity_id": "company:tsmc"}
```

Capture locale, sans réseau :

```sh
python scripts/download_international.py --source tsmc --config tsmc.json \
  --snapshot selected-tsmc.html --received-at 2026-10-12T12:00:00Z \
  --output data/fundamentals/tsmc_selected.parquet
```

Capture réseau explicitement demandée :

```sh
python scripts/download_international.py --source tsmc --config tsmc.json \
  --run --snapshot-id tsmc-2026-capture-001 \
  --output data/fundamentals/tsmc_capture_001.parquet
```

`--features-config` reçoit `{decisions: [...], series: [...]}` et exige
`--features-output ...parquet` ; grille bornée à 512 décisions × 32 séries.
Les sorties sont immuables. Résultat partiel : `*.partial.parquet` avec son
manifeste. Raw : `data/raw/international/<collection_id>/`.

Autres configurations, noms exacts :

- MOPS : `companies={code: entity_id}` ; `publication_times` facultatif par
  `code:YYYY-MM`, `encoding` facultatif pour CSV.
- TWSE : `stock_no`, `month=YYYY-MM`, `entity_id`, `close_schedule` facultatif.
- DART : `corp_code`, `year`, `report_code`, `entity_id`, `accounts` ; chaque compte
  a `account_nm/fs_div/sj_div/amount_field/metric/period_start/period_end/period_kind/currency`.
- Eurostat : `dataset`, `dimensions` contenant toutes les dimensions hors temps,
  `start/end`, `entity_id/metric` ; `freq/unit/geo` obligatoires.
- BoJ : `db/code/start/end/entity_id/metric/unit/frequency` ; bornes conformes au
  manuel, `YYYYQQ` pour trimestre, `YYYYMM` pour monthly/daily.
- e-Stat : `table_id/dimensions/start_code/end_code/period_map/entity_id/metric/unit/frequency`.
- NBS : `url/entity_id/table_index/columns/period_map/header_evidence` ; chaque
  métrique pointe vers `{index, unit}` ; publication/encodage facultatifs.
- USGS : `url/material_id/countries/country_column/period_columns/unit` ;
  `file_name` facultatif sélectionne un seul CSV ScienceBase via `name`,
  `header_row`, `value_suffixes`, publication/encodage facultatifs.
- GDELT : `query/start/end/entity_id/max_records` ; dates aware.
- GDELT Events : `entity_id/country_codes/root_codes` ; `--snapshot` TSV requis.

Les exemples de formes fournisseur sont dans `tests/test_international.py` ; les
valeurs sont synthétiques. [Exemple exécutable](../examples/international_features.py).

## Références vérifiées et limites opérationnelles

Contrats de parsing : [TSMC IR](https://investor.tsmc.com/english/monthly-revenue/2026),
[TWSE Swagger](https://openapi.twse.com.tw/),
[DART comptes majeurs](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DE003&apiId=AE00033),
[Eurostat JSON-stat](https://ec.europa.eu/eurostat/fr/web/user-guides/data-browser/api-data-access/api-getting-started/api),
[BoJ API](https://www.stat-search.boj.or.jp/info/api_manual_en.pdf),
[e-Stat API](https://www.e-stat.go.jp/api/api-info/e-stat-manual),
[NBS PMI](https://www.stats.gov.cn/english/PressRelease/202607/t20260701_1964047.html),
[USGS MCS CSV](https://www.usgs.gov/data/mineral-commodity-summaries-2026-data-release),
[GDELT DOC](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/amp/),
[GDELT Events 2.0](https://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf).

Pour chaque source utilisée, vérifier un petit échantillon réel, l'accès, les quotas et
les droits, puis qualifier les codes et calendriers. L'extraction de texte demande un
modèle évalué sur corpus réel. Un historique fiable exige des archives de versions,
publications et identifiants. Voir aussi les [exports DL](model-datasets.md) et [COT et
consensus sous licence](p3-sources.md).

[Vérification Lot 4 : tests, packaging et limites](archive/lot4-verification.md).
