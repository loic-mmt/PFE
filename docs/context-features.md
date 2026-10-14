# Références, macro et fondamentaux

Ces APIs calculent les références de marché, séries macro révisables et ratios
financiers à partir de données locales. Les collecteurs sont facultatifs. Guidance,
extraction LLM, graphes et embeddings sont décrits au [Lot
3](market-data-roadmap.md#lot-3--extraction-documentaire-graphes-événements).

## Contrat temporel commun

Les observations gardent `entity_id`, `observation_date`, `published_at`,
`available_at`, `value`, `source_id`, `revision_id`, ainsi que `metric`, `unit`,
`schema_version` et les métadonnées propres à la source. Dates économiques et
publication sont distinctes. Une décision utilise seulement les valeurs déjà
**disponibles**, avec sélection de la révision alors connue.

- **Réception, par défaut** : disponibilité à la capture, publication inconnue
  laissée nulle. Les observations sont utilisables à partir de cette réception.
- **Reconstruction explicite** : vintages ALFRED, acceptance SEC ou carte de
  clôtures Cboe, selon les conventions ci-dessous. `point_in_time=False` demeure :
  aucune garantie de livraison historique ou d'absence de correction rétroactive.
- Publications/décisions horodatées : fuseau requis, normalisation UTC. Une date
  seule ne devient pas une heure de publication.
- Révisions tardives : recalcul des statistiques au moment où elles deviennent
  disponibles ; aucune modification des décisions antérieures.
- Pas de remplissage des valeurs inconnues par zéro. Masques et âges exposent les
  données manquantes/périmées. Les fenêtres macro/références comptent des périodes
  distinctes, sans multiplier les observations par forward-fill sur les décisions.

Les APIs de features reçoivent un `DatetimeIndex` conscient du fuseau, ou une
DataFrame de décisions avec `decision_at` (et `entity_id` pour les fondamentaux
ou la macro multi-entités). Elles préservent l'index d'entrée. Les fondamentaux
sur un `DatetimeIndex` exigent des décisions triées, uniques. Les APIs historiques
sur OHLCV restent inchangées ; `compute_fred_macro_features` utilise leurs heures
de clôture, tandis que `compute_fred_features` reçoit des décisions explicites.

```bash
uv run python examples/context_features.py  # Synthétique ; aucune clé/réseau
```

## Références : benchmarks, secteurs, FX, VIX et matières premières

[`data/reference_assets.py`](../src/pfe/data/reference_assets.py) impose une
identité exacte et stable : `instrument_id`, `symbol`, `asset_class`, `currency`,
`calendar`, `frequency`, `price_adjustment`, unité et source. Les unités restent
celles de l'entrée. Une référence ne mélange pas implicitement plusieurs sources,
conventions de cotation ou émetteurs.

- `reference_assets_from_daily` adapte les barres canoniques du Lot 1 : vraie
  clôture de séance dans `observed_at`, réception et qualité conservées. Une
  entreprise/facteur manquant exige un ID explicite ; ticker courant ≠ issuer
  historique. Utiliser le calendrier exact choisi lors de l'ingestion.
- `reference_assets_from_observations` adapte **une série sélectionnée** FRED,
  EIA, Cboe ou Pink Sheet. Déclarer instrument/symbole/type ; calendrier/fréquence
  manquants exigent une convention explicite. Les métadonnées existantes ne sont
  pas écrasées silencieusement. Les moyennes mensuelles restent mensuelles.
- `normalize_reference_assets` / `load_reference_assets` valident des entrées
  canoniques ou CSV/Parquet. L'import CSV préserve les identifiants textuels.
- FX : `base_currency`, `quote_currency` et unité `QUOTE/BASE` requis. USD/JPY
  signifie JPY par USD ; cours strictement positif. Aucune inversion cachée.

[`features/reference.py`](../src/pfe/features/reference.py) :

```python
from pfe import compute_reference_features

features = compute_reference_features(
    decisions,
    references=references,
    target=stock_reference,
    windows=(5, 20),
    groups=("returns", "relative", "risk"),
    max_age="7D",
)
```

Groupes `levels`, `returns`, `relative`, `risk`, `exposure` ; sélection exacte par
`names`, ou `compute_reference_feature(decisions, name, ...)`. Les sorties sont
préfixées `reference[INSTRUMENT].` et accompagnées de `mask.<feature>`.

| Sortie | Convention |
|---|---|
| `level`, `mean_wN`, `std_wN`, `zscore_wN` | N périodes ; écart-type population ; warm-up/variance nulle → NaN |
| `return`, `log_return`, `momentum_wN` | Rendement du prix fourni ; momentum nécessite N+1 périodes |
| `relative_strength_wN` | `(1 + rendement cible)/(1 + rendement référence) - 1` |
| `beta_wN`, `corr_wN`, `sample_count_wN` | N rendements avec les deux extrémités disponibles et exactement appariées |
| `fx_effect`, `energy_effect` | Sensibilité déclarée par l'appelant × variation de référence |
| `age_seconds` | Ancienneté de disponibilité de la période courante |
| `observation_age_seconds` | Ancienneté de la clôture économique, si explicitement connue |
| `feature_age_seconds` | Temps depuis le dernier recalcul sparse |
| `point_in_time`, `price_basis_known` | Qualité/provenance conservée ; ne qualifie pas couverture ou corporate actions |

Beta/corr et force relative exigent des `observed_at` **identiques**, mêmes
calendrier, fréquence et devise. Deux cours sur la même date mais clôturés à des
heures différentes ne sont pas des horizons appariés. Sans vrai timestamp de
clôture, ces sorties restent NaN. `max_age` contrôle aussi la cible et la réception
du benchmark à la clôture effectivement appariée, même si un cours non apparié
plus récent a été reçu entre-temps. Prix bruts,
split-adjusted et total return gardent leur convention : ajuster/réconcilier
explicitement au Lot 1 avant de comparer si nécessaire.
Le flag `point_in_time` décrit la période courante, pas une certification de tous
les inputs d'une fenêtre ; qualifier aussi l'historique utilisé.

Les expositions utilisent les coefficients fournis par l’appelant :
`fx_exposures={"USDJPY": 0.5}`, `energy_exposures={"WTI": -0.2}` et
`reporting_currency="USD"`. Une exposition JPY exprimée en USD inverse le cours USD/JPY
; EUR/USD est direct pour une exposition EUR en USD. Une énergie dans une autre devise
exige une conversion préalable explicite. Ces sorties mesurent les sensibilités aux
facteurs selon ces coefficients.

Pour les ETF/benchmarks : utiliser `scripts/download_data.py --timeframe 1d` du Lot 1 et
les symboles qualifiés. Fournir les identifiants VIX/FX/futures/indice et leurs
calendriers. Les instruments étrangers suivent le même contrat, avec calendrier local
explicite ; leur disponibilité commerciale/historique reste à qualifier.

## FRED et ALFRED

[`data/fred.py`](../src/pfe/data/fred.py) conserve observations, unités/fréquence,
ajustement saisonnier, bornes real-time, titres et métadonnées de releases.
[`data/alfred.py`](../src/pfe/data/alfred.py) utilise les mêmes endpoints FRED v1
avec une plage real-time explicitement bornée. Source commune `fred` ; toutes les
vintages sont transmises au moteur macro existant, sans sélection latest-only.

`normalize_fred_observations(rows, metadata, received_at=...)` accepte les valeurs
numériques source et `.` comme valeur nulle. `last_updated` décrit la mise à jour
courante de la série ; la publication de chaque observation exige une autre horloge. Les
dates de release gardent leur précision date. Selon FRED, elles peuvent différer de la
date de disponibilité sur FRED.

Politique `vintage_next_day` : `available_at` = minuit **du lendemain** de
`realtime_start`, en `America/New_York` par défaut, convertible explicitement.
C'est une convention conservatrice à la journée ; elle ne permet pas de backtester
une décision intraday au moment exact d'une annonce. `published_at` reste null,
`availability_basis="reconstructed"`, `point_in_time=False`.
En mode réception, sélectionner une seule date real-time : plusieurs versions du
même point reçues ensemble seraient ambiguës et sont rejetées.

```python
from pfe import compute_fred_features

macro = compute_fred_features(
    decisions,
    observations=vintages,
    pairs=(("DGS10", "DGS2"),),
    names=("macro.DGS10.level", "macro.pair[DGS10,DGS2].spread"),
)
```

Spreads : unité, fréquence et ajustement saisonnier identiques requis. Deux taux
`Percent` donnent des **points de pourcentage**, sans division implicite par 100. Chaque
série conserve une entité/fréquence/base saisonnière stable au cours de son historique ;
un changement exige une réconciliation explicite. Le moteur macro fournit niveaux,
changements, révisions, âges et statistiques trailing. DGS2/DGS10, DFF et FEDFUNDS
restent des séries distinctes ; choisir les IDs officiels et leur profondeur. Consensus
et prévisions nécessitent leurs propres observations source. Une révision nulle après un
point numérique retire le point dans le bridge FRED et le moteur macro ; une
restauration ultérieure reste une révision distincte. Le point retiré reste manquant
dans les fenêtres jusqu’à sa restauration. Le [calendrier
historisé](reusable-pipelines.md) ajoute versions, reports, annulations et surprise avec
forecast strictement connu avant publication vérifiée.

### Collecte et replay

```bash
# Une capture actuelle : realtime-start == realtime-end.
uv run --env-file .env python scripts/download_fred.py --run \
  --series-id DGS10 --start 2025-01-01 --end 2026-10-12 \
  --realtime-start 2026-10-12 --realtime-end 2026-10-12 \
  --snapshot-id dgs10-20261012

# Vintages historiques, convention date-only explicite.
uv run --env-file .env python scripts/download_fred.py --run \
  --series-id CPIAUCSL --start 2024-01-01 --end 2026-10-12 \
  --realtime-start 2024-01-01 --realtime-end 2026-10-12 \
  --availability vintage_next_day --snapshot-id cpi-vintages-20261012
```

Clé `FRED_API_KEY` pour les nouveaux GET ; replay complet sans clé. Mode
`--snapshot fixture.json` : tableau `responses`, avec `endpoint`, `payload`,
`received_at`, `offset` pour les pages et `release_id` pour `/release/dates`.
Aucun HTTP en mode snapshot. `collect_fred_series` retourne
`(collection_id, observations, summary)` ; la CLI cumule les révisions dans
`data/macro/us_macro_vintages.parquet` en conservant la première réception d'une
révision identique. Une convention de disponibilité, fréquence ou base
saisonnière incompatible est rejetée avant modification de l'historique.

Budgets par défaut : 20 000 lignes source, 1 MiB total, 20 nouveaux GET/invocation,
1 000 lignes/page. Lignes/octets comprennent les pages en cache. Les endpoints
series, observations, vintagedates, series/release et release/dates sont archivés
avec pagination contrôlée. Cap atteint → `pagination_complete=False` et curseur
conservé ; absence de série/période jamais inférée d'un échantillon tronqué.
Un nouveau snapshot-ID capture un nouveau téléchargement, le même ID rejoue le
cache et les réceptions originales. La profondeur/licence ICE reste distincte.

Sources : [API observations](https://fred.stlouisfed.org/docs/api/fred/series_observations.html),
[vintages](https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html),
[real-time](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html),
[release dates](https://fred.stlouisfed.org/docs/api/fred/release_dates.html).

## EIA, Cboe et Banque mondiale

Chaque source possède son adaptateur et son schéma de parsing.

| Source / module | Ce qui est implémenté | Temporalité / limites |
|---|---|---|
| [EIA](../src/pfe/data/eia.py) | API v2 : route/facets/champ exacts, unités, fréquence, pagination offset/length/total | Réception par défaut ; publication seulement depuis une colonne explicitement désignée et horodatée ; vintages historiques à fournir séparément |
| [Cboe](../src/pfe/data/cboe.py) | CSV VIX : DATE et colonnes OPEN/HIGH/LOW/CLOSE sélectionnées | Réception ; publication par version explicitement fournie ; reconstruction seulement avec carte de clôtures déclarée, pas calendrier actions supposé |
| [Pink Sheet](../src/pfe/data/world_bank.py) | Table mensuelle ou XLSX : feuille, header, colonne période et unités sélectionnés explicitement | Observations mensuelles sélectionnées, jamais cours daily ; disponibilité à réception ou publication horodatée de la version du fichier |

`ContextLimits(max_rows=25000, max_bytes=1048576, max_requests=8, page_size=5000)`
borne les collectes ; `ContextCollection` contient observations, `complete`,
`issues`, `metadata` et `collection_id`. Les fichiers complets et réponses sont
versionnés dans `RawStore`. Les caps produisent un résultat déclaré incomplet
ou un refus explicite ; les octets HTTP sont bornés avant décodage, les lignes
avant normalisation. La CLI écrit les sorties partielles avec suffixe.
Les clés et échos EIA `request.params.api_key` sont retirés **avant stockage**.
Les métadonnées de requête conservées sont expurgées, jamais la clé d'API.

`parse_pink_sheet` fonctionne sans dépendance Excel. Pour XLSX :

```bash
uv sync --extra commodities  # openpyxl facultatif, importé seulement pour XLSX
```

`parse_pink_sheet_xlsx` exige `sheet_name`, `header_row` (zéro-based),
`period_column` (nom exact ou index zéro-based), `column_specs` et `received_at`.
Les formats
période sont explicites (`YYYY-MM` ou `YYYYMmm`). `data_start_row` et
`data_end_row` délimitent les lignes zéro-based, borne de fin exclue ; pas d'heuristique de titres,
unités ou notes. ZIP décompressé ≤ 50 MiB par défaut, colonnes ≤ 256. Les cellules
formules ne sont pas calculées ; leurs valeurs mises en cache sont celles du
fichier. Un changement de disposition échoue au lieu de sélectionner autre chose.
Les observations Pink Sheet gardent début du mois dans `observation_date` et fin
du mois dans `period_end` ; aucune de ces dates n'établit la publication.

Exemple de config Cboe (`cboe.json`) :

```json
{"metrics": ["CLOSE"]}
```

```bash
uv run python scripts/download_context.py --source cboe --run \
  --config cboe.json --snapshot-id vix-20261012 \
  --output data/market/vix.parquet
```

Pour EIA, config = champs d'`EIAQuery` : `route`, `entity_id`, `value_field`,
`frequency`, `facets`, `start`, `end`, `unit` si connue, `calendar`, éventuelle
`published_column`, `availability_basis`. Route **relative** API v2, sans
`/data/` final ; un seul code par facet, une série exacte par requête. Les
métadonnées dimensionnelles et unités doivent rester identiques entre les
observations et pages ; les périodes hors bornes inclusives sont rejetées.
Fréquences prises en charge : daily, weekly, monthly, quarterly, annual.
Clé `EIA_API_KEY` pour collecte en ligne.
Pour Pink Sheet, config = sélection XLSX ci-dessus et `column_specs` :
`{"Copper": {"entity_id": "commodity:copper", "unit": "USD/mt", "currency": "USD"}}`.
Le nom de colonne, l'unité et le header doivent être vérifiés dans la version.

Offline : `--snapshot fichier --received-at 2026-10-12T12:00:00Z` pour CSV/XLSX ou JSON
EIA brut. Un snapshot EIA multi-pages fournit `pages` avec `payload` et `received_at`
par page. Rejouer la même identité ne redownload pas un fichier modifié à la même URL.
Nouveau snapshot pour nouvelle version. Les observations peuvent ensuite être adaptées
en références en conservant les horaires fournis et les heures inconnues. Choisir un
output par capture, puis sélectionner les séries et fusionner leurs révisions pour
constituer l’historique. Les fichiers existants sont protégés de l'écrasement. La
reconstruction Cboe décrit des horaires supposés pour des valeurs de la version
téléchargée ; elle ne reconstitue pas les révisions des anciennes versions.

Sources officielles : [EIA v2](https://www.eia.gov/opendata/documentation.php),
[Cboe historique](https://www.cboe.com/tradable_products/vix/vix_historical_data),
[commodity markets](https://www.worldbank.org/en/research/commodity-markets).
Le registre conserve accès gratuit et notes de droits distinctes :
[EIA réutilisation](https://www.eia.gov/about/copyrights_reuse.php),
[Cboe termes](https://www.cboe.com/terms),
[Banque mondiale conditions](https://data.worldbank.org/summary-terms-of-use).
Accès public ≠ permission générale de distribution/training/commercialisation.

## SEC : filings et Company Facts

[`data/sec.py`](../src/pfe/data/sec.py) fournit `SECClient`, `SECData`,
`normalize_sec_filings` et `normalize_company_facts`.

- Submissions récentes et pages historiques sélectionnées ; accession, form,
  documents principaux et `acceptanceDateTime`. Les documents HTML/textuels
  sont collectables/extractibles via le [Lot 3](document-features.md).
- Company Facts : taxonomie/concept, unité source, start/end, accession,
  contexte, base comptable, EPS basic/diluted et état reported actual.
- `fy/fp` appartiennent au **filing**, conservés comme `filing_fiscal_*` ; ce ne
  sont pas automatiquement l'année/quarter d'un fait comparatif. `period_kind`
  est une classification par durée, marquée `quality.period_inferred`, avec dates
  exactes conservées. Les calendriers fiscaux atypiques exigent qualification.
- Les concepts de CA concurrents restent distincts : aucune fusion automatique
  de `Revenues`, `SalesRevenueNet` et autres tags. Custom/segments restent en
  contexte séparé ; leur absence ne prouve pas absence d'exposition.
- Acceptance consciente du fuseau : `accepted_at` conserve l'acceptation SEC,
  distincte de la publication effective. `published_at` reste **null par défaut** :
  la [SEC signale des dépôts diffusés le jour ouvré suivant](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data).
  Seule une carte `publication_times={accession: timestamp aware vérifié}` fournie
  à `normalize_sec_filings` renseigne une publication. Acceptance absente, naïve ou date seule :
  null. `filed` conserve sa précision au jour pour filtrer et ordonner les versions.
- Réception commune : sélectionner la dernière déclaration identifiable d'un
  même point. Tous les accns source restent en raw ; conflit impossible à
  départager → erreur. Réconstruction `--availability publication --latency 60s`
  est un nom legacy : utilise `accepted_at+latency` lorsque la publication est
  inconnue mais reste **non-PIT**, avec `published_at=null` ; corrections
  post-acceptance et délai réel de disponibilité non qualifiés. Une publication
  explicitement vérifiée remplace ce point d'ancrage, sans preuve de réception
  historique du pipeline.

```bash
uv run python scripts/download_sec.py \
  --cik 0000320193 --user-agent 'MyFeatureLibrary contact@example.org' \
  --audit-id sec-20261012 --max-pages 20 --max-filings 1000
```

Utiliser un contact réel. `SECClient` limite chaque client à 8 GET/s par défaut,
toujours strictement sous 10/s ; partager le client entre callers parallèles.
4 MiB **par réponse**, 20 pages et 1 000 filings par défaut ; les caps sont
visibles dans `SECData.complete` et le manifest. `start` inclusif/`end` exclusif
filtrent la date de filing. `--recent-only` omet les pages historiques.
Company Facts JSON plus gros que le cap échoue explicitement ; augmenter le
budget pour un émetteur après inspection, aucune lecture bulk sans borne.

Raw `data/raw/sec/<collection-id>/`. Sorties immutables par capture dans
`data/fundamentals/sec/company_facts_<collection-id>.parquet` et
`filings_<collection-id>.parquet`. Rejouer le même audit réutilise raw/réception
sans HTTP. Nouveau audit pour nouvelles valeurs. Cumuler les versions avec
`merge_company_facts` avant les features : révision identique → première réception,
révision nouvelle → nouvelle disponibilité ; conventions incompatibles refusées.

[SEC APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
et [accès automatisé](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data).
CIK identifie un déposant, aucun univers historique complet de titres n'est créé.

## Features fondamentales

[`features/fundamentals.py`](../src/pfe/features/fundamentals.py) :

```python
from pfe import compute_fundamental_features

fundamentals = compute_fundamental_features(
    decisions,
    facts=facts,
    entity_id="sec:cik:0000320193",
    period_kind="quarter",
    basis="us_gaap",
    eps_basis="diluted",
    currency="USD",
    concepts={"revenue": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"},
)
```

Sélection exacte par `names` ou `compute_fundamental_feature`. Sorties
`fundamentals.*` : CA/EPS, net income, operating income, gross profit,
assets/liabilities/equity/cash, total debt **si explicitement fournie**,
long-term debt, operating cash flow, CapEx ; croissance CA/EPS YoY, marges,
ratios dette/actifs/fonds propres, liabilities/assets, long-term debt/assets,
OCF/revenue, intensité CapEx, FCF et net debt. Dette long terme n'est pas renommée
dette totale ; SEC Company Facts n'est pas une comptabilité consolidée exhaustive.

Règles de calcul :

1. Seulement `reported_actual`, périmètre entity, base et EPS sélectionnés.
   Guidance/analyst estimate ne deviennent pas des résultats réels.
2. La période cible est la plus récente alors connue. Ratios de flux : mêmes
   start/end, unité/devise, base, source et périmètre. Ratios de stocks : même date.
   Tag/source/devise ambigus → NaN + qualité, ou sélection explicite.
3. Quarter dérivé : différence de flux additifs YTD (et Q4 annuel−9 mois), mêmes
   fiscal start/concept/base/unité/source ; toutes les jambes connues. Jamais
   différence d'EPS pondéré. `derived_quarter` indique la dérivation.
4. FCF = operating cash flow − CapEx avec convention cash outflow positive,
   mêmes période et bases. `free_cash_flow_derived` l'indique ; aucun nombre absent
   n'est présenté comme extrait. Net debt = dette totale − cash compatible.
5. YoY : même concept/base, périodes de durée comparable et espacées d'environ
   un an (tolérance 15 jours pour années 52/53 semaines). Croissance =
   `(current - prior) / abs(prior)` ; zéro/missing → NaN, changement de signe
   conserve cette convention explicitement.
6. Âge par métrique, âge économique, `coverage_fraction`, qualité ambiguë/incomplète
   et masques. Couverture = fraction d'inputs présents parmi les niveaux définis,
   aucune preuve de complétude comptable de l'émetteur. `max_age` masque les inputs
   de la période cible trop anciens, après sélection de cette période ; pas de
   substitution par une période antérieure nouvellement révisée. Les comparatifs
   historiques déjà connus restent utilisables pour YoY lorsque l'input courant
   satisfait ce seuil.

## Validation et reste à faire

[Vérification du 12 octobre 2026 : périmètre, correctifs, tests et limites](archive/lot2-verification.md).

Fixtures couvrent disponibilité juste avant/à publication, révision future sans
changement passé, révision d'une ancienne période, batch simultané, unités et
périodes incompatibles, horizons benchmark décalés, sens FX, données manquantes,
cache sans réseau, reprise de pagination, budgets, expurgation des clés,
formulaires SEC, bases/EPS/contexts et calculs manuels. Aucune collecte live
fournisseur n'est utilisée pour affirmer une couverture réelle ou un droit.

Restent à qualifier/implémenter :

- Captures réelles par source et série : profondeur, limites compte, droits,
  calendriers, conventions FX/ajustements, publication et latence. Faire de petits
  audits avant historique ; conserver source/manifest et comparer au document.
- Heures de releases vérifiées par source et snapshots contemporains pour PIT
  intraday ; les dates ALFRED/release ne suffisent pas. Le retrait numérique→null
  et le calendrier versionné sont implémentés et testés localement.
- Taxonomies financières hors mappings US GAAP, concepts custom, segments,
  fiscal calendars atypiques et dette consolidée : mapping explicite contrôlé
  sur filings, aucune déduction globale à partir de la seule taxonomie. Le
  [moteur de mappings explicites et de dette disjointe](reusable-pipelines.md)
  est implémenté ; la couverture comptable réelle reste à qualifier.
- [Extraction documentaire](document-features.md) : documents, prompts A–D, preuves, backend
  LLM facultatif et graphe/events historisés ; modèle réel à qualifier.
- [Sources internationales](international-features.md) : connecteurs étrangers ciblés et GDELT,
  décisions aware, liens documentés et masques ; sources réelles à qualifier.
- [Datasets modèles](model-datasets.md) : fenêtres/panels, graphes/events, splits
  purgés, scaler train et exports bornés ; [COT et consensus](p3-sources.md).
  Sources live et packages premium restent à qualifier avant collecte réelle.

Voir [inventaire et lots](market-data-roadmap.md), [prompts](llm-extraction-prompts.md),
[contrats](qualification.md) et [Lot 1](daily-market.md).
