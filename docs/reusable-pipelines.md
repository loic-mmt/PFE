# Bibliothèque réutilisable, plusieurs timeframes et petites vérifications

Le projet consommateur choisit les données, l'univers, les fréquences et le stockage,
puis appelle les APIs de collecte et de calcul. L'import de `pfe` charge la
bibliothèque. Les exemples sont synthétiques ; les captures réelles servent aux
vérifications ciblées.

## Datasets multi-timeframes

Les APIs publiques sont `TimeframeSpec`, `build_timeframe_observations` et
`build_timeframe_dataset`. Elles acceptent des OHLCV locaux avec index UTC
timezone-aware et une colonne `available_at` pour chaque bougie source. L'index
représente l'ouverture de la bougie ; elle ne peut être disponible avant sa clôture. Les
prix, volumes, unités, conventions d'ajustement et instruments restent ceux de la source
fournie.

```python
from pfe import TimeframeSpec, build_timeframe_dataset

specs = [
    TimeframeSpec(
        frequency=freq,
        names=("close", "returns.simple"),
        units={"close": "USD", "returns.simple": "fraction"},
    )
    for freq in ("1min", "5min", "1h")
]
result = build_timeframe_dataset(
    {"issuer:example": minute_bars},
    specs,
    source_frequency="1min",
    source_id="my_provider",
    revision_id="snapshot-001",
    decisions=decision_times,
    lookback=32,
    label_feature="tf[1min].close",
    label_horizon=5,
)
# result.features.observations / result.features.registry
# result.panel / result.sequences / result.labels
```

- Les métriques portent des noms distincts, par exemple `tf[5min].close`.
  Les noms du catalogue PFE s'appliquent ; seuls les calculs sélectionnés sont
  effectués. La grille de décision peut différer des timeframes des features.
- `lookback`, `stride` et `label_horizon` comptent les pas de cette grille,
  pas des jours. Un horizon de cinq pas sur une grille de minutes vaut cinq
  décisions futures. Les labels restent séparés des inputs.
- Seules les bougies cibles complètes produisent des valeurs utilisables.
  Trous et bougies incomplètes sont signalés par des masques, avec valeurs manquantes conservées.
  L'upsampling est refusé. Les indicateurs suivent `gap_policy="reset"` par défaut.
- La dernière feature connue peut persister entre deux clôtures. Pour borner sa
  durée d'utilisation, déclarer `FeatureSpec.max_age` dans le registre utilisé
  par `build_feature_panel`. Le registre retourné ne fixe pas de durée universelle.
- La disponibilité des agrégats respecte les réceptions des entrées. Celle des
  indicateurs utilise conservativement le maximum des réceptions de leur préfixe
  source ; une entrée tardive peut donc retarder plusieurs indicateurs suivants.
- `availability_policy="bar_close"` constitue une reconstruction explicite,
  marquée non qualifiée PIT. La politique par défaut exige les réceptions réelles.
  Même avec des réceptions fournies, la couverture historique n'est pas certifiée
  automatiquement : les observations produites gardent `point_in_time=False`.

Durées fixes : secondes/minutes/heures, ou jours de 24 h pour un marché continu.
`mode="session"` exige `schedules` et `session_metadata` par entité ; il réutilise
les calendriers de séances, demi-séances et changements DST du Lot 1.
Un jour fixe de 24 h ne représente pas une séance actions. Les features path/profile
intrabar sont disponibles en mode fixe ; le mode séance utilise le catalogue daily.

Voir [exemple autonome](../examples/reusable_pipeline.py),
[tests de fréquences et de disponibilité](../tests/test_timeframe_datasets.py).
`build_timeframe_observations` reste borné en mémoire pour les OHLCV fournis ;
le flux partitionné ci-dessous commence aux observations canoniques déjà calculées.

## Panels et fenêtres partitionnés

`iter_parquet_observations` lit les record batches, avec contrôle préalable de la
taille décompressée des row groups. Les fichiers et leurs batches doivent suivre
`available_at`, globalement ; les anciennes périodes économiques peuvent être
révisées tardivement. Préparer ce tri lors de l'écriture des observations.

`iter_panel_partitions` conserve l'état as-of nécessaire, le chevauchement des
fenêtres et le lookahead réservé aux labels. `iter_partitioned_sequences` reproduit
les fenêtres du calcul complet, y compris aux frontières et avec `stride > 1`.
Il garde l'axe des décisions en mémoire, puis matérialise les valeurs par partition.
Chaque buffer, panel et séquence est plafonné ; le RSS total peut inclure plusieurs
buffers simultanés et les allocations de pandas/Arrow. Ce n'est pas une limite OS
globale de RAM.

```python
from pfe import (
    iter_parquet_observations,
    iter_partitioned_sequences,
    fit_partitioned_scaler,
    save_partitioned_dataset,
)


def partitions():
    return iter_partitioned_sequences(
        decisions,
        iter_parquet_observations(paths, batch_rows=256),
        entity_ids=entities,
        registry=registry,
        lookback=32,
        label_feature="tf[1min].close",
        label_horizon=5,
        partition_size=128,
    )


scaler = fit_partitioned_scaler(partitions(), train_end=train_end)
save_partitioned_dataset(
    output,
    partitions(),
    scaler=scaler,
    max_archive_bytes=8_388_608,
)
```

Deux passes permettent un seul scaler global ajusté sur les cellules uniques du
train. Les fenêtres dont le label ou sa disponibilité franchit `train_end` sont
exclues. Les paramètres, compteurs et empreinte du train sont conservés ; le writer
revérifie cette empreinte. Pas d'ajustement indépendant par shard.

Les indices des séquences restent locaux au shard : ajouter `global_offset` pour
obtenir l'indice sur l'axe global. `core_start/core_stop` délimitent les décisions
propres à la partition ; le panel contient aussi les contextes chevauchants.
`iter_saved_partitions` vérifie les SHA256 et charge un shard à la fois.
Le manifeste global n'est publié qu'après succès de tous les shards. Une reprise
rejoue la même factory et accepte uniquement des fichiers identiques ; changer la
config ou les entrées exige un nouveau répertoire.

CLI locale, sans collecte :

```bash
uv run python scripts/build_partitioned_dataset.py \
  --observations observations-001.parquet observations-002.parquet \
  --config dataset.json --output data/datasets/partitioned \
  --partition-size 128 --max-bytes 8388608 --max-archive-bytes 8388608
```

Config : `decisions`, `entity_ids`, `features` (liste de `FeatureSpec`), `lookback`,
`stride`, éventuellement `labels: {price_feature, horizon}`, et `scale: true` avec
`train_end`. `DatasetLimits` et les limites d'archive sont distincts et explicites.
Ce writer produit les séquences GRU/TCN ; les tenseurs GNN/events existants restent
disponibles via les APIs du [Lot 5](model-datasets.md).

## Orchestration et exports

`run_pipeline` exécute un DAG de `PipelineStep` ; les dépendances passent par
`needs`. Chaque opération porte une révision de code. Identité de cache : config,
SHA256 des fichiers déclarés et manifestes des dépendances. Les chemins publiés
sont relatifs au répertoire d'exécution ; fichiers existants immutables.

Opérations fournies : `local_tables`, `collect`, `merge_observations`, `timeframes`,
`financial_mappings`, `dataset`, `graph`, `graph_bindings`, `graph_features`,
`graph_pairs`, `graph_dataset`. Les [opérations graphes](graph-features.md) acceptent
des relations structurées qualifiées et exportent les datasets GNN. `collect`
sélectionne FRED, EIA, une source internationale existante ou le security master.
`outputs` associe les noms de tables à leurs destinations, par exemple
`macro/us_macro_vintages.parquet`. L'opération `dataset` publie les artefacts du builder
complet ; le writer partitionné est une API/CLI séparée. Le JSON sélectionne les
opérations enregistrées dans le runner.

Un appelant Python peut enregistrer un `PipelineHandler(callable, revision)` pour
enchaîner d'autres APIs de la bibliothèque. Il doit déclarer ses fichiers externes
dans `files`, conserver une révision correcte et faire passer ses accès réseau et
raw par `PipelineContext`. Un callable de confiance qui contourne ce contexte
n'est pas contrôlé par les budgets réseau du runner.

```json
{
  "limits": {"max_bytes": 8388608, "max_download_bytes": 2097152, "max_requests": 16},
  "steps": [
    {"name": "local", "operation": "local_tables",
     "parameters": {"paths": {"observations": "observations.parquet"}}},
    {"name": "export", "operation": "merge_observations", "needs": ["local"],
     "outputs": {"observations": "market/global_reference_assets.parquet"}}
  ]
}
```

```bash
uv run python scripts/run_pipeline.py --config pipeline.json --output data/runs/example
# Ajouter --allow-network uniquement pour une collecte choisie.
# Fournir les clés dans l'environnement, jamais dans le JSON.
```

Une source partielle rend ses dépendances et le manifeste global partiels. Une
interruption conserve les étapes déjà validées. Le replay vérifie les tables et
artefacts et rejette les caches corrompus. Limites globales : requêtes, réponses,
téléchargements, taille d'archive ; limites par table : lignes et taille en mémoire. Le
runner conserve les résultats des étapes en RAM ; utiliser les APIs partitionnées pour
les grands panels. Une étape partielle déjà publiée reste un snapshot immutable : son
replay ne relance pas automatiquement une collecte. Changer son périmètre ou ses caps
exige un nouveau run. En revanche, une étape interrompue avant validation reprend à
partir des pages raw déjà reçues dans le même run.

## Macro, univers, cross-section et fondamentaux

| Besoin | API et garanties ajoutées |
|---|---|
| C01 : retrait macro | `normalize_macro_series`, bridge FRED et moteur sparse acceptent numérique → null → numérique. Le retrait masque la valeur ; pas de résurrection de l'ancienne version ni de réduction implicite des fenêtres |
| C02 : calendrier historisé | `normalize_macro_calendar_history`, `macro_calendar_asof`, `compute_macro_features(calendar_history=...)` : replanification, annulation, actual/forecast/previous révisés et horloges par champ. Surprise avec forecast strictement connu avant `release_at` vérifié ; pas de surprise si publication inconnue |
| C03/C08 : master | `parse_security_snapshot` / `collect_security_master` pour SEC, Nasdaq listed/otherlisted et Alpha Vantage `LISTING_STATUS`. Identités permanentes explicitement fournies ; réception et dates source distinctes ; `update_security_master` historise les changements, `security_universe_asof` filtre à la décision. Absence d'un ticker ≠ radiation |
| C04 : cross-section | `compute_cross_sectional_features` : rang percentile, z-score population, écart à moyenne et effectif, univers puis groupes connus as-of. `normalize_classifications` / `classifications_asof` historisent les groupes. Périodes économiques identiques par défaut ; masques et effectif minimum |
| C05 : comptabilité | `FinancialMapping`, `apply_financial_mappings` : concepts IFRS/custom, contexte segment, unité, base EPS et périodes fiscales explicitement justifiés. `DebtRecipe` / `derive_debt_totals` : composants déclarés disjoints, même filing/unité/contexte ; pas de double comptage avec un total direct |

Le calendrier historisé contient une **version d'état complète** par événement et
révision, pas un patch de champs. `available_at` indique quand la version est connue ;
les champs actual/forecast/previous non nuls exigent leurs horloges. Une correction
du forecast après annonce ne modifie pas le consensus de référence de la surprise.
`calendar_history` et l'ancien argument `events` sont mutuellement exclusifs.

Le master demande une correspondance ticker → `entity_id`, `instrument_id`, `venue`,
`currency`. Les IDs permanents doivent venir d'un référentiel qualifié. Les annuaires
courants décrivent les univers actuels ; les anciens statuts s'importent depuis une
archive via le contrat bitemporel. SEC exige un `User-Agent` avec contact réel, Alpha
Vantage une clé facultative `ALPHAVANTAGE_API_KEY`. Nasdaq ne demande pas de clé pour
ces fichiers publics. Si plusieurs sources contredisent le statut, ticker, venue ou
devise d'un même instrument, la sélection d'univers exige un `source_id` explicite ;
filtrer les statuts actifs ne doit pas cacher une radiation déclarée par une autre
source.

Les mappings comptables portent `reference` et `available_at` ; leur connaissance
retarde la feature au besoin. Ils ne modifient pas les nombres source. Les segments
s'utilisent avec `compute_fundamental_features(entity_scope="segment", context=...)`.
Chaque concept IFRS/custom exige un mapping justifié pour le contexte retenu.

Contrats publics de ces adaptateurs :
[SEC mappings](https://www.sec.gov/files/company_tickers.json),
[définitions Nasdaq](https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs),
[Alpha Vantage listing status](https://www.alphavantage.co/documentation/#listing-status).

## Petites vérifications des services

`scripts/verify_integrations.py` teste les quatre clés facultatives et le modèle Qwen
déjà installé. Lecture optionnelle des valeurs littérales dans `.env`, sans
shell/interpolation ni affichage des secrets. Par invocation : 12 requêtes HTTPS, 64
Kio/réponse, 512 Kio transférés et 2 Mio d'archive au maximum ; requêtes sans retry
automatique. Les contrôles utilisent les dépendances et le modèle déjà installés ; le
serveur Ollama doit être actif.

```bash
uv run python scripts/verify_integrations.py --env-file .env \
  --snapshot-id verification-001 --output data/verification/001
```

EIA utilise deux journées WTI ; OpenDART un compte annuel Samsung ; e-Stat une table
du catalogue et une cellule, sans prétendre qualifier son mapping économique.
FRED teste deux observations et leurs métadonnées bornées. Qwen travaille sur un
court texte synthétique ; réponses, rejets et replay sont conservés. Une réponse
du modèle ou un JSON valide ne démontre pas la qualité financière de l'extraction.
Les résultats datés sont décrits dans [la vérification immédiate](immediate-verification.md).

Les calculs demandent des entrées qualifiées : historique PIT, couverture et identités
de l'univers, corpus LLM annoté et mappings comptables validés. Revere, J-Quants/KRX
natifs et les autres feeds licenciés nécessitent un adaptateur selon leurs
spécifications et accès ; leurs données peuvent déjà entrer par les imports canoniques.