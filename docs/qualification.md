# Contrats, sources, stockage et qualification

Cette couche fournit les contrats communs, le registre des sources, le security master
et le stockage raw. Les loaders prix/macro/options gardent leurs schémas spécialisés.
Historique et univers proviennent des données collectées.

## Utilisation

```bash
uv run --no-sync python examples/data_qualification.py

# Audit à partir d'un catalogue sauvegardé : aucun réseau ni clé.
uv run --no-sync python scripts/qualify_lse.py \
  --snapshot discovery.json --audit-id offline-20261011 \
  --symbol QQQ@etf --symbol SPY@etf

# Audit réseau explicite : quatre endpoints metadata, jamais history/dataset/export.
uv run --env-file .env python scripts/qualify_lse.py \
  --run --audit-id lse-20261011 --symbol QQQ@etf \
  --max-rows 25000 --max-bytes 33554432 --max-requests 4
```

`LSE_API_KEY` requise uniquement pour exécuter `--run` non déjà enregistré.
Identité déterministe : source, dataset, paramètres, révision source/licence et
version de schéma. Même identité → reprise/réutilisation ; nouvel audit → nouvel
`--audit-id`. Changer fichier offline ou limites crée aussi une autre collection.
Les clés restent hors des fichiers et des messages d'erreur fournisseur.

Plafonds par défaut : 25 000 lignes par catalogue, 1 MiB de réponses cumulées, quatre
requêtes par invocation. Le fichier offline, chaque page sauvegardée et le rapport ont
également un plafond d'octets. Une réponse trop grande est rejetée. Un catalogue au
plafond de lignes, ou avec une continuation, est marqué incomplet. `/catalog` retourne
une liste en une requête. Les pages déjà reçues peuvent être reprises après échec d'un
endpoint.

Exemple de `discovery.json` synthétique pour le mode offline :

```json
{
  "provider": "LSE",
  "audited_at": "2026-10-11T12:00:00Z",
  "catalog": [{"symbol": "QQQ", "dataset": "etf", "country": "US", "unit": "USD"}],
  "catalog_complete": false,
  "catalog_authoritative": false,
  "reference": [],
  "reference_complete": false,
  "reference_authoritative": false
}
```

Les flags d’un échantillon valent False. True exige une preuve de couverture du
catalogue. Dans cette fixture, une entrée absente est `unqualified`.

## Observations et révisions

API : `normalize_observations(frame)`, `observations_asof(frame, decision_at)`.
Constante : `OBSERVATION_COLUMNS`.

Colonnes obligatoires : les sept champs utilisateur, plus `metric`, `unit`,
`schema_version`. Les valeurs `published_at` et `revision_id` peuvent être null.
Un numéro de révision non null est nécessaire pour distinguer plusieurs versions
d'une même observation ; deux versions simultanées ambiguës sont rejetées.

| Champ | Contrat |
|---|---|
| `entity_id`, `source_id`, `metric`, `unit`, `schema_version` | Strings non vides, sans espaces externes |
| `observation_date` | Date civile ISO ou instant timezone-aware ; sortie ISO, précision date conservée |
| `published_at` | Instant timezone-aware ou null ; normalisé UTC |
| `available_at` | Instant timezone-aware obligatoire, jamais avant publication connue |
| `value` | Nombre fini, objet/tableau JSON ou null ; bool/nombre en string rejetés au niveau principal |
| `revision_id` | String nullable ; immutable pour une observation |

Extras : `instrument_id`, `source_document_id`, `ingested_at`, `currency`,
`availability_basis`, `point_in_time`. Metadata supplémentaires conservées.
Valeur JSON copiée profondément ; NaN imbriqué, infini, cycles et objets non JSON
rejetés. Les périodes futures sont permises pour guidance ; elles ne modifient
jamais la disponibilité.

Provenance par défaut : `availability_basis="unknown"`, `point_in_time=False`. Le
producteur doit justifier le flag par les preuves de disponibilité de sa source.
`receipt` exige `ingested_at` et disponibilité au moins égale à ingestion ;
`publication` exige publication connue. `reconstructed` et `event_time` permettent
backfill/latence documentés sans confondre ingestion actuelle et connaissance
historique. Les instants de disponibilité exigent un timestamp avec fuseau ; timestamps
naive, epochs numériques et dates seules sont rejetés.

As-of conserve toutes les périodes économiques ; il sélectionne la dernière version
connue par `(entity_id, metric, observation_date, source_id, instrument_id)`. Une
révision tardive d'une ancienne période ne remplace pas le niveau de la période plus
récente. Les sources sont traitées séparément, avec unités cohérentes entre vintages. La
réconciliation et les conversions d’unité ou de devise sont à faire en amont.

```python
from pfe import normalize_observations, observations_asof

history = normalize_observations(frame)
known = observations_asof(history, "2026-01-02T13:30:00Z")
# Toutes les lignes satisfont available_at <= decision_at.
```

## Security master bitemporel

API : `normalize_security_master(frame)`,
`security_master_asof(frame, decision_at, effective_at=None)`.
Colonnes : `SECURITY_MASTER_COLUMNS` ; métadonnées CIK/share class facultatives.

- `entity_id` = entreprise ; `instrument_id` = titre ; `mapping_id` = intervalle
  de mapping corrigible. Ce sont trois identifiants différents.
- Mapping : ticker, venue, currency, status (`active`, `suspended`, `delisted`,
  `inactive`), `valid_from`, `valid_to`, disponibilité, source, révision, schéma.
- Validité économique `[valid_from, valid_to)` ; borne finale null = intervalle
  ouvert. Instants aware UTC, pas rétrodatation depuis catalogue actuel.
- Une correction conserve `mapping_id`, ajoute une nouvelle révision et clôture
  éventuellement l'ancien intervalle. Changement de ticker/statut → nouvel
  intervalle avec un autre mapping ID. CIK string de 1–10 chiffres → 10 chiffres.
- Appliquer toutes les révisions simultanées avant validation : clôturer ancien
  ticker et ouvrir nouveau ticker dans un même batch ne produit pas faux conflit.
- Vérifier chevauchements à chaque état historique. Ticker/venue ne peut pointer
  simultanément vers deux instruments/entreprises ; sources concordantes restent
  distinctes. Les conflits doivent être résolus explicitement.
- As-of choisit d'abord versions connues, puis validité économique. Filtrer
  l'intervalle avant la correction pourrait ressusciter ancien mapping.
- Tous statuts sont retournés, y compris delisted ; sélectionner `active` relève
  de la politique d'univers. Interroger passé avec connaissance actuelle via
  `effective_at` ne prouve pas que mapping était connu à cette date passée.

`collect_security_master` / `parse_security_snapshot` acceptent SEC, Nasdaq
listed/otherlisted et Alpha Vantage `LISTING_STATUS`. Identités permanentes fournies
explicitement, captures à réception, changements versionnés et filtre
`security_universe_asof` : voir [API et limites](reusable-pipelines.md). Une radiation
exige un statut fourni par la source. Pour un univers ancien, utiliser des archives de
cotation plutôt qu’un annuaire courant.

## Registre des sources

`SourceRecord`, `SourceRegistry`, `default_source_registry()` : déclarations
immutables par `(source_id, revision_id)`, permissions explicites, URL/conditions,
audit timestamp, version et disponibilité connue. Re-register identique = no-op ;
modifier même révision = erreur. `get(source_id, revision_id)` garde accès ancien ;
sélection sans version exige ordre d'audit connu et non ambigu.

Registre initial : `lse`, `sec`, `fred`, `local` (audit du 11 octobre 2026). Lot 2
ajoute `eia`, `cboe`, `world_bank` (audit du 12 octobre 2026), avec accès, conditions et
limites propres aux sources. Les droits de training/distribution et les accès du compte
sont à vérifier pour chaque dataset. Une source locale peut avoir une URL null. URLs
publiques sans credentials/query, droits inconnus = null/tuple vide. FRED nécessite
qualification par série ; SDK LSE MIT ≠ licence data. `to_dict()` produit JSON ;
`SourceRegistry.load(path)` charge un fichier borné. Metadata source complète intégrée à
l'identité et au manifeste RawStore.

## Raw immutable et manifestes

```python
from pfe import RawStore, default_source_registry

store = RawStore("data/raw/collections", max_page_bytes=1048576)
collection = store.begin(
    "lse",
    "discovery",
    {"audit_id": "audit-1"},
    source_metadata=default_source_registry().get("lse").to_dict(),
)
store.write_page(collection, "catalog", payload, row_count=len(payload))
store.write_page(collection, "final", report, complete=True)
manifest = store.manifest(collection)
```

Layout : `root/<collection SHA256>/manifest.json`, `pages/<page-ID SHA256>.bin`
et `.json` pour le record immutable. JSON canonique, octets bruts inchangés,
checksum SHA256, taille, ordre, row count, receipt time, metadata et cursor.
Rejouer même page/body/metadata garde premier receipt time sans double compteur.
Toute différence exige une nouvelle page/vintage ou collection ; une collection
complète n'accepte que replay identique.

Écriture : journal record → raw → manifeste, fichiers temporaires fsync et
publication atomique. Verrous threads/processus ; lock libéré après crash.
`recover(collection_id)` récupère un journal/raw non commité avant reprise ; raw manquant ou hash
différent ne devient jamais page valide. Une page en attente doit être récupérée
avant append d'une autre page. `RawStoreError`, `ImmutablePageError`,
`CorruptStoreError` distinguent invalidité, conflit et corruption.

Rejouer `recover()` ne change ni page ni receipt time. Si le raw n'a pas été
publié, rejouer le `write_page()` original permet de terminer l'écriture ; sans
ses octets originaux, démarrer un nouvel audit. La CLI récupère les pages
journalisées avant tout refetch réseau.

`next_cursor` est continuation opaque ; `resume_cursor()` seul ne prouve pas
complétude. `manifest["complete"]` explicite, False par défaut ; une page capped/
incomplete ou avec continuation ne peut finir la collection. Reprise générique
par page disponible aux futurs connecteurs ; pages LSE discovery = endpoints.
Le store ne valide pas sémantique des fenêtres d'un fournisseur : l'adaptateur
doit définir bornes, schéma et preuves de complétude.

## Matrice de qualification

`qualify_lse_snapshot(snapshot, symbols=..., limits=...)` pur offline ;
`collect_lse_discovery` transport injecté ; transport HTTP stdlib GET borné sur
`/vault/meta`, `/vault/catalog`, `/vault/reference`, `/vault/usage` uniquement. Le
transport refuse les redirections, filtre les credentials et contrôle le plafond
d’octets. Les requêtes se limitent aux quatre endpoints metadata.

Rapport : les **34 besoins actuels** du roadmap, priorités/modèles, méthodes SDK,
état d'implémentation daté, preuves catalogue/référence, matrice instrument×dataset,
first/last/ticks/unit/devise/timezone, ajustements, provenance et quota.

- `available` : référencement metadata exact uniquement ; pas qualification
  financière, disponibilité daily, couverture complète ou droit du compte.
- `absent` : code exact non trouvé, seulement catalogue daté, complet et autoritatif.
  Jamais conclusion « entreprise/facteur absent sous tous les alias ».
- `unqualified` : alias/identity incomplet, ambiguïté, unités/PIT ou couverture inconnus.
- `local` : calcul local ; inputs et code correspondant restent à qualifier.
- `not-applicable` : besoin attribué ici à une source externe, pas preuve universelle
  d'absence chez LSE.

`SYMBOL@DATASET` résout plusieurs classes ; aucun uppercasing/alias automatique.
Catalogue partiel, doublons, invalid rows, pagination ou cap empêchent conclusions
d'absence. Références qualifiées séparément. Snapshots actuels n'établissent ni dates de
publication, ni vintages, ni anciens listings. `entitlement="unknown"`. Statut
d'ajustement split des candles actions/ETF cite SDK 0.14 ; raw ticks et dividendes
restent non qualifiés. Le rapport décrit les métadonnées à la date de l’audit. Relancer
l’audit pour actualiser cet état ; les captures de données marché ont leurs propres
manifestes.

## Validation

À livraison du Lot 0 le 12 octobre 2026, Python 3.12 : **165 tests Lot 0** et
**549 tests globaux** passent. Ruff lint/format, exemple offline, build wheel et
sdist réussis ; les cinq modules Lot 0 sont présents dans les deux archives, la
CLI dans le sdist. Aucun appel authentifié LSE n'a été exécuté ; le rapport d'un
compte réel se produit avec `--run` et un nouvel `--audit-id`.

Tests offline : avant/pile à disponibilité, révision tardive d'ancienne période,
réceptions simultanées, corrections ticker/delisting, source revisions, replay
sans doublon, byte/row/request caps, corruption/hash, crash/recovery,
concurrence threads/processus, erreur réseau/reprise et mode offline sans clé.

```bash
uv run --no-sync pytest -q tests/test_contracts.py tests/test_security_master.py \
  tests/test_sources.py tests/test_storage.py tests/test_qualification.py \
  tests/test_qualify_lse.py
make check
make build
```
