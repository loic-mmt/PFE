# Prix daily, corporate actions et features

Les prix portent une séance économique, des horaires de marché réels et une
disponibilité explicite. Les indicateurs existants sont réutilisés par instrument ;
les fenêtres daily comptent des séances. Les corporate actions produisent des
rendements réconciliés distincts des prix d'entrée. Toutes les données de tests
et de l'exemple sont synthétiques.

## Installation et exemple

```bash
uv sync --locked --extra lse --extra plots --extra calendars
uv run --no-sync python examples/daily_market.py
```

L'extra `calendars` verrouille `exchange-calendars` 4.13.2. L'import de `pfe` et
les calendriers injectés fonctionnent sans cet extra. Les calendriers nommés
calculent localement les horaires ; ils ne téléchargent pas de données marché.
La bibliothèque de calendriers est communautaire : vérifier les exceptions avec
la bourse et conserver la version utilisée. Les [horaires NYSE](https://www.nyse.com/trade/hours-calendars)
et son [calendrier officiel 2026](https://www.nyse.com/publicdocs/nyse/ICE_NYSE_2026_Yearly_Trading_Calendar.pdf)
servent de référence aux séances US ; l'adaptateur suit les [APIs exchange_calendars](https://github.com/gerrymanoim/exchange_calendars).

## Séances et prix

API publique : `exchange_schedule(start, end, calendar="XNYS")`,
`normalize_session_schedule(frame)`, `normalize_daily_bars(frame, schedule=...)`,
`load_daily_bars(path, schedule=...)`, `aggregate_session_bars(frame, schedule=...)`.

`exchange_schedule` prend des dates de séance inclusives. Un calendrier injecté contient
`session_date`, `open_at`, `close_at` timezone-aware ; pauses facultatives via
`break_start_at`, `break_end_at`. Les sessions et pauses fournies sont validées. Une
plage de jours non cotés retourne un calendrier vide. Chaque instrument doit utiliser
son calendrier applicable : le ticker seul ne détermine pas sa bourse.

Un frame daily normalisé contient notamment :

| Colonnes | Convention |
|---|---|
| `instrument_id`, `entity_id` | Identités séparées ; entreprise nullable si non résolue |
| `session_date`, `observation_date` | Même date économique ISO, pas une disponibilité à minuit |
| `timestamp`, `session_open`, `session_close` | Open réel et clôture réelle, UTC |
| `open`, `high`, `low`, `close`, `volume` | Valeurs connues validées ; valeurs manquantes conservées |
| `source_id`, `revision_id`, `schema_version` | Provenance/version ; revision nullable pour un snapshot |
| `published_at`, `ingested_at`, `available_at` | Publication/ingestion/disponibilité distinctes |
| `currency` | Nullable si inconnue ; aucune conversion implicite |
| `price_adjustment`, `volume_adjustment` | `raw`, `split_adjusted`, `unknown` ; prix aussi `total_return` |
| `session_scope` | `rth`, `extended`, `all`, `unknown` |
| `quality.*` | Complétude prix/volume/séance, couverture et éligibilité explicites |

`label="session_date"` lit une date de séance ou une étiquette à minuit ; ce choix
est une déclaration du producteur. `label="open"` et `"close"` exigent un instant
aware correspondant exactement au calendrier. `available_at` n'est jamais avant
la clôture réelle ni une publication connue. En mode `receipt`, ingestion aware
requise et disponibilité au moins égale à réception. `event_time` et
`reconstructed` sont des hypothèses historiques déclarées ; `point_in_time`
reste False par défaut. Le calendrier courant ne prouve pas quand une ancienne
modification d'horaire avait été annoncée.

Un snapshot native OHLCV ne prouve pas la complétude RTH :
`quality.session_complete` reste nullable. Le collecteur marque durablement
incomplète une barre reçue avant sa clôture ; attendre ne transforme pas ce
snapshot en barre finale. Obtenir la valeur finale exige un nouveau snapshot.
L'API daily accepte une seule version par instrument/séance ; les snapshots raw
restent distincts, les versions à utiliser doivent être sélectionnées en amont.

## Agrégation intraday RTH

`aggregate_session_bars` agrège les intervalles RTH réels, y compris demi-séances
et pauses. Les timestamps d'entrée représentent les opens ; leur fréquence doit
diviser les intervalles actifs. Les observations hors séance sont soit exclues
avec compteurs dans `attrs`, soit rejetées par `extended="raise"`.

- `partial="keep"` conserve une séance incomplète avec masques, et les séances
  attendues entièrement manquantes sous forme de NaN.
- `partial="drop"` omet les séances incomplètes ; `"raise"` les rejette.
- `coverage` facultatif contient `instrument_id`, `first_session`, `last_session`.
  Il borne les périodes attendues ; aucune prolongation après la dernière borne.
- Sans couverture explicite, seules les séances entre première et dernière
  observation RTH sont considérées. Une fin de fichier ne prouve pas une radiation.

Prix et volumes ne sont pas remplis. La complétude signifie que la grille source
attendue est présente ; elle ne certifie pas l'intégralité du tape fournisseur.
La disponibilité de la sortie tient compte de la clôture, des entrées et de la
latence déclarée. L'historique de listings/radiations réel reste non qualifié
sans données du security master ; une couverture déclarée n'en crée pas un.

## Télécharger daily et plusieurs instruments

Le [SDK LSE](https://github.com/londonstrategicedge/lse-data/blob/main/lse/client.py)
documente `candles(timeframe="1d")` et les références splits/dividendes. Le mode
daily utilise les mêmes GET avec un plafond d'octets avant décodage JSON.

```bash
uv run --extra lse --extra calendars --env-file .env python scripts/download_data.py \
  --timeframe 1d --tickers SPY QQQ --dataset etf \
  --calendar XNYS --daily-label session_date --currency USD \
  --identity SPY=us:spy --identity QQQ=us:qqq \
  --start 2026-03-01 --end 2026-04-01 --snapshot-id daily-20261012
```

Ici `XNYS`, USD, les IDs et le label sont des choix explicites à vérifier pour le
flux utilisé. `--exchange TICKER=CALENDAR` permet des calendriers différents ;
`--entity TICKER=ENTITY_ID` apporte une identité d'entreprise vérifiée. Sans ID
explicite, `lse_symbol:TICKER` est un identifiant de snapshot fournisseur, pas un
security master historique. `--ticker` répétable/comma et `--tickers` acceptent
plusieurs symboles ; casse et validation du mode historique sont conservées.

Prix/volume/scope sont inconnus par défaut. `--price-adjustment split_adjusted` est
permis pour `--dataset stocks|etf`, avec la preuve du [changelog
SDK](https://github.com/londonstrategicedge/lse-data/blob/main/CHANGELOG.md). Le volume
et les dividendes ne sont pas qualifiés par cette preuve. Les prix ajustés conservent ce
statut ; RTH et profondeur fournisseur restent non qualifiés. La réception est le
défaut. `--availability-basis event_time` indique une reconstruction avec `--latency`
explicite, sans certifier une livraison passée.

Bornes locales : start inclusif, end exclusif. Daily exige `YYYY-MM-DD` ; les
bornes inclusives éventuellement renvoyées par le serveur sont filtrées. Défauts :
fenêtres de 31 jours, 5 000 lignes/requête, 250 000 lignes, 8 MiB et 64 requêtes
par invocation. Les budgets sont partagés entre instruments, y compris les
octets/lignes des pages réutilisées. Une page au cap échoue explicitement ;
reprendre un audit ne transforme pas une page plafonnée en couverture complète.

Raw/receipts/manifeste/checksums : `data/raw/lse/snapshots/<collection>/`. Processed :
`data/processed/<TICKER>_1d_<window>[_snapshot-id].parquet`. Même requête → reprise sans
refetch ni nouveaux reçus ; nouveau snapshot → nouvel ID. Les outputs sont publiés
atomiquement sans remplacer un fichier différent. `manifest.complete` indique que la
normalisation locale est terminée. La couverture historique du fournisseur demande une
vérification séparée. Les listes de séances manquantes figurent dans le résumé.

Le mode `1m` et ses anciens chemins restent disponibles. Pour rejouer cette
fenêtre, utiliser `--resume`. Son transfert Parquet SDK est borné par la fenêtre
(`--max-export-days`, 31 jours par défaut) puis vérifié en taille avant lecture ;
le SDK ne permet pas de plafonner les octets réseau avant transfert. Cette limite
est enregistrée dans le résumé. Les nouveaux GET daily/actions sont bornés avant
lecture complète. Aucun transfert de marché réel n'a été exécuté pour ces tests.

## Features par instrument/séance

```python
from pfe import compute_daily_features, exchange_schedule, load_daily_bars

schedule = exchange_schedule("2026-03-01", "2026-04-01", calendar="XNYS")
daily = load_daily_bars("daily.parquet", schedule=schedule)
features = compute_daily_features(
    daily,
    schedule=schedule,
    windows=(5, 20),
    names=("returns.simple", "technical.rsi_w14", "trend.sma_w20"),
)
```

Les noms sont ceux du catalogue existant ; `groups` et `names` sont exclusifs.
Les fenêtres comptent les séances de chaque instrument. Vendredi→lundi, DST,
jour férié et demi-séance ne réinitialisent pas les indicateurs. Une séance
attendue manquante ou inéligible réinitialise par défaut ; `gap_policy="allow"`
permet explicitement de la franchir. Les NaN de chauffe restent NaN.
`allow_partial=True` autorise explicitement des barres partielles finies.

L'index ordinal utilisé par les noyaux est interne ; la sortie conserve les
dates/heures réelles. Les features d'horloge intraday `calendar.*`/`session.*`
sont exclues. Les prix sont utilisés dans leur convention déclarée : les
indicateurs sur prix bruts gardent les effets des splits. Les rendements
réconciliés ci-dessous sont une sortie distincte à choisir pour les modèles.

`bar_available_at` conserve la disponibilité de la barre ; `available_at` des
features est au moins celle du préfixe utilisé. Un ancien prix reçu tardivement
ne devient pas utilisable plus tôt. `decision_at` filtre les barres encore
inconnues avant calcul. Les conventions devise/ajustement/scope doivent rester
cohérentes dans la série ; aucun état ne traverse les instruments.

`mask.<feature>` indique une valeur connue, y compris le masque de chauffe ;
`include_masks=False` évite cet export. `quality.age_seconds` mesure le temps
depuis clôture au cutoff explicite, sinon à disponibilité de la feature.
`quality.availability_lag_seconds` conserve le délai du prix source ;
`quality.receipt_age_seconds` est nullable lorsque réception inconnue ou backfill
postérieur au cutoff. Qualité, NaN, masques et âges restent dans l'export.

```bash
uv run --extra calendars python scripts/build_features.py \
  --daily --calendar XNYS --input daily.parquet --output features.parquet \
  --features returns.simple technical.rsi_w14 trend.sma_w20 --catalog columns.csv
```

`--schedule` CSV/Parquet remplace `--calendar` pour un calendrier injecté. Daily
n'utilise pas origin/offset, agrégation de durées ni streaming intrabar ; ces
options gardent leur rôle dans le mode intraday existant.

## Corporate actions et rendements

API : `ACTION_COLUMNS`, `normalize_corporate_actions`, `corporate_actions_asof`,
`adapt_lse_actions`, `collect_lse_actions`, `corporate_action_returns`.

Les IDs d'action/source/révision sont immutables. Annonce, publication, disponibilité et
date économique sont distinctes. Une date d'annonce sans heure est conservée dans la
preuve brute avec sa précision au jour. Les corrections de dates et annulations sont
résolues après sélection de la dernière version connue, avant filtrage de la date
effective.

Ratio split = nouvelles actions par ancienne action ; ratio positif. Dividende =
montant non négatif, devise et ex-date explicites. `dividend_basis` indique
`pre_split_share` ou `post_split_share`, avec `amount_unit="currency_per_share"`.
Une convention inconnue reste inconnue ; le calcul cash la rejette.

`corporate_action_returns` attend des prix par session avec identité, devise,
clôture et disponibilité. Il exige prix `raw` ou `split_adjusted` et convention
de volume explicite. Il ne modifie ni OHLCV ni volume :

- `input_price_return` : rendement des prix fournis.
- `price_return` : pour prix bruts, facteur split connu/effectif dans l'intervalle
  courant ; prix déjà split-adjusted utilisés directement, sans double correction.
- `total_return` : ajout optionnel des cash dividends explicitement qualifiés,
  uniquement sur prix bruts à même devise et même base par action.
- IDs, facteurs et statut de réconciliation sont conservés ; couverture des
  actions False par défaut. Aucune action connue ne prouve pas son absence.

Les anciens rendements ne sont pas réécrits par une action reçue plus tard.
Les splits futurs ne sont pas appliqués aux niveaux de prix historiques. Les
prix `unknown`/`total_return`, les conversions FX implicites, et l'ajout de cash
dividends à une échelle ajustée non qualifiée sont rejetés. Sélectionner une
source d'actions lorsqu'il existe plusieurs sources non réconciliées.

## Collecte actions et mappings

Le SDK documente les endpoints/date filters ; les champs numériques et conventions
ne sont pas garantis dans cette documentation. `field_map` est donc obligatoire,
avec direction du ratio explicitement déclarée. Ce mapping synthétique illustre
la forme, pas le schéma réel LSE :

```json
{
  "split": {"symbol": "ticker", "action_id": "id", "effective_date": "date", "ratio": "factor"},
  "dividend": {"symbol": "ticker", "action_id": "id", "effective_date": "date", "amount": "cash", "currency": "ccy"}
}
```

```bash
uv run --env-file .env python scripts/download_actions.py \
  --run --snapshot-id actions-20261012 --symbol AAPL \
  --entity-id issuer:verified-id --instrument-id security:verified-id \
  --start 2026-01-01 --end 2026-03-31 --field-map verified-fields.json \
  --ratio-convention new_per_old --dividend-basis post_split_share
```

`--snapshot provider.json` remplace `--run` pour un JSON offline contenant `split` et
`dividend`, chacun une liste de rows fournisseur. Deux GET bornés, 5 000 rows/requête et
1 MiB total par défaut. Une page atteignant le plafond est marquée incomplète. La
collecte traite les deux réponses et leurs identifiants fournis, sans pagination. Les
ex-dates divs utilisent le `effective_date` documenté par LSE ; les bornes serveur
restent non qualifiées.

Raw : `data/raw/lse/actions/<collection>/` ; normalisé :
`data/market/corporate_actions.parquet`. Le Parquet conserve l'historique des
révisions ; revoir le même fait n'ajoute pas une copie ni un reçu plus tardif.
Un même payload ne peut être réinterprété avec un ratio inversé. La reprise après
échec réutilise les pages stockées, y compris sans clé si toutes sont présentes.
Changer d'audit nécessite un autre `--snapshot-id` ; les anciens raw restent lisibles.
Complétude locale et couverture fournisseur sont séparées : `window_complete`
et les preuves d'absence restent non qualifiés. Le catalogue actuel n'établit ni
publication historique ni profondeur réelle des actions.

## Validation

Vérifié localement le 12 octobre 2026, Python 3.12 : **636 tests globaux passent**,
Ruff lint/format OK, exemple daily et quickstart OK, wheel/sdist construits et
contenu vérifié. L'extra calendrier est installé et présent dans le lockfile/CI.

Tests synthétiques : calendrier XNYS/DST/holiday/early close, gaps/pauses/partial,
multi-instruments, NaN et chauffe, disponibilité tardive, absence de fuite future,
split/dividende/révision/annulation, conventions et unités, pages bornées, crash/
reprise/absence de clé, publication des outputs et compatibilité du mode `1m`.
L'exemple public couvre action et ETF, split, cash dividend, DST et early close.
Listings/radiations fournisseur restent déclarés non qualifiés.

```bash
make check
make example
make build
```
