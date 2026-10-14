# COT, consensus et archives options

Ce module collecte les rapports COT, importe des estimations sous licence et archive les
chaînes options. Les imports sont locaux ; les collectes passent par les APIs dédiées.
Exemples et tests utilisent des données synthétiques. Les données professionnelles
demandent une licence et une qualification du package utilisé.

## Captures bornées et replay

`CaptureLimits` : 1 000 observations, 512 Kio par réponse, 4 Mio pour raw + Parquet
normalisé + résumé, hors petits manifestes. Les collecteurs effectuent une requête
au maximum par capture, sans pagination ni bulk. `RawStore` conserve réponse,
réception, hash, requête sans secrets et complétude. Un échec après réception se
reprend depuis le raw sans nouvel appel. Sortie `ObservationCapture` : identité,
observations, `complete`, `issues`, métadonnées.

`merge_observation_captures(*captures)` cumule les captures qui se chevauchent, garde la
première disponibilité d'une même version et refuse une version dont valeur ou unité
change. Utiliser ce merge pour dédupliquer les vintages.

## CFTC COT

`COTQuery` exige rapport, code marché à six caractères, entité canonique, bornes
économiques et mapping métrique → colonne fournisseur. Les six jeux Socrata Legacy,
Disaggregated et TFF, futures-only/combined sont distincts dans `COT_DATASETS`. Chaque
requête sélectionne un seul rapport et ses catégories.

```python
import pfe

query = pfe.COTQuery(
    report="legacy_futures",
    market_code="001602",
    entity_id="factor:declared_cot_market",
    start="2026-06-01",
    end="2026-06-30",
    columns={"open_interest": "open_interest_all"},
)
capture = pfe.collect_cot(
    query,
    store,
    snapshot_id="selected-cot-001",
    offline=(selected_json_bytes, "2026-07-01T12:00:00Z"),
)
```

Sans `offline`, GET HTTPS filtré par marché et période sur le PRE officiel. Le plafond
compte les observations scalaires : trois colonnes autorisent trois fois moins de lignes
source. Un résultat qui atteint le plafond est marqué partiel.

`report_date_as_yyyy_mm_dd` indique la date des positions ; la publication a son propre
horodatage. `publication_times={economic_date: aware_release}` est une carte de
publications vérifiées, avec exceptions de calendrier. Sans carte, publication inconnue
et disponibilité à réception. Une capture latest garde sa disponibilité à réception,
même avec cette carte.

`compute_cot_features` choisit le dernier rapport économique disponible. Les métriques
long/short/OI doivent partager la même version, unité et période. Net, net/OI et
variation à sept jours exacts sont calculés ; absence de rapport ou de jambe →
masque/NaN. Fraîcheur par disponibilité 14D et observation 21D par défaut, paramétrable.
Ces deux limites contrôlent aussi l’âge économique des positions. Les résultats
décrivent les positions d’un groupe COT.

Références : [PRE et catégories](https://publicreporting.cftc.gov/stories/s/r4w3-av2u),
[guide PRE](https://publicreporting.cftc.gov/stories/s/COT-Help/p2fg-u73y/),
[calendrier CFTC](https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm).
Le calendrier prévoit généralement vendredi 15 h 30 New York pour les positions
du mardi ; des reports et publications exceptionnelles existent. Le code n'en
déduit jamais automatiquement un horodatage historique.

## Consensus et estimations sous licence

`EstimateLicense` déclare fournisseur (`factset`, `ibes`, `licensed_export`), dataset et
référence de licence non secrète. `qualified_historical_pit=True` exige une référence de
qualification du package exact. L’utilisateur fournit ces références pour un accès dont
il dispose.

`LicensedEstimateAdapter` accepte JSON/CSV ou un client authentifié fourni par
l'utilisateur. L’utilisateur décrit les champs de son export avec `column_map` et
`value_maps`, y compris les booléens CSV et les valeurs nulles. Par défaut, format
canonique :

- Entité, métrique, début/fin de période fiscale, quarter/annual/YTD/instant.
- Devise, échelle, GAAP/non-GAAP, EPS basic/diluted/non applicable, segment nullable.
- Classe `analyst_estimate` ou `reported_actual` ; guidance rejetée ici.
- Statistique `mean`, `median`, `detail` ou `actual` ; `analyst_id` requis pour
  detail, nul pour consensus/actual ; `active` booléen conserve les retraits.
- `snapshot_at`, `published_at` nullable, valeur, dispersion/count facultatifs.
- `announcement_at` facultatif, mais obligatoire pour calculer une surprise réelle.
  La publication d’une révision et l’annonce initiale ont des horloges séparées.

Les chiffres et unités restent inchangés. Les instantanés et analystes constituent
des partitions distinctes ; les révisions du même instantané sont historisées.
Sans package PIT qualifié, toute l'histoire importée aujourd'hui n'est utilisable
qu'à réception, avec `point_in_time=False`. Avec qualification, chaque version
doit fournir son vrai `available_at`, au moins égal au snapshot/publication.

```python
license = pfe.EstimateLicense("factset", "selected_authorized_export", "internal-license-reference")
adapter = pfe.LicensedEstimateAdapter(license)
capture = adapter.capture(
    store,
    snapshot_id="authorized-export-001",
    query={"entity_ids": ["company:A"]},
    offline=(export_bytes, "2026-10-14T12:00:00Z"),
)
```

Pour un client propriétaire déjà disponible, `fetch(max_bytes=...)` retourne une
`HTTPResponse` de données sans écho d'authentification. Ce client doit borner les
octets réseau avant transfert. Aucun secret ne doit apparaître dans la config,
la référence de licence ou l'export ; les clés d'authentification sont rejetées.

`compute_estimate_features` exige les neuf champs `ESTIMATE_CONTEXT`, même période,
devise/échelle/base/segment et fournisseur. Mean, écart-type, analyst_count, dispersion
et révision de consensus sont disponibles. Une moyenne de détails utilise le dernier
état connu de chaque analyste, retraits compris, et porte `available_analysts_only`. La
couverture totale des analystes reste inconnue.

Une surprise nécessite un actual connu et une annonce prouvée. Baseline = dernière
moyenne disponible strictement avant l'annonce, snapshot lui aussi antérieur. Une
estimation enregistrée après l'annonce est exclue, même si son snapshot économique est
ancien. Formule relative : `(actual - estimate) / abs(estimate)` ; dénominateur nul →
NaN. Actual retiré ou annonce inconnue → surprise masquée. Un changement de couverture
peut modifier le consensus, même à estimations individuelles constantes.

[FactSet consensus/detail/actual/guidance](https://insight.factset.com/resources/factset-consensus-estimates-datafeed),
[FactSet package PIT](https://insight.factset.com/hubfs/Resources%20Section/White%20Papers/ID11996_point_in_time.pdf),
[LSEG I/B/E/S et packages historiques](https://www.lseg.com/content/dam/data-analytics/en_us/documents/brochures/lseg-data-for-quant-research-brochure.pdf).
Choisir le bon package et ses conventions avant de qualifier une profondeur PIT.

## Options prospectives

`collect_option_chain(transport, store, underlying=..., snapshot_id=..., limit=20)`
utilise le transport natif LSE borné, sans SDK importé. La requête sélectionne un ticker
et, au besoin, type/expiry/strikes ; elle s’effectue en une page, à la demande de
l’appelant. Même requête/snapshot_id → replay sans clé ni nouvel appel.

`archive_option_sample(payload, ...)` archive aussi les petits exports existants
chain/flow/candles. Le snapshot économique d'origine reste distinct de réception ;
aucune ligne ne devient disponible avant réception. Les timestamps UTC textuels
du vault sont normalisés selon le contrat SDK existant, avec raw conservé.
`merge_option_chains` réunit les snapshots compatibles, rejoue sans doublon et
refuse les corrections ambiguës d'un même snapshot. Garder ces corrections raw.

Le moteur `compute_option_features` existant est réutilisé. Champs bid/ask/OI absents
restent NaN. Volumes, premiums et ratios portent sur les contrats capturés. La
couverture du marché et du NBBO dépend de la source. L'historique prints/contrats, les
conventions et droits restent à qualifier. Les captures options utilisent LSE ; l'ajout
d'une autre source dépendra des champs et de la profondeur requis.

## CLI

```sh
python scripts/download_p3.py --source cot --config cot.json \
  --snapshot selected-cot.json --received-at 2026-10-14T12:00:00Z \
  --snapshot-id cot-selected-001
```

`--run` remplace `--snapshot` pour COT ou chaîne options. Options utilise `LSE_API_KEY`
déjà configurée. Consensus CLI : exports locaux uniquement, avec config `{license,
query, column_map?, value_maps?, format?}` ; clients propriétaires via Python.
Raw/Parquet/manifeste dans `data/raw/p3/<collection_id>/`. La CLI affiche l’identité de
collection et masque les credentials. Pour une collecte périodique, choisir fréquence,
périmètre et budget.

Datasets et labels : [model-datasets.md](model-datasets.md).

[Bilan de vérification Lot 5 : tests, packaging et limites](archive/lot5-verification.md).
