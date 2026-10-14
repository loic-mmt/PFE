# Calendrier, politique US et conflits

Implémentation du 14 octobre 2026. Calculs indépendants du timeframe ; les
features sont évaluées aux décisions explicites, ou sur les vraies dates de
session pour le daily. Aucun LLM, modèle supplémentaire ou gros téléchargement.

## API et sorties

| API | Sorties et conditions |
|---|---|
| `compute_calendar_features(decisions, timezone=..., schedule=...)` | Année civile/ISO, mois, semaine ISO, trimestre, jour du mois/année/semaine, week-end, frontières de périodes, encodages sin/cos. Position et séances restantes par mois/trimestre/année si calendrier complet |
| `compute_features(..., groups=("calendar",))` / sélection par nom | Les nouvelles colonnes calendrier sont disponibles avec les features intraday existantes ; mêmes conventions d'index de début de barre |
| `compute_daily_features(..., groups=("calendar",))` | Calendrier sur `session_date` réel, même lorsque OHLCV est manquant. Aucun calcul sur l'index ordinal interne. Les horloges/sessions intraday restent exclues |
| `compute_political_features(decisions, history=None)` | Cycle électoral US 1–4, sin/cos, années présidentielle/midterm, jours avant/après élections, fenêtre post-midterm de neuf mois. Partis présidentiels et majorités House/Senate, tenure et gouvernement unifié/divisé avec historique fourni |
| `compute_conflict_features(decisions, observations, ...)` | Événements, conflits distincts, pays concernés, décès best/low/high, couverture, variations entre fenêtres, conflits nouvellement observés, par type de violence ; scopes global/pays/région. Classification annuelle « guerre » distincte |
| `compute_graph_conflict_features(decisions, observations, history, ...)` | Activité/décès pondérés par fractions pays déclarées, séparées en revenus, production, fournisseurs, coûts et matières premières. Couverture du mapping et poids ambigus restent visibles |
| `context_feature_observations(features, units=..., ...)` | Colonnes numériques sélectionnées converties en observations canoniques, utilisables par `build_feature_panel` et les datasets existants. Broadcast uniquement vers des `entity_ids` explicitement choisis |

Le calendrier civil est inclus dans le catalogue daily par défaut. Les colonnes
`calendar.hour_*` et `session.*` demeurent intraday. Les anciennes API et noms
restent utilisables ; les sélections par nom évitent de calculer les autres groupes.

Les positions de séances comptent les dates du calendrier fourni, sans inventer
de jours fériés. Les bornes `schedule.attrs['start']`/`['end']` doivent couvrir
la période entière, ou la plage de séances doit la dépasser des deux côtés.
Sinon les positions/restants restent NaN et `calendar.mask.schedule_*` vaut zéro.
Une colonne optionnelle `schedule.available_at` impose que toutes les séances
du programme considéré soient déjà connues. Utiliser les programmes annoncés
à l'époque pour les fermetures exceptionnelles ; un calendrier final rétrospectif
ne prouve pas que ces fermetures étaient prévisibles.

## Cycle présidentiel et résultats politiques

Le cycle décrit des **années civiles électorales**, pas le mandat personnel d'un
président : 2001=1, 2002=2, 2003=3, 2004=4. Les features acceptent 1950–2195 et
le calendrier régulier 1946–2200 ; les changements historiques de loi et élections
spéciales sont hors périmètre. Dates selon la règle moderne du mardi suivant le
premier lundi de novembre. Sources : [USAGov midterms](https://www.usa.gov/midterm-elections)
et [présidentielles](https://www.usa.gov/presidential-election-process).

`politics.us.post_midterm_9m` commence à minuit New York le lendemain du scrutin
et se termine neuf **mois calendaires** plus tard, borne exclusive. Pour le
8 novembre 2022 : `[2022-11-09, 2023-08-09)`. Aucun label « bullish » n'est imposé.
La propriété prédictive doit être évaluée hors échantillon, sans sélectionner
une fenêtre favorable après observation des rendements.

Les états politiques sont importés via `normalize_political_history(frame)` :

```python
history = pd.DataFrame(
    [
        dict(
            record_id="synthetic-transition",
            state_kind="president",
            party="republican",
            effective_from="2025-01-20T17:00:00Z",
            available_at="2025-01-20T17:05:00Z",
            source_id="synthetic",
            revision_id="1",
        )
    ]
)
features = compute_political_features(decisions, history=history)
```

Exemple synthétique, pas un historique réel qualifié. `state_kind` : president,
house, senate. Partis : democratic, republican, other, split, unknown. Champs
obligatoires : record_id/state_kind/party/effective_from/available_at/source_id/
revision_id ; `effective_to` exclusif et publication/provenance facultatifs.
Instants aware explicites ; aucune heure n'est inventée pour une date seule.

Les enregistrements doivent être disponibles **et** effectifs. Les versions sont
immuables ; la plus récente par identité est résolue au cutoff. Une correction
d'une ancienne période ne remplace pas un état plus récent. Plusieurs sources
simultanées restent ambiguës ; `source_id` peut sélectionner une source exacte.
Partis encodés par catégories, sans ordre numérique artificiel. Sans historique,
cycle calculable, partis et gouvernement restent manquants avec masques nuls.
Les reconstructions historiques ne deviennent pas automatiquement PIT.
Une chambre `split` reste une catégorie explicite ; gouvernement unifié/divisé
reste manquant sans contrôle partisan établi, notamment pour les égalités du Sénat.

## UCDP : collecte bornée et distinctions

[Documentation API](https://ucdp.uu.se/apidocs/index.html),
[définitions](https://www.uu.se/en/department/peace-and-conflict-research/research/ucdp/ucdp-definitions/),
[codebook GED](https://ucdp.uu.se/downloads/ged/ged261.pdf),
[Candidate](https://ucdp.uu.se/downloads/candidateged/ucdp-candidate-codebook1.5.pdf).
API gratuite avec token demandé au fournisseur. `UCDP_API_TOKEN` est transmis
seulement dans `x-ucdp-access-token`, jamais dans requêtes archivées/manifests.

```python
query = UCDPQuery("gedevents", "26.1", "2025-01-01", "2025-01-31", countries=(678,))
capture = collect_ucdp(
    query,
    RawStore("data/verification/ucdp"),
    snapshot_id="selected-page-1",
    api_token=os.environ["UCDP_API_TOKEN"],
    limits=CaptureLimits(max_rows=100, max_response_bytes=131072),
)
features = compute_conflict_features(decisions, capture, countries=(678,))
```

Une page par appel, version obligatoire, budgets octets/lignes/archive,
aucun retry ni pagination automatique. `page` sélectionne explicitement une autre
page. `complete` décrit la requête retournée ; une pagination incomplète est
signalée. `countries` utilise les codes **Gleditsch–Ward**, pas ISO.
Les filtres GED StartDate/EndDate portent sur `date_end`, vérifié après réception.
Le resource `ucdpprioconflict` utilise des années complètes et fournit
`intensity_level` : minor=1, war=2. « Guerre » correspond à la classification
annuelle UCDP state-based, seuil de 1 000 morts liés aux combats ; aucune
classification instantanée n'est reconstruite depuis un simple article.
Choisir une version couvrant réellement la période demandée : la version annuelle
GED 26.1 contient l'historique jusqu'à 2025 ; pour 2026, sélectionner le millésime
Candidate mensuel/trimestriel adapté et vérifier sa couverture. Une réponse vide
hors couverture ne prouve pas l'absence de conflits.
Les versions Candidate mensuelles (`26.0.8`) et trimestrielles (`26.01.26.06`)
sont acceptées. Un événement à cheval sur deux mois peut apparaître dans plusieurs
releases ; l'identité d'événement, ses vintages et la première disponibilité
doivent être conservés lors de leur réunion.

`parse_ucdp` accepte des bytes sans réseau. Disponibilité au plus tôt à réception
et après toute publication explicitement renseignée. `observation_date` est la
date économique, jamais une disponibilité fictive. Les corrections de date
gardent l'identité d'événement ; doublons/vintages conflictuels sont rejetés.
`active=False` permet une rétractation explicite. Une omission dans une capture
partielle ne signifie pas terminaison/rétractation. Pour réunir des captures
identiques, utiliser `merge_observation_captures`, qui préserve la première réception.

Fenêtres par défaut 30/90 jours entiers, sur dates économiques UTC `date_end`,
après filtrage `available_at`. Séparer state_based/non_state/one_sided : leurs
IDs et événements ne sont pas un compteur unique de guerres. Un conflit avec
plusieurs événements compte une seule fois dans son type. « Newly observed »
signifie première disponibilité dans le corpus, pas début historique du conflit.
Les variations comparent deux fenêtres économiques adjacentes, au même cutoff.
Décès manquants restent manquants, jamais zéro.
`age_seconds` mesure l'âge depuis réception ; `observation_age_days` mesure l'âge
du dernier événement économique connu. Une capture récente peut contenir des
observations anciennes, notamment avec le délai des publications UCDP.

Sans source connue, compteurs manquants ; une capture GED vide connue peut donner
zéro **dans cette requête**. `complete_global_universe` reste zéro. Une source
annuelle seule ne produit pas de faux zéros d'activité événementielle. Les guerres
annuelles deviennent visibles seulement après disponibilité et clôture de leur
année économique ; la dernière année publiée n'est pas l'année courante par défaut.

Pour le graph, `country_nodes={678: 'country:YE'}` est un crosswalk explicite.
Seules les arêtes connues `exposure.revenue`, `exposure.production`,
`exposure.supplier`, `exposure.operating_cost`, `exposure.raw_material`
participent. Pas de localisation déduite du siège ou du secteur. Pas de milieu
de plage, de pourcentage inventé ou de renormalisation de poids incomplets.
Périodes/scopes mélangés ou déclarations répétées donnent NaN/masque nul.
Les sommes pondérées sont des statistiques par pays multipliées par les fractions
connues ; elles ne sont ni probabilités ni décomptes globaux de conflits distincts.

## Pipeline et vérification

Handlers : `calendar_features`, `political_features`, `conflict_features`,
`graph_conflict_features` et `collect` avec `kind='ucdp'`. Ils retournent
`features` ; `observation_options` fournit éventuellement units/entity_ids/
source_id/revision_id à l'adaptateur de datasets. Les historiques, calendriers
et observations sont référencés par `étape.table`. Réseau uniquement opt-in,
cache rejouable sans credentials, budgets globaux inchangés.

```bash
uv run python examples/calendar_politics_conflicts.py
```

Tests hors ligne : ISO/leap/DST et vrais jours de session, frontières exactes des
midterms, états effectifs mais non encore disponibles, révisions et retraits,
conflits distincts vs événements, publications annuelles tardives, requêtes
incomplètes/vides, plafonds, absence de token dans l'archive, expositions graph,
conversion en feature panel et replay du pipeline.
Un extrait minuscule de l'exemple officiel de documentation UCDP vérifie aussi
le parsing des champs réels ; ce n'est pas une capture live de l'API authentifiée.

## Données / qualification restant à fournir

- Historique réel des partis/majorités, changements intra-mandat et heures de
  disponibilité qualifiées. Import opérationnel ; aucun historique politique
  complet préinstallé ni résultats futurs générés.
- Token UCDP pour un test live restreint. Parsing, transport, budgets et replay
  testés hors ligne ; cela ne qualifie pas l'accès réel du compte.
- Vintages historiques UCDP et dates de publication qualifiées pour le backtest.
  Un fichier moderne contenant 2010 reçu aujourd'hui reste disponible aujourd'hui.
- Univers/couverture et rétractions lors de changements de version. Aucun statut
  de paix/fin de guerre déduit d'un manque de données ou d'une capture incomplète.
- Crosswalks pays/entreprises et véritables pourcentages d'exposition. Propagation
  indirecte via fournisseurs exige des chemins géographiques documentés ; cette
  API propage les expositions pays directes déclarées.
- Validation prédictive du cycle post-midterm et des variables géopolitiques.
  L'implémentation des features ne valide pas une stratégie de trading.
