# Datasets pour GRU/TCN, GNN et Event Encoder

`pfe.datasets` produit des tableaux NumPy et des artefacts NPZ/JSON, utilisables par un
projet d’entraînement. PyTorch/TensorFlow sont facultatifs. Sources P3 :
[p3-sources.md](p3-sources.md). Les moteurs de features existants restent utilisables
indépendamment des datasets.

Extension du 14 octobre : [APIs multi-timeframes, orchestration et partitions](reusable-pipelines.md).
`iter_panel_partitions` / `iter_partitioned_sequences` évitent le panel complet,
avec fenêtres chevauchantes et labels identiques. `fit_partitioned_scaler` ajuste
un seul scaler train global ; `save_partitioned_dataset` écrit des shards bornés.
Le builder complet décrit ci-dessous reste disponible pour les petits inputs.

## Registry et provenance

`FeatureSpec` décrit nom, métrique canonique, unité, source, fenêtre, dépendances,
fréquence facultative, durée maximale de disponibilité et `instrument_id` facultatif.
`FeatureRegistry` impose les noms uniques et fournit une identité déterministe. Les
dépendances du registry sont des noms d’entrées, à fournir avant la construction du
panel.

```python
import pfe

registry = pfe.FeatureRegistry(
    [
        pfe.FeatureSpec(
            "momentum_20",
            "momentum_20",
            "fraction",
            "local",
            window="20 sessions",
            dependencies=("adjusted_close",),
            max_age="5D",
        )
    ]
)
panel = pfe.build_feature_panel(
    decisions,
    observations,
    entity_ids=entity_ids,
    registry=registry,
)
```

Entrées : observations canoniques des Lots 0–4, valeurs numériques déjà calculées.
Entité, métrique, unité et source exactes doivent correspondre au registry. Le dernier
état connu de la période économique la plus récente est choisi à chaque décision aware.
Révision tardive d'une période ancienne ne remplace pas la période récente. Source,
devise et échelle doivent correspondre à la spécification. Si une entité possède
plusieurs instruments pour la sélection, renseigner `instrument_id` dans la
spécification ; une sélection ambiguë est rejetée. L'identité de l'instrument reste dans
les preuves. `quality.feature_eligible` accepte uniquement des booléens ou une absence
(défaut : éligible) ; la chaîne `"False"` est rejetée.

`FeaturePanel` expose les axes `[decision, entity, feature]` :

- `values`, `mask` : valeur et présence/fraîcheur/compatibilité.
- `available_at_ns`, `observation_at_ns` : UTC ns ; `NaT` int64 pour temps inconnu.
- `provenance_codes` : index dans la table de preuves, `-1` si aucune observation.
- Table de preuves : entité, métrique, unité, source, révision, document, date
  économique avec sa précision d'origine, instrument, base de disponibilité, flag PIT.
- Décisions, identités explicites et registry. L'empreinte inclut valeurs, masques,
  horloges, axes et preuves, pas seulement les paramètres de calcul.

Un état nul retire la valeur ; une donnée périmée garde sa provenance et ses horaires,
avec une valeur masquée. Les fenêtres numériques représentent les manquants par zéro
accompagné d’un masque faux.

Un fait de guidance/consensus attaché à une période future n'est pas directement
un input numérique de panel. Le transformer d'abord à la décision avec les APIs
fondamentaux/statements/estimates, puis publier ses features. La passerelle
`feature_frame_to_observations(frame, registry, available_at=..., revision_id=...)`
accepte l'index `(decision_at, entity_id)` et exige un DataFrame de disponibilité
par cellule. Elle ne déduit jamais cette disponibilité de la date de ligne.
Conserver les délais source/calcul/revue dans les temps fournis.

Le flag PIT reprend les déclarations des sources. Préparer l’univers à chaque date avec
le security master, radiations comprises, et qualifier la couverture des données avant
de construire le panel.

## Fenêtres et labels

```python
labels = pfe.forward_return_labels(panel, price_feature="close", horizon=5)
dataset = pfe.build_sequence_dataset(panel, lookback=32, labels=labels)
```

`SequenceDataset.x` et `.mask` : `[sample, time, feature]`, ordre déterministe
décision puis entité. GRU utilise ce layout ; un TCN nécessitant channels-first
peut transposer les deux derniers axes. End index, entity index, décision et début
de fenêtre permettent de retrouver chaque cellule et sa provenance dans le panel.

Le lookback inclut uniquement la décision courante et son passé. `stride` avance dans la
grille fournie ; il ne crée pas de jours de marché. Les fenêtres conservent les
manquants. Chaque fenêtre exige un lookback complet ; les observations manquantes à
l’intérieur gardent leurs masques. Pour des lots plus petits :

```python
for batch in pfe.iter_sequence_batches(
    panel,
    lookback=32,
    labels=labels,
    batch_size=64,
):
    consume(batch.x, batch.mask, batch.y, batch.y_mask)
```

Les batches ont exactement les mêmes fenêtres que l'export complet, sans perte aux
frontières des lots. Le panel complet reste en mémoire. Utiliser les APIs partitionnées
pour des volumes plus grands. Segmenter une source en amont exige de conserver l'overlap
de calcul et de fenêtre nécessaire.

`ForwardLabels` est séparé : rendement simple prix futur/prix courant − 1, horizon en
pas de grille, masque, début/fin d'intervalle et disponibilité du label. Les prix
doivent être positifs, connus, et l'observation future doit avancer ; un prix stale
répété ne devient pas un nouveau label. La queue sans horizon futur reste masquée. Les
labels sont stockés séparément de `x` et du registry des inputs.

Choisir les conventions d'ajustement/rendement en amont. Ces labels ne corrigent
pas automatiquement splits/dividendes, univers, devises ou prix incorrects. Le
prix futur du label est celui connu à la décision future de la grille ; il ne
réécrit pas les entrées passées lors d'un restatement ultérieur.

## Splits temporels et normalisation

```python
split = pfe.temporal_split(
    dataset,
    train_end="2024-01-01T00:00:00Z",
    validation_end="2025-01-01T00:00:00Z",
    embargo="7D",
)
scaler = pfe.TrainOnlyScaler.fit(dataset, split)
x_scaled, scaled_mask = scaler.transform(dataset)
```

Intervalles de décision semi-ouverts : train avant sa borne ; validation après
borne train + embargo ; test après borne validation + embargo. Les labels doivent
finir **et être disponibles strictement avant** la prochaine borne. Un label qui
touche cette borne est purgé. Labels absents/queue future et trous d'embargo
figurent dans `split.purged`. Aucune séparation aléatoire des lignes.

La validation est purgée à sa propre borne de test. Les inputs de lookback peuvent
utiliser l'histoire antérieure à la frontière, comme en inférence réelle. Cette purge
sépare les horizons de labels. Des observations proches peuvent garder une dépendance
statistique.

Le scaler accepte uniquement un `TemporalSplit` de ce dataset, non vide en train, avec
indices cohérents avec les bornes purgées. Moyenne/variance par feature sur les cellules
uniques des fenêtres train : pas de surpondération des overlaps. Statistiques
validation/test et labels exclus du fit. Variance nulle → échelle 1. Feature jamais
observée en train → reste masquée après transformation, même si elle apparaît en
validation. Les zéros accompagnés d’un masque faux représentent des données manquantes.
`transform_panel` applique ces statistiques figées aux axes du panel ; le registry doit
correspondre.

Les sorties normalisées ont une unité standardisée ; le registry conserve l'unité
source avec les paramètres mean/scale dans le manifeste. Un modèle évalué hors
échantillon doit utiliser le scaler du train correspondant, sans refit sur le test.

## GNN

[Extension graphes](graph-features.md) : `build_graph_dataset` assemble les cibles
et contextes, `edge_attr` / `edge_attr_mask` préservent séparément les poids,
bounds et âges ; `iter_graph_partitions` / `save_partitioned_graph_dataset`
exportent un snapshot par décision sans dupliquer les contextes chevauchants.

```python
for snapshot in pfe.iter_graph_tensors(panel, graph_history):
    consume_graph(snapshot.x, snapshot.mask, snapshot.edge_index)
```

Un snapshot par décision. Nœuds connus dans le graphe et présents dans l'univers
explicite du panel, triés par ID. Nœuds/edges inconnus à la décision sont exclus.
`edge_index` : `[2, edge]`, direction et types conservés. Node types, identités,
masques, availability des edges et documents justificatifs accompagnent les valeurs.
Poids = pourcentage connu / 100 ; propriétaire et nature revenue/supply/exposure
conservés. Plusieurs poids incompatibles ou inconnus → NaN. Les poids du réseau partiel
gardent leur dénominateur d’origine.

Les features des nœuds restent en unités source. Pour normaliser, appliquer les
statistiques train figées à ces mêmes features ; aucune estimation globale sur
les snapshots validation/test n'est faite par `iter_graph_tensors`.

## Event Encoder

`build_event_tensor(decisions, event_history, entity_ids=..., max_events=16)`
réutilise les événements/liens historisés et la revue du Lot 3. Axes
`[decision, entity, event]` ; vecteurs ajoutent l'axe dimension.

Derniers événements connus et liés, triés par première disponibilité puis ID,
queue de taille bornée. Padding explicite : IDs vides, type/statut −1, masque
d'événement faux ; aucun lien société déduit du seul texte d'un titre. Récence
mesurée depuis première disponibilité événement/entité. Magnitude et unité
préservées séparément ; `magnitude_mask` distingue absence et zéro réel. Valeurs
continues paddées à zéro ; masques à appliquer avant le modèle. Remapper les codes
de padding selon la convention de l'Embedding de votre framework.

Les embeddings sont facultatifs : sans enrichissement, dimension zéro et aucun backend
LLM. Avec enrichissements : digest `model_revision`, dimension et `model_available_at`
explicites, document/hash texte exacts, réponse accepted et `is_model_output=True`.
Modèle **et réponse d'inférence** disponibles avant décision, sinon vecteur masqué.
L’encodage conserve les dates de disponibilité et la couverture des sources.

## Export reproductible et lecture

```python
pfe.save_model_dataset(
    "data/datasets/run_001",
    panel,
    sequences=dataset,
    split=split,
    scaler=scaler,
    configuration={"universe_revision": "declared-001"},
)
arrays, manifest = pfe.load_model_dataset("data/datasets/run_001")
```

`arrays.npz` : tableaux sans pickle, panel/provenance codes, inputs, labels,
splits et inputs normalisés facultatifs ; graphes ragged et events facultatifs.
`manifest.json` : registry, sources/révisions/documents, configuration, axes,
paramètres du scaler train, empreintes du panel/dataset et SHA256 des tableaux.
Maps type/statut Event Encoder et nature/propriétaire des poids GNN conservés.
Une exportation du scaler exige son split d'apprentissage correspondant.
L'empreinte des séquences inclut leurs valeurs, masques et horloges ; modifier
ces tableaux invalide les anciens splits/scalers. L'export vérifie les fenêtres
et les features des nœuds contre le panel d'origine, ainsi que les axes temps/
entités des événements. Un graphe provenant d'un autre panel est rejeté, même
si ses dimensions correspondent. Les tableaux contenant des objets sont
refusés avant toute écriture.

Fichiers immuables : même contenu rejoué identique ; contenu différent exige
un autre dossier. Publication atomique par fichier, manifeste publié après les
tableaux. Sans manifeste valide, pas de dataset complet. La lecture vérifie hash,
inventaire, types, shapes et allocations NPY/ZIP bornées, `allow_pickle=False`.

Budget `DatasetLimits` par défaut : 8 Mio cumulés pour tableaux/export + manifeste,
4 096 samples, 4 096 temps, 128 entités, 256 features, 8 192 edges par snapshot,
32 événements maximum par séquence. Les allocations sont contrôlées avant les
grands tableaux ; une exportation multimodale doit tenir dans le budget total.
Changer ces plafonds explicitement pour un besoin et un espace disque connus.

## CLI locale

`scripts/build_dataset.py --observations ...parquet --config ...json --output ...`
ne fait aucun accès réseau. Config : `decisions`, `entity_ids`, `features`
(spécifications registry), `lookback`, `stride?`, `labels?={price_feature,horizon}`,
`split?={train_end,validation_end,test_end?,embargo?}`, `scale?`.

Inputs supplémentaires : `--nodes/--edges` (graphes Lot 3),
`--events/--event-links`, `--enrichments` JSON facultatif et `event_config` pour
digest/dimension/horloge du modèle. Parquet source borné avant lecture à 20 000
lignes et taille décompressée limitée ; config et fichiers ont aussi un plafond.
Le manifeste conserve les SHA256 des fichiers et la configuration.

[Exemple complet, sans réseau](../examples/model_datasets.py) : données
synthétiques, labels, purge, scaler train et replay d'un export temporaire.

Reste opérationnel : collecter des inputs réels autorisés, qualifier leurs
horloges/couvertures/conventions, constituer l'univers historique et évaluer le modèle
choisi. Les accès premium et la qualification des données relèvent du projet
consommateur.

[Bilan de vérification Lot 5 : tests, packaging et limites](archive/lot5-verification.md).
