# Graphes : features disponibles, datasets GNN et travaux différés

Ces APIs calculent les features de graphes et préparent les artefacts GNN pour un projet
consommateur. Le calcul utilise les relations et observations fournies ; l’entraînement
et la collecte sont des étapes séparées.

## 1. Correctifs et raccordements réalisés

| Sujet | Livraison |
|---|---|
| Intraday | Comparaison des observations en timestamps UTC, y compris dates seules et offsets différents ; une valeur du jour n'est plus exclue par une comparaison avec une date sans heure |
| Identités | `entity_registry_from_security_master` crée un registre de snapshot ; `bind_graph_observations` relie des identifiants explicitement typés à l'émetteur et conserve l'instrument et la latence du mapping |
| Relations sans LLM | `import_graph_relations` importe des relations structurées explicitement qualifiées par l'appelant, avec versions, direction, preuves et poids inconnus conservés |
| Assemblage | `build_graph_dataset` rassemble entreprises, voisins et contextes, features numériques choisies et topologie ; les entités cibles restent séparées de l'univers de contexte |
| Pipeline commun | Étapes `graph`, `graph_bindings`, `graph_features`, `graph_pairs`, `graph_dataset`, avec caches, exports et budgets existants |
| Attributs d'arêtes | `GraphTensor.edge_attr`, `edge_attr_mask`, `edge_attr_names` : dépendances CA/approvisionnement, exposition, bornes, âges et qualification PIT séparés |
| Partitions GNN | `iter_graph_partitions`, `save_partitioned_graph_dataset`, lecture via `iter_saved_partitions` ; une sortie par décision du cœur, pas de duplication des contextes chevauchants |

Tables et tensors conservent les preuves et métadonnées. Le graphe décrit les relations
observées ; sa couverture est **partielle**. Les relations structurées et les faits LLM
revus ont des imports séparés.

## 2. Familles de features

`GRAPH_FEATURE_GROUPS` contient `degree`, `dependency`, `exposure`, `neighbors`,
`quality`, `channels`, `two_hop`, `topology`, `dynamics`. `groups` permet de
sélectionner les familles coûteuses.

### Fournisseurs, clients et concurrents

```python
features = pfe.compute_graph_features(
    decisions,
    history,
    entity_ids=companies,
    node_features=observations,
    node_feature_units={"tf[5min].returns.simple": "fraction"},
    groups=("channels",),
)
```

Namespaces `graph.neighbors.suppliers.*`, `.customers.*`, `.competitors.*` :
effectif, moyenne, écart-type population, minimum/maximum, couverture, âges,
masques et somme pondérée connue par métrique. Les fournisseurs sont les nœuds
amont du flux et les clients les nœuds aval, même si la source a décrit la relation
depuis l'autre entreprise. Les concurrents utilisent les relations explicitement
déclarées et une agrégation non pondérée.

Chaque poids garde son propriétaire et son dénominateur : CA du fournisseur ou
approvisionnements du client. Les poids manquants restent inconnus ; les poids connus
gardent leurs valeurs, sans renormalisation.

Par défaut, les voisins doivent partager la même période économique.
`require_same_observation=False` autorise explicitement les dernières valeurs
as-of de périodes différentes, avec le flag `mixed_periods` visible.
`node_max_age` borne l'âge de disponibilité ; `node_max_observation_age` borne
l'âge économique. Les horizons et unités doivent rester comparables.

### Deux sauts

`graph.two_hop.suppliers.*` / `.customers.*` : nombre de chemins, nombre de nœuds
uniques au second saut strict, chemins pondérés connus et couverture. Les nœuds
déjà directement reliés et le nœud lui-même sont exclus du second saut.

`known_path_coefficient_sum` additionne les produits de deux poids uniquement pour le
même type de dépendance et des périodes/scopes compatibles ; une période de poids
inconnue empêche ce calcul. Le résultat est un **coefficient mathématique de chemin**.
Il décrit les poids fournis ; des probabilités ou pourcentages économiques consolidés
demanderaient d'autres hypothèses sur leurs dénominateurs. Les chemins dont un poids
manque restent comptés dans la topologie.

### Voisins et expositions communes

```python
pairs = pfe.compute_graph_pair_features(
    decisions,
    history,
    pairs=[("company:A", "company:B")],
)
```

Sortie `(decision_at, left_entity_id, right_entity_id)` : voisins communs sur l'union
amont/aval, Jaccard, fournisseurs/clients communs ; cibles pays, matières, industries et
facteurs communes avec Jaccard. Le chevauchement de poids connus additionne les minima
seulement pour le même type de relation, scope et période connue. Le résultat mesure le
chevauchement des coefficients fournis. L’appelant sélectionne les paires à calculer.
Cette table peut être publiée par l'étape `graph_pairs`. Elle n'est pas ajoutée
automatiquement à chaque arête du tensor GNN.

### Centralités et PageRank

`graph.topology.*` : degrés entrants/sortants, degré total normalisé par `N-1`,
centralité harmonique, closeness avec correction pour nœuds inaccessibles,
betweenness dirigée normalisée, PageRank et indicateurs de convergence.

Par défaut : nœuds entreprises, flux fournisseur/client, arcs dirigés non pondérés,
preuves dupliquées réduites à un arc. `topology_node_types` et `topology_relations`
permettent une autre sélection explicite. Closeness/harmonic mesurent les chemins
sortants. PageRank : damping 0,85, masse des dangling nodes redistribuée uniformément,
tolérance L1 `1e-10`, 100 itérations au maximum. En cas de non-convergence, le score est
masqué et le flag est renseigné.

Les centralités portent sur les nœuds et relations du sous-graphe fourni.

### Évolution du réseau

`graph.dynamics.*_w30D`, avec `dynamics_windows=("7D", "30D")` configurable :
`new_edges`, `disappeared_edges`, `terminated_edges`, `weight_changes`.
Les transitions sont rejouées aux instants de connaissance, aux bornes économiques
connues et aux expirations. Une relation créée puis terminée entre deux décisions
est comptée. Une expiration/revue retirée est une disparition ; elle ne devient
pas une terminaison commerciale sans preuve. Une réouverture n'entraîne pas le
recomptage d'une ancienne terminaison lors de son expiration suivante.

Ces compteurs portent sur les états des arêtes déclarées ; des sources indépendantes
peuvent décrire le même lien. Une nouvelle connaissance d'une ancienne relation
n'est pas présentée comme sa date réelle de création commerciale.

## 3. Entrées et assemblage

### Import structuré

Champs requis par relation : `relationship_id`, `company_id`, `related_entity_id`,
`relation_type`, `source_id`, `revision_id`, `available_at`, `source_document_id`,
`evidence_quote`, `qualification_reference`. Ce champ référence la qualification
réalisée par l’appelant. Les nœuds doivent être déclarés explicitement dans
`EntityRegistry`.

`company_id` porte le rôle : `supplier` signifie qu'il fournit l'autre entreprise ;
`customer` signifie qu'il achète à l'autre. Les flux sont orientés fournisseur → client.
`manufacturer` et supplier sont deux types de relation séparés. Versions `existing` /
`terminated`, dates économiques facultatives, publication vérifiée facultative, poids
0–100 ou null. `available_at` ne précède jamais preuve ni résolution d'identité ; une
date de publication seule n'est pas transformée en heure. Révision identique → contenu
immutable et première disponibilité conservée.

```python
history = pfe.import_graph_relations(relation_records, entity_registry)
dataset = pfe.build_graph_dataset(
    decisions,
    observations,
    history,
    registry=feature_registry,
    entity_ids=target_companies,
    include_context=True,
    graph_feature_units={"graph.topology.pagerank": "score"},
    graph_feature_options={"groups": ("topology",)},
)
for graph in dataset.snapshots():
    consume(graph.x, graph.mask, graph.edge_index, graph.edge_attr, graph.edge_attr_mask)
```

Les contextes sont les autres nœuds connus avant la dernière décision, puis filtrés à
chaque snapshot selon leur disponibilité. Les nœuds sans feature disponible restent
présents avec masque faux. La matrice conserve les types de nœuds et un registre commun
de features. Elle peut être adaptée au framework GNN choisi. La normalisation reste
celle ajustée sur le train avec les APIs existantes.

`graph_feature_units` sélectionne les features structurelles à ajouter aux inputs. Leurs
disponibilités dérivées sont conservativement celles des décisions de calcul ; leurs
preuves hashent seulement les versions alors connues. La qualification PIT dépend des
preuves historiques des sources. Les features de paires restent une table séparée.

`edge_attr` comporte huit colonnes : les trois poids distincts en fractions,
bornes basse/haute de l'exposition en fractions, âge de preuve et de disponibilité
en secondes, flag PIT. Inconnus paddés à zéro avec masque faux. L'ancien
`edge_weight` scalaire reste manquant si plusieurs poids sont présents ; les
nouveaux canaux les préservent séparément. Nature, propriétaire, scope/période,
documents et citations restent dans les métadonnées de provenance.

Les identités n'effectuent pas de fusion automatique parent/filiale, ADR/action
ou dénomination ambiguë. `entity_registry_from_security_master` opère sur un snapshot
as-of ; la validité économique des alias d'`EntityRegistry` reste à précision date.
Les changements de symbole intraday demandent des registres de snapshot distincts.

### Partitions et pipeline

```python
parts = pfe.iter_graph_partitions(
    decisions,
    pfe.iter_parquet_observations(paths),
    history,
    entity_ids=companies,
    registry=feature_registry,
    partition_size=64,
    lookback=8,
)
pfe.save_partitioned_graph_dataset(
    directory,
    parts,
    history,
    max_archive_bytes=8_388_608,
)
```

Le flux commence aux observations canoniques déjà calculées et triées globalement par
`available_at`. Calculer les features OHLCV/graphes avant cette étape partitionnée. Le
panel est borné par partition, l'axe des décisions et l'historique de topologie restent
en mémoire. Les snapshots des chevauchements ne sont pas exportés deux fois. Reprise
immutable et vérification SHA256 par shard ; le manifeste complet est publié après
réussite de tous les shards.

Dans `run_pipeline`, `graph` prend `relations="source.relations"` et
`entities="source.entities"` ou un master explicitement sélectionné.
`graph_features` prend `history="graph"`, `observations="source.observations"`,
`decisions`, `entity_ids`, `feature_units`, `options`.
`graph_dataset` prend `history`, une liste de références `observations`, `directory`
et `config` : `decisions`, `entity_ids`, `features`, éventuellement
`graph_feature_units`, `graph_feature_options`, `partition_size`.
Les sorties nodes/edges et les tables de features sont publiables via `outputs`.

Le runner commun conserve ses tables d'étapes en RAM : il sert les petits runs
bornés. Utiliser l'API partitionnée avec des fichiers/batches externes pour les
grands panels. `GraphFeatureLimits` borne nœuds, arêtes, opérations, epochs, décisions,
paires, lignes d'historique et cellules de sortie. Les centralités et le replay
peuvent atteindre leur budget avant les plafonds individuels de nœuds/arêtes.

## 4. Vérification

Validation finale : **1 096 tests passent**, 13 exemples offline passent, Ruff valide le
code et le format des 211 fichiers contrôlés, `git diff --check` passe. Wheel et sdist
construits hors ligne avec les dépendances déjà en cache. Les 87 modules Python du wheel
correspondent aux sources ; import isolé, features du graphe et export/lecture GNN
vérifiés avec frameworks DL, LLM et NetworkX bloqués. Cette vérification ajoute
seulement l’extrait public décrit ci-dessous.

[Tests ciblés](../tests/test_graph_extensions.py) : valeurs manuelles sur chaînes,
directions des poids, données intraday et futures, alignement strict/as-of, stale,
centralités/PageRank, deux sauts, pays communs, changements intermédiaires,
terminaison/réouverture/expiration, immutabilité, contexte et attributs GNN,
export partitionné, pipeline avec cache/replay. L'exemple
[`graph_features.py`](../examples/graph_features.py) est synthétique et sans réseau.

Un court [extrait public vérifié](../tests/fixtures/graph/public_relation.json) reprend
une relation de fabrication Blackwell/TSMC, d'après le [communiqué NVIDIA du 18 mars
2024](https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing/).
L'extrait a été consulté le 14 octobre 2026. Sa publication garde sa précision date, sa
disponibilité est celle du contrôle actuel et `point_in_time=False`. Le poids de
dépendance est inconnu. Le contrôle porte sur cette relation ; la couverture du réseau,
la synchronisation des prix et la qualité LLM demandent des validations séparées.

## 5. Backlog figé : prérequis externes, non réalisés

| ID | Travail différé | Prérequis / critère pour reprendre |
|---|---|---|
| G-D01 | Choisir et qualifier un meilleur LLM pour relations/expositions | Modèle fourni/installé explicitement, corpus réel annoté, précision/rappel/direction et revue contrôlés. Le backend générique existe ; aucun autre modèle installé ou testé ici |
| G-D02 | Couverture réelle des fournisseurs/clients/concurrents et poids | Rapports/corpus sélectionnés ou export autorisé ; extraction/validation, dénominateurs, périodes et suivi. Les champs non publiés restent inconnus |
| G-D03 | Historique industriel véritablement PIT | Archives de relations et identités avec changements/horaires disponibles à l'époque ; le snapshot courant ne les reconstruit pas |
| G-D04 | Correspondances ambiguës d'entités et taxonomies mondiales | Crosswalk documenté pour filiales, ADR/classes, pays, industries et matières ; validation des cas ambigus. Raccords explicites disponibles dans le code |
| G-D05 | Connecteur natif de réseau industriel propriétaire | Spécification et exemple légal du produit retenu, accès/licence et conventions historiques. Endpoint et droits Revere à configurer selon le contrat retenu |
| G-D06 | Qualification end-to-end sur un univers réel représentatif | Univers, données synchronisées et vérité terrain sélectionnés ; vérification couverture/poids/séances. Le petit extrait public et les données synthétiques ne remplacent pas ce corpus |
| G-D07 | Benchmark grande échelle et optimisation incrémentale ciblée | Volumétrie et contraintes du consommateur définies, données et machine disponibles ; les calculs présents sont bornés et refusent les dépassements |

Ce backlog **différé** porte sur les données et la qualité du modèle. Un premier dataset
peut utiliser les relations structurées et observations déjà qualifiées.