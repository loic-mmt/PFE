# Documents, extraction locale, graphes et événements

La bibliothèque archive les documents, extrait les contrats A–D et conserve les versions
des faits pour calculer les features à chaque instant de décision. Les tests utilisent
des textes synthétiques et des transports injectés.

Les [besoins et sources](market-data-roadmap.md) et les
[prompts utilisateur originaux](llm-extraction-prompts.md) restent la référence.
Les fondamentaux XBRL déjà structurés passent par le [Lot 2](context-features.md),
avant d'utiliser un LLM pour guidance, relations, expositions et articles.

## APIs

| API | Rôle |
|---|---|
| `DocumentStore`, `DocumentLimits`, `SourceDocument` | TXT/HTML, couche texte PDF facultative, URL explicite, un filing SEC sélectionné, hash/versions/droits/horloges |
| `Extractor`, `ExtractionLimits`, `ExtractionResult` | JSON importé hors ligne ou backend injecté ; chunks, cache, réponses brutes, rejets et état de complétude |
| `extraction_schema`, `extraction_messages` | Schémas et prompts A–D versionnés, tous les champs originaux conservés |
| `OllamaBackend` | Modèle GGUF local déjà installé et identifié par digest, inférence facultative |
| `adjudicate_fact`, `evaluate_extractions` | Revue sémantique nommée, précision/rappel, erreurs par champ et seuils explicites |
| `EntityRegistry`, `build_graph_history`, `graph_asof` | Alias explicites, graphe partiel orienté, versions de preuve et validité économique |
| `merge_graph_declarations` | Relations déterministes issues de référentiels qualifiés : industrie, facteurs, pays, matières |
| `build_event_history`, `events_asof` | Events distincts, versions, statuts, liens aux entreprises et identité explicite entre documents |
| `compute_graph_features` | Degrés, dépendances connues, expositions, joins voisins, couverture et masques |
| `compute_event_features` | Comptes/statuts/fenêtres, âge, magnitudes compatibles, sentiment et embeddings facultatifs |
| `statements_frame`, `compute_statement_features` | Résultats/guidance/estimations séparés, valeurs/plages dans leurs unités originales |

Toutes ces API sont accessibles depuis `pfe`. Les familles `graph`, `events` et
`statements` apparaissent dans `feature_catalog()` ; passer un résultat au
catalogue donne ses colonnes exactes.

## Budgets et disque

| Limite par instance/exécution | Défaut |
|---|---:|
| Document brut | 512 KiB |
| Texte UTF-8 parsé | 256 KiB |
| Documents/requêtes de collecte | 20 |
| Payloads documents bruts + texte/métadonnées | 8 MiB |
| Téléchargements cumulés d'une instance | 8 MiB |
| PDF | 20 pages, couche texte uniquement |
| Chunk / chevauchement | 12 000 / 400 caractères |
| Chunks par document | 32 |
| Inférences/imports JSON non cachés | 64 |
| Réponse d'extraction / faits par réponse | 64 KiB / 100 |
| Payloads d'archive d'extraction | 8 MiB |
| Embedding | 2 048 dimensions maximum |
| JSON d'entrée/sortie CLI | 4 MiB |
| Parquet d'entrée/sortie CLI | 8 MiB par fichier |
| Grille de features CLI | 512 décisions × 32 entités maximum |

Les manifests et index ajoutent un petit coût de métadonnées. L'extracteur réserve une
marge avant chaque appel pour le JSON échappé et les faits validés. Un dépassement
provoque une erreur. La collecte traite les documents demandés individuellement ;
crawling et archives ZIP SEC sont hors périmètre. Modèle, OCR et dépendances doivent
être installés séparément. La CLI écrit les fichiers demandés et conserve les sorties
brutes immuables. Les dataclasses de limites permettent d'ajuster les budgets au
périmètre choisi.

TXT/HTML utilisent la bibliothèque standard. L'import PDF exige `pypdf` et signale son
absence. Les tests PDF utilisent un lecteur injecté. Tableaux complexes et scans
demandent un prétraitement séparé, avec OCR si nécessaire.

## Contrats et horloges

Les fichiers structurés exposent `entity_id`, `observation_date`, `published_at`,
`available_at`, `value`, `source_id`, `revision_id`. Les schémas documentaires,
graphes et events autorisent une période économique inconnue (`null`), plutôt que
la remplacer par une date de publication. Leur `value` contient des références
d'archive ou un `record_json` à l'intérieur d'un objet structuré uniforme pour
Parquet ; le JSON conserve aussi les plages et les champs A–D complets.
Les résultats financiers ont un `value` numérique nullable et des bornes séparées.
Ces schémas ne sont pas directement des observations scalaires du Lot 0.

Un document garde son corps original, son texte parsé, hashes, URL, `logical_id`,
droits, version de parsing, première réception, publication explicitement fournie et
éventuelle acceptation SEC. Même contenu/métadonnées → même version et première
réception conservée. Contenu ou métadonnées modifiés → nouvelle version. Acceptation SEC
et publication sont deux horloges distinctes. La date `filed` conserve sa précision au
jour.

Le défaut est conservateur :

```text
document.available_at = première réception
fait.available_at    = réception de la réponse d'extraction
fait revu.available_at = instant de la revue
edge/link.available_at = max(fait, identités nécessaires)
embedding/sentiment.available_at = réception du calcul local
```

Un article devient utilisable après sa réception, son traitement et la revue requise,
même si sa publication est ancienne. Les snapshots passés restent stables lorsque de
nouveaux documents arrivent. Reconstituer les latences historiques demanderait une
politique distincte et qualifiée.

Les alias demandent `alias`, `entity_id`, `node_type`, `available_at`. Types :
`company`, `country`, `industry`, `commodity`, `factor`. `valid_from`/`valid_to` sont
économiques, sous forme de dates, intervalle `[from,to)`. Sans période économique
explicite d'une relation, la liaison utilise sa date de connaissance. Le premier mapping
compatible fixe l'identité ; ambiguïté → issue, pas de jointure floue. Un mapping connu
plus tard retarde l'arête/le lien. Les pays/matières sans mapping reçoivent des IDs
dérivés de leur libellé exact. Les synonymes, comme « US » et « USA », exigent un
mapping explicite.

La provenance d'identité suit le Lot 0 : `point_in_time=False` et
`availability_basis="unknown"` par défaut. Une assertion PIT explicite doit être
accompagnée de sa politique ; `receipt` exige `ingested_at`. Les features graphe,
events et statements conservent la qualification de leurs mappings.

## Extraction et qualification

`financial-extraction-1` / `financial-prompts-1` :

- A : enveloppe `relationships`, rôle de `source_company` par rapport à
  `related_entity`, `effective_status` supplémentaire. Fournisseur A de B → A→B ;
  client A de B → B→A. Une entité anonyme conserve `null` comme identité.
- B : `exposures`, pourcentage scalaire ou objet `lower_bound`/`upper_bound`.
  Siège ≠ revenus ; géographie de production et vente distinctes.
- C : `statements`, `fiscal_period_label` supplémentaire. Actual, guidance et
  estimation restent distincts. Dates ISO ou dates complètes explicitement écrites
  en anglais/français peuvent être normalisées ; un trimestre/« FY2026 » seul laisse les bornes de période inconnues.
  Les plages et unités sont conservées telles quelles.
- D : `events`, publication prouvée dans le texte ou métadonnées explicitement
  fournies, sinon `null`. Mécanisme économique absent → `null`.

Validation : enveloppe/champs/types/enums stricts, JSON sans clés dupliquées ni nombres
non finis, document/hash, citation exacte dans le chunk, offsets/page, attribution
explicite, présence des nombres/units/périodes et preuve en % pour les
dépendances/expositions. Contradictions anglaises évidentes de direction,
guidance/consensus et rumeur sont rejetées. Les descriptions et mécanismes extraits
doivent figurer dans la preuve citée.

Après validation de la citation, les faits sont `unreviewed` jusqu'à la revue
sémantique. `adjudicate_fact` enregistre un reviewer, notes, horodatage et verdict après
contrôle des champs, dénominateurs, attribution, direction et périodes. Par défaut,
graphes/events et features financières utilisent seulement les faits approuvés. Le mode
API `require_review=False` permet une exploration explicitement provisoire ; les statuts
restent visibles. Les outputs sentiment/embedding sont des calculs de modèle, séparés
des faits financiers.

Réponses brutes et rejets sont archivés, même pour citation invalide ou JSON
inexploitable. `complete` décrit l'achèvement des appels/chunks ; il ne mesure ni rappel
ni correction sémantique. Une liste vide validée, un rejet et un appel incomplet sont
des états distincts. Une erreur de transport ou d'inférence laisse un manifest incomplet
; seule une liste vide validée indique l'absence d'événement extrait.

Cache : identité du document/hash/disponibilité, tâche, offsets du chunk,
versions de prompt/schéma, backend/modèle/digest/options et limites. Rejouer une
entrée complète conserve le premier traitement. Après interruption entre réponse
et résultat, reprise de la réponse stockée sans nouvel appel au modèle.

## Ollama facultatif

Le modèle doit déjà être présent. `model_revision` est le SHA256 complet renvoyé
par [`GET /api/tags`](https://docs.ollama.com/api/tags). Le backend contrôle ce
digest avant/après inférence et consulte les informations GGUF/context du modèle
via [`POST /api/show`](https://github.com/ollama/ollama/blob/main/docs/api.md#show-model-information).
Endpoint HTTP loopback, proxies/redirections désactivés ; alias cloud refusés.

[`POST /api/chat`](https://docs.ollama.com/api/chat) utilise `format=schema`,
`stream=False`, température 0, sortie plafonnée, `keep_alive=0` et aucun outil. Contexte
demandé ≤32 768 tokens et ≤contexte déclaré du modèle. Un contrôle conservateur des
octets UTF-8, template, système et réserve de sortie refuse un prompt trop grand avant
inférence. Cette borne contrôle la taille du contexte. Le rappel du modèle et le
comportement de son template/tokenizer doivent être évalués sur un corpus annoté.
`done_reason` différent de `stop` → sortie incomplète en quarantaine. Les [structured
outputs](https://docs.ollama.com/capabilities/structured-outputs) passent ensuite par la
validation indépendante.

[`POST /api/embed`](https://docs.ollama.com/api/embed) utilise `truncate=False` :
entrée trop grande → erreur du serveur, plutôt qu'embedding d'un texte écourté.
Enrichissement d'un document court seulement ; digest, dimension, hash du texte et
horloge conservés. Un seul digest et une dimension configurée pour agréger les
embeddings ; les scores sentiment sont qualifiés comme outputs du modèle.

## Graphes et features

Extension du 14 octobre : [graphes et datasets GNN](graph-features.md).
Intraday corrigé, canaux fournisseurs/clients/concurrents, chemins à deux sauts,
voisins/expositions communes, centralités/PageRank, évolution du réseau, import
structuré sans LLM, raccord master et exports partitionnés. Le backlog externe
et la qualification d'un meilleur LLM sont explicitement différés dans ce document.

Les arêtes gardent le rôle original, l'entreprise propriétaire de la dépendance,
le flux orienté, citation/document, versions et poids nullable. Les relations
incertaines/plans n'activent pas le graphe confirmé. Une absence de mention ne
ferme pas une relation. Fermeture sur terminaison documentée ou `effective_to`
explicite ; expiration uniquement avec `max_age` choisi par l'appelant.
Une rétractation de revue retire cette preuve, sans supprimer une autre preuve
approuvée. L'ordre des états utilise la date économique explicite ou celle de la
source, puis sa publication/réception originale : revue tardive d'une ancienne
preuve ≠ réactivation d'une relation terminée. `max_age` porte sur la preuve source.
Un snapshot économique passé sans début explicite exclut les états seulement
observés plus tard. Les historiques complets restent archivés.

Degrés par rôle/relation et `supply_flow` dédupliqué, membres d'industrie, pays,
matières et facteurs. `known_weight_sum` et `concentration_known = sum((pct/100)^2)`
décrivent uniquement les poids connus, sans les renormaliser. Scopes produits ou
périodes incompatibles → masque/NaN ; poids contradictoires simultanés → NaN, avec
toutes les références de preuve. Une plage n'est pas réduite à son milieu. Couverture
sur les relations déclarées, poids dépassant un total de 1 et masque de données sont
visibles. Le graphe décrit les relations observées : un degré nul indique qu'aucune
relation n'est connue dans les entrées fournies. Les dénominateurs et le contexte
temporel des poids doivent être vérifiés lors de la revue.

Les joins voisins attendent des observations du Lot 0 et une configuration
`node_feature_units={"momentum_20":"fraction", ...}` : les noms doivent préciser
l'horizon. Pas de mélange de sources ambiguës, unités ou dates économiques ;
périodes de marché décalées → NaN et masque. Moyenne des voisins disponibles et
somme pondérée des seuls coefficients connus, couverture et âge. Les métriques
prix/volatilité se calculent avec les Lots 1–2 avant leur conversion en observations.
`neighbor_node_types` et `neighbor_relations` permettent aussi de propager des
inputs pays/industrie/matière/facteur explicitement liés ; coefficients d'exposition
connus, sans renormalisation. Âge source/économique et qualification des voisins
sont visibles, séparément de la qualification topologique.

Les déclarations déterministes de référentiels utilisent `merge_graph_declarations` :
observations qualifiées, unité `relationship`, `metric="graph.<relation>.<target_id>"`,
`value={target_entity_id, relation_type, active, weight_pct}`, citation et document
source. Relations autorisées : `industry_member`, `factor_exposure`, `country_exposure`,
`commodity_exposure`. Fermetures et révisions conservent leur historique ; les poids
proviennent des déclarations qualifiées.

## Events et features financières

Une extraction event strictement identique est dédupliquée. Fusionner deux extraits
exige une équivalence démontrée. Des produits ou magnitudes inconnus ne suffisent pas à
identifier deux lancements comme le même événement. `event_ids={fact_id: event_id}` fixe
explicitement les équivalences entre articles ou extraits. Tous les documents/versions
restent traçables.

Les fenêtres utilisent la première disponibilité de l'event pour l'entité. Une nouvelle
publication explicitement liée au même event ne remet pas son âge à zéro.
`publication_age_seconds` reste séparé de l'âge de première disponibilité ; publication
inconnue → NaN. Statuts `confirmed`, `announced_plan`, `rumor` séparés pour comptes et
magnitudes ; magnitudes agrégées seulement avec type/unité configurés.
Sentiment/embeddings agrègent les articles disponibles dans le corpus, avec
modèle/hash/horloge compatibles. Zéro compte indique seulement le corpus fourni.

Les statements ont leur propre namespace. Chaque specification exige `metric`,
`statement_type`, `currency`, `unit_scale`, `segment_name`, et peut fixer une période
fiscale ou une source. Une révision tardive d'un ancien actual ne remplace pas la
période la plus récente. La priorité des statements utilise leur publication ou
réception originale ; une revue tardive n'actualise pas un ancien statement. Âges de
disponibilité, source et période fiscale sont séparés. Guidance et estimations exigent
une sélection de période et de source ; l'ambiguïté produit NaN. Le calcul de surprise
demande un consensus historique qualifié, fourni séparément des estimations
individuelles.

## Commandes

Exemple complet synthétique, sans réseau ni modèle :

```bash
python examples/document_features.py
python scripts/text_pipeline.py --help
```

Importer un document local (512 KiB maximum) :

```bash
python scripts/text_pipeline.py import rapport.txt --logical-id acme-report \
  --source-id company_ir --rights "usage local à vérifier"
```

Le programme affiche `document_id` et crée l'index/metadata Parquet. La commande `url`
demande un URL précis, un `--allowed-host` et un `--user-agent` ; `sec` demande
CIK/accession/nom du document et User-Agent identifié avec email. Elles collectent
uniquement les documents désignés. Une instance réutilise le transport et son throttle
(0,15 s entre départs) ; le client SEC reste sous 10 requêtes/s sur cette instance.

JSON hors ligne pour une tâche/un document :

```bash
python scripts/text_pipeline.py extract --document-id DOCUMENT_ID \
  --tasks A --response-json relations.json
```

Ollama déjà installé et modèle déjà présent, digest obtenu séparément :

```bash
python scripts/text_pipeline.py extract --document-id DOCUMENT_ID \
  --model MODELE_LOCAL --model-revision SHA256_COMPLET \
  --chunk-chars 1500 --num-predict 1024 --tasks A B C D
```

Découpage plus petit pour petits contextes ; document trop long pour le plafond
de chunks → choisir des sections locales explicitement. Pour sentiment/embedding,
utiliser `enrich --kind sentiment|embedding` avec modèle/digest adaptés.

Fichier de revue JSON :

```json
[{"fact_id":"fact:...","approved":true,"reviewer":"nom",
  "notes":"Attribution, direction, valeur, dénominateur et période vérifiés."}]
```

```bash
python scripts/text_pipeline.py review --facts data/extraction/facts.json \
  --reviews reviews.json
python scripts/text_pipeline.py build --facts data/extraction/facts.json \
  --entities entities.json --decisions decisions.json --entity-id COMPANY_ID
```

`entities.json` est la liste d'alias historiques décrite plus haut.
`decisions.json` est une liste de timestamps aware croissants, uniques.
Options : `--event-ids`, `--graph-declarations`, `--node-features`,
`--node-feature-units`, `--neighbor-relations`, `--neighbor-node-type`,
`--statement-specifications`, `--enrichments`,
`--model-revision`, `--embedding-dimensions`.

Sorties :

```text
data/metadata/source_documents.parquet + documents.json
data/raw/documents/                 corps, texte, manifests immuables
data/raw/extraction/                réponses brutes, faits/rejets, cache
data/extraction/facts.json          faits et revues append-only
data/extraction/rejected.json + runs.json
data/graph/nodes.parquet + edges_history.parquet + node_features.parquet
data/events/events_structured.parquet + event_entity_links.parquet
data/events/event_features.parquet
data/events/event_embeddings.parquet        si demandé, vectors en value_json
data/fundamentals/company_statements.parquet
data/fundamentals/management_guidance.parquet
data/fundamentals/analyst_statements.parquet
data/fundamentals/statement_features.parquet si demandé
data/metadata/graph_event_issues.json
```

`company_facts.parquet` du Lot 2 et `analyst_estimates_pit.parquet` de futurs
snapshots de consensus ne sont pas écrasés par des extractions textuelles.
Index faits/revues sans écrasement de versions conflictuelles ; Parquet dérivé
régénéré atomiquement à partir de l'historique complet fourni.

## Vérification et suite

[Rapport historique du Lot 3](archive/lot3-verification.md). Pour les contrôles
consolidés et leurs limites, voir [la vérification des extensions](immediate-verification.md).

`tests/fixtures/extraction_corpus.json` : 18 cas synthétiques annotés, 11 faits
attendus, cas anonymes/plans/plages/classes financières/rumeurs/absence/instructions
source et cinq contradictions. Le replay JSON valide les mécanismes de précision
et rappel ; il n'évalue pas un LLM réel. L'évaluateur compare tous les champs,
preuve/document inclus ; doublons comptés comme faux positifs, corpus vide non qualifié.

```bash
python scripts/text_pipeline.py evaluate --predicted predicted.json \
  --expected annotated.json --minimum-precision 0.99 --minimum-recall 0.90
```

Les seuils sont choisis par l'utilisateur pour son usage, sans valeurs implicites.
Échec → code retour 2. Chaque enregistrement d'évaluation a `case_id`, `task`,
`record`. Pour qualifier un modèle : annoter un échantillon représentatif de vrais
filings/news, exécuter le digest/options/prompts figés, mesurer erreurs par champ,
direction et classe, puis contrôler les exemples rejetés et manquants.

Tests couvrent invariance au futur, alias tardifs/ambigus, arêtes orientées,
terminaison/revue, poids manquants/plages, unités/périodes de voisins, events
distincts/doublons/horloges, versions, contexte/digest/localité Ollama, reprise,
JSON/citations/champs invalides, budgets, outputs facultatifs et CLI Parquet.

Avant d'utiliser des données réelles, qualifier leurs droits, publications et identités,
puis évaluer le modèle sur un corpus annoté. Voir aussi les [sources internationales et
GDELT](international-features.md) et les [datasets, splits et
embeddings](model-datasets.md). Une reconstruction des latences historiques demande une
politique séparée des captures effectives.