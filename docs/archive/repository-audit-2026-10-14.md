# Revue du dépôt — 14 octobre 2026

> Archive historique : résultats et limites au moment de cette vérification.
> Pour l’état actuel, consulter la [roadmap](../market-data-roadmap.md#9-travail-restant-et-méthode)
> et les documents d’API qui y sont liés.

**14 bugs reproduits et corrigés. 1 044 tests passent, dont 74 régressions ajoutées.**
Le passage couvre le checkout courant, fichiers non suivis compris, et les Lots
0–5. Les modifications déjà présentes ont été conservées. Aucun commit, collecte
live, ajout de dépendance ou téléchargement de modèle n'a été effectué.

## Standards

Trois violations confirmées des contrats documentés dans
[development.md](../development.md) : bars normalisées acceptées par les APIs,
valeurs manquantes préservées et complétude explicite des collectes.

| Bug reproduit | Correction | Régression |
|---|---|---|
| **P1** : index UTC stocké en microsecondes accepté par l'API sélectionnée mais rejeté par l'API groupée ; calculs de grille comparant microsecondes et nanosecondes | `normalize_bars` produit un index UTC nanoseconde, sans modifier les instants ; calcul groupé et agrégation cohérents | `test_bars.py`, `test_aggregation.py` : secondes, millisecondes, microsecondes et nanosecondes |
| **P2** : `period_start=np.nan`, `pd.NA` ou `NaT` déclenche un `TypeError` pour revenue/EPS/assets | Périodes manquantes canonicalisées en `None` ; niveau conservé, croissance sans période comparable masquée | `test_fundamentals.py` : 12 combinaisons métrique/représentation nulle |
| **P2** : transports LSE acceptant HTTP 206 ou un corps tronqué, pouvant laisser qualifier une collecte incomplète comme complète | Validation partagée du statut HTTP, Content-Range et Content-Length ; erreurs de lecture sans corps ni clé, réponses fermées | `test_lse_transport.py` : 28 cas metadata/rows, troncature, limites, fermeture et confidentialité |

Autre vérification : comparaison synthétique de **1 195 features streaming** avec le
calcul complet, fenêtres 2/3/7, lots irréguliers, trous et politiques reset/allow. Les
sorties concordent à la tolérance `1e-8`. Les changements portent sur les bugs
reproduits.

## Spec

Onze défauts confirmés par rapport aux contrats de chronologie, preuves,
comparabilité et export des Lots 3–5.

| Bug reproduit | Correction | Contrat et tests |
|---|---|---|
| **P1** : annonce future permettant d'utiliser une estimation après la décision ; ajouter une donnée future changeait la surprise de 0,5 à −96,5 | Annonce exigée au plus tard à la décision ; actual disponible avant son annonce refusé à l'import ; invariance future testée | [P3](../p3-sources.md), `test_premium_sources.py` |
| **P1** : secret dans un CSV refusé par le parseur, mais déjà écrit dans l'archive raw | Prévalidation CSV/JSON avant toute persistance du corps ; CSV autorisés toujours acceptés | [P3](../p3-sources.md), `test_premium_sources.py`, captures offline et client injecté |
| **P2** : nouvelle revue d'un événement du 2 janvier le faisant apparaître récent après revue le 20 janvier | Première disponibilité de l'événement et de son lien à l'entité conservée ; changement de revue ne remet pas l'âge à zéro | [Documents](../document-features.md), `test_graph_events.py` |
| **P2** : variation hebdomadaire COT calculée avec long/short précédents appartenant à des révisions différentes | Même révision exigée aussi pour le rapport précédent ; variation incompatible masquée | [P3](../p3-sources.md), `test_premium_sources.py` |
| **P2** : revenu TSMC 100 puis marqueur explicite « − » conservant le niveau 100 | Null historisé pour un mois achevé à Taïwan ; cellules vides et futurs mois non publiés restent exclus | [International](../international-features.md), `test_international.py`, retrait et frontière locale de fin de mois |
| **P2** : comptes DART Jan–Dec puis Apr–Mar chevauchants comparés comme deux années civiles, donnant un faux YoY de 50 % | Fréquence annual uniquement pour Jan1–Dec31 d'une même année ; autres périodes fiscales fournissent des niveaux, sans comparaison civile inventée | [International](../international-features.md), `test_international.py` |
| **P2** : deux instruments du même émetteur donnant un prix 10 ou 100 selon l'ordre des lignes | `FeatureSpec.instrument_id` permet une sélection exacte ; ambiguïté rejetée et identité conservée dans les preuves/passerelles | [Datasets](../model-datasets.md), `test_datasets.py` |
| **P2** : `quality.feature_eligible="False"` devenant vrai et `pd.NA` provoquant une erreur | Booléens explicites exigés ; absence du flag traitée comme son défaut documenté ; aucune conversion par vérité Python | [Datasets](../model-datasets.md), `test_datasets.py` |
| **P2** : modification des inputs, masques ou horloges d'une séquence sans modification de son empreinte | Empreinte couvrant tous les tableaux ; anciens splits/scalers invalidés après modification | [Datasets](../model-datasets.md), `test_datasets.py` |
| **P2** : export complet mélangeant panel, fenêtres modifiées, graphe d'un autre panel ou axes événements incompatibles | Vérification provenance, identités, fenêtres, valeurs et horloges contre le panel ; rejet avant écriture | [Datasets](../model-datasets.md), `test_datasets.py`, `test_structured_datasets.py` |
| **P2** : export de tableaux `object` créant un NPZ nécessitant pickle, incompatible avec son propre lecteur | Tableaux contenant des objets refusés avant publication | [Datasets](../model-datasets.md), `test_datasets.py` |

## Vérification finale

| Contrôle | Résultat |
|---|---|
| Suite avant corrections | 970 tests passés ; les nouveaux cas n'étaient pas couverts |
| Suite après corrections | **1 044 tests passés en 54,29 s** |
| Lint / format | Ruff OK ; 180 fichiers formatés |
| Diff | `git diff --check` OK |
| Exemples offline | 11 passés ; exemple international rejoué après ses corrections |
| Build offline | Wheel natif + sdist reconstruits, uniquement depuis les dépendances déjà en cache |
| Wheel vérifié | `pfe-0.1.0-cp312-cp312-macosx_26_0_arm64.whl`, 389 138 octets ; sources Python identiques au checkout |
| Sdist | 786 767 octets |
| Import isolé du wheel | Import public et natif, bars microsecondes et pipeline datasets OK ; checkout editable exclu, SDK optionnels et frameworks DL bloqués |
| Environnement | Python 3.12.14, NumPy 2.3.5, pandas 2.3.3, macOS arm64 |

Commandes de vérification :

```sh
PYTHONPYCACHEPREFIX=/tmp/pfe-lot2-verification/pycache \
  /tmp/pfe-lot2-verification/venv/bin/python -m pytest -q
/tmp/pfe-lot2-verification/venv/bin/ruff check .
/tmp/pfe-lot2-verification/venv/bin/ruff format --check .
git diff --check
uv build --offline --python /tmp/pfe-lot2-verification/venv/bin/python
```

Les scénarios explorés ont confirmé les bugs listés ci-dessus. Les tests utilisent des
données synthétiques et des transports injectés ; les données fournisseur, droits,
couverture PIT et précision du modèle local demandent la qualification décrite dans la
[roadmap](../market-data-roadmap.md). Python 3.13/3.14 reste à tester localement.

Standards : **3 corrigés**, principal risque corrigé : précision temporelle UTC.
Spec : **11 corrigés**, principaux risques corrigés : fuite temporelle de surprise
et persistance prématurée de secrets CSV.
