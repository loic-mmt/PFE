# Calcul ciblé et grands volumes

L'API par groupes reste disponible : `compute_features(..., groups=...)`,
`enrich_bars(...)` et `build_features(..., groups=...)` gardent leur comportement.
Le calcul indépendant est une API supplémentaire.

Les noyaux TPO/intrabar et callbacks glissants utilisent maintenant C++ par
défaut. Les mesures de cette page précèdent ce port ; voir [le benchmark
Python/C++](cpp.md) pour les comparaisons actuelles sur les mêmes sorties.

```python
from pfe import compute_feature, compute_selected_features, compute_features

rsi = compute_feature(bars, "technical.rsi", window=14)
selected = compute_selected_features(
    bars, ("returns.log", "statistics.close_mean_w20", "technical.rsi_w14")
)
# Même sélection possible dans l'API existante.
selected = compute_features(bars, names=("returns.log", "technical.rsi_w14"))
# Les groupes restent utiles lorsqu'on veut toute une famille.
technical = compute_features(bars, windows=(14,), groups=("technical",))
```

Les noms sont ceux du catalogue existant, fenêtres incluses (`_w14`). `window=`
ajoute ce suffixe à un nom sans fenêtre. `names=` et `groups=` sont exclusifs ;
avec `names=`, les fenêtres viennent des noms, et non de `windows=`. Les fenêtres
comptent les observations. Une feature rend une Series ; une sélection rend
exactement les colonnes demandées, sans les autres features de son groupe.

```python
rv = compute_feature(bars, "path.realized_variance", frequency="7min")
target = compute_selected_features(
    bars,
    ("path.realized_variance", "profile.tpo_poc", "technical.rsi_w14"),
    frequency="45min",
    segments=5,
    bins=10,
)
```

En mode agrégé, les fenêtres OHLCV portent sur les bougies cible. Les features
`path.*` / `profile.*` utilisent les bougies source. Le DataFrame ajoute toujours
`available_at` pour indiquer la disponibilité au close ; une Series expose cette
convention dans ses attributs. `aggregate_selected_features` permet aussi de
sélectionner des colonnes OHLCV/qualité et d'utiliser `partial='keep'` / `as_of`.
`multi_timeframe_features(..., names=...)` accepte la même sélection.

## Travail réalisé seulement à la demande

Les formules des groupes servent de source aux plans de dépendances : seuls les
nœuds nécessaires aux sorties choisies sont évalués. Les calculs intermédiaires
partagés sont mémorisés pendant l'appel et libérés après leur dernière
utilisation. Les caches persistants contiennent seulement des plans et des
métadonnées ; ils ne conservent aucune donnée d'entrée.

Les indicateurs TA-Lib utilisent leurs noyaux natifs. Un noyau multi-sorties
comme MACD calcule intrinsèquement ses trois valeurs, mais ses autres indicateurs
ne sont jamais appelés. Plusieurs sorties de ce même noyau partagent un seul
appel. Les régressions ciblées évitent les résidus lorsque seule la pente est
requise. Les WMA utilisent TA-Lib ; les récurrences personnalisées et boucles
numériques utilisent Numba. La première utilisation d'un noyau JIT peut coûter
plus cher ; les mesures ci-dessous séparent ce démarrage du traitement.

Les OHLCV cible, la variance réalisée, les semivariances et plusieurs colonnes de
qualité utilisent des réductions NumPy sur toutes les bougies, sans DataFrame par
bougie cible. Les autres descripteurs intrabar évaluent seulement leur plan local.
Les vérifications OHLCV utilisent des temporaires limités à 100 000 observations.
L'entrée ciblée doit être un OHLCV numérique, trié, unique, avec index temporel
avec fuseau : appeler `normalize_bars` au chargement si nécessaire.
`validate=False` saute seulement les contrôles numériques pour une entrée déjà
validée ; les contrôles d'index restent actifs.

## Mémoire bornée par lots

```python
from pfe import iter_bars, iter_features

batches = iter_bars("data/processed/QQQ.parquet", batch_size=100_000)
for features in iter_features(batches, ("returns.log", "technical.rsi_w14", "technical.ema_w20")):
    consume(features)  # écrire/traiter puis libérer chaque lot
```

`iter_bars` projette seulement OHLCV et timestamps du Parquet. `iter_features`
conserve le recouvrement nécessaire aux fenêtres finies et l'état exact RSI,
EMA et ATR entre lots. `gap_policy='reset'` réinitialise cet état aux trous, même
entre deux lots ; `allow` enchaîne les observations. Aucun recalcul de tout
l'historique, aucune approximation par une courte chauffe d'EMA/RSI.

Pour l'agrégation, fournir `frequency=` à `iter_features` : seule la dernière
bougie provisoire est retenue entre lots. Toutes les features intrabar/profil
sont disponibles, avec les mêmes conventions d'origine et de disponibilité.
Les features OHLCV cible disposent du même support que les features source.

**Limites explicites :** le streaming des fenêtres finies et des trois
récurrences ci-dessus est disponible. Les autres récurrences natives (MACD,
ADX…), les états infinis personnalisés, les features calendar/session et
complexity sont rejetés en streaming. Leur calcul individuel complet reste
possible. Les bougies cible en streaming exigent `partial='drop'`, sans `as_of`.
La mémoire dépend aussi des fenêtres demandées et de la durée cible : choisir
une fenêtre gigantesque ou une seule bougie englobant tout le fichier exige de
conserver son historique. Ne pas convertir le générateur en liste ni concaténer
tous les lots si l'objectif est de borner la mémoire.

```bash
uv run python scripts/build_features.py \
  --input data/processed/QQQ.parquet \
  --output data/processed/QQQ_features_7m.parquet \
  --frequency 7min \
  --features path.realized_variance technical.rsi_w14 \
  --batch-size 100000
```

Sans `--batch-size`, `--features` fonctionne aussi avec CSV ou Parquet en mémoire.
Les options historiques `--groups` et `--path-only` restent disponibles. L'écriture
par lots est atomique : un échec ne remplace pas le fichier de sortie existant.

## Mesures reproductibles

Mesures locales sur macOS arm64 / Python 3.12.14, données synthétiques,
fenêtre 14 ; processus séparés, exécution séquentielle. Temps de calcul après
initialisation ; RSS inclut Python, dépendances, entrée, sortie et temporaires.

| Cas | Lignes source | Colonnes | Temps | Pic RSS |
| --- | ---: | ---: | ---: | ---: |
| Groupe technical | 50 000 | 149 | 0,792 s | 442 Mio |
| RSI seul | 50 000 | 1 | 0,0010 s | 155 Mio |
| Groupe statistics | 50 000 | 170 | 14,285 s | 405 Mio |
| Moyenne close seule | 50 000 | 1 | 0,0011 s | 158 Mio |
| RSI seul | 1 000 000 | 1 | 0,019 s | 227 Mio |
| Moyenne close seule | 1 000 000 | 1 | 0,020 s | 229 Mio |
| Variance réalisée, cible 7 min | 1 000 000 | 1 | 0,028 s | 258 Mio |
| RSI, lots 100 000 | 100 000 000 | 1 | 6,509 s | 203 Mio |

La dernière mesure inclut génération des données synthétiques, validation,
calcul et consommation des sorties, sans conservation globale ni I/O disque.
Initialisation du noyau RSI : 0,077 s avec cache JIT disponible. Les autres
mesures excluent la création de l'entrée. Ces chiffres sont indicatifs : taille
des fenêtres, nombre de sorties, trous, disque et matériel changent le résultat.

```bash
uv run python scripts/benchmark_features.py --case group-rsi --rows 50000
uv run python scripts/benchmark_features.py --case selected-rsi --rows 50000
uv run python scripts/benchmark_features.py --case stream-rsi \
  --rows 100000000 --batch-size 100000
```

Sélectionner une feature supprime le travail sans rapport ; cela ne change pas
sa complexité mathématique. Sample entropy utilise des distances entre motifs,
certains descripteurs utilisent FFT/Haar ou des statistiques par fenêtre. Leurs
coûts dépendent de la fenêtre et peuvent rester élevés sur des centaines de
millions de lignes. Le benchmark RSI n'est pas une estimation de ces calculs.
