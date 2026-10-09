# Noyaux C++ et benchmark

L'API Python et le calcul par groupes restent disponibles. Une installation
compilée utilise désormais C++ par défaut pour ces parties :

- reconstruction TPO et tous les descripteurs de profil (POC, value area,
  concentration, transitions, présence par prix/segment…) ;
- parcours intrabar : drawdown/drawup et récupération, runs, régressions,
  corrélations, segments temporels, volume et résolutions internes ;
- callbacks glissants : skew/kurtosis, MAD, régression, âge des extrema,
  drawdown maximal, runs, sommes pondérées (ALMA, WMA, HMA, pente des transforms) ;
- descripteurs de complexité : sample/permutation/sign entropy, crossings,
  Petrosian, Hurst, expected shortfall, tail fraction, pente et détails Haar.

Les rolling mean/min/max/quantile/correlation déjà vectorisés restent dans
pandas/NumPy, les indicateurs TA-Lib dans leur bibliothèque native, et les
récurrences Numba restent disponibles. Les FFT et ajustements AR/PACF gardent
leurs implémentations NumPy/LAPACK. Leur parcours de fenêtres Python n'a pas été
remplacé par un moteur FFT C++ dans cette modification.

## Architecture et installation

`cpp/native.cpp` contient les noyaux C++17. `pybind11` échange les buffers NumPy ;
`scikit-build-core` et CMake construisent une wheel native spécifique à la
plateforme et à la version Python. Compilation Release, sans fast-math, avec
contraction FMA désactivée sur Clang/GCC pour limiter les changements numériques.
Les boucles C++ libèrent le GIL. Aucune création de DataFrame ni callback Python
par fenêtre dans les noyaux portés.

```bash
uv sync --extra lse --extra plots --reinstall-package pfe
uv build
```

Un compilateur C++17 est nécessaire pour construire depuis les sources : outils
Xcode sur macOS, GCC/Clang sur Linux, MSVC sur Windows. Les dépendances de build
Python sont installées automatiquement par uv. Le checkout conserve `venv/` et
le lien `.venv` utilisés précédemment. Aucune bibliothèque C++ externe de calcul
n'est requise. Les builds et binaires générés restent dans les dossiers ignorés.

```python
from pfe import compute_feature, feature_backend

# C++ par défaut ; signature et noms inchangés.
mad = compute_feature(bars, "statistics.close_mad", window=20)

# Référence, pour comparer ou déboguer ; contexte local au thread/task.
with feature_backend("python"):
    reference = compute_feature(bars, "statistics.close_mad", window=20)
```

`feature_backend("cpp")` échoue explicitement si l'extension n'est pas installée.
Un checkout importé sans build peut utiliser les références Python ; une
installation normale construit l'extension. Les wrappers vérifient types,
longueurs et frontières des buffers avant accès aux pointeurs.

## Calcul ciblé et gros fichiers

L'agrégation ciblée passe tous les groupes à un unique appel C++, au lieu de
construire un DataFrame et de rappeler Python pour chaque bougie. Chaque groupe
évalue les noms demandés, avec caches locaux pour ses primitives partagées. La
mémoire de travail des parcours est proportionnelle à la bougie cible ; celle
des fenêtres glissantes à leur longueur. Les sorties restent proportionnelles
au nombre de lignes et de features. `iter_bars` / `iter_features` continuent de
borner la mémoire par lots avec les mêmes garanties d'état et de disponibilité.

Les inputs float64 contigus sont transmis sans copie. Un autre dtype ou un
stride incompatible nécessite une conversion locale par pybind11. Les kernels
ne modifient jamais l'entrée. Les NaN de chauffe et les gaps restent présents.

Deux exceptions conservent la sémantique NumPy antérieure :

- `path.top3_moves_volume_share` utilise encore le tri NumPy, dont l'ordre des
  égalités dépend de sa version/plateforme. Aucun tri différent n'est substitué.
- Pour `structure.triangle_convergence`, les pentes presque égales sont
  recomparées avec les dot-products NumPy d'origine. Cela évite qu'un epsilon de
  rounding inverse une décision binaire, notamment sur une série plate.

Les moments glissants reproduisent l'ordre de réduction float64 de NumPy pour
les fenêtres constantes, afin de préserver également ses conventions NaN.
La référence Python reste testée avec des tolérances numériques, sans promesse
d'identité bit à bit entre compilateurs. Aucun historique futur n'est utilisé.

## Reproduire les mesures

```bash
uv run python scripts/benchmark_cpp.py --rows 100000 --repeat 3
```

Le script vérifie d'abord les mêmes sorties Python/C++ sur 2 000 observations.
Chaque backend s'exécute ensuite dans un processus séparé, séquentiellement, sur
le même dataset synthétique (seed fixe). Il enregistre trois temps, leur médiane,
les NaN, checksum, warmup, pic RSS et versions dans
[`benchmarks/cpp.json`](benchmarks/cpp.json). Les checksums et comptes NaN du jeu
complet sont aussi comparés. Les tests détaillés comparent chaque colonne.

Les temps incluent validation et API Python, mais excluent génération des données,
imports, compilation, warmup et I/O disque. Benchmark des profils/parcours :
bougies source 1 min, cible 30 min, 10 niveaux de prix et 5 segments. Fenêtres
glissantes : 20 observations. Les chiffres comparent des sorties équivalentes,
avec les primitives NumPy/Numba déjà présentes dans la référence ; ils ne
comparent pas une seule feature à un groupe entier.

Mesures locales : macOS arm64, Python 3.12.14, NumPy 2.3.5, pandas 2.3.3. **100 000 lignes source**, médiane de trois exécutions.

| Cas | Référence | C++ | Gain | RSS référence / C++ |
| --- | ---: | ---: | ---: | ---: |
| TPO : 6 descripteurs | 1.6296 s | 0.0070 s | ×234.1 | 205 / 165 Mio |
| Parcours : 8 descripteurs | 0.8766 s | 0.0234 s | ×37.5 | 166 / 165 Mio |
| MAD close | 1.3516 s | 0.0159 s | ×85.0 | 164 / 164 Mio |
| Pente de régression close | 0.1495 s | 0.0080 s | ×18.7 | 196 / 164 Mio |
| Sample entropy des rendements | 4.3849 s | 0.0286 s | ×153.3 | 163 / 163 Mio |
| Groupe enrichi complet (286 descripteurs + OHLCV) | 5.5670 s | 0.5522 s | ×10.1 | 310 / 215 Mio |

## Limites

Le gain dépend des features, fenêtres, entrées, compiler et matériel. C++ retire
les callbacks Python et allocations répétées ; il ne supprime pas la complexité
mathématique. Sample entropy reste quadratique dans la longueur de fenêtre.
Un profil plus dense augmente le coût de reconstruction et le nombre de sorties.
Une stratégie dominée par lecture Parquet ou kernels natifs déjà rapides peut
obtenir un gain global beaucoup plus faible que ces mesures ciblées.
