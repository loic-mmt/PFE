# Vérification des extensions immédiates — 14 octobre 2026

Périmètre : [C01–C08](market-data-roadmap.md#extensions-c01-c08), APIs multi-timeframes
et préparation des datasets pour un projet consommateur. Le contrôle utilise les
dépendances et le modèle déjà installés, ainsi que de petites captures.

## Vérification du code

Les tests couvrent notamment : retraits et révisions macro, calendriers reportés
ou annulés, forecast figé avant annonce vérifiée, changements du master,
conflits de statuts entre sources, univers/groupes as-of, rangs cross-section,
mappings IFRS/custom/segments, périodes fiscales et dette sans double comptage.

Les tests multi-timeframes vérifient secondes, minutes, heures, jours continus,
séances DST/demi-séances, entrées tardives, trous, bougies incomplètes et absence
de fuite future. Les partitions reproduisent les valeurs, masques, fenêtres,
labels et stride du builder complet ; le scaler global utilise uniquement les
cellules train uniques après purge des labels. DAG et CLIs testent interruption,
reprise, cache altéré, sources partielles, caps et exports locaux.

Commandes de validation finale :

```bash
python -m pytest -q --tb=short
ruff check .
ruff format --check .
git diff --check
uv build --offline --python /tmp/pfe-lot2-verification/venv/bin/python
```

Résultat final : **1 083 tests passent en 56,66 s**. Ruff valide le code et le format
des 205 fichiers contrôlés ; `git diff --check` passe. Les douze exemples sans plotting,
dont `examples/reusable_pipeline.py`, passent sans réseau. Wheel et sdist sont
construits hors ligne depuis le cache existant, sans `uv sync` ni installation. Les **84
modules Python** du wheel sont identiques aux sources ; import isolé sans checkout,
extension native et dataset multi-timeframes validés, dépendances optionnelles et
frameworks DL bloqués pendant ce contrôle.

## Petites captures réelles et modèle local

Clés lues depuis `.env` uniquement pour les quatre services autorisés, sans shell,
affichage ni persistance des valeurs. Le script `verify_integrations.py` plafonne
chaque invocation à 12 requêtes HTTPS, 64 Kio par réponse, 512 Kio transférés et
2 Mio d'archive. Les reprises utilisent les captures existantes.

| Service | Résultat observé | Portée démontrée |
|---|---|---|
| FRED | Échec HTTP 500 répété sur la petite requête | Accès réel et clé non validés ; collecteur et replay couverts offline |
| EIA | Deux valeurs WTI des 2 et 3 janvier 2025, collection complète, replay identique | Authentification, réponse v2 étroitement filtrée, parser et stockage existants |
| OpenDART anglais | Un compte `Sales`, CFS/IS, Samsung 2024, KRW, replay identique | Clé, compte explicitement sélectionné et convention annuelle de ce rapport. Le premier essai `Revenue` était vide ; le libellé a été corrigé d'après la réponse réelle |
| e-Stat | Une table du catalogue et une cellule, reçues et archivées | Authentification et schéma de cette réponse validés ; mapping économique général et historique PIT à qualifier. Parsers testés offline |
| Qwen `qwen2.5:3b-instruct` | Prompts A/B/C/D exécutés, réponses terminées, **0 fait accepté, 3 rejets**, replay identique | Backend local, format structuré, validation, archivage des rejets et cache. Précision/rappel financier sur corpus réel non mesurés |

Digest Ollama vérifié avant/après inférence :
`357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`. Contexte configuré
8192, sortie plafonnée à 768 tokens par prompt, texte synthétique court. Le modèle était
déjà installé ; le serveur lancé pour le test a été arrêté après vérification. Les
rejets de Qwen portent entre autres sur des informations non soutenues par les citations
et des attributions d'entité. Ces faits sont enregistrés comme rejets et exclus des
features.

Le [rapport JSON sans secrets](benchmarks/immediate-integrations.json) conserve
les résultats et identités de collections. Ses compteurs HTTP concernent la
dernière invocation : les trois sources réussies étaient alors rejouées depuis
le cache, donc zéro nouvelle réponse enregistrée. Les tentatives FRED précédentes
et les réceptions initiales restent décrites dans le tableau ci-dessus.

Captures temporaires : environ 140 Kio logiques de fichiers, environ 350 Kio de blocs
disque, sous `/tmp/pfe-immediate-smoke-20261014`. Aucun raw fournisseur ajouté au Git.
Wheel et sdist : environ 1,3 Mio ensemble. Le volume disposait d’environ 12 Gio après
les tests.

## Validations restantes

L'historique et les horaires PIT, l'exhaustivité des univers, les conventions de chaque
fournisseur, la couverture comptable et la qualité d'un modèle local exigent les entrées
et corpus propres au projet consommateur. Les sources licenciées demandent leurs accès
et spécifications. Revere et J-Quants/KRX nécessitent des connecteurs supplémentaires ;
C08 couvre les adaptateurs décrits dans la roadmap.

Les erreurs de service et rejets d’extraction restent dans les rapports. La
qualification de chaque source dépend de ses captures et de leur couverture.