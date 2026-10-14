# Données de marché et features DL : inventaire, sources, feuille de route

Audit initial du **11 octobre 2026**, état consolidé le **14 octobre 2026**. PFE calcule
des features prix, macro, fondamentales, graphes et événements pour GRU, TCN, GNN et
Event Encoder. Un LLM local peut extraire les faits des textes.

**LSE désigne London Strategic Edge**, fournisseur du repo. **LSEG** désigne
London Stock Exchange Group, fournisseur distinct, notamment d'I/B/E/S.

Les moteurs et datasets sont testés offline. Les [extensions
C01–C08](reusable-pipelines.md) couvrent retraits/calendriers macro, snapshots de
master, cross-section, mappings comptables, orchestration, partitions et
multi-timeframes. L'usage sur données réelles demande de qualifier les archives, leur
couverture et leurs horloges. L'[état et les prérequis](#9-travail-restant-et-méthode)
détaille le code disponible, les accès, le stockage et les validations nécessaires.

## 1. Périmètre demandé

Les 34 besoins ci-dessous sont classés par priorité P0–P3. « Daily » désigne la
fréquence d'observation ; la publication peut survenir plus tard. Le projet consommateur
choisit les fréquences — secondes, minutes, heures ou séances — et la grille de
décision.

| Données | Pays / zone | Source et fréquence souhaitées | Modèle / usage | Priorité |
|---|---|---|---|---|
| OHLCV actions US | USA | Historique de marché / daily | GRU, TCN, GNN | P0 |
| Ajustements splits et dividendes | USA | Historique des corporate actions | Prétraitement | P0 |
| Tickers, identifiants, historique de cotation | USA | SEC, fournisseur de marché | Prétraitement | P0 |
| RSI, ATR, MACD, moyennes mobiles, momentum | USA | Calcul local / daily | GRU, TCN | P0 |
| Volatilité, volume relatif, drawdown | USA | Calcul local / daily | GRU, TCN, GNN | P0 |
| Rendements SPY, QQQ, IWM | USA | Marchés / daily | Macro, GNN | P1 |
| ETF sectoriels, SOXX, XLF, XLE, etc. | USA | Marchés / daily | Macro, GNN | P1 |
| VIX et indices de volatilité | USA | [Cboe](https://www.cboe.com/tradable_products/vix/vix_historical_data/) / daily | Macro | P1 |
| Taux 2 ans, 10 ans, courbe des taux | USA | [FRED](https://fred.stlouisfed.org/) / daily | Macro | P1 |
| Spreads de crédit | USA | FRED / daily | Macro | P1 |
| Fed Funds, CPI, PPI, chômage | USA | [ALFRED](https://alfred.stlouisfed.org/) / publications | Macro | P1 |
| Dollar, USD/JPY, USD/CNY, USD/TWD, USD/KRW | USA, Japon, Chine, Taïwan, Corée | Marchés FX / daily | Macro, GNN | P1 |
| Pétrole, gaz, cuivre, aluminium, or | Monde | Marchés, [EIA](https://www.eia.gov/opendata/), [Banque mondiale](https://www.worldbank.org/en/research/commodity-markets) | Macro, GNN | P1 |
| CA, EPS, marges, dette, cash-flow réels | USA | [SEC XBRL](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) / trimestriel | Fondamentaux, GNN | P1 |
| Guidance CA, EPS, marge, CapEx | USA | 8-K, earnings calls / irrégulier | LLM → Fondamentaux | P1 |
| Fournisseurs et clients déclarés | Monde | Rapports annuels / irrégulier | LLM → GNN | P1 |
| Exposition géographique des revenus | Monde | 10-K, rapports / annuel | LLM → GNN | P1 |
| Exposition matières premières et production | Monde | Rapports / annuel | LLM → GNN | P1 |
| Relations sectorielles et concurrentielles | Monde | SEC, référentiels, extraction | GNN | P1 |
| Prix et rendements des principaux fournisseurs étrangers | Taïwan, Corée, Japon, Europe | Bourses locales / daily | GNN, Macro | P2 |
| CA mensuel des grandes fonderies, notamment TSMC | Taïwan | [TSMC IR](https://investor.tsmc.com/english/monthly-revenue/2026) / mensuel | GNN, Fondamentaux | P2 |
| Résultats des entreprises étrangères stratégiques | Taïwan, Corée, Europe | [MOPS](https://mops.twse.com.tw/), [OpenDART](https://engopendart.fss.or.kr/intro/main.do), rapports | LLM, GNN | P2 |
| PMI, production industrielle, activité manufacturière | Chine | [NBS](https://www.stats.gov.cn/english/) / mensuel | Macro, GNN | P2 |
| Inflation, industrie, commerce extérieur | UE | [Eurostat](https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/) / publications | Macro | P2 |
| Tankan, taux, inflation, production | Japon | [BoJ](https://www.stat-search.boj.or.jp/index_en.html) / publications | Macro | P2 |
| Production et dépendances minières | Monde, par pays producteur | [USGS](https://www.usgs.gov/centers/national-minerals-information-center) / annuel | GNN structurel | P2 |
| Événements géopolitiques et économiques | Monde | [GDELT](https://gdeltproject.org/data.html) / événements | LLM, Event Encoder | P2 |
| Nouvelles propres aux entreprises | Monde | SEC 8-K, communiqués, fournisseur news | LLM, Event Encoder | P2 |
| Embeddings textuels et sentiment | Monde | Calcul local / à publication | Event Encoder | P2 |
| Consensus historiques de CA et EPS | Monde | [FactSet](https://insight.factset.com/resources/factset-consensus-estimates-datafeed), [I/B/E/S](https://www.lseg.com/en/data-catalogue/company-data/ibes-estimates/broker-estimates) / snapshots | Fondamentaux, Event Encoder | P3 |
| Révisions d'analystes, dispersion, surprises | Monde | Calcul sur historiques de consensus | Fondamentaux | P3 |
| Relations fournisseur-client industrielles complètes | Monde | FactSet Revere / historique | GNN | P3 |
| Positionnement sur futures | USA, marchés mondiaux | [CFTC COT](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm) / hebdomadaire | Macro | P3 |
| Options, skew et volatilité implicite | USA | Cboe, fournisseurs spécialisés | Macro, GRU | P3 |

## 2. Ce qui existe réellement ici

L'audit porte sur le checkout, **y compris les modifications non commitées** et les
fichiers options. Les références décrivent ce code, dont la publication dans une release
demande une vérification séparée.

| Domaine | Implémentation | Couverture / manque |
|---|---|---|
| OHLCV | [download_data.py](../scripts/download_data.py), [bars.py](../src/pfe/data/bars.py), [daily.py](../src/pfe/data/daily.py) | `1m` SDK et daily REST natif, plusieurs tickers/invocation ; snapshots immutables, processed par instrument et provenance daily |
| Indicateurs | [api.py](../src/pfe/features/api.py), [standard.py](../src/pfe/features/standard.py), [technical.py](../src/pfe/features/technical.py) | RSI/ATR/MACD, rendements, moyennes, momentum, volatilité, volume relatif, drawdown ; TA-Lib et extensions |
| Intrabar | [aggregation.py](../src/pfe/features/aggregation.py), [intrabar.py](../src/pfe/features/intrabar.py), [profiles.py](../src/pfe/features/profiles.py) | Bougies enrichies, trajectoire, TPO, qualité ; agrégation à durée fixe |
| Sélection / volume de calcul | [selected.py](../src/pfe/features/selected.py), [streaming.py](../src/pfe/features/streaming.py), [C++](../cpp/native.cpp) | Calcul par nom/groupe, lots et accélération ; aucune collecte massive multi-source orchestrée |
| Modèles linéaires | [linear.py](../src/pfe/features/linear.py) | OLS/Ridge rolling, labels connus au moment du fit ; sorties utilisables comme features |
| Macro | [data/fred.py](../src/pfe/data/fred.py), [data/alfred.py](../src/pfe/data/alfred.py), [features/macro.py](../src/pfe/features/macro.py) ; LSE existant | Observations/vintages/releases FRED bornées, unités et politiques date-only explicites ; moteur sparse et spreads compatibles. Heures PIT non qualifiées |
| Références / commodities | [reference_assets.py](../src/pfe/data/reference_assets.py), [features/reference.py](../src/pfe/features/reference.py), [EIA](../src/pfe/data/eia.py), [Cboe](../src/pfe/data/cboe.py), [World Bank](../src/pfe/data/world_bank.py) | Contrats, collecteurs versionnés, returns/beta/corr/force relative/expositions déclarées ; horizons exacts, moyennes mensuelles distinctes du daily |
| Options | [download_options.py](../scripts/download_options.py), [data/options.py](../src/pfe/data/options.py), [features/options.py](../src/pfe/features/options.py) | Activité, IV/greeks, skew/terme, flow, contrats, qualité ; OI/bid/ask restent manquants si source absente |
| Splits/dividendes | [corporate_actions.py](../src/pfe/data/corporate_actions.py), [download_actions.py](../scripts/download_actions.py) | Contrats, collecte bornée, révisions/as-of et rendements réconciliés ; conventions numériques/mappings explicites, couverture historique non qualifiée |
| Séances daily | [sessions.py](../src/pfe/data/sessions.py), [features/daily.py](../src/pfe/features/daily.py) | Calendriers réels/injectés, agrégation RTH, fenêtres par séance/instrument, masques/âges ; aucun remplissage |
| Identifiants et univers | [security_master.py](../src/pfe/data/security_master.py), [snapshots](../src/pfe/data/security_sources.py) | Contrat bitemporel, collecteurs SEC/Nasdaq/Alpha Vantage, changements et sélection as-of ; archives anciennes et couverture réelle non certifiées |
| Fondamentaux | [data/sec.py](../src/pfe/data/sec.py), [features/fundamentals.py](../src/pfe/features/fundamentals.py) | Submissions/companyfacts, accessions, réception/reconstruction explicite, croissance/marges/levier/cash-flow/CapEx ; bases et périodes séparées, couverture limitée aux concepts fournis |
| Guidance | [Lot 3](document-features.md), [statements](../src/pfe/features/statements.py) | Extraction A–D locale/offline, preuves et revue, valeurs/plages actual/guidance/estimate séparées ; consensus historique fourni séparément |
| Graphes / événements / LLM | [documents](../src/pfe/data/documents.py), [extraction](../src/pfe/extraction/engine.py), [graphes](../src/pfe/data/graph.py), [events](../src/pfe/data/events.py) | Imports/collecte bornés, Ollama facultatif digest local, graphes/events historisés, features et outputs locaux ; corpus synthétique, modèle réel à qualifier |
| Sources internationales | [Lot 4](international-features.md) | TSMC/MOPS/TWSE, OpenDART, Eurostat, BoJ/e-Stat, NBS/USGS et GDELT ciblés ; couverture/horloges/droits réels à qualifier |
| COT / consensus | [Lot 5 P3](p3-sources.md) | COT par rapport/marché/période ; exports/clients licenciés facultatifs, comparabilité et surprise pré-annonce ; accès premium à configurer selon le contrat |
| Datasets DL | [Lot 5 modèles](model-datasets.md), [extensions](reusable-pipelines.md) | Multi-timeframes, panels/fenêtres partitionnés, GNN/events, provenance, masques, labels séparés, purge et scaler train global ; NPZ/JSON bornés |

Introspection offline : `compute_features` sur 128 bougies synthétiques, fenêtres `(5,
20, 60)`, tous groupes → **1 912 colonnes** ; `enrich_bars(..., "7min")` par défaut →
**293 colonnes**, OHLCV et `available_at` inclus. Valeurs de warm-up peuvent rester NaN.
Le catalogue exact dépend des fenêtres, segments, bins et noms sélectionnés ;
`feature_catalog(result)` inventorie une sortie réelle. La couverture des données et les
biais se vérifient séparément de ce catalogue. Voir [features](features.md),
[macro](macro.md), [options](options.md), [performance](performance.md).

### Données locales observées, sans nouvel appel API

| Fichier / répertoire | Contenu constaté |
|---|---|
| `data/processed/QQQ_1m_20260831T000000Z_20260910T000000Z.parquet` | 6 607 bougies ; 2026-08-31 08:00 → 2026-09-09 23:59 UTC |
| `data/raw/lse/QQQ/1m/20260831T000000Z_20260910T000000Z/` | Export brut correspondant, même nombre de lignes |
| `data/raw/lse/macro/sample.json` | 1 observation CPI, 0 Fed Funds, 20 événements calendrier |
| `data/raw/lse/macro/sample_series.json` | 6 observations Fed Funds + 6 CPI |
| `data/raw/lse/options/QQQ_chain_sample.json` | 10 contrats de chaîne |
| `data/raw/lse/options/probe.json` | 6 contrats + 6 prints + 6 bougies de contrat ; contrats de chaîne déjà expirés |

Fichiers privés ignorés par Git, donc absents d'un clone neuf. Ce n'est ni un
univers actions US, ni une base daily, ni une archive de chaînes ou de vintages.
Les noms exacts des fichiers doivent être recontrôlés si la collecte évolue.

### Limites temporelles et daily

- OHLCV-only (`normalize_bars`) garde cinq colonnes et index. Le contrat daily
  conserve identités, provenance, ajustements et qualité ; un ID de
  snapshot fournisseur ne remplace pas un identifiant permanent de security master.
- Bougies indexées à l'ouverture ; features utilisables à la clôture. Macro LSE
  et chaîne options sont alignées à la **réception**, sans remplir le passé.
  Les normalizers acceptent des timestamps explicites, mais le drapeau
  `point_in_time=True` fourni par un appelant n'est pas une preuve indépendante.
- Calendrier macro : une version par événement ; pas de replay complet des
  changements de planning ni de toutes les révisions de résultats.
- Options : chaîne courante ≠ chaîne historique ; prints datés ≠ surface NBBO
  synchrone. Expirations recalculées, échantillons plafonnés ; pas de signe dealer
  ni direction agresseur déduits des calls/puts.
- `source_frequency="1d"` décrit une durée fixe, pas les horaires d'une bourse.
  `gap_policy="reset"` réinitialise aux trous : weekends/fériés daily doivent
  être traités par l'adaptateur de séances :
  `compute_daily_features`/`aggregate_session_bars`. Horaires réels, demi-séances,
  extended hours et disponibilité explicites ; une agrégation UTC de 24 h
  ne suffit pas pour une bougie US daily.

## 3. Inventaire LSE : offert ≠ déjà intégré

La [databank officielle](https://londonstrategicedge.com/data/) annonce gratuitement
3 979 actions, 25 ETF, 62 paires FX, 23 matières premières, 54 futures, 19 indices,
202 séries obligataires, 14 640 séries macro, options sur 3 186 sous-jacents US.
Plan annoncé : 10 exports/heure, jusqu'à 1 000 000 lignes/export ; REST/WebSocket
inclus. Quotas et couverture évoluent ; vérifier compte et réponse serveur.

Le [SDK officiel](https://github.com/londonstrategicedge/lse-data) annonce actions
US depuis 2003, FX depuis 2009, prints options depuis 2014 : profondeur générale,
pas garantie par symbole. Licence SDK MIT distincte des droits data : recherche,
trading et entraînement propres, même commerciaux, autorisés selon README ;
redistribution/revente interdites. Garder données privées et exemples synthétiques.

Repo verrouillé sur `lse-data==0.14.0` dans [uv.lock](../uv.lock), contrainte
`>=0.14,<0.15` dans [pyproject.toml](../pyproject.toml). Le [changelog 0.14](https://raw.githubusercontent.com/londonstrategicedge/lse-data/main/CHANGELOG.md)
annonce bougies actions/ETF **déjà ajustées splits**, catalogue authentifié et
calendrier vault limité aux événements courants. Aucun engagement total return
ou historique de publication/vintages : ne pas doubler l'ajustement splits.

| Besoin | API SDK offerte | Intégration PFE |
|---|---|---|
| OHLCV daily / intraday | `candles(..., timeframe="1d")`, `history()` | Collecteur daily natif et `1m`, multi-tickers, calendriers et provenance ; [Lot 1](daily-market.md) |
| Splits / dividendes | `splits()`, `dividends()` | Présente : collecte bornée, mappings/conventions explicites, historique de révisions |
| Profils, secteur, industrie | `company_profiles()` | Absente ; profil courant ≠ master historique |
| États financiers | `financial_reports()` | Absente ; périodes FY/Q, dates publication à vérifier |
| Ratios / fondamentaux | `fundamentals()` | Absente ; snapshot courant |
| Macro / taux | `economics()`, `series()`, `bond_yields()` | Échantillons economics + calendrier ; pas connecteur yields dédié |
| Calendrier | `economic_calendar()` | Partielle, échantillon courant reçu |
| Positionnement futures | `cot()` | Absente |
| Chaînes / prints / contrats options | `options()`, `options_flow()`, `option_candles()` | Présente, bornée ; historique chaînes absent |

Méthodes et signatures : [client
SDK](https://raw.githubusercontent.com/londonstrategicedge/lse-data/main/lse/client.py).
Découverte : `catalog()`, `vault_meta()`, `datasets()`, `reference()` ; tables de
référence exportables par `dataset()` : [vault
SDK](https://raw.githubusercontent.com/londonstrategicedge/lse-data/main/lse/vault.py).
[qualify_lse.py](../scripts/qualify_lse.py) conserve cet inventaire. L'implémentation a
été vérifiée offline ; le compte demande un audit authentifié.

**À vérifier sur catalogue authentifié** : SPY/QQQ/IWM, SOXX/XLF/XLE et ETF sectoriels,
VIX/VXN/RVX, US2Y/US10Y, indice dollar et quatre paires FX citées, pétrole/gaz/cuivre/aluminium/or,
cours locaux des fournisseurs étrangers, séries de crédit et macro nationales.
Une classe FX/ETF annoncée ne prouve pas un symbole précis. ADR US et action locale
sont deux instruments distincts.

Non documenté dans les interfaces examinées : master historique sans biais de survie,
corporate actions PIT, guidance structurée, graphe fournisseur/client, expositions
géographiques, articles/transcripts, consensus historiques PIT. Leur disponibilité doit
être confirmée auprès du fournisseur. Cet audit repose sur la documentation publique ;
les droits du compte restent à vérifier.

## 4. Sources gratuites et limites

« Gratuit » désigne ici un accès sans achat identifié. Quotas, droits d'usage et de
redistribution, stockage, calcul et validation doivent être vérifiés.

Les connecteurs LSE, FRED/ALFRED, SEC, EIA, Cboe VIX et Pink Sheet sont implémentés avec
les limites décrites au [Lot 2](context-features.md). Les [connecteurs internationaux et
GDELT](international-features.md) utilisent des sélections explicites et des imports
locaux limités. Les alternatives et flux exhaustifs ci-dessous demandent une
qualification dédiée. « Gratuit partiel » signifie couverture ou droits limités, sans
équivalence garantie avec un produit professionnel.

| Besoin couvert | Source gratuite / accès | Fréquence et limite utile pour implémenter |
|---|---|---|
| Actions/ETF US et références SPY/QQQ/IWM/sectoriels | LSE ; alternative [Alpaca plans](https://docs.alpaca.markets/us/docs/about-market-data-api), [FAQ feed](https://docs.alpaca.markets/us/docs/market-data-faq) | Alpaca : historique depuis 2016 annoncé, temps réel gratuit IEX ; FAQ autorise historique SIP sans abonnement avec `end` retardé d'au moins 15 min. Tester feed/compte. IEX seul ≠ volumes consolidés ; radiations et univers ancien à qualifier |
| Splits/dividendes | LSE ; annonces IR/filings publics | Registre, collecte et facteurs versionnés implémentés au Lot 1 ; pas preuve que tout historique soit brut, dividend-adjusted ou PIT |
| Identifiants/cotations | [SEC submissions](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), [Nasdaq directories](https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs), [Alpha Vantage](https://www.alphavantage.co/documentation/) `LISTING_STATUS` | CIK = déposant ; annuaire courant ≠ univers historique. Capturer snapshots. AV documente actifs/delistés à date ; [clé gratuite](https://www.alphavantage.co/support/api-key/) généralement 25 requêtes/jour ; daily adjusted premium |
| Indicateurs, volatilité, volume relatif, drawdown | Calcul local PFE | Calculables sur données qualifiées ; benchmark beta/corr avec horizons appariés et cross-section univers/secteur implémentés. BETA/CORREL TA-Lib high/low ≠ benchmark externe |
| VIX/autres indices volatilité | [Cboe historiques](https://www.cboe.com/tradable_products/vix/vix_historical_data/) | Clôtures daily, VIX depuis 1990 ; pas surface options ni prix exécutables |
| Taux US/courbe | [FRED DGS2](https://fred.stlouisfed.org/series/DGS2), [DGS10](https://fred.stlouisfed.org/series/DGS10), [Treasury CSV/XML](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve) | Daily ; dériver `DGS10 - DGS2` après disponibilité de chaque jambe ; unités points de pourcentage |
| Fed Funds/CPI/PPI/chômage | [FRED/ALFRED real-time](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html), [BLS API](https://www.bls.gov/developers/home.htm), [calendrier BLS](https://www.bls.gov/schedule/2026/home.htm) | API FRED avec clé ; BLS sans clé ou inscrit. Vintages + archives release ; taux cible, effective daily [DFF](https://fred.stlouisfed.org/series/DFF) et moyenne mensuelle [FEDFUNDS](https://fred.stlouisfed.org/series/FEDFUNDS) distincts |
| Spreads crédit | [FRED ICE HY OAS](https://fred.stlouisfed.org/series/BAMLH0A0HYM2) | Gratuit partiel : depuis avril 2026, trois ans d'observations seulement ; profondeur via ICE. Notes imposent usage interne et accord pour redistribution |
| Dollar et FX asiatiques | [Dollar broad DTWEXBGS](https://fred.stlouisfed.org/series/DTWEXBGS), [JPY](https://fred.stlouisfed.org/series/DEXJPUS), [CNY](https://fred.stlouisfed.org/series/DEXCHUS), [TWD](https://fred.stlouisfed.org/series/DEXTAUS), [KRW](https://fred.stlouisfed.org/series/DEXKOUS) | Série dollar broad ≠ DXY. Paires en monnaie étrangère pour 1 USD ; observations H.10 quotidiennes à midi NY, [publication hebdomadaire](https://alfred.stlouisfed.org/category?cid=94&ob=t&od=desc&t=trade-weighted), pas close tradable |
| Pétrole/gaz/stocks/production US | [EIA API v2](https://www.eia.gov/opendata/documentation.php), [clé et bulk](https://www.eia.gov/opendata/register.php) | Clé gratuite API, bulk sans clé ; fréquences selon série. Prix, stocks et publications séparés ; latest API ne garantit pas vintages |
| Cuivre/aluminium/or/énergie mondiale | [World Bank Pink Sheet](https://www.worldbank.org/en/research/commodity-markets), marché LSE à qualifier | Banque mondiale mensuel/annuel : pas remplacement daily de tous métaux. Benchmark/unité/devise explicites ; forecasts Outlook séparées des observations |
| Résultats US et émetteurs étrangers déposants SEC | [SEC XBRL et filings](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | JSON sans clé. Company Facts couvre faits d'entité et taxonomies standard ; segments, géographies/custom → filing original. `frames` latest n'est pas jointure PIT |
| Guidance, relations, expositions, concurrence | 10-K/10-Q/8-K/20-F/6-K et IR publics ; extraction locale | Texte brut gratuit partiel. Clients parfois anonymes ; couverture réelle inconnue. Publication, orientation relation et citations indispensables. Calls/transcripts non universellement publics |
| Cours fournisseurs étrangers | [TWSE historique](https://www.twse.com.tw/en/trading/historical/stock-day.html?fr=cg57256C0201M02), [OpenAPI](https://openapi.twse.com.tw/), [J-Quants](https://www.jpx.co.jp/english/corporate/news/news-releases/6020/20230403-01.html), [Corée FSC](https://www.data.go.kr/data/15094808/openapi.do) | TWSE : daily depuis 2010 annoncé. JPX : ancien plan gratuit 2 ans/12 semaines de retard ; conditions 2026 à revalider. Corée : J+1 ouvré après 13h, non-commercial, modification et redistribution interdites selon page FSC ; réutilisation pour features à qualifier, commercial via KRX. Europe/chaque venue à qualifier ; aucune couverture gratuite globale démontrée |
| CA mensuel TSMC | [TSMC IR](https://investor.tsmc.com/english/monthly-revenue/2026) | Millions TWD, données consolidées mensuelles ; joindre communiqué daté, pas date artificielle en fin de mois |
| Résultats Taïwan/Corée/Europe | [MOPS](https://mops.twse.com.tw/), [OpenDART](https://engopendart.fss.or.kr/intro/main.do), rapports IR | MOPS rapports/XBRL/revenus mensuels, calendrier ROC ; DART clé, [conditions gratuites](https://engopendart.fss.or.kr/uss/umt/EgovMberInsertView.do), [quotas variables](https://engopendart.fss.or.kr/guide/detail.do?apiGrpCd=DE001&apiId=AE00001). Europe : parse rapports émetteurs, pas API unique équivalente SEC supposée |
| PMI/industrie Chine | [NBS National Data](https://data.stats.gov.cn/english/?cn=C01), [communiqué PMI](https://www.stats.gov.cn/english/PressRelease/202606/t20260601_1963851.html) | Public mensuel ; PMI NBS ≠ PMI privé Caixin/S&P. Distinguer mensuel et cumul janvier ; API stable non confirmée |
| Inflation/industrie/commerce UE | [Eurostat API](https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/api-introduction) | Gratuit JSON-stat/SDMX ; DB **dernière version seulement**, sans historique des versions. Snapshots futurs + archives releases pour PIT |
| Tankan/taux/activité Japon | [BoJ API annoncée 18-02-2026](https://www.boj.or.jp/en/statistics/outline/notice_2026/not260218a.htm), [manuel](https://www.stat-search.boj.or.jp/info/api_manual_en.pdf), [e-Stat](https://www.e-stat.go.jp/api/api/index.php/en/api-dev/how_to_use) | BoJ JSON/CSV ; e-Stat appId inscrit. Inflation/production : agence statistique compétente, pas toutes séries chez BoJ. Révisions à archiver |
| Production/dépendances minières | [USGS MCS](https://pubs.usgs.gov/publication/mcs2026), [CSV 2026](https://www.usgs.gov/data/mineral-commodity-summaries-2026-data-release) | Annuel, CSV CC0 ; PDF initial 06-02-2026, CSV publié 18-09-2026. Versions et dates de fichiers distinctes ; aucune disponibilité au 1er janvier supposée |
| Événements monde/news sociétés | [GDELT](https://gdeltproject.org/data.html), SEC/IR | Events/GKG toutes 15 min, archives libres ; détection ≠ première publication, lien ≠ droit sur article. Corpus news exhaustif absent ; BigQuery peut facturer calcul |
| Embeddings/sentiment | Calcul local, [Ollama embeddings](https://docs.ollama.com/capabilities/embeddings) | Pas achat data nécessaire ; licence du modèle, compute et dimension/version à conserver |
| Consensus/révisions/dispersion entreprise | Quelques citations IR/news ; snapshots futurs autorisés | Partiel ; ne reconstitue pas ancien consensus multi-analystes. Surprise macro existante ≠ surprise earnings entreprise |
| Relations industrielles | Rapports et extraction locale | Graphe déclaré partiel, liens/poids inconnus restent inconnus ; ne pas créer fournisseur à partir simple mention |
| COT | [CFTC archives](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm), [calendrier](https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm) ; LSE | Gratuit hebdo. Positions généralement mardi, publication vendredi 15:30 `America/New_York`, exceptions calendaires ; distinguer Legacy/Disaggregated/TFF et futures-only/combiné |
| Options/skew/IV | LSE ; [Alpaca options](https://docs.alpaca.markets/us/docs/historical-option-data) | LSE modèles IV/greeks et prints ; Alpaca historique depuis février 2024 annoncé, feed indicatif gratuit ≠ OPRA intégral. Archive surfaces/OI/NBBO longue profondeur à qualifier |

### Conséquences de qualification

- [FRED release dates](https://fred.stlouisfed.org/docs/api/fred/release_dates.html)
  précise que date release source ≠ date disponibilité FRED. ALFRED fournit
  dates de vintages, pas toutes heures intraday : vérifier calendrier/archives,
  puis conserver incertitude et latence.
- SEC : conserver accession et timestamp d'acceptation depuis dépôt ; extrait
  Company Facts avec date `filed` ne suffit pas pour fixer heure. Respecter
  [accès automatisé SEC](https://www.sec.gov/files/about/webmaster-faq.htm)
  et identifier le client ; limite annoncée 10 requêtes/s.
- Les quotas des downloaders PFE bornent **l'échantillon**, pas le dataset.
  Pagination/reprise, contrôles de couverture et sauvegarde de releases restent
  nécessaires avant entraînement historique.
- J-Quants a évolué vers [API V2 et API key en 2026](https://www.jpx.co.jp/english/corporate/news/news-releases/6020/20260119.html).
  Ancienne annonce gratuite ne fixe pas droits/prix actuels ; portail actuel
  inaccessible à la recherche (403). Valider avant choisir source opérationnelle.

## 5. Ce qui nécessite une licence payante

**« Forcément payant » ne s'applique pas à une famille entière.** Produits
FactSet/I/B/E/S sous licence ; établissement déjà abonné peut absorber coût.
Les autres lignes deviennent payantes selon couverture, qualité, droits et
profondeur voulus. Aucun prix non publié ni abonnement de l'utilisateur présumé.

| Besoin | Produit payant / preuve | Alternative gratuite et écart |
|---|---|---|
| Consensus CA/EPS historiques PIT | [I/B/E/S](https://www.lseg.com/en/data-catalogue/company-data/ibes-estimates/broker-estimates), [FactSet Estimates](https://insight.factset.com/resources/factset-consensus-estimates-datafeed) | Guidance et estimations citées dans articles, snapshots prospectifs ; pas historique exhaustif de consensus/analystes |
| Révisions, dispersion, surprises earnings | Historique estimates individuel/consensus/actuals licencié | Calcul local gratuit si inputs obtenus ; sans anciens snapshots, pas reconstruction fiable |
| Réseau industriel enrichi et historisé | [FactSet Supply Chain PIT](https://insight.factset.com/supply-chain-signals-enhancing-the-customer-momentum-strategy-with-network-centrality) | Rapports + LLM : déclarations partielles. Même payant, graphe économique réellement complet et pourcentages de dépendance non garantis |
| Chaînes options profondes/NBBO/OI/analytics | [Cboe Option Quotes](https://datashop.cboe.com/option-quote-intervals) : achats historiques, IV/greeks en option | LSE gratuit couvre déjà une partie ; latest trades/greeks ≠ surfaces historiques synchrones, exécution ou OI exhaustif |
| News/transcripts complets, normalisés et archivés | Flux professionnel sous licence, fournisseur à qualifier | SEC/IR/GDELT partiels ; existence d'un article public ne crée pas droit sur un feed intégral |
| Profondeur crédit ICE et marchés étrangers/commercial | [Restrictions FRED ICE](https://fred.stlouisfed.org/series/BAMLH0A0HYM2), services de bourses | Fenêtre gratuite/proxies, annonces/IR, données locales restreintes ; pas équivalence indice ni droit commercial automatique |
| Univers historique de qualité, actions radiées et identifiants | Produit fournisseur à qualifier sur contrat et échantillon | Master public reconstruit possible ; complétude historique/delistings/delisting returns non établie ici |

Pour I/B/E/S, demander **package PIT**, pas simplement « historical » : la
[brochure LSEG](https://www.lseg.com/content/dam/data-analytics/en_us/documents/brochures/lseg-data-for-quant-research-brochure.pdf)
distingue snapshots quotidiens PIT avec livraison hebdomadaire, Historical,
Daily Means et Current. Une livraison hebdomadaire ne signifie pas que ces
données ont été reçues chaque jour par votre pipeline.
La [méthodologie FactSet PIT](https://insight.factset.com/hubfs/Resources%20Section/White%20Papers/ID11996_point_in_time.pdf)
décrit des snapshots de consensus à minuit marché local ; vérifier activation,
corrections, couverture et granularité du feed acheté.

Avant achat : demander extrait comprenant une révision, société radiée,
changement de ticker/devise, dates fiscal/publication/activation et droits ML.
Comparer au MVP gratuit sur besoin exact. Pas besoin de payer pour COT,
indicateurs locaux, embeddings ou toute extraction de rapports publics.

## 6. Contrat point-in-time et stockage cible

### Champs indispensables

| Champ demandé | Signification / règle |
|---|---|
| `entity_id` | Entreprise ou facteur ; stable. Instrument coté relié séparément à l'entité |
| `observation_date` | Période économique de la valeur ; jamais utilisée seule comme disponibilité |
| `published_at` | Publication effective ; nullable si inconnue, timezone et précision conservées |
| `available_at` | Premier instant justifié d'utilisation par le pipeline |
| `value` | Nombre ou structure typée ; valeur manquante explicite |
| `source_id` | Source d'origine, reliée au registre des sources et droits |
| `revision_id` | Version pour données révisables ; null si non applicable |

Lot 0 impose aussi `metric`, `unit`, `schema_version` et valide les compléments
`instrument_id`, `currency`, `ingested_at`, `availability_basis`, `point_in_time`,
`source_document_id` lorsqu'ils sont fournis ; [contrat exact](qualification.md).
`period_start`, `period_end` et `quality_flags` restent des extensions à définir
par les futurs adaptateurs. Structure numérique et JSON validées séparément ;
unicité selon entité/métrique/période/source/version.
Conserver vocabulaire existant (`observation_at` macro, `bar_at` options) derrière
adaptateurs, sans renommer silencieusement les APIs publiques.

### Disponibilité et révisions

1. **Collecte réelle** : `available_at` reflète réception et traitement ; conserver
   publication séparément. Donnée sans publication historique prouvée : réception,
   `availability_basis="receipt"`, provenance historique non vérifiée.
2. **Reconstruction historique** : utiliser publication authentifiée et délai
   explicitement documenté. `ingested_at` reste date du backfill actuel ; ne pas
   prétendre qu'il représente une livraison passée. Modélisation de latence
   identifiée comme hypothèse, distincte d'un historique de livraisons mesuré.
3. À décision `t`, retenir uniquement versions `available_at <= t`. Pour une
   période révisée, sélectionner dernière version déjà connue ; ne pas écraser
   ni antidater l'ancienne. Joindre par disponibilité, jamais par seule période.
4. Alignement multi-pays : clôture locale, fuseau, DST et délai après publication.
   Feature TSMC asiatique n'arrive aux actions US qu'après publication vérifiée.
5. Horodatage connu au jour seulement : conserver cette précision et
   documenter la convention de disponibilité retenue. `published_at` peut rester
   null et l'utilisation reposer sur réception.
6. Calendrier : historiser annonce du planning, report et annulation. Consensus
   antérieur à release requis pour surprise ; forecast observé après release
   ne peut pas devenir forecast d'origine.

Features dérivées : disponibilité au moins celle de toutes leurs entrées et de
leur calcul en mode réel. Ne pas confondre date de fait, période fiscale,
publication, ingestion et activation d'une arête de graphe.

### Arborescence demandée

```text
data/
  market/
    us_stocks_daily.parquet
    global_reference_assets.parquet
    corporate_actions.parquet
  macro/
    us_macro_vintages.parquet
    global_macro.parquet
    commodities.parquet
  fundamentals/
    company_facts.parquet
    management_guidance.parquet
    analyst_estimates_pit.parquet
  graph/
    nodes.parquet
    edges_history.parquet
    node_features.parquet
  events/
    events_structured.parquet
    event_entity_links.parquet
    event_embeddings.parquet
  metadata/
    security_master.parquet
    source_documents.parquet
    publication_calendar.parquet
```

Arborescence **cible**, pas état actuel. Garder en plus `raw/` immuable et sorties
`processed/` existantes pendant migration. Pour grande profondeur, noms `.parquet`
peuvent devenir datasets partitionnés par source/année/instrument ; contrat logique
inchangé. Manifestes raw et registre source versionné sont disponibles ; chaque futur
collecteur devra s'y raccorder.

| Table cible | Contenu / contraintes |
|---|---|
| Marché | OHLCV + instrument, séance, clôture/availability, devise, statut splits/dividendes, couverture |
| Corporate actions | Type, ratio/montant, annonce, ex-date, record/pay dates si disponibles, version et source |
| Macro | Période, valeur, unité, vintage, publication et availability ; snapshots conservés |
| Fondamentaux | Concept/basis comptable, durée, fiscalité de période, accession, actual/guidance/estimate séparés |
| Security master | Entité ↔ instrument ↔ ticker/venue/devise avec périodes de validité ET dates de connaissance |
| Documents | ID, URL, hash, texte brut/normalisé, langue, publication/réception, droits et version |
| Graphe | Arêtes orientées, type, preuves, poids nullable, période économique et historique de connaissance |
| Événements | Faits distincts, statut, magnitude/unités, liens entités, provenance, timestamps |
| Embeddings | Document/chunk, vecteur, modèle/digest/dimension, hash texte, version et disponibilité |

Ne pas publier données LSE dans Git. Métadonnées partageables et fixtures restent
soumises aux droits de chaque source.

## 7. LLM local optionnel

Prompts système et A–D **conservés intégralement** dans
[llm-extraction-prompts.md](llm-extraction-prompts.md).
Ils couvrent relations, géographies/coûts, résultats/guidance et événements.

Le connecteur collecte les documents ; le LLM extrait les faits présents dans le texte
et les métadonnées fournis. La sortie est limitée à ces faits, avec leurs preuves.

La [chaîne documentaire](document-features.md) suit ces étapes. Qualifier le modèle
retenu sur un corpus réel avant usage :

1. Extraire SEC/IR/news → archiver texte, hash, ID, dates et droits.
2. Découper par sections/tableaux avec chevauchement borné ; conserver offsets.
3. `backend="none"` par défaut ; `backend="ollama"` opt-in, endpoint local et
   modèle installés/configurés explicitement. Importer anciens JSON reste possible.
4. Appeler extraction avec prompt commun, tâche A/B/C/D et schéma JSON versionné.
   Valider sortie : enums, nullable, nombres, unités, période, citation exacte et ID.
5. Vérifier faits contre source, dédupliquer, résoudre identités sans confondre
   sociétés, filiales, ADR, pays de siège et exposition commerciale.
6. Enregistrer seulement faits validés et version du modèle/prompt/schéma.
   Sorties rejetées séparées pour revue ; aucun ajout silencieux à l'entraînement.
7. Embeddings et sentiment : étape locale distincte. Sentiment = estimation de
   modèle, pas fait vérifié ni direction de rendement.

Champs supplémentaires nécessaires : représentation des ranges B, libellé fiscal
C, statut relation effective/projet A. Schémas objets A/B/C/D à stabiliser ; vide
signifie tableau vide de l'enveloppe. Mécanisme économique D = null si non écrit.
Les citations valides limitent les hallucinations sans certifier tous les champs.

[Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
permet schéma dans `format` pour `/api/chat`, température 0 et validation indépendante.
`/api/embed` pour vecteurs : [documentation
embeddings](https://docs.ollama.com/capabilities/embeddings). L'[adaptateur
documentaire](document-features.md) utilise un serveur et un modèle Ollama installés
séparément.

## 8. Contextes étrangers par secteur

| Secteur US tradé | Contextes étrangers à envisager |
|---|---|
| Semi-conducteurs | Taïwan, Corée du Sud, Japon, Pays-Bas, Chine |
| Automobile | Chine, Europe, Japon, matières premières et énergie |
| Énergie | OPEP, pétrole mondial, gaz, production et stocks US |
| Banque et assurance | Taux US, spreads, Europe, liquidité mondiale |
| Industrie / aéronautique | Chaînes d'approvisionnement, Europe, Japon, marchés de l'énergie |
| Consommation | Chine, Europe, devises, inflation, importations |
| Logiciels / cloud | CapEx technologique, dépenses des entreprises, résultats des grands clients |

Table de contexte utilisateur : sélection de candidats, pas exposition prouvée
pour toutes les sociétés du secteur. Une feature étrangère doit garder sa source,
son calendrier et une relation documentée si elle reçoit un poids d'exposition.

## 9. Travail restant et méthode

Périmètre : bibliothèque de features, collecte et préparation des inputs. Broker,
slippage, frais de short et politique d’exécution sont exclus à la demande utilisateur.
Chaque statut distingue **code présent / code testé / source réelle testée /
couverture qualifiée**. Les lots et extensions sont testés offline ; cela ne
qualifie pas automatiquement les archives, accès et données d’un projet consommateur.
Les [petits contrôles réels](immediate-verification.md) conservent leurs résultats
et limites ; les [rapports historiques](archive/README.md) gardent les anciennes preuves.

### Lot 0 — contrats et qualification avant collecte

[Contrats, stockage et qualification](qualification.md) : observations versionnées,
replay, complétude et master bitemporel. Vérifier les révisions tardives, les
événements simultanés et l’absence de donnée avant disponibilité. Une archive
ancienne de cotations/radiations reste à fournir et qualifier.

### Lot 1 — P0 prix, corporate actions et features

[Prix daily, séances et actions](daily-market.md) : multi-tickers, clôtures réelles,
RTH, demi-séances, DST et facteurs connus à la décision. Qualifier les ajustements
prix/volume et éviter le double ajustement. Le contrôle réel doit inclure split,
dividende et radiation si la source les fournit, avec qualité et masques exportés.

### Lot 2 — P1 références, macro et fondamentaux

[Références, macro et états financiers](context-features.md) : benchmarks appariés,
vintages, FRED/ALFRED, EIA/Cboe/World Bank et SEC. Qualifier heures de publication,
unités, périodes fiscales et bases comptables. Révisions futures sans modification
du passé ; guidance distincte d’un consensus historique pré-annonce.

### Lot 3 — extraction documentaire, graphes, événements

[Documents et extraction](document-features.md), [prompts](llm-extraction-prompts.md)
et [features graphes](graph-features.md) : preuves, direction des relations, revue,
rejets et horloges conservés. Évaluer précision/rappel sur un corpus réel annoté ;
les fixtures ne qualifient pas un LLM. La collecte industrielle et un meilleur
modèle restent différés sous G-D01–G-D07, avec leurs prérequis dans la doc graphes.

### Lot 4 — P2 sources internationales ciblées

[Adaptateurs internationaux](international-features.md) : TSMC/MOPS/TWSE,
OpenDART, Eurostat, BoJ/e-Stat, NBS, USGS et GDELT ciblés. Qualifier langues,
devises, calendriers, identités et liens sociétés sur les pays/séries choisis.
J-Quants/KRX natifs et flux news exhaustifs restent des intégrations à spécifier.

### Lot 5 — P3 et datasets prêts pour modèles

[Sources P3](p3-sources.md) et [datasets modèles](model-datasets.md) : COT publié,
options prospectives, imports consensus licenciés facultatifs, GRU/TCN, GNN et
Event Encoder. Labels séparés, provenance et masques, purge des horizons et scaler
train global. Les [pipelines réutilisables](reusable-pipelines.md) ajoutent DAG,
multi-timeframes et partitions ; qualifier univers, historiques et packages réels.

### Extensions C01-C08

Les implémentations sont vérifiées sur de petites fixtures locales, avec les dépendances
actuelles. Les APIs et commandes sont centralisées dans
[reusable-pipelines.md](reusable-pipelines.md). Le tableau sépare les critères de
clôture du code des prérequis d’usage réel.

| ID | Extension | État actuel | Critère de clôture du code | Prérequis pour l'usage réel |
|---|---|---|---|---|
| C01 | Révisions macro numérique → null | Implémenté : moteur sparse et bridge FRED/ALFRED retirent/restaurent un point | Tests retrait, restauration, fenêtres et invariance future | Vintages et horaires source qualifiés ; clé FRED uniquement si collecte API |
| C02 | Calendrier macro historisé | Implémenté : états complets versionnés, report/annulation, horloges actual/forecast/previous | Tests as-of, révisions et forecast gelé strictement avant annonce vérifiée | Snapshots/archives réels et horaires de publication ; accès selon fournisseur |
| C03 | Security master et univers à date | Implémenté : SEC/Nasdaq/Alpha Vantage, identités déclarées, différences de snapshots et univers as-of | Tests ticker/statut/radiation/replay ; aucun delisting déduit du silence | Référentiels et archives réels ; annuaires courants ne recréent pas l'ancien univers |
| C04 | Features cross-section | Implémenté : rang percentile, z-score, écart moyen/effectif univers/secteur ; groupes bitemporels | Tests masques, effectif, groupes/univers as-of et invariance future | Univers et classifications historisés ; classification utilisateur possible sans GICS |
| C05 | Fondamentaux plus étendus | Implémenté : mappings IFRS/custom explicites, segments, périodes fiscales déclarées et recettes de dette disjointes | Tests contextes/horloges/périodes, retraits et priorité du total direct | Filings représentatifs et contrôle comptable ; aucun mapping universel annoncé |
| C06 | Orchestration multi-source et sorties communes | Implémenté : DAG d'opérations versionnées, config/API/CLI, reprise, caps globaux, déduplication et exports choisis | Tests pipeline local multi-timeframes→dataset, interruption, partialité et cache altéré | Clés seulement pour les collectes activées ; autres APIs raccordables par handlers Python explicites |
| C07 | Panels/datasets partitionnés | Implémenté : reader Parquet par batches, état as-of, overlap/lookahead, writer immutable et scaler train global | Équivalence fenêtres/labels/stride avec calcul complet ; caps et empreinte du train vérifiés | Benchmark à grande échelle selon données ; buffers plafonnés, pas limite OS globale de RAM |
| C08 | Adaptateurs de formats spécifiés | Implémenté : Alpha Vantage `LISTING_STATUS`, SEC et Nasdaq ; parsers, préflight, caps et replay | Tests petits exports JSON/CSV/pipe, footer, statuts, secrets et identités | Alpha Vantage : clé facultative non demandée pour le socle ; autres produits natifs à spécifier/licencier séparément |


### Clés et comptes pour la collecte réelle

Ces conditions concernent la collecte réseau ; replay et calcul local fonctionnent sans
clés. Après l'inventaire initial offline, quatre clés ont servi à de petits tests
autorisés. Leurs valeurs sont exclues des sorties et captures. La
[vérification](immediate-verification.md) conserve les résultats ; les accès premium
demandent un contrôle du package souscrit.

| Source / usage | État du code ici | Prérequis réseau | Paiement / limite |
|---|---|---|---|
| LSE : prix, actions, références, macro/options selon catalogue | Collecteurs existants | `LSE_API_KEY`, compte et symboles/produits réellement disponibles | Clé annoncée gratuite ; couverture et quotas du compte à qualifier. [Source LSE](https://londonstrategicedge.com/free-market-data-api/) |
| FRED / ALFRED | Collecteur existant, API série/vintage | `FRED_API_KEY`, compte FRED | Clé d'identification requise ; droits et profondeur restent propres à la série, notamment données de tiers. [API v1](https://fred.stlouisfed.org/docs/api/api_key.html) |
| EIA | Collecteur existant | `EIA_API_KEY`, inscription | Clé gratuite. Bulk sans clé non nécessaire pour commencer. [Documentation officielle](https://www.eia.gov/opendata/documentation/APIv2.1.0.pdf) |
| OpenDART | Adaptateur ciblé existant | `OPENDART_API_KEY`, inscription ; code société, rapport et comptes choisis | Service gratuit en principe ; conditions et limites du compte. [Inscription](https://engopendart.fss.or.kr/uss/umt/EgovMberInsertView.do) |
| e-Stat | Adaptateur ciblé existant | `ESTAT_APP_ID`, inscription et ID d'application ; codes dimensionnels sélectionnés | Pas de package payant imposé ici. [FAQ officielle](https://www.e-stat.go.jp/api/api-dev/faq) |
| SEC submissions/companyfacts | Collecteur existant | Réseau et `User-Agent` identifiant application + contact réel, passé au client ou à la CLI | Pas de clé ni authentification pour `data.sec.gov`. Débit régulé. [API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), [accès](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data) |
| Cboe VIX, World Bank, TSMC, MOPS/TWSE, Eurostat, BoJ, NBS, USGS, GDELT DOC, COT | Adaptateurs ciblés existants | Pas de clé imposée par les chemins actuels du code ; URL/code exact, réseau, quotas et droits à vérifier par source | Pas d'achat nécessaire pour tenter un petit échantillon public ; accès public ne certifie pas tout droit ou toute profondeur. GDELT Events ici reste un import de petit TSV local |
| Nasdaq directories / SEC mappings vers master | Collecteur C03 implémenté ; vérification offline | Pas de clé nécessaire pour les fichiers publics retenus ; SEC demande un contact réel | Annuaire courant, pas garantie d'archives historiques. [Définitions Nasdaq](https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs) |
| Alpha Vantage `LISTING_STATUS`, alternative master | Parseur/collecteur dédié C08 implémenté ; vérification offline | `ALPHAVANTAGE_API_KEY` facultative, compte et quota ; pas nécessaire pour SEC/Nasdaq | Actifs/radiés à date documentés ; complétude et disponibilité historique à qualifier. [Contrat](https://www.alphavantage.co/documentation/#listing-status) |
| J-Quants natif, facultatif | Pas de connecteur ici | Compte + clé API V2 ; nouvelle config à définir, pas variable déjà supportée par PFE | Plan gratuit limité ; plans payants selon profondeur/fraîcheur/champs. [API V2](https://www.jpx.co.jp/english/corporate/news/news-releases/6020/20260119.html), [plans actuels](https://jpx-jquants.com/ja) |
| Autres flux Corée / Europe | Imports canoniques possibles ; connecteurs propres aux places non livrés | Fournisseur et usage à choisir, codes/venues/calendriers ; clé ou contrat selon service | Ni gratuité globale, ni abonnement universel requis : qualification au cas par cas |

J-Quants affiche au contrôle du 14 octobre 2026 un plan gratuit de deux années
excluant les douze dernières semaines. Ce plan n'est pas un accès récent complet.
Les sources étrangères déjà obtenues légalement peuvent aussi entrer par les
contrats de références existants, sans développer cette alternative native.

### Accès licenciés selon le produit

L'accès dépend du produit retenu et du contrat du compte. Un établissement déjà abonné
peut fournir un accès autorisé ; le calcul local des features utilise ensuite les
données fournies.

| Besoin réel | Code déjà présent / code restant | Prérequis externe |
|---|---|---|
| Consensus CA/EPS historiques, révisions et dispersion multi-analystes | `LicensedEstimateAdapter`, imports CSV/JSON, client injecté et features existent | Export ou client FactSet/I/B/E/S autorisé, package historique PIT exact, mappings et exemple réel. Une clé quelconque ne donne pas le package. [FactSet Estimates](https://insight.factset.com/resources/factset-consensus-estimates-datafeed), [packages I/B/E/S](https://www.lseg.com/content/dam/data-analytics/en_us/documents/brochures/lseg-data-for-quant-research-brochure.pdf) |
| Réseau industriel professionnel historisé, FactSet Revere/Supply Chain | Graphe et extraction documentaire existent ; adaptateur dédié au produit absent | Licence/export/spécification/identifiants et temporalité du package ; couverture et pourcentages non garantis même avec abonnement. [Catalogue FactSet](https://www.factset.com/marketplace/catalog) |
| Archives options synchrones profondes, NBBO, OI/champs spécifiques | Moteur et captures prospectives LSE existent ; compléter ingestion du produit choisi | Achat d'archive ou abonnement autorisé si les champs/profondeur manquent dans la source existante ; gros volume possible. [Cboe Option Quotes](https://datashop.cboe.com/option-quote-intervals) |
| News/transcripts professionnels normalisés et historisés | Import documents, extraction et events existent ; raccordement au feed choisi absent | Contrat et droit d'utiliser/conserver les textes, API/export spécifique. SEC/IR/GDELT permettent un corpus partiel ; aucun achat nécessaire pour cette version partielle |
| Historique de radiations/master ou profondeur crédit/places étrangères supérieure aux sources publiques | Contrats et moteurs réutilisables ; intégration selon produit | Archive accessible, gratuite ou sous licence, attestant la couverture voulue ; un fournisseur payant est une option, pas preuve automatique d'exhaustivité |

Les surprises demandent un historique de consensus pré-annonce qualifié. Les graphes
documentaires couvrent les seules relations présentes dans les rapports.

### Ressources et stockage

État local observé lors de l’inventaire du **14 octobre 2026** : environ **13 Gio disponibles** sur le
volume du repo (`df -h .`). `ollama` est dans le PATH. Un manifeste
`qwen2.5:3b-instruct` existe ; les **5 blobs référencés sont présents avec les tailles
déclarées**, environ **1,797 Gio** logiques déjà stockés. Le contenu des poids n'a
pas été lu lors de cet inventaire initial, leur hash intégral n'a pas été recalculé.
Une petite inférence Qwen a ensuite été vérifiée, sans pull de modèle ;
voir [preuve et limites](immediate-verification.md). RAM et qualité financière
ne sont pas qualifiées par ce contrôle.

| Travail | Disque supplémentaire | Autre ressource / limite |
|---|---|---|
| C01–C08, petits tests et exemples | Fichiers de code et petites fixtures ; aucun dataset lourd obligatoire | Dépendances Python actuelles et temps de développement |
| Évaluation extraction/sentiment avec le Qwen déjà présent | Aucun nouveau poids nécessaire si ce modèle est retenu et compatible ; corpus et réponses à archiver | Serveur local fonctionnel, digest, RAM/contexte, temps CPU/GPU et annotation humaine ; qualité non démontrée |
| Embeddings réels | Aucun si un modèle compatible est déjà fourni ; sinon poids du modèle choisi à installer | Aucun modèle d'embeddings distinct observé dans les manifests ; Qwen n'est pas présumé compatible avec `/api/embed`. [API modèles](https://docs.ollama.com/api/tags), [embeddings](https://docs.ollama.com/capabilities/embeddings) |
| PDF texte | Bibliothèque `pypdf` si absente ; taille exacte à vérifier avant installation | Chemin optionnel existant, ne traite pas automatiquement les scans |
| OCR/scans/tableaux complexes | Moteur OCR, dépendances et éventuellement poids/langues à choisir ; sorties temporaires par page | CPU/GPU, RAM, corpus de validation et revue ; extension facultative pour les documents qui l'exigent |
| Captures publiques ciblées | Souvent petits fichiers avec les caps actuels ; une réponse complète peut dépasser le cap | Choisir explicitement budget, fréquence et rétention ; certaines sources ne filtrent pas le fichier côté serveur |
| Années d'actions, centaines de features et univers large | Peut dépasser le volume libre, même si prix daily seuls restent plus petits | APIs partitionnées C07 disponibles ; le builder complet reste en RAM. Choisir colonnes/périodes/budgets dans le repo consommateur |
| Options intraday, archives news/PDF monde ou ticks | Potentiellement très important ; stockage externe/distant selon périmètre | Bande passante, décompression et copies raw/processed/exports à compter |

**Estimations de mémoire**, pour 2 520 séances en float64, hors compression Parquet :

- 100 titres × 100 features : valeurs seules **0,188 Gio** ; arrays actuels d'un
  panel avec masques/horloges/provenance **0,681 Gio**, hors tables de preuves.
- 5 000 titres × 500 features : valeurs seules **46,939 Gio** ; arrays du panel
  **170,153 Gio**, hors preuves et autres copies. Cela dépasse la capacité d’un panel
  complet en RAM sur ce volume ; C07 permet un traitement partitionné sous budget.
- Base du calcul : `T × N × F × 8` octets pour les seules valeurs ; le panel
  actuel compte environ 29 octets par cellule + l'axe temporel. Les fenêtres
  GRU/TCN matérialisées peuvent ajouter des copies importantes.

À titre de référence fournisseur, Cboe estime une année SPY à la minute avec
calculs à **20 GB compressés / 145 GB décompressés**, contre 1 200 GB compressés
pour le marché entier à la minute sans calculs. Ces chiffres décrivent le produit
Cboe sélectionné, pas les petits samples PFE. [Estimations officielles](https://datashop.cboe.com/option-quote-intervals).

Les petites captures de qualification tiennent dans l'espace disponible. Le stockage
d'un historique étendu dépendra du périmètre choisi.

### Qualification et configuration restantes

- **Preuve historique** : horaires de publication et versions réellement
  disponibles. Une API latest-only interrogée aujourd'hui ne reconstitue pas ce
  qui était connu dix ans auparavant. Archives ou collecte prospective requises.
- **Identités** : relier sociétés, filiales, classes, ADR, CIK, venues et changements
  de ticker. Ambiguïtés à résoudre sur documents/référentiels ; pas seulement du code.
- **Comptabilité** : vérifier concepts IFRS/custom, segments, consolidation de dette,
  périodes fiscales et EPS sur de vrais états. Un parseur syntaxiquement valide
  ne certifie pas cette équivalence économique.
- **Évaluation LLM** : constituer un petit corpus réel annoté, fixer des seuils,
  mesurer précision/rappel par champ et direction de relation, auditer les rejets.
  L'évaluateur existe ; le corpus synthétique ne qualifie pas le modèle local.
- **Revue des faits** : le pipeline existant exige une revue nommée avant usage
  confirmé. Une politique alternative d'acceptation automatique doit être explicite
  et validée ; elle ne découle pas de la seule présence du modèle.
- **Configuration** : univers, séries exactes, base d'ajustement, fréquences, délais,
  secteurs et budgets. Certains choix sont fournis par l'utilisateur/fournisseur,
  d'autres peuvent être documentés depuis un petit audit de source.
- **Exécution régulière** : machine active et mécanisme de lancement si des captures
  prospectives périodiques sont voulues. Configurer ce lancement séparément de l'inventaire.
- **Droits effectifs** : usage des textes, redistribution, conservation et
  entraînement selon la source choisie. Gratuité d'accès, licence du SDK et licence
  de données sont des choses distinctes.

### Ordre pratique

Contrats/master → daily/actions/features → références + macro PIT → SEC et
extraction locale → graphes/events → contextes étrangers → sources premium selon
besoin confirmé. Choisir univers, séries, fréquences, sorties et budgets dans le
repo consommateur ; les lots peuvent se chevaucher sous leurs garanties temporelles.

## 10. Limites de l'audit

L'audit initial porte sur le code, les fichiers locaux et la documentation publique. Les
statistiques sont celles annoncées par les fournisseurs ; volumes réels et droits du
compte demandent des captures et contrôles dédiés avant intégration. La section 2 et les
lots décrivent les socles existants ; l'[état et
prérequis](#9-travail-restant-et-méthode) indique les travaux et validations restants.
Les [contrôles d'intégration](immediate-verification.md) précisent les tests réalisés,
les services joints et la couverture vérifiée.

