# PROJ-IASD01 — Rapport Scientifique de Consolidation de l'Expérience D

## 1. Objectif de l'Expérience D

L'expérience D a été conçue pour surmonter les limitations fondamentales identifiées lors des phases 2a à 2h :
1. **Élimination des fuites d'information par le préfixe** : Dans le dataset brut PhiUSIIL, 100 % des URLs étiquetées comme légitimes commencent par `https://www.`, tandis que les URLs de phishing présentent des schémas et préfixes hétérogènes. Une simple règle syntaxique triviale (`PrefixRule`) atteint 98.8 % de F1-Score sur le test, révélant un biais d'échantillonnage majeur du dataset.
2. **Reproductibilité stricte et élimination des artefacts** : L'analyse des compteurs de caractères du CSV brut (NoOfLettersInURL, etc.) a montré un décalage d'un caractère et des définitions non reproductibles depuis la chaîne URL brute.
3. **Élimination de la fuite intra-domaine** : 991 hôtes distincts possédaient des variantes dans la colonne `Domain` (ex: `www.paypal.com` et `paypal.com`), conduisant à une fuite entre plis lors d'une séparation basée uniquement sur `Domain`.
4. **Hypothèse opérationnelle de D** : Évaluer la capacité discriminante intrinsèque d'un modèle de classification supervisée entraîné **strictement sur des caractéristiques syntaxiques et statistiques de l'hôte normalisé** (`Host`), en isolant complètement les hôtes entre ensembles d'entraînement et de test.

---

## 2. Description des 10 caractéristiques HOST_ONLY

L'hôte est normalisé via la fonction `extract_host(url)` : suppression du schéma (`http://`, `https://`), suppression du préfixe `www.` initial éventuel, suppression des ports, des identifiants d'authentification (`user:pass@`), du chemin, de la requête et des fragments, suppression du point final éventuel, et conversion en minuscules.

Les 10 caractéristiques extraites sont :

1. **HostLength** : Longueur totale en caractères de l'hôte normalisé.
2. **HostTLDLength** : Longueur en caractères de l'extension de premier niveau (TLD extrait via `extract_tld_naive`).
3. **HostNoOfSubDomain** : Nombre de sous-domaines (composants du nom d'hôte séparés par des points, moins le domaine et le TLD).
4. **HostIsIP** : Indicateur binaire (1 si l'hôte est une adresse IPv4 valide, 0 sinon).
5. **HostNoOfDigits** : Nombre total de chiffres numériques (`0`-`9`) dans l'hôte.
6. **HostDigitRatio** : Ratio du nombre de chiffres sur la longueur de l'hôte (`HostNoOfDigits / HostLength`).
7. **HostNoOfLetters** : Nombre total de lettres alphabétiques (`a`-`z`) dans l'hôte.
8. **HostLetterRatio** : Ratio du nombre de lettres sur la longueur de l'hôte (`HostNoOfLetters / HostLength`).
9. **HostNoOfHyphens** : Nombre de tirets (`-`) présents dans l'hôte.
10. **HostLongestLabel** : Longueur en caractères du plus long label (segment séparé par un point) de l'hôte.

*(Note d'intégrité : La variable candidate `HostNoOfOtherSpecial` s'est révélée strictement constante à 0 sur l'ensemble des 235 370 observations du dataset après normalisation standard de l'hôte et a été exclue de l'espace des caractéristiques pour éviter toute variable dégénérée).*

---

## 3. Dataset et protocole de séparation par Host

- **Dataset source** : PhiUSIIL Phishing URL Dataset (UCI Machine Learning Repository).
- **Déduplication** : 235 795 observations brutes $\to$ 235 370 observations uniques (425 doublons stricts supprimés).
- **Nombre total d'hôtes distincts** : 219 094 hôtes uniques.
- **Protocole de split** : `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` avec regroupement strict par **`Host`** (`GROUP_COL = "Host"`).
- **Cloisonnement garanti** : Aucun hôte n'est commun au jeu d'entraînement et au jeu de test ($\text{Train} \cap \text{Test} = \emptyset$).
- **Répartition des volumes** :
  - **Train split** : 188 492 observations (80.08 %).
  - **Test split** : 46 878 observations (19.92 %).
- **Validation croisée (Train)** : 5 plis groupés par `Host` (`GroupKFold(n_splits=5)`), assurant qu'aucun hôte n'apparaît simultanément dans les plis d'entraînement et le pli de validation.

---

## 4. Modèles comparés

Tous les modèles ont été construits au sein de pipelines `scikit-learn` intégrant un prétraitement standard (`StandardScaler` pour les modèles sensibles à l'échelle : LogisticRegression, KNN, LinearSVC, SVC_RBF ; pas de mise à l'échelle pour DecisionTree et RandomForest).

1. **DummyClassifier** : Baseline statistique naïve (`strategy="prior"`).
2. **LogisticRegression** : Régression logistique avec régularisation L2 ($C \in \{0.1, 1.0, 10.0\}$).
3. **DecisionTree** : Arbre de décision CART (`max_depth` $\in \{5, 10, 20, \text{None}\}$).
4. **RandomForest** : Forêt aléatoire de 100 ou 200 arbres (`max_depth` $\in \{20, \text{None}\}$).
5. **KNN** : Plus proches voisins ($k \in \{5, 15\}$), évalué sur sous-échantillon stratifié de 39 999 observations.
6. **LinearSVC** : Support Vector Classifier linéaire (`max_iter=2000`).
7. **SVC_RBF** : Support Vector Classifier à noyau RBF (`probability=True`), évalué sur sous-échantillon stratifié de 29 999 observations.
8. **PrefixRule** : Baseline heuristique externe (`URL ne commençant pas par 'https://www.' => phishing`).

---

## 5. Tableau de Validation Croisée (5 plis groupés par `Host` sur Train)

*Source : `reports/tables/cv_results_D.csv`. Métrique de sélection principale : Average Precision (PR-AUC).*

| Modèle | Average Precision (PR-AUC) | ROC-AUC | Précision | Rappel | F1-Score | Hyperparamètres retenus |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **DummyClassifier** | $0.4281 \pm 0.0010$ | $0.5000 \pm 0.0000$ | — | — | $0.0000 \pm 0.0000$ | `strategy='prior'` |
| **LogisticRegression** | $0.8223 \pm 0.0148$ | $0.7929 \pm 0.0219$ | $0.9033 \pm 0.0045$ | $0.5762 \pm 0.0200$ | $0.7034 \pm 0.0158$ | `C=0.1` |
| **DecisionTree** | $0.8487 \pm 0.0067$ | $0.8387 \pm 0.0049$ | $0.8901 \pm 0.0050$ | $0.6158 \pm 0.0246$ | $0.7277 \pm 0.0182$ | `max_depth=20` |
| **RandomForest** *(retenu)* | $\mathbf{0.8544 \pm 0.0069}$ | $\mathbf{0.8419 \pm 0.0047}$ | $0.8879 \pm 0.0048$ | $0.6206 \pm 0.0252$ | $\mathbf{0.7303 \pm 0.0184}$ | `max_depth=20, n_estimators=200` |
| **KNN** *(39 999 obs.)* | $0.8260 \pm 0.0066$ | $0.8250 \pm 0.0039$ | $0.8590 \pm 0.0139$ | $0.6197 \pm 0.0233$ | $0.7198 \pm 0.0184$ | `n_neighbors=15` |
| **LinearSVC** | N/A *(pas de proba)* | $0.7929 \pm 0.0218$ | — | — | $0.7000 \pm 0.0157$ | `max_iter=2000, C=1.0` |
| **SVC_RBF** *(29 999 obs.)* | $0.8359 \pm 0.0068$ | $0.8178 \pm 0.0073$ | — | — | $0.7172 \pm 0.0196$ | `kernel='rbf', C=1.0` |

---

## 6. Tableau de Test Final (46 878 observations)

*Source : `reports/tables/test_metrics_D.csv`. Seuil standard appliqué.*

| Modèle | Seuil | PR-AUC | ROC-AUC | Précision | Rappel | F1-Score | TP | FP | TN | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DummyClassifier** | 0.5000 | 0.4229 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 27 052 | 19 826 |
| **LogisticRegression** | 0.5000 | 0.8335 | 0.8115 | 0.9086 | 0.6028 | 0.7248 | 11 952 | 1 203 | 25 849 | 7 874 |
| **DecisionTree** | 0.5000 | 0.8535 | 0.8444 | 0.8852 | 0.6357 | 0.7400 | 12 604 | 1 635 | 25 417 | 7 222 |
| **RandomForest** | 0.5000 | **0.8585** | **0.8473** | 0.8845 | 0.6367 | **0.7404** | 12 624 | 1 649 | 25 403 | 7 202 |
| **KNN** | 0.5000 | 0.8287 | 0.8264 | 0.8623 | 0.6402 | 0.7348 | 12 693 | 2 027 | 25 025 | 7 133 |
| **LinearSVC** | 0.0000 | 0.8335 | 0.8115 | 0.9161 | 0.5842 | 0.7134 | 11 582 | 1 061 | 25 991 | 8 244 |
| **SVC_RBF** | 0.5000 | 0.8417 | 0.8245 | 0.8984 | 0.6314 | 0.7416 | 12 519 | 1 416 | 25 636 | 7 307 |
| **PrefixRule** | *N/A* | **0.9863** | **0.9882** | **1.0000** | **0.9763** | **0.9880** | **19 357** | **0** | **27 052** | **469** |

---

## 7. Comparaison avec PrefixRule

Sur l'ensemble du jeu de test global :
- **PrefixRule** commet seulement **469 erreurs** (0 faux positif, 469 faux négatifs).
- **RandomForest (D)** commet **8 851 erreurs** (1 649 faux positifs, 7 202 faux négatifs).

Cependant, cette supériorité apparente de PrefixRule est un **artefact de construction du dataset PhiUSIIL** : les collecteurs ont extrait les sites légitimes uniquement parmi des domaines de premier plan préfixés par `https://www.` sans chemin, tandis que les sites de phishing provenaient de sources hétérogènes. Lorsque le préfixe est neutralisé (voir Section 8), l'heuristique de préfixe s'effondre totalement.

---

## 8. Résultats sur les Sous-Groupes d'URLs (G1 et G2)

Les sous-groupes sont définis depuis l'URL brute :
- **G1** : URLs commençant exactement par `https://www.` (27 521 obs. test : 27 052 légitimes, 469 phishing ; prévalence : 1.7042 %).
- **G2** : G1 ET sans chemin non vide (hors `/` final), sans requête ni fragment (27 252 obs. test : 27 052 légitimes, 200 phishing ; prévalence : 0.7339 %).

*Les deux sous-groupes comptent plus de 100 observations de phishing en test (469 $\ge$ 100 et 200 $\ge$ 100).*

### Tableau comparatif sous-groupes G1 et G2
*(Source : `reports/tables/subgroups_evaluation_D.csv` ; IC 95 % via 1 000 rééchantillonnages d'hôtes avec remise)*

| Sous-groupe | Modèle | Prévalence | PR-AUC [IC 95 %] | ROC-AUC [IC 95 %] | FP @ 0.5 | FN @ 0.5 | FP @ 0.9352 | FN @ 0.9352 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **G1** | **RandomForest** | 1.7042 % | 0.2727 [0.2231, 0.3302] | 0.7250 [0.6873, 0.7603] | 1 649 | 286 | 91 | 340 |
| **G1** | **LogisticRegression** | 1.7042 % | 0.2840 [0.2349, 0.3382] | 0.7046 [0.6630, 0.7425] | 1 203 | 290 | N/A | N/A |
| **G1** | **DecisionTree** | 1.7042 % | 0.2040 [0.1642, 0.2538] | 0.7248 [0.6867, 0.7604] | 1 635 | 286 | N/A | N/A |
| **G1** | **KNN** | 1.7042 % | 0.2131 [0.1698, 0.2634] | 0.7263 [0.6930, 0.7578] | 2 027 | 280 | N/A | N/A |
| **G1** | **LinearSVC** | 1.7042 % | 0.2839 [0.2349, 0.3375] | 0.7056 [0.6633, 0.7433] | 1 061 | 293 | N/A | N/A |
| **G1** | **SVC_RBF** | 1.7042 % | 0.2372 [0.1890, 0.2919] | 0.6848 [0.6365, 0.7309] | 1 416 | 282 | N/A | N/A |
| **G1** | **PrefixRule** | 1.7042 % | **0.0170** [0.0150, 0.0192] | **0.5000** [0.5000, 0.5000] | **0** | **469** | **0** | **469** |
| | | | | | | | | |
| **G2** | **RandomForest** | 0.7339 % | 0.2254 [0.1624, 0.2958] | 0.7657 [0.7252, 0.8064] | 1 649 | 108 | 91 | 139 |
| **G2** | **LogisticRegression** | 0.7339 % | 0.2454 [0.1798, 0.3133] | 0.7469 [0.7053, 0.7881] | 1 203 | 111 | N/A | N/A |
| **G2** | **DecisionTree** | 0.7339 % | 0.1444 [0.1014, 0.1907] | 0.7649 [0.7245, 0.8051] | 1 635 | 108 | N/A | N/A |
| **G2** | **KNN** | 0.7339 % | 0.1542 [0.1067, 0.2087] | 0.7516 [0.7081, 0.7936] | 2 027 | 103 | N/A | N/A |
| **G2** | **LinearSVC** | 0.7339 % | 0.2452 [0.1791, 0.3142] | 0.7458 [0.7034, 0.7870] | 1 061 | 112 | N/A | N/A |
| **G2** | **SVC_RBF** | 0.7339 % | 0.1637 [0.1103, 0.2285] | 0.7301 [0.6851, 0.7726] | 1 416 | 107 | N/A | N/A |
| **G2** | **PrefixRule** | 0.7339 % | **0.0073** [0.0063, 0.0084] | **0.5000** [0.5000, 0.5000] | **0** | **200** | **0** | **200** |

**Constat majeur sur G1 et G2** :
- **PrefixRule** est totalement aveugle sur G1 et G2 : elle prédit 100 % de légitimes, manquant l'intégralité des phishings (469 FN sur G1, 200 FN sur G2), avec un ROC-AUC strictement égal à 0.5000 et un PR-AUC égal à la prévalence brute.
- **Les modèles de Machine Learning (Expérience D)** conservent un signal discriminant réel (ROC-AUC de 0.725 à 0.766 sur RandomForest, PR-AUC de 0.22 à 0.28, soit plus de 15 à 30 fois la prévalence de base), démontrant que les caractéristiques d'hôtes capturent une structure informative indépendante du préfixe.

---

## 9. Comparaison des Politiques de Seuil (RandomForest D)

Le seuil haute précision `0.9352` a été déterminé sur les prédictions hors-pli du **TRAIN** (`cross_val_predict`, 5 plis groupés par Host) sous la contrainte $\text{Précision} \ge 0.99$ (atteignant $99.00\%$ de précision et $47.46\%$ de rappel sur le Train OOF).

### Comparaison sur le jeu de Test (46 878 obs.) :

| Politique de seuil | Seuil | TP | FP | TN | FN | Précision | Rappel | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard (0.5)** | 0.5000 | 12 624 | 1 649 | 25 403 | 7 202 | 88.45 % | 63.67 % | 0.7404 |
| **Haute Précision (Train OOF)** | **0.9352** | **9 399** | **91** | **26 961** | **10 427** | **99.04 %** | **47.41 %** | **0.6412** |

- Au seuil standard 0.5, le modèle privilégie un F1-score équilibré (0.7404) avec 1 649 fausses alertes.
- Au seuil haute précision 0.9352, les faux positifs s'effondrent de **1 649 à 91** (précision test de 99.04 %, conforme à la cible train $\ge 99\%$), au prix d'un rappel réduit à 47.41 % (10 427 phishings non détectés).

---

## 10. Tests Statistiques de McNemar (sur jeu de test, seuil 0.5)

Test binomial exact bidirectionnel (`scipy.stats.binomtest`) comparant les décisions binaires du modèle retenu (**RandomForest D**) aux modèles alternatifs sur les 46 878 observations de test :

1. **RandomForest vs DecisionTree** :
   - Paires discordantes : $b = 46$ (RF faux, DT vrai), $c = 52$ (RF vrai, DT faux), total = 98.
   - **p-value = 0.6137** ($p > 0.05$) : La différence de performance entre la forêt aléatoire et l'arbre de décision n'est pas statistiquement significative au seuil $\alpha = 0.05$.

2. **RandomForest vs SVC_RBF** :
   - Paires discordantes : $b = 1090$, $c = 962$, total = 2 052.
   - **p-value = 0.00504** ($p < 0.01$) : Différence statistiquement significative.

3. **RandomForest vs LogisticRegression** :
   - Paires discordantes : $b = 1641$, $c = 1867$, total = 3 508.
   - **p-value = 0.000145** ($p < 0.001$) : Différence hautement significative en faveur de RandomForest.

---

## 11. Intervalles de Confiance par Cluster Bootstrap

Les intervalles de confiance à 95 % ont été obtenus par **rééchantillonnage avec remise des grappes d'hôtes (`Host`)** sur 1 000 réplications (graine pseudo-aléatoire fixée à 42), selon la méthode des percentiles empiriques (percentiles 2.5 % et 97.5 %).

- Ce protocole respecte la dépendance intra-hôte : toutes les URLs associées à un même hôte sont incluses ensemble lorsqu'un hôte est échantillonné.
- Sur les sous-groupes G1 et G2, les intervalles de ROC-AUC et de PR-AUC confirment que les performances sont strictement supérieures au hasard :
  - G1 RandomForest : PR-AUC $\in [0.2231, 0.3302]$ (vs prévalence $0.0170$) ; ROC-AUC $\in [0.6873, 0.7603]$.
  - G2 RandomForest : PR-AUC $\in [0.1624, 0.2958]$ (vs prévalence $0.0073$) ; ROC-AUC $\in [0.7252, 0.8064]$.

---

## 12. Limitations Scientifiques et Biais du Dataset

1. **Biais de collecte massif sur le préfixe** :
   Dans PhiUSIIL, 100 % des 134 850 sites légitimes commencent par `https://www.`, contre seulement 2.45 % des sites de phishing. Un modèle entraîné sur l'URL complète sans contrôle apprend ce raccourci statistique artificiel.
2. **Absence de chemin dans les sites légitimes** :
   100 % des URLs légitimes du dataset sont des hôtes nus (`https://www.domain.tld/` sans aucun chemin d'accès), alors que les attaques de phishing modernes ciblent des sous-pages profondes.
3. **Validité externe non garantie** :
   Les performances rapportées sur PhiUSIIL **ne constituent en aucun cas une garantie de précision en environnement réel sur Internet**. Dans un contexte ouvert de cybersécurité :
   - Les attaquants enregistrent massivement des certificats SSL/TLS valides (`https://`) et peuvent configurer des sous-domaines `www.` ;
   - Le taux de base du phishing sur le Web ouvert est très inférieur à la prévalence artificielle du dataset ($\sim 42\%$).
4. **Conclusion méthodologique** :
   L'expérience D démontre qu'en éliminant les fuites et les artefacts du dataset (par isolation stricte des hôtes et extraction de 10 descripteurs morphologiques de l'hôte), on obtient un modèle d'ingénierie robuste, transparent et vérifiable, tout en révélant la véritable difficulté du problème masquée par les biais de PhiUSIIL.
