# RAPPORT COMPARATIF SCIENTIFIQUE — MODÈLES AVANCÉS (PHASE 4)

**Auteur :** Fateh (@fateh1818), Membre 4  
**Projet :** Détection des Sites Web de Phishing par Classification  
**Branche :** `feature/siad19-advanced-research`  
**Dataset :** PhiUSIIL (235 370 observations — 188 492 Train / 46 878 Test)  
**Partitionnement :** Strict par `Host` (`GroupShuffleSplit` / 5-Fold `GroupKFold`, 0 fuite)  
**Espace de caractéristiques :** 18 variables `URL_ONLY` recalculées en temps réel  

---

## 1. Vue d'Ensemble et Méthodologie

Ce rapport présente l'évaluation comparative rigoureuse entre la référence historique (Random Forest Expérience C) et trois algorithmes avancés de Gradient Boosted Decision Trees :
- **XGBoost 3.4.1** (`tree_method='hist'`)
- **LightGBM 4.7.0** (`num_leaves=63`)
- **CatBoost 1.2.10** (`depth=7`, ordered boosting)

### 1.1 Protocole d'Étanchéité Expérimentale
1. **Zéro fuite de données :** Train (188 492) et Test (46 878) découpés par `GroupShuffleSplit` sur `Host` (exactement identiques aux partitions historiques du projet).
2. **Prédictions Out-of-Fold (OOF) :** Calculées via un 5-Fold `GroupKFold` sur le Train. Aucune donnée du jeu de test n'a été vue pendant le réglage des seuils ou l'ajustement de la calibration.
3. **Calibration Post-Hoc :** Ajustement de modèles de Platt Scaling (régression logistique) et de régression Isotonique exclusivement sur les probabilités OOF du Train.
4. **Optimisation des Seuils :** Sélection du seuil optimal maximisant le F1-score sur les probabilités OOF, puis gel et évaluation stricte sur le Test.

---

## 2. Résultats Globaux Comparatifs sur le Jeu de Test (46 878 URLs)

| Modèle | Train CV AP (5-Fold) | Temps Entraînement | Test PR-AUC | Test ROC-AUC | Test F1 (s=0.5) | Précision (s=0.5) | Rappel (s=0.5) | FP / FN (s=0.5) | Seuil Optimal OOF | Test F1 (optimal) | Test FP / FN (opt) | Test Brier Score (Raw) | Test Brier Score (Isotonique) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForest (Baseline C)** | $0.998057 \pm 0.000159$ | 4.66 s | 0.997916 | 0.997390 | 0.993650 | 0.996853 | 0.990467 | 62 / 189 | 0.56 | 0.993850 | 52 / 191 | 0.005011 | 0.004735 |
| **XGBoost** | $\mathbf{0.998340 \pm 0.000104}$ | 1.69 s | $\mathbf{0.998200}$ | $\mathbf{0.997769}$ | 0.994404 | 0.998474 | 0.990366 | **30** / 191 | 0.69 | $\mathbf{0.994604}$ | **19** / 194 | 0.004244 | $\mathbf{0.004172}$ |
| **LightGBM** | $0.998296 \pm 0.000103$ | **1.12 s** | 0.998186 | 0.997742 | $\mathbf{0.994430}$ | 0.998272 | 0.990618 | 34 / 186 | 0.68 | 0.994554 | **19** / 196 | $\mathbf{0.004242}$ | 0.004181 |
| **CatBoost** | $0.998228 \pm 0.000135$ | 4.74 s | 0.998090 | 0.997575 | 0.993400 | 0.996095 | $\mathbf{0.990719}$ | 77 / **184** | 0.56 | 0.993826 | 57 / 187 | 0.005227 | 0.004794 |

### Enseignements Clés :
- **Supériorité Statistique de XGBoost & LightGBM :** Les deux modèles dépassent la Random Forest de référence sur l'AP en validation croisée ($0.99834$ vs $0.99806$), sur le PR-AUC Test ($0.99820$ vs $0.99792$) et sur le ROC-AUC Test ($0.99777$ vs $0.99739$).
- **Réduction Drastique des Faux Positifs :** À seuil standard ($0.5$), XGBoost divise par deux les faux positifs (de 62 à **30**). Avec le seuil optimal sélectionné sur validation ($0.69$), XGBoost et LightGBM n'engendrent plus que **19 faux positifs** sur 27 052 sites légitimes (précision de **99.90%**).
- **Rapidité d'Entraînement :** LightGBM et XGBoost s'entraînent 3 à 4 fois plus rapidement que Random Forest et CatBoost sur les 188 492 observations.

---

## 3. Analyse Approfondie sur les Sous-Groupes Vulnérables (G1 et G2)

Les sous-groupes permettent d'auditer la résistance des modèles face au biais du préfixe `https://www.` :
- **G1 (27 521 obs — 469 phishing, prévalence 1.70%) :** URLs commençant par `https://www.`
- **G2 (27 252 obs — 200 phishing, prévalence 0.73%) :** URLs de G1 ne comportant aucun chemin (hosts nus HTTPS). C'est le cas le plus difficile pour un modèle syntaxique.
- **G1_strict (269 obs — 269 phishing, prévalence 100%) :** Population disjointe de G2 ($G1 \setminus G2$, URLs sous `https://www.` avec un chemin/query).

| Modèle | Sous-Groupe | n total | n phishing | PR-AUC | PR-AUC IC 95% (Cluster Bootstrap) | ROC-AUC | F1 ($s=0.5$) | FP / FN ($s=0.5$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForest Baseline** | G1 | 27 521 | 469 | 0.8426 | $[0.8074, 0.8736]$ | 0.9368 | 0.8381 | 62 / 86 |
| | **G2 (Cas Critique)** | 27 252 | 200 | 0.5521 | $[0.4822, 0.6145]$ | 0.8517 | 0.6064 | 62 / 86 |
| | G1_strict | 269 | 269 | 1.0000 | — | 1.0000 | 1.0000 | 0 / 0 |
| **XGBoost** | G1 | 27 521 | 469 | 0.8499 | $[0.8142, 0.8798]$ | $\mathbf{0.9394}$ | $\mathbf{0.8633}$ | **30** / 90 |
| | **G2 (Cas Critique)** | 27 252 | 200 | 0.5943 | $[0.5270, 0.6539]$ | $\mathbf{0.8578}$ | 0.6471 | **30** / 90 |
| | G1_strict | 269 | 269 | 1.0000 | — | 1.0000 | 1.0000 | 0 / 0 |
| **LightGBM** | G1 | 27 521 | 469 | $\mathbf{0.8504}$ | $[0.8144, 0.8804]$ | 0.9384 | 0.8620 | 34 / 88 |
| | **G2 (Cas Critique)** | 27 252 | 200 | $\mathbf{0.5968}$ | $[0.5295, 0.6557]$ | 0.8556 | $\mathbf{0.6474}$ | 34 / 88 |
| | G1_strict | 269 | 269 | 1.0000 | — | 1.0000 | 1.0000 | 0 / 0 |
| **CatBoost** | G1 | 27 521 | 469 | 0.8363 | $[0.8009, 0.8671]$ | 0.9335 | 0.8207 | 77 / 89 |
| | **G2 (Cas Critique)** | 27 252 | 200 | 0.5237 | $[0.4488, 0.5882]$ | 0.8440 | 0.5722 | 77 / 89 |
| | G1_strict | 269 | 269 | 1.0000 | — | 1.0000 | 1.0000 | 0 / 0 |

### Analyse de Robustesse :
- **Gain Majeur sur G2 :** LightGBM et XGBoost font passer le PR-AUC de **0.5521 à ~0.596**, soit un gain net de **+4.4 points**. L'intervalle de confiance à 95% monte à $[0.53, 0.66]$.
- **Faux Positifs sur G2 :** Sur ce sous-ensemble de sites légitimes, les faux positifs chutent de **62 à 30** pour XGBoost (-51.6%).
- **G1_strict :** Les 269 attaques dotées d'un chemin ou d'une query sous `https://www.` sont détectées avec un score de perfection absolue ($100\%$ de rappel et $0$ faux positifs) par tous les modèles.

---

## 4. Étude de Calibration des Probabilités

Le score de Brier mesure l'écart quadratique moyen entre les probabilités prédites et la vérité terrain (plus il est faible, mieux le modèle est calibré) :

| Modèle | Brier Score Brut (OOF) | Brier Score Platt (OOF) | Brier Score Isotonique (OOF) | Brier Score Brut (Test) | Brier Score Platt (Test) | Brier Score Isotonique (Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForest Baseline** | 0.004912 | 0.004780 | 0.004544 | 0.005011 | 0.004892 | 0.004735 |
| **XGBoost** | 0.004156 | 0.004171 | 0.004070 | 0.004244 | 0.004265 | $\mathbf{0.004172}$ |
| **LightGBM** | $\mathbf{0.004111}$ | $\mathbf{0.004113}$ | $\mathbf{0.004027}$ | $\mathbf{0.004242}$ | $\mathbf{0.004248}$ | 0.004181 |
| **CatBoost** | 0.005323 | 0.005144 | 0.004878 | 0.005227 | 0.005053 | 0.004794 |

### Conclusions sur la Calibration :
1. **Calibration Native Supérieure :** LightGBM et XGBoost affichent nativement un Brier Score de $0.0042$, nettement meilleur que celui de Random Forest ($0.0050$). Leurs probabilités reflètent plus fidèlement le risque réel de phishing.
2. **Apport de la Régression Isotonique :** La régression isotonique permet de réduire encore l'erreur de probabilité pour tous les modèles sans perturber le classement des scores (monotonie garantie).

---

## 5. Benchmark Opérationnel de Latence et Empreinte Mémoire

Mesures réalisées sur CPU (30 répétitions par batch, warm-up exclu) :

| Modèle | 1 URL (Total / P50) | Batch 10 (par URL) | Batch 100 (par URL) | Batch 1000 (par URL) | Pic Mémoire (1000 URLs) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RandomForest Baseline** | 43.4 ms | 4.54 ms | 0.45 ms | 0.054 ms (54.2 ms total) | 1 038 KB |
| **XGBoost** | 6.69 ms | 0.65 ms | 0.068 ms | 0.009 ms (9.3 ms total) | 671 KB |
| **LightGBM** | 6.92 ms | 0.72 ms | 0.076 ms | 0.009 ms (9.3 ms total) | **562 KB** |
| **CatBoost** | **4.08 ms** | **0.40 ms** | **0.040 ms** | **0.005 ms** (4.7 ms total) | 572 KB |

### Synthèse Opérationnelle :
- **Inférence Unitaire :** CatBoost est le plus rapide avec seulement **4.08 ms** par prédiction unitaire, suivi de près par XGBoost (6.69 ms) et LightGBM (6.92 ms). Random Forest est 6 à 10 fois plus lent (43.4 ms).
- **Inférence par Lots (Batch 1000) :** Tous les modèles de boosting traitent 1 000 URLs en moins de 10 millisecondes (soit moins de $10\ \mu\text{s}$ par URL).
- **Consommation Mémoire :** LightGBM et CatBoost consomment presque **deux fois moins de mémoire vive** en pic que la Random Forest.

---

## 6. Synthèse Finale et Modèle Champion Recommandé

### Champion Retenu : **XGBoost (ou LightGBM)**
1. **Précision Exceptionnelle :** Divise les fausses alertes par 2 (de 62 à 30 à seuil standard, et jusqu'à 19 à seuil OOF optimal).
2. **Robustesse Améliorée :** Amélioration substantielle du sous-groupe G2 ($PR\text{-}AUC: 0.594$ vs $0.552$).
3. **Efficacité Opérationnelle :** Latence réduite d'un facteur 6 en production unitaire et entraînement 3× plus rapide.
4. **Intégrité Totale :** 100% hors-ligne, zéro fuite de données, validé par 96 tests unitaires et d'intégration.
