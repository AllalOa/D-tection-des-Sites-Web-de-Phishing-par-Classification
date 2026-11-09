# PROJ-IASD01 — Rapport Comparatif Descriptif : Expérience C vs Expérience D

**Date d'exécution** : 2026-10-29  
**Auteur** : Ingénieur Machine Learning Senior (Classification & Validation Expérimentale)  
**Statut** : Rapport confirmatoire post-évaluation de la Phase 5b.

---

## 1. Cadre méthodologique et avertissement d'intégrité

> [!IMPORTANT]
> **Déclaration d'antériorité et nature de l'analyse** :
> Les performances de l'expérience D (**HOST_ONLY**, 10 variables) ont été établies et figées lors de la Phase 3 (documentées dans 
eports/PHASE3_D_RESULTS.md et 
eports/tables/test_metrics_D.csv).
> L'entraînement et l'évaluation de l'expérience C (**URL_ONLY**, 18 variables recalculées) ont été autorisés et exécutés lors de la Phase 5b sans modifier ni les données, ni les partitions, ni le protocole de l'expérience D.
> La présente comparaison est **strictement descriptive et confirmatoire**. Aucun modèle, aucun hyperparamètre et aucun seuil n'ont été ajustés ou modifiés rétroactivement après consultation des scores sur le jeu de test.

---

## 2. Définition des espaces de caractéristiques

| Paramètre | Expérience C (URL_ONLY) | Expérience D (HOST_ONLY) | Différence méthodologique |
|---|:---:|:---:|---|
| **Nombre de variables** | **18** | **10** | C dispose de 8 variables supplémentaires liées au protocole, au chemin et aux paramètres |
| **Périmètre d'analyse** | URL brute complète (schéma + hôte + chemin + requête) | Nom d'hôte normalisé uniquement (extract_host) | D exclut volontairement le protocole, le chemin, les requêtes et les fragments |
| **Variables clés** | IsHTTPS, URLLength, NoOfSubDomain, DomainLength, compteurs de caractères | HostLength, HostTLDLength, HostNoOfSubDomain, HostIsIP, ratios de chiffres/lettres | C capture les motifs dans le chemin et le protocole ; D mesure uniquement la réputation lexicale de l'hôte |
| **Biais de protocole** | Présent (IsHTTPS = 1 pour 100 % des légitimes) | Éliminé (schéma supprimé par normalisation) | D est conçu pour neutraliser l'artefact de collecte https://www. |

---

## 3. Modèles retenus par Validation Croisée (Train CV)

Les deux expériences ont utilisé la même règle d'arbitrage : sélection exclusive selon la moyenne d'**Average Precision (PR-AUC)** sur 5 plis GroupKFold groupés par Host sur le Train (188 492 observations).

| Expérience | Meilleur modèle retenu | Hyperparamètres optimaux | Train CV AP (PR-AUC) | Train CV ROC-AUC | Train CV F1-score |
|---|---|---|:---:|:---:|:---:|
| **Expérience C** | **RandomForest** | 
_estimators=200, max_depth=20 | **0.9981 ± 0.0002** | **0.9975 ± 0.0002** | **0.9941 ± 0.0004** |
| **Expérience D** | **RandomForest** | 
_estimators=200, max_depth=20 | **0.8544 ± 0.0069** | **0.8419 ± 0.0047** | **0.7303 ± 0.0184** |

*Note sur les durées d'entraînement* :
- **RandomForest C** : 223.9 secondes (GridSearchCV, 4 combinaisons × 5 plis = 20 fits sur 188 492 observations, 18 variables).
- **RandomForest D** : environ 210 secondes (même protocole, 10 variables).

---

## 4. Tableau comparatif complet sur le Jeu de Test (46 878 observations)

*Effectifs réels du jeu de test : 27 052 légitimes (0) et 19 826 phishing (1).*

| Métrique d'évaluation | Expérience C (RandomForest) | Expérience D (RandomForest) | Écart (C − D) | Baseline PrefixRule |
|---|:---:|:---:|:---:|:---:|
| **Seuil de décision standard** | **0.5000** | **0.5000** | 0.0000 | *Règle binaire* |
| **PR-AUC (Average Precision)** | **0.9979** | **0.8585** | **+0.1394** | 0.9863 |
| **ROC-AUC** | **0.9974** | **0.8473** | **+0.1501** | 0.9882 |
| **Précision** | **0.9969** | **0.8845** | **+0.1124** | 1.0000 |
| **Rappel** | **0.9905** | **0.6367** | **+0.3538** | 0.9763 |
| **F1-score** | **0.9936** | **0.7404** | **+0.2532** | 0.9880 |
| **Vrais Positifs (TP)** | **19 637** | **12 624** | +7 013 | 19 357 |
| **Faux Positifs (FP)** | **62** | **1 649** | **-1 587** | **0** |
| **Vrais Négatifs (TN)** | **26 990** | **25 403** | +1 587 | 27 052 |
| **Faux Négatifs (FN)** | **189** | **7 202** | **-7 013** | 469 |
| **Total erreurs (FP + FN)** | **251** | **8 851** | **-8 600** | 469 |

---

## 5. Analyse de robustesse sur les sous-groupes G1 et G2

Les sous-groupes permettent de diagnostiquer la dépendance des modèles aux artefacts syntaxiques du dataset :
- **G1** : URLs commençant par https://www..
- **G2** : URLs de G1 dépourvues de chemin non vide, de requête ou de fragment (page d'accueil pure).

*Les intervalles de confiance à 95 % sont calculés par bootstrap groupé sur Host (1 000 tirages, graine 42).*

### Sous-groupe G1 (27 521 observations, dont 469 phishing — prévalence 1.70 %)

| Modèle | PR-AUC [IC 95 %] | ROC-AUC [IC 95 %] | FP (seuil 0.5) | FN (seuil 0.5) |
|---|:---:|:---:|:---:|:---:|
| **RandomForest C** | **0.8426** [0.8069, 0.8724] | **0.9368** [0.9187, 0.9533] | **62** | **86** |
| **RandomForest D** | 0.2727 [0.2231, 0.3302] | 0.7250 [0.6873, 0.7603] | 1 649 | 286 |
| **PrefixRule** | 0.0170 [0.0150, 0.0192] | 0.5000 [0.5000, 0.5000] | 0 | 469 (100 %) |

### Sous-groupe G2 (27 252 observations, dont 200 phishing — prévalence 0.73 %)

| Modèle | PR-AUC [IC 95 %] | ROC-AUC [IC 95 %] | FP (seuil 0.5) | FN (seuil 0.5) |
|---|:---:|:---:|:---:|:---:|
| **RandomForest C** | **0.5521** [0.4823, 0.6213] | **0.8517** [0.8124, 0.8861] | **62** | **86** |
| **RandomForest D** | 0.2254 [0.1624, 0.2958] | 0.7657 [0.7252, 0.8064] | 1 649 | 108 |
| **PrefixRule** | 0.0073 [0.0063, 0.0084] | 0.5000 [0.5000, 0.5000] | 0 | 200 (100 %) |

---

## 6. Analyse des seuils haute précision

Dans les applications de sécurité web (ex: blocage automatique d'URL), un faux positif pénalise lourdement l'expérience utilisateur. Un seuil haute précision (précision $\ge 0.99$ sur Train OOF) a été audité :

- **Expérience D (RandomForest)** :
  - Seuil fixé sur Train OOF : **0.9352**.
  - Performance Test à 0.9352 : Précision = **0.9858**, Rappel = **0.3340** (6 622 TP, 95 FP, 13 204 FN).
  - Bilan : Chute majeure du rappel nécessaire pour approcher 99 % de précision sur les seules caractéristiques d'hôte.

- **Expérience C (RandomForest)** :
  - Au seuil standard **0.5000**, la précision sur Train OOF est déjà de **0.9971** (Rappel = 0.9910).
  - Sur le Test à **0.5000**, la précision confirmatoire est de **0.9969** pour un rappel de **0.9905** (62 FP seulement).
  - Un ajustement de seuil haute précision n'est donc pas nécessaire pour C, le seuil nominal 0.5 respectant d'emblée la contrainte opérationnelle.

---

## 7. Interprétation scientifique et enseignements

1. **Supériorité des caractéristiques de l'URL complète** :
   L'expérience C surpasse très nettement l'expérience D sur toutes les métriques globales (F1 de 0.9936 vs 0.7404, erreurs réduites de 8 851 à 251). Cela démontre que les structures de chemin, de paramètres et de longueur d'URL apportent une information discriminante cruciale qui n'est pas présente dans le nom d'hôte seul.

2. **Impact du biais d'échantillonnage révélé par G1 et G2** :
   L'analyse de permutation confirme que IsHTTPS (importance 0.1814) et URLLength (importance 0.1360) sont les moteurs dominants du modèle C. Lorsque le préfixe https://www. est neutralisé dans G1, la PR-AUC de C passe de 0.9979 à 0.8426. Dans G2 (absence de chemin), elle chute à 0.5521. Néanmoins, même dans ces sous-groupes hostiles, C reste très nettement supérieur à D (0.5521 vs 0.2254 dans G2) et à PrefixRule qui échoue totalement.
