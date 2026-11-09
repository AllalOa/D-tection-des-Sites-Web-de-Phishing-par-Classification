# PROJ-IASD01 — Synthèse Exécutive Finale des Résultats (A, B, B', C, D)

**Projet :** Détection des Sites Web de Phishing par Classification Automatisée  
**Jeu de données :** Benchmark PhiUSIIL (235 370 observations dédupliquées)  
**Date :** 2026-10-30 (Harmonisé en Phase 8a-bis)  
**Auteur :** Ingénieur Machine Learning Senior, Rédacteur Scientifique  

---

## 1. Tableau Récapitulatif Global des 5 Expériences

Toutes les expériences ont été évaluées sur le même jeu de Test indépendant de 46 878 observations (19 826 phishing, 27 052 légitimes), sous partitionnement strict par hôte (`GroupShuffleSplit` / `GroupKFold(n_splits=5)`, 0 hôte commun) :

| Indicateur Évalué | A (FULL_FEATURES) | B (URL_PAGE) | B' (PURIFIED_CONTENT) | C (URL_ONLY) | D (HOST_ONLY) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Espace de Représentation** | 50 vars originales + TLD | 44 vars originales + TLD | 38 vars originales (sans TLD)* | 18 vars URL recalculées | 10 vars Host recalculées |
| **Origine des Données** | CSV PhiUSIIL brut | CSV PhiUSIIL brut | CSV PhiUSIIL brut | Extraction Python URL | Extraction Python Host |
| **Meilleur Modèle Retenu** | **RandomForest** | **RandomForest** | **RandomForest** | **RandomForest** | **RandomForest** |
| **Hyperparamètres Optimaux** | `max_depth=None`, `n_est=100` | `max_depth=None`, `n_est=200` | `max_depth=20`, `n_est=200` | `max_depth=20`, `n_est=200` | `max_depth=20`, `n_est=200` |
| **Train CV Average Precision** | $1.000000 \pm 0.000000$ | $0.999999 \pm 0.000001$ | $0.999994 \pm 0.000006$ | $0.998057 \pm 0.000159$ | $0.854412 \pm 0.006942$ |
| **Train CV F1-Score ($s=0.5$)** | $0.999975 \pm 0.000023$ | $0.999795 \pm 0.000061$ | $0.999417 \pm 0.000119$ | $0.994052 \pm 0.000411$ | $0.730297 \pm 0.018412$ |
| **Temps d'Entraînement RF** | 356.4 s | 558.3 s | 323.1 s | 223.9 s | ~ 210 s |
| **Test PR-AUC (Average Precision)**| **1.000000** | **0.99999998** | **0.999996** | **0.997916** | **0.858522** |
| **Test ROC-AUC** | **1.000000** | **0.99999998** | **0.999997** | **0.997390** | **0.847339** |
| **Test F1-Score ($s=0.5$)** | **1.000000** | **0.999899** | **0.999420** | **0.993650** | **0.740432** |
| **Test Précision ($s=0.5$)** | 1.000000 | 0.999899 | 0.999344 | 0.996853 | 0.884467 |
| **Test Rappel ($s=0.5$)** | 1.000000 | 0.999899 | 0.999496 | 0.990467 | 0.636740 |
| **Vrais Positifs (TP)** | 19 826 | 19 824 | 19 816 | 19 637 | 12 624 |
| **Faux Positifs (FP)** | **0** | **2** | **13** | **62** | **1 649** |
| **Vrais Négatifs (TN)** | 27 052 | 27 050 | 27 039 | 26 990 | 25 403 |
| **Faux Négatifs (FN)** | **0** | **2** | **10** | **189** | **7 202** |
| **Sous-Groupe G1 PR-AUC** | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 0.9983 [0.9951, 0.9999] | 0.8426 [0.8069, 0.8724] | 0.2727 [0.2231, 0.3302] |
| **Sous-Groupe G2 PR-AUC** | 1.0000 [1.0000, 1.0000] | 1.0000 [0.9998, 1.0000] | 0.9998 [0.9992, 1.0000] | 0.5521 [0.4823, 0.6213] | 0.2254 [0.1624, 0.2958] |
| **Exécution Hors Ligne Pure** | Non (Crawl web) | Non (Crawl web) | Non (Crawl web) | **Oui (< 1 ms)** | **Oui (< 0.1 ms)** |

*\* L'audit de conformité de la Phase 8a-bis a établi que B' a été réellement entraînée avec ses 38 caractéristiques numériques sans le transformateur TLD, suite à une condition de filtrage CLI stricte dans `train.py`.*

---

## 2. Enseignements Méthodologiques et Comparatifs

1. **Expérience A (Perfection Formelle) :**  
   Atteint $\text{F1}=1.0000, \text{FP}=0, \text{FN}=0$, mais exploite des variables dérivées à forte redondance descriptive et 26 variables de contenu collectées conjointement.
2. **Expérience B (Suppression des Variables Dérivées) :**  
   Démontre que la suppression des 6 variables dérivées ne fait apparaître que 4 erreurs sur 46 878 tests ($\text{F1}=0.9999$).
3. **Expérience B' (Purification et Absence de TLD) :**  
   B' élimine 6 variables d'URL non reproductibles en plus des variables dérivées. En raison d'une déviation au protocole, B' n'utilise pas non plus le TLD. Les deux expériences B et B' diffèrent donc de **7 entrées brutes**. Malgré cela, B' conserve une performance exceptionnelle ($\text{F1}=0.9994$, 23 erreurs sur Test).
4. **Expérience C (Autonomie Opérationnelle Hors Ligne) :**  
   Avec seulement 18 caractéristiques extraites de la chaîne URL en temps sub-milliseconde, C atteint $\text{F1}=0.9936$ et $\text{PR-AUC}=0.9979$, sans aucune dépendance réseau.
5. **Expérience D (Insuffisance de l'Hôte Seul) :**  
   La modélisation basée sur le seul nom d'hôte ne détecte pas 36.3 % des attaques ($\text{FN}=7\,202$), confirmant qu'une infrastructure saine peut héberger des URLs malveillantes.

---

## 3. Précautions d'Interprétation Scientifique

- **Biais du Préfixe `https://www.` :** 100 % des sites légitimes de Test débutent par `https://www.` contre seulement 2.36 % des sites de phishing dans PhiUSIIL. Une règle statique naïve atteint $\text{F1}=0.9880$, illustrant la forte présence de raccourcis statistiques dans le corpus.
- **Contraintes du Scraping en Production :** L'analyse de page (A, B, B') suppose un site en ligne et accessible sans masquage (*cloaking*), deux hypothèses souvent mises à mal par l'évasion adverse et la courte durée de vie des sites frauduleux.
- **Comparaison B' vs C :** B' et C ont des représentations d'URL distinctes ; leur comparaison illustre une différence globale de configuration et non une preuve causale stricte de l'apport des seules variables PAGE.

---

## 4. Recommandation Architecturale : Proposition Conceptuelle de Cascade

Une architecture en cascade à deux niveaux est envisagée pour équilibrer latence et profondeur :
- **Niveau 1 :** Filtrage instantané via l'Expérience C (URL pure, $< 1$ ms local).
- **Niveau 2 :** Analyse approfondie via l'Expérience B' pour les cas litigieux ou les URLs courtes sous HTTPS (profil G2).

> **Avertissement méthodologique :**  
> Cette cascade et les seuils d'incertitude indicatifs constituent une proposition conceptuelle et **n'ont pas fait l'objet d'une validation expérimentale formelle**.
