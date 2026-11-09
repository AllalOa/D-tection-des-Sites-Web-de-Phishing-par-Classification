# RAPPORT COMPARATIF SCIENTIFIQUE — EXPÉRIENCES A, B, C ET D

**Projet :** PROJ-IASD01 — Détection des Sites Web de Phishing par Classification  
**Date :** 2026-10-30  
**Auteur :** Machine Learning Senior Engineer  
**Statut :** Évaluation finale validée — Expériences A, B, C et D  

---

## 1. Vue d'Ensemble et Objectifs des Quatre Expériences

Dans le cadre du projet PROJ-IASD01, quatre approches de modélisation ont été évaluées de manière rigoureuse sur le corpus PhiUSIIL (235 370 observations dédupliquées, 188 492 Train, 46 878 Test) avec un partitionnement strict par nom d'hôte (`GROUP_COL = "Host"`) sans aucun chevauchement :

1. **Expérience A — Modèle Complet (FULL FEATURES — 50 variables numériques originales + TLD) :**
   - Évalue la borne supérieure théorique de performance lorsque l'ensemble des métriques de contenu HTML, de code JavaScript, de requêtes réseau, d'URL et les **6 variables dérivées** (`URLSimilarityIndex`, etc.) sont exploitées.
2. **Expérience B — Modèle sans Variables Dérivées (CONTENT_FREE — 44 variables numériques originales + TLD) :**
   - Évalue la performance obtenue en combinant les 18 variables d'URL originales et les 26 variables de contenu de page, mais en **excluant rigoureusement les 6 variables dérivées**. Cette expérience isole le rôle des variables dérivées face aux variables de contenu brutes.
3. **Expérience C — Modèle Autonome d'URL (URL_ONLY — 18 variables recalculées) :**
   - Évalue la détection de phishing à partir de la seule chaîne d'URL brute (schéma, hôte, chemin, paramètres de requête, fragments), recalculée de manière déterministe et transparente sans accès réseau.
4. **Expérience D — Modèle Autonome d'Hôte (HOST_ONLY — 10 variables recalculées) :**
   - Évalue le pouvoir discriminant résiduel lorsque le modèle est restreint au seul nom d'hôte normalisé (FQDN sans protocole, port, chemin ou paramètres).

---

## 2. Tableau Comparatif Transversal des Meilleurs Modèles

Pour chaque expérience, le modèle optimal sélectionné par validation croisée (`GroupKFold` 5 plis par Hôte sur le Train) selon le critère d'Average Precision est un **RandomForest** :

| Métrique / Propriété | Expérience A (FULL_FEATURES) | Expérience B (CONTENT_FREE) | Expérience C (URL_ONLY) | Expérience D (HOST_ONLY) |
|:---|:---:|:---:|:---:|:---:|
| **Espace de caractéristiques** | 50 numériques + TLD (51 colonnes) | 44 numériques + TLD (45 colonnes) | 18 numériques recalculées | 10 numériques recalculées |
| **Nature des variables** | Précollectées (URL + Page + Dérivées) | Précollectées (URL + Page, sans Dérivées) | Recalculées statiques (chaîne URL) | Recalculées statiques (nom d'hôte) |
| **Modèle optimal retenu par CV** | **RandomForest** (`None`, 100) | **RandomForest** (`None`, 200) | **RandomForest** (20, 200) | **RandomForest** (20, 200) |
| **Temps d'entraînement RF (GridSearch)** | 356.4 s | 558.3 s | 223.9 s | env. 210 s |
| **Train CV AP (PR-AUC)** | **1.0000 ± 0.0000** | **1.0000 ± 0.0000** (0.999999) | **0.9981 ± 0.0002** | **0.8544 ± 0.0069** |
| **Train CV ROC-AUC** | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 (0.999999) | 0.9975 ± 0.0002 | 0.8419 ± 0.0047 |
| **Train CV F1-score (seuil 0.5)** | 0.999975 ± 0.000023 | 0.999795 ± 0.000061 | 0.9941 ± 0.0004 | 0.7303 ± 0.0184 |
| **Test PR-AUC (scores continus)** | **1.0000** | **1.0000** (0.99999998) | **0.9979** | **0.8585** |
| **Test ROC-AUC** | 1.0000 | 1.0000 (0.99999998) | 0.9974 | 0.8473 |
| **Test Precision (seuil 0.5)** | 1.0000 | 0.9999 (0.999899) | 0.9969 | 0.8845 |
| **Test Recall (seuil 0.5)** | 1.0000 | 0.9999 (0.999899) | 0.9905 | 0.6367 |
| **Test F1-score (seuil 0.5)** | **1.0000** | **0.9999** (0.999899) | **0.9936** | **0.7404** |
| **Faux Positifs (FP) Test** | **0** | **2** | **62** | **1 649** |
| **Faux Négatifs (FN) Test** | **0** | **2** | **189** | **7 202** |
| **G1 PR-AUC [IC 95%]** | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] | 0.8426 [0.8069, 0.8724] | 0.2727 [0.2231, 0.3302] |
| **G1 Erreurs (FP / FN)** | FP = 0 / FN = 0 | FP = 2 / FN = 0 | FP = 62 / FN = 86 | FP = 1 649 / FN = 286 |
| **G2 PR-AUC [IC 95%]** | 1.0000 [1.0000, 1.0000] | 1.0000 [0.9998, 1.0000] | 0.5521 [0.4823, 0.6213] | 0.2254 [0.1624, 0.2958] |
| **G2 Erreurs (FP / FN)** | FP = 0 / FN = 0 | FP = 2 / FN = 0 | FP = 62 / FN = 86 | FP = 1 649 / FN = 108 |
| **Déploiement autonome hors ligne** | **NON** (crawl web requis) | **NON** (crawl web requis) | **OUI** (inspection URL < 1 ms) | **OUI** (inspection FQDN < 0.1 ms) |

---

## 3. Analyse Critique des Résultats et Contribution des Variables Dérivées

### 3.1. Comparaison A vs B : L'impact de l'exclusion des 6 variables dérivées
- **Observation majeure :** En retirant les 6 variables dérivées (notamment `URLSimilarityIndex`, `CharContinuationRate`, `TLDLegitimateProb`), la performance de l'Expérience B demeure extrêmement élevée :
  - **PR-AUC Test :** passe de 1.0000 (A) à **0.99999998** (B).
  - **F1 Test :** passe de 1.0000 (A) à **0.999899** (B).
  - **Erreurs totales sur 46 878 observations :** passent de 0 erreur (A) à seulement **4 erreurs** (2 faux positifs, 2 faux négatifs dans B).
- **Interprétation scientifique :**
  - Dans l'expérience A, la variable `URLSimilarityIndex` servait de raccourci prépondérant (`DecisionTree(depth=1)` obtenait déjà 0.9956 en AP).
  - L'expérience B démontre que **les 26 caractéristiques de page (`PAGE_FEATURES`) combinées aux caractéristiques d'URL suffisent presque intégralement à reproduire la quasi-séparation parfaite** du dataset PhiUSIIL.
  - Les variables de contenu (`LineOfCode`, `NoOfExternalRef`, `NoOfImage`, `HasSocialNet`, etc.) portent une séparation massive qui rend les signaux dérivés largement redondants dans ce dataset spécifique.

### 3.2. Analyse du Biais de Scraping et Limite de Généralisation
- **Présence d'un artefact structurel de collecte :**
  - Bien que B exclue les raccourcis statistiques de `DERIVED_FEATURES`, elle conserve l'ensemble des 26 `PAGE_FEATURES`.
  - Comme démontré dans l'analyse exploratoire (Phase 4), les sites de phishing collectés dans PhiUSIIL présentaient une proportion importante de pages inaccessibles, suspendues ou vides lors du crawling (`LineOfCode` médian = 0 en phishing contre 1 319 en légitime).
  - Par conséquent, la performance quasi-parfaite de B ($F1 \approx 0.9999$) reflète en partie ce **biais de disponibilité de contenu** inhérent aux collectes asynchrones de phishing, et **ne saurait être extrapolée comme une garantie d'infaillibilité sur des attaques de phishing actives servies en production**.
- **Contrainte opérationnelle :**
  - Contrairement aux expériences C et D qui opèrent de façon statique à très haut débit (résolveur DNS ou proxy périmétrique), l'expérience B requiert l'exécution complète d'une pile de capture web (rendu DOM, téléchargement des ressources CSS/JS/images), ce qui limite son déploiement à des cas d'usage asynchrones (ex: sandbox, analyse forensique).

### 3.3. Hiérarchie des Performances et Dégradation sur les Sous-Groupes (G1 et G2)
- Sur les sous-groupes complexes G1 (URLs en `https://www.`) et G2 (racines épurées) :
  - **Expérience A & B :** Maintiennent une PR-AUC quasi parfaite ($\approx 1.0000$) grâce aux variables de contenu HTML qui restent discriminantes même quand l'URL semble parfaitement légitime en surface.
  - **Expérience C (URL_ONLY) :** Chute à PR-AUC = 0.8426 sur G1 et 0.5521 sur G2, car l'URL ne porte plus de token textuel évident (pas de chemin ni de requête sur G2).
  - **Expérience D (HOST_ONLY) :** Chute sévèrement à PR-AUC = 0.2727 sur G1 et 0.2254 sur G2, démontrant qu'un nom d'hôte seul ne peut discriminer un site frauduleux usurpant la forme d'un domaine racine sécurisé.
