# RAPPORT COMPARATIF SCIENTIFIQUE — EXPÉRIENCES A, C ET D

**Projet :** PROJ-IASD01 — Détection des Sites Web de Phishing par Classification  
**Date :** 2026-10-30  
**Auteur :** Machine Learning Senior Engineer  
**Statut :** Évaluation finale validée — Expériences A, C et D  

---

## 1. Vue d'Ensemble et Objectifs des Trois Expériences

Dans le cadre du projet PROJ-IASD01, trois approches de modélisation ont été évaluées de manière rigoureuse sur le corpus PhiUSIIL (235 370 observations dédupliquées, 188 492 Train, 46 878 Test) avec un partitionnement strict par nom d'hôte (`GROUP_COL = "Host"`) sans aucun chevauchement :

1. **Expérience A — Modèle Complet (FULL FEATURES — 50 variables numériques + TLD) :**
   - Évalue la borne supérieure théorique de performance lorsque l'ensemble des métriques de contenu HTML, de structure JavaScript, de requêtes web et de chaîne d'URL sont précollectées.
2. **Expérience C — Modèle Autonome d'URL (URL_ONLY — 18 variables recalculées) :**
   - Évalue la détection de phishing à partir de la seule chaîne d'URL brute (schéma, hôte, chemin, paramètres de requête, fragments), recalculée de manière déterministe et transparente sans accès réseau.
3. **Expérience D — Modèle Autonome d'Hôte (HOST_ONLY — 10 variables recalculées) :**
   - Évalue le pouvoir discriminant résiduel lorsque le modèle est restreint au seul nom d'hôte normalisé (FQDN sans protocole, port, chemin ou paramètres).

---

## 2. Tableau Comparatif Transversal des Meilleurs Modèles

Pour chaque expérience, le modèle optimal sélectionné par validation croisée (GroupKFold 5 plis par Hôte sur le Train) est un **RandomForest** :

| Métrique / Caractéristique | Expérience A (FULL_FEATURES) | Expérience C (URL_ONLY) | Expérience D (HOST_ONLY) |
|:---|:---:|:---:|:---:|
| **Espace de caractéristiques** | 50 numériques + 1 catégorielle `TLD` (51 colonnes) | 18 numériques recalculées | 10 numériques recalculées |
| **Nature des variables** | Précollectées (URL + HTML + JS + TLD) | Recalculées statiques (chaîne URL) | Recalculées statiques (nom d'hôte) |
| **Modèle optimal retenu par CV** | **RandomForest** (`max_depth=None`, `n_est=100`) | **RandomForest** (`max_depth=20`, `n_est=200`) | **RandomForest** (`max_depth=20`, `n_est=200`) |
| **Temps d'entraînement RF (GridSearch)** | 356.4 s | 223.9 s | env. 210 s |
| **Train CV AP (PR-AUC)** | **1.0000 ± 0.0000** | **0.9981 ± 0.0002** | **0.8544 ± 0.0069** |
| **Train CV ROC-AUC** | 1.0000 ± 0.0000 | 0.9975 ± 0.0002 | 0.8419 ± 0.0047 |
| **Train CV F1-score (seuil 0.5)** | 0.999975 ± 0.000023 | 0.9941 ± 0.0004 | 0.7303 ± 0.0184 |
| **Test PR-AUC (scores continus)** | **1.0000** | **0.9979** | **0.8585** |
| **Test ROC-AUC** | 1.0000 | 0.9974 | 0.8473 |
| **Test Precision (seuil 0.5)** | 1.0000 | 0.9969 | 0.8845 |
| **Test Recall (seuil 0.5)** | 1.0000 | 0.9905 | 0.6367 |
| **Test F1-score (seuil 0.5)** | **1.0000** | **0.9936** | **0.7404** |
| **Faux Positifs (FP) Test** | **0** | **62** | **1 649** |
| **Faux Négatifs (FN) Test** | **0** | **189** | **7 202** |
| **Sous-groupe G1 PR-AUC [IC 95%]** | 1.0000 [1.0000, 1.0000] | 0.8426 [0.8069, 0.8724] | 0.2727 [0.2231, 0.3302] |
| **Sous-groupe G1 Erreurs (FP / FN)** | FP = 0 / FN = 0 | FP = 62 / FN = 86 | FP = 1 649 / FN = 286 |
| **Sous-groupe G2 PR-AUC [IC 95%]** | 1.0000 [1.0000, 1.0000] | 0.5521 [0.4823, 0.6213] | 0.2254 [0.1624, 0.2958] |
| **Sous-groupe G2 Erreurs (FP / FN)** | FP = 0 / FN = 0 | FP = 62 / FN = 86 | FP = 1 649 / FN = 108 |
| **Déploiement autonome hors ligne** | **NON** (requête HTTP & rendu web obligatoires) | **OUI** (inspection de chaîne < 1 ms) | **OUI** (inspection FQDN < 0.1 ms) |

---

## 3. Analyse Critique des Résultats et Compromis Opérationnels

### 3.1. Performance Absolue de l'Expérience A (FULL FEATURES)
- L'expérience A atteint une séparation parfaite sur le jeu de test : **PR-AUC = 1.0000**, **F1 = 1.0000**, avec **0 faux positif** et **0 faux négatif** sur 46 878 observations.
- Cette performance s'explique par la combinaison des 50 caractéristiques :
  - Les caractéristiques HTML (`LineOfCode`, `LargestLineLength`, `HasTitle`, balises `<iframe>`, balises de formulaire) fournissent un signal déterminant sur la nature de la page.
  - Les variables dérivées (`URLSimilarityIndex`, `DomainTitleMatchScore`) offrent un pouvoir de discrimination direct entre marques légitimes et imitations.
- **Mise en garde méthodologique (Biais de collecte Page) :** Cette quasi-perfection statistique doit être interprétée avec prudence. Dans le dataset PhiUSIIL, les sites de phishing ayant été scrapés après leur signalement comportaient potentiellement des pages d'erreur HTTP ou des bannières de suspension de l'hébergeur. Dans un environnement opérationnel temps réel, un modèle exploitant ces variables exige un téléchargement réseau complet et expose le système d'inspection au risque d'attaques actives (scripts malveillants, détection de robots).

### 3.2. Viabilité Opérationnelle de l'Expérience C (URL_ONLY)
- L'expérience C atteint **F1 = 0.9936** et **PR-AUC = 0.9979** sans aucun appel réseau.
- Elle ne concède que **62 fausses alertes** et **189 faux négatifs** sur 46 878 URLs, tout en s'exécutant en moins d'une milliseconde par URL.
- **Conclusion opérationnelle :** Pour un déploiement en passerelle de messagerie, résolveur DNS ou extension de navigateur, l'expérience C offre le meilleur compromis sécurité / débit / efficacité.

### 3.3. Rôle de l'Expérience D (HOST_ONLY)
- L'expérience D montre les limites du filtrage au niveau nom d'hôte seul (**F1 = 0.7404**, **FN = 7 202**).
- Beaucoup d'attaques de phishing modernes utilisent des domaines légitimes compromis ou des plateformes de cloud partagé (ex: `sites.google.com`, `firebaseapp.com`) où seul le chemin ou la requête permet de détecter la menace. L'expérience D ignore structurellement ces signaux.

---

## 4. Synthèse Décisionnelle

| Critère de Choix | Expérience A | Expérience C | Expérience D |
|:---|:---:|:---:|:---:|
| **Taux de détection (Rappel)** | **100.00 %** | 99.05 % | 63.67 % |
| **Précision (Faux positifs min)** | **100.00 %** | 99.69 % | 88.45 % |
| **Vitesse d'inférence** | Lente (téléchargement HTTP requis) | Ultra-rapide (< 1 ms) | Instantanée (< 0.1 ms) |
| **Sécurité d'exécution** | Risque de contact réseau | Zéro contact réseau | Zéro contact réseau |
| **Robustesse aux domaines partagés** | Excellente | Excellente | Faible |
| **Cas d'usage recommandé** | Sandbox forensique / Investigation asynchrone | Passerelle réseau en ligne / Navigateur | Filtrage DNS préliminaire par réputation |
