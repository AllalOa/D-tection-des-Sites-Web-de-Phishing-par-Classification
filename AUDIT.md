# AUDIT COMPLET — PROJ-IASD01
**Projet :** Détection des sites web de phishing par classification supervisée  
**Dataset :** PhiUSIIL Phishing URL Dataset (~235 795 lignes, UCI id 967)  
**Date d'audit :** Novembre 2026  
**Auditeur :** Ingénieur Data Mining / MLOps Senior  
**Note actuelle observée :** 15.0 / 20 (ML: 7.6/10 · UI: 3.2/4 · Git: 2.3/3 · Doc: 1.9/3)  
**Objectif visé :** 18.0 à 20.0 / 20

---

## 1. Synthèse Exécutive & Diagnostic des Notes

L'évaluation actuelle du chargé de cours attribue au projet la note de **15.0 / 20**, ventilée comme suit :
* **Rigueur technique & Data Mining :** 7.6 / 10
* **Application interactive (Flask/Streamlit) :** 3.2 / 4
* **Traçabilité Git & collaboration :** 2.3 / 3
* **Rapport scientifique & README :** 1.9 / 3

Le commentaire de synthèse relève expressément :  
> *« Commit 2d415e7 : Refonte webapp v2 déployée, rapport de 1 Mo (1517 lignes). Sanction mineure pour nom de fichier dupliqué "(2).pdf" et disparités de contribution... »*

En outre, deux membres de l'équipe (**mohamedmakhloufi2004** / IASD15 et **fateh1818** / SIAD19) sont pénalisés à **57% d'impact (8.52/20)** en raison d'un manque d'engagement perçu sur des briques critiques (explicabilité, intégration).

---

## 2. Analyse Détaillée Point par Point du Barème

### 2.1. Rigueur Technique & Data Mining (Score actuel : 7.6 / 10 | Potentiel : 9.5 - 10 / 10)

| Critère du barème | État actuel dans le repo | Statut | Anomalies & Manques constatés |
|---|---|---|---|
| **Pipeline scikit-learn hermétique** | `Pipeline` + `ColumnTransformer` implémentés dans `src/preprocessing.py`. | **Partiel** | Le pipeline sépare log1p et TLDGrouper mais n'est pas exposé sous une API unique persistée avec validation de contrat de bout en bout. |
| **Encodage / Standardisation dans le pipeline** | `TLDGrouper` et `StandardScaler` intégrés dans `build_pipeline()`. | **Conforme** | Appris uniquement sur `X_train`. |
| **Aucune fuite de données (Data Leakage)** | `split.py` utilise `GroupShuffleSplit` et `GroupKFold` par nom d'hôte/domaine. | **Partiel** | Absence de test unitaire automatisé dédié attestant mathématiquement de l'étanchéité stricte (vérification d'absence de contamination train/test sur target et variables). |
| **Split avant prétraitement** | Le découpage est réalisé avant tout ajustement de transformer. | **Conforme** | Respecté dans les scripts `src/train.py` et `src/split.py`. |
| **Validation croisée robuste** | 5-fold GroupKFold implémenté. | **Conforme** | Résultats sauvegardés dans `reports/tables/cv_results_*.csv`. |
| **Métriques adaptées (F1, ROC-AUC, PR-AUC, Confusion Matrix)** | PR-AUC, ROC-AUC, F1, matrices exportées. | **Conforme** | Évaluation multi-expériences (A, B, B', C, D). |
| **Descripteurs lexicaux en temps réel** | `src/url_features.py` calcule 18 features offline. | **À enrichir** | Manque l'extraction explicite des motifs suspects lexicaux et de tokenisation avancée requis pour l'inférence temps réel. |
| **Explicabilité locale (SHAP / LIME)** | Absent du pipeline actuel. | **Manquant** | **TÂCHE RÉSERVÉE** pour le membre **mohamedmakhloufi2004 (IASD15)**. |

---

### 2.2. Application Interactive Flask / Streamlit (Score actuel : 3.2 / 4 | Potentiel : 3.8 - 4 / 4)

Le barème académique exige formellement **au moins 4 volets interactifs** :
1. **Contrôle qualité des données :** Détection et visualisation des doublons (425 doublons exacts documentés), contrôle des valeurs manquantes (0/56), typage strict, statistiques d'intégrité.
2. **EDA interactive (≥ 10 figures pertinentes) :** Distributions univariées, déséquilibre de classe, corrélations de Pearson/Spearman, biais de structure (préfixe `https://www.`, variables de page artificielles), top TLDs.
3. **Validation statistique (≥ 4 tests d'hypothèses formels) :** 
   - Test du Chi-deux ($\chi^2$) d'indépendance pour les variables catégorielles/binaires.
   - Test ANOVA à 1 facteur ou Kruskal-Wallis pour les comparaisons multi-groupes (ex. par TLD ou par protocole).
   - Test de normalité de Shapiro-Wilk (ou D'Agostino-Pearson) pour justifier les approches non paramétriques.
   - Test non-paramétrique de Mann-Whitney $U$ pour les variables asymétriques/à queue lourde.
   *Avec formulation systématique de $H_0$, $H_1$, p-value, corrections multi-tests (Bonferroni, Benjamini-Hochberg) et taille d'effet (Cramer's V, Rank-Biserial r).*
4. **Benchmark de modèles (baseline vs ensembles) + Inférence par lot :**
   - Comparaison explicite DummyClassifier, DecisionTree (baseline règle simple), LogisticRegression, RandomForest et Gradient Boosting (XGBoost/LightGBM/CatBoost).
   - Formulaire d'import de fichier CSV de test avec exécution du pipeline en streaming et bouton d'export des prédictions (CSV enrichi avec classe et probabilité).
   - Onglet interactif « Tester une URL » avec prédiction instantanée et alertes heuristiques.

**État actuel de l'application Flask (`webapp/webapp/app.py`) :**
* Dispose d'un tableau de bord général (`/`), d'une vue de comparaison globale (`/experiments`), d'une évaluation par expérience (`/evaluation/<exp>`), d'une page sur les limites (`/limites`), et d'un formulaire mono-URL (`/predict`).
* **Lacunes majeures identifiées :**
  - Pas de volet dédié « Contrôle Qualité » interactif.
  - L'EDA n'est pas centralisée sous forme de galerie interactive riche (≥ 10 figures dynamiques).
  - La validation statistique formelle (les 4 tests requis avec $H_0/H_1$) n'a pas de vue dédiée dans l'interface web.
  - Absence totale d'inférence par lot (import CSV / export des prédictions).
  - Absence du volet explicabilité locale SHAP (réservé à IASD15).

---

### 2.3. Traçabilité Git & Collaboration (Score actuel : 2.3 / 3 | Potentiel : 2.8 - 3 / 3)

| Élément audité | Constat | Impact |
|---|---|---|
| **Structure des commits** | Commits récents en Conventional Commits (`feat:`, `refactor:`, `docs:`), mais historique avec des fusions non documentées. | Modéré |
| **Fichier `.gitignore`** | **ABSENT** à la racine du dépôt. Risque critique d'ajouter par erreur des artefacts volumineux (`.joblib`, `.csv`, `__pycache__`). | Élevé |
| **Disparités de contribution** | Deux membres (IASD15 et SIAD19) sont affichés à 57% d'impact (8.52/20) sur la plateforme d'évaluation. | Élevé |
| **Branche de travail** | Nécessité d'opérer sur une branche dédiée `feat/amelioration-note` sans push sauvage. | Conforme aux consignes |
| **Plan de travail collaboratif** | Absence d'un document de répartition claire des responsabilités restantes. | Modéré |

---

### 2.4. Rapport Scientifique & Documentation (Score actuel : 1.9 / 3 | Potentiel : 2.8 - 3 / 3)

| Élément audité | Constat | Impact |
|---|---|---|
| **Nom de fichier avec anomalie de téléchargement** | Présence dans `Documentation/` d'un fichier nommé `Documentation_PROJ-IASD01_Detection_Phishing (2).pdf`. Sanctionné explicitement par le chargé de cours. | Critique |
| **Doublons de documentation** | `Documentation/README.md` est une copie conforme inutile du `README.md` racine. | Modéré |
| **Fichier `requirements.txt` racine** | **ABSENT** à la racine du projet (seul un fichier incomplet existe dans `webapp/webapp/requirements.txt`). Le `README.md` racine invite pourtant à taper `pip install -r requirements.txt`. | Élevé |
| **Références obsolètes dans `README.md`** | Mentionne des notebooks inexistants (`03_modelisation.ipynb`, `04_evaluation.ipynb`) et des chemins non nettoyés. | Modéré |
| **Rapport scientifique formel (`docs/Rapport_Projet.md` / PDF)** | Le rapport existant de 17 pages doit être refondu et standardisé dans `docs/Rapport_Phishing_IASD01.pdf` (15 à 30 pages) avec toutes les sections académiques requises (contexte, données, EDA, tests d'hypothèses, modélisation, protocoles A à D, benchmarks, limites, conclusion, bibliographie). | Élevé |

---

## 3. Audit d'Exécution du Code et de la Suite de Tests

### 3.1. Environnement d'Exécution
* **Système d'exploitation :** Windows 11
* **Python :** 3.14.2 (compatible 3.11 / 3.13)
* **Packages clés présents :** `scikit-learn 1.8.0`, `pandas 2.3.3`, `numpy 2.4.0`, `scipy 1.17.0`, `statsmodels 0.15.0`, `Flask 3.1.3`, `pypdf 6.6.2`, `pytest 9.2.0`.

### 3.2. Erreurs et Échecs de la Suite Pytest (23 échecs sur 327 tests)
1. **Module fantôme introuvable :**
   - `tests/test_build_features.py` : `from src.build_features import build_features` déclenche `ModuleNotFoundError: No module named 'src.build_features'`. Le module `src/build_features.py` n'a jamais été implémenté.
2. **Scripts et manifestes manquants :**
   - `tests/test_delivery_ml.py` : échec car `delivery_ml/MANIFEST_SHA256.csv` et `scripts/demo_predict.py` sont introuvables.
3. **Artefacts de modèles avancés non versionnés :**
   - `tests/advanced/test_advanced_models.py` : échec car `models/advanced/randomforest_baseline_url_only.joblib` est absent.
4. **Tests nécessitant les données brutes lourdes :**
   - Tests de partitionnement et d'intégrité de hash dans `test_phase5c.py`, `test_phase6a.py`, `test_phase6c.py`, `test_preparation_phase5a.py` : recherchent `data/processed/url_features_all.csv` ou le CSV brut de 235 795 lignes (qui ne doit pas être versionné sur GitHub sans LFS).
5. **Seuil de latence strict :**
   - `test_advanced_research.py` : `mean_ms` (179 ms) > 100 ms sur machine locale sous Windows.

---

## 4. Matrice de Priorisation des Problèmes par Impact

| Priorité | Problème identifié | Rubrique | Impact sur la note | Action corrective requise |
|:---:|---|:---:|:---:|---|
| **P0** | Fichier dupliqué `(2).pdf` et duplication `Documentation/` | Doc (/3) | -0.6 pt | Supprimer le dossier `Documentation/`, renommer et consolider le rapport dans `docs/Rapport_Phishing_IASD01.pdf`. |
| **P0** | Absence de `requirements.txt` et `.gitignore` à la racine | Git & Doc | -0.5 pt | Créer `requirements.txt` versionné et `.gitignore` complet. |
| **P1** | Volets obligatoires manquants dans la Webapp (Qualité, EDA dynamique, Tests stats, Inférence par lot) | UI (/4) | -0.8 pt | Enrichir l'application Flask avec les 4 volets complets + inférence par lot (CSV import/export). |
| **P1** | Pipeline scikit-learn non scellé + absence de test anti-fuite | ML (/10) | -0.8 pt | Refactoriser et sceller le Pipeline/ColumnTransformer dans `src/preprocessing.py`, ajouter `tests/test_data_leakage.py`. |
| **P1** | Tests de normalité et multi-groupes manquants (Shapiro-Wilk, Kruskal-Wallis/ANOVA) | ML & UI | -0.5 pt | Calculer et intégrer les 4 tests statistiques requis dans `reports/tables/` et dans l'interface web. |
| **P1** | Modules ou scripts manquants faisant échouer `pytest` (`build_features.py`, `demo_predict.py`) | ML & Qualité | -0.6 pt | Créer `src/build_features.py` et les scripts utilitaires pour rétablir 100% de succès aux tests applicables. |
| **P2** | Enrichissement des descripteurs lexicaux temps réel (`src/url_features.py`) | ML (/10) | -0.3 pt | Ajouter les descripteurs lexicaux manquants (mots-clés suspects, entropie, tokens). |
| **P2** | Rédaction du rapport scientifique complet `docs/Rapport_Projet.md` + conversion PDF | Doc (/3) | -0.5 pt | Rédiger le rapport complet de 15 à 30 pages selon les standards académiques. |
| **P2** | Pénalité individuelle de contribution sur IASD15 (@mohamedmakhloufi2004) | Git (/3) | -0.7 pt | Créer `MEMBER_TASKS.md` pour lui confier la responsabilité exclusive du module SHAP/LIME, des tests unitaires associés et des sections dédiées du rapport. |

---

## 5. Délimitation Stricte des Tâches pour IASD15 (@mohamedmakhloufi2004)

Pour corriger la note de contribution de **mohamedmakhloufi2004 (IASD15)** de 8.52/20 vers 18-20/20, les travaux suivants lui sont **STRICTEMENT RÉSERVÉS** et ne sont pas exécutés par l'ingénieur d'audit :
1. **Module `src/explain.py` :** Implémentation du moteur d'explicabilité locale basé sur **SHAP** (`TreeExplainer` pour RandomForest / gradient boosting) et optionnellement **LIME** (`LimeTabularExplainer`).
2. **Intégration du Volet Explicabilité dans la Webapp :** Création de l'interface ou des graphiques (waterfall plot, force plot, summary plot) permettant à l'utilisateur de comprendre pourquoi une URL spécifique a été classée comme phishing.
3. **Tests Unitaires d'Explicabilité :** Rédaction de `tests/test_explain.py` validant les explications locales, la somme des valeurs SHAP et la consistance des attributions.
4. **Sections du Rapport Scientifique :** Rédaction des sections « 8. Explicabilité Locale des Prédictions » et « 9. Limites Opérationnelles et Adversariales » dans le rapport final.

Le cahier des charges détaillé de ces travaux est consigné dans `MEMBER_TASKS.md`.
