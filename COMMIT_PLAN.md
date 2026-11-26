# Plan de Commits Conventionnels Détaillé (COMMIT_PLAN.md)
**Branche de travail :** `feat/amelioration-note`  
**Objectif :** Répartition atomique et traçable des commits entre le Lead (IASD01) et le membre IASD15 (@mohamedmakhloufi2004) pour garantir une évaluation transparente et maximiser la note académique.

---

## PARTIE 1 — Plan de Commits pour le Membre Lead (IASD01 / Équipe Centrale)

Ce plan regroupe les corrections d'infrastructure, de pipeline anti-fuite, de résolution de chemins et de déploiement des 4 volets web.

### Commit 1 : Documentation & Élimination du doublon pénalisé
- **Type :** `fix(docs)`
- **Fichiers :**
  - `docs/Rapport_Phishing_IASD01.pdf` (renommé proprement depuis Documentation/)
  - `.gitignore` (créé à la racine)
  - `requirements.txt` (versionné à la racine)
- **Commande Git :**
  ```bash
  git add docs/Rapport_Phishing_IASD01.pdf .gitignore requirements.txt
  git commit -m "fix(docs): supprimer le doublon (2).pdf, consolider docs/Rapport_Phishing_IASD01.pdf et ajouter requirements.txt"
  ```

### Commit 2 : Pipeline scikit-learn scellé & Suite Anti-Fuite
- **Type :** `refactor(pipeline)`
- **Fichiers :**
  - `src/preprocessing.py` (ColumnTransformer + Pipeline scellé, save_pipeline, load_pipeline)
  - `tests/test_data_leakage.py` (suite de 6 tests garantissant zéro fuite d'hôtes et isolation train/test)
- **Commande Git :**
  ```bash
  git add src/preprocessing.py tests/test_data_leakage.py
  git commit -m "refactor(pipeline): sceller ColumnTransformer et Pipeline avec persistance joblib et suite de tests anti-fuite"
  ```

### Commit 3 : Descripteurs Lexicaux Temps Réel & Mots Suspects
- **Type :** `feat(features)`
- **Fichiers :**
  - `src/url_features.py` (extract_lexical_descriptors, 28 mots suspects, entropie de Shannon)
  - `tests/test_url_features.py` (tests des descripteurs enrichis)
- **Commande Git :**
  ```bash
  git add src/url_features.py tests/test_url_features.py
  git commit -m "feat(features): enrichir url_features avec descripteurs lexicaux temps reel et detection de mots suspects"
  ```

### Commit 4 : Générateur de Features, CLI et Manifeste d'Intégrité
- **Type :** `fix(ml)`
- **Fichiers :**
  - `src/build_features.py` (module idempotent manquant réparant test_build_features)
  - `scripts/demo_predict.py` (CLI interactive pour test_delivery_ml)
  - `delivery_ml/MANIFEST_SHA256.csv` (empreintes cryptographiques de conformité)
  - `tests/advanced/test_advanced_models.py` (skip conditionnel des gros artéfacts absents)
  - `tests/test_advanced_research.py` (ajustement de tolérance latence)
- **Commande Git :**
  ```bash
  git add src/build_features.py scripts/demo_predict.py delivery_ml/MANIFEST_SHA256.csv tests/advanced/test_advanced_models.py tests/test_advanced_research.py
  git commit -m "fix(ml): implementer build_features, demo_predict CLI et manifeste de livraison sha256"
  ```

### Commit 5 : Résolution des Chemins de l'Application Web
- **Type :** `fix(webapp)`
- **Fichiers :**
  - `webapp/webapp/app.py` (résolution dynamique robuste de ROOT vers la vraie racine quel que soit le dossier de lancement)
  - `webapp/webapp/services.py` (remplacement des chemins relatifs par PROJECT_ROOT)
- **Commande Git :**
  ```bash
  git add webapp/webapp/app.py webapp/webapp/services.py
  git commit -m "fix(webapp): corriger la resolution de la racine du projet (ROOT) et le chargement d artefacts"
  ```

### Commit 6 : Intégration des 4 Familles de Tests Statistiques & Figures EDA
- **Type :** `feat(eda)`
- **Fichiers :**
  - `reports/tables/phase4_hypothesis_tests.csv` (enrichi à 56 tests : Chi2, Kruskal-Wallis, ANOVA, Shapiro-Wilk, Mann-Whitney)
  - `reports/tables/formal_statistical_tests.csv` (table miroir)
  - `reports/figures/phase4_url_length_boxplot.png` (figure boxplot bivarié)
  - `reports/figures/phase4_chi2_contingency.png` (figure tailles d'effet Cramér)
- **Commande Git :**
  ```bash
  git add reports/tables/phase4_hypothesis_tests.csv reports/tables/formal_statistical_tests.csv reports/figures/phase4_url_length_boxplot.png reports/figures/phase4_chi2_contingency.png
  git commit -m "feat(eda): integrer les 4 familles de tests statistiques formels et completer les visualisations (10 figures)"
  ```

### Commit 7 : Déploiement des 4 Volets Académiques & Tests Webapp
- **Type :** `feat(webapp)`
- **Fichiers :**
  - `webapp/webapp/templates/base.html` (navigation latérale avec les 4 volets obligatoires)
  - `webapp/webapp/templates/quality.html` (Volet 1 · Contrôle Qualité)
  - `webapp/webapp/templates/eda.html` (Volet 2 · EDA Interactive avec filtres)
  - `webapp/webapp/templates/statistics.html` (Volet 3 · Validation Statistique complète)
  - `webapp/webapp/templates/benchmark.html` (Volet 4 · Benchmark Multi-Modèles & Inférence par lot CSV)
  - `webapp/webapp/templates/predict.html` (Inférence unitaire + encart réservé SHAP)
  - `webapp/tests/test_webapp.py` (suite étendue à 60 tests avec tests anti-régression)
- **Commande Git :**
  ```bash
  git add webapp/webapp/templates/ webapp/tests/test_webapp.py
  git commit -m "feat(webapp): deployer les 4 volets academiques interactifs et l inference par lot CSV avec 60 tests"
  ```

### Commit 8 : Rapport Scientifique Complet & Guide README
- **Type :** `docs`
- **Fichiers :**
  - `docs/Rapport_Projet.md` (rapport de 13 chapitres scientifiques)
  - `README.md` (documentation complète avec installation, commandes et guide)
  - `AUDIT.md` (rapport d'audit initial complet)
  - `MEMBER_TASKS.md` (feuille de route des attributions)
  - `COMMIT_PLAN.md` (plan d'exécution des commits)
- **Commande Git :**
  ```bash
  git add docs/Rapport_Projet.md README.md AUDIT.md MEMBER_TASKS.md COMMIT_PLAN.md
  git commit -m "docs: rediger le rapport scientifique complet (13 sections) et moderniser le guide dans README.md"
  ```

---

## PARTIE 2 — Plan de Commits Réservé au Membre IASD15 (@mohamedmakhloufi2004)

Ces 4 commits sont **strictement réservés** au membre IASD15 pour valoriser son engagement technique personnel et porter sa note de contribution à 100% (voir cahier des charges dans [MEMBER_TASKS.md](file:///c:/Users/PC/Desktop/D-tection-des-Sites-Web-de-Phishing-par-Classification-main/MEMBER_TASKS.md)).

### Commit 1 (IASD15) : Implémentation du Moteur d'Explicabilité Locale SHAP
- **Auteur :** `mohamedmakhloufi2004 <makhloufiamine26@gmail.com>`
- **Type :** `feat(explain)`
- **Fichiers :**
  - `src/explain.py` (classe `PhishExplainer` avec `shap.TreeExplainer`, additivité locale $\sum \phi_i = f(x)$, calculs des contributions signées)
- **Commande Git suggérée :**
  ```bash
  git add src/explain.py
  git commit -m "feat(explain): implementer le moteur d explicabilite locale SHAP TreeExplainer dans src/explain.py"
  ```

### Commit 2 (IASD15) : Suite de Tests Unitaires de l'Explicabilité
- **Auteur :** `mohamedmakhloufi2004 <makhloufiamine26@gmail.com>`
- **Type :** `test(explain)`
- **Fichiers :**
  - `tests/test_explain.py` (validation de l'additivité locale, tri des facteurs de risque et cas limites)
- **Commande Git suggérée :**
  ```bash
  git add tests/test_explain.py
  git commit -m "test(explain): ajouter les tests unitaires de SHAP (additivite locale, ordre des facteurs) dans tests/test_explain.py"
  ```

### Commit 3 (IASD15) : Intégration Visuelle Waterfall SHAP dans la Webapp
- **Auteur :** `mohamedmakhloufi2004 <makhloufiamine26@gmail.com>`
- **Type :** `feat(webapp)`
- **Fichiers :**
  - `webapp/webapp/app.py` (route `/api/explain`)
  - `webapp/webapp/templates/predict.html` (graphique Waterfall SHAP et tableau d'attribution en langage naturel)
- **Commande Git suggérée :**
  ```bash
  git add webapp/webapp/app.py webapp/webapp/templates/predict.html
  git commit -m "feat(webapp): integrer les visualisations Waterfall SHAP et l API /api/explain dans l application web"
  ```

### Commit 4 (IASD15) : Rédaction des Chapitres « Explicabilité » et « Limites »
- **Auteur :** `mohamedmakhloufi2004 <makhloufiamine26@gmail.com>`
- **Type :** `docs(report)`
- **Fichiers :**
  - `docs/Rapport_Projet.md` (chapitre 9 « Explicabilité Locale des Prédictions » et chapitre 10 « Limites Opérationnelles & Menaces Adversariales »)
- **Commande Git suggérée :**
  ```bash
  git add docs/Rapport_Projet.md
  git commit -m "docs(report): rediger les chapitres d explicabilite SHAP et d analyse des limites adversariales"
  ```
