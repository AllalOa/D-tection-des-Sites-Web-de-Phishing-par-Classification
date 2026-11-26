# PROJ-IASD01 — Détection des Sites Web de Phishing par Classification

Système complet de Machine Learning et Data Mining (méthodologie CRISP-DM) pour la détection en temps réel de sites web de phishing à partir du dataset **PhiUSIIL** (UCI id 967, ~235 795 lignes).

**Environnement :** Python 3.11 / 3.13 / 3.14 | **scikit-learn :** 1.8.0 | **pandas :** 2.3.3 | **Flask :** 3.1.3

---

## 1. Installation Rapide

```bash
# 1. Cloner ou ouvrir le dossier du projet
cd D-tection-des-Sites-Web-de-Phishing-par-Classification

# 2. Créer et activer un environnement virtuel (optionnel mais recommandé)
python -m venv venv
source venv/bin/activate        # Linux / MacOS
.\venv\Scripts\activate         # Windows

# 3. Installer les dépendances certifiées
pip install -r requirements.txt
```

---

## 2. Lancement de l'Application Web Interactive (PhishGuard)

L'application web Flask comprend **les 4 volets académiques formels du barème** ainsi que l'inférence par lot et le test d'URL unitaire :

```bash
# Lancement direct en mode local
python webapp/webapp/app.py
```
Accéder ensuite à l'interface dans votre navigateur : **http://127.0.0.1:5000**

### Structure des 4 Volets dans l'Application :
1. **Volet 1 · Contrôle Qualité (`/quality`) :**
   - Audit des 235 795 observations, suppression des 425 doublons stricts (235 370 lignes nettes).
   - Contrôle d'exhaustivité (0 valeur manquante sur 56 variables).
   - Taxonomie des 4 familles de caractéristiques (18 URL_ONLY, 32 PAGE, 6 DERIVED, méta).
2. **Volet 2 · EDA Interactive (`/eda`) :**
   - Galerie analytique de **10 figures scientifiques haute résolution** (distributions univariées, corrélations de Spearman vs Pearson, boxplots de longueur, tailles d'effet Cramér, biais de protocole `https://www.`, etc.).
   - Filtres interactifs par catégorie.
3. **Volet 3 · Validation Statistique Formelle (`/statistiques`) :**
   - Les 4 familles de tests d'hypothèses obligatoires :
     * **Test du Chi-deux ($\chi^2$) d'indépendance** sur 19 variables binaires ($V$ de Cramér).
     * **Kruskal-Wallis $H$ & ANOVA à 1 facteur** sur les groupes multi-TLD ($p < 10^{-15}$).
     * **Test de normalité de Shapiro-Wilk** ($W < 0.85$, $p < 0.001$, justifiant les approches non-paramétriques).
     * **Test de Mann-Whitney $U$** sur 31 variables continues ($r$ bisérial de rang).
   - Formulation explicite de $H_0$, $H_1$, statistiques, p-values brutes et ajustées (Bonferroni, Benjamini-Hochberg FDR) et interprétations scientifiques.
4. **Volet 4 · Benchmark Multi-Modèles & Inférence par Lot (`/benchmark`) :**
   - Comparaison complète : DummyClassifier (baseline naïve), DecisionTree depth=1, LogisticRegression, DecisionTree CART, RandomForest (Champion déployable), XGBoost, LightGBM, CatBoost.
   - **Moteur d'inférence par lot :** Téléversez un fichier CSV contenant une liste d'URLs pour obtenir la classification instantanée avec export des résultats au format CSV enrichi.
5. **Inférence Temps Réel & Descripteurs Lexicaux (`/predict`) :**
   - Test instantané d'une URL quelconque (hors ligne, < 1ms) avec extraction de descripteurs lexicaux (IP, HTTPS, caractères spéciaux, entropie, mots suspects).
   - Espace réservé pour le volet d'explicabilité locale SHAP (confié au membre IASD15).

---

## 3. Lancement des Tests Automatisés (Pytest)

La suite de tests unitaires et d'intégration couvre l'intégrité des données, l'hermétisme des pipelines, l'anti-fuite de données et l'application web :

```bash
# Exécution ciblée des tests de l'application web (60 tests passants)
pytest -o pythonpath=. webapp/tests/test_webapp.py -v

# Exécution ciblée du test anti-fuite de données (Data Leakage)
pytest -o pythonpath=. tests/test_data_leakage.py -v

# Exécution ciblée de l'extraction des descripteurs lexicaux d'URL
pytest -o pythonpath=. tests/test_url_features.py -v
```

> **Note sur le jeu de données brut (`data/raw/`) :**  
> Le fichier brut `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv` (~80 Mo, 235 795 lignes) est ignoré par Git (conforme aux bonnes pratiques MLOps). Les modèles sérialisés (`models/*.joblib`), les tables de métriques et les figures sont inclus pour permettre le déploiement direct. Pour régénérer les modèles à partir de zéro après téléchargement du CSV source :
> ```bash
> python -m src.build_features
> python -m src.train --experiment C
> python -m src.train --experiment D
> ```

---

## 4. Prédiction en Ligne de Commande (CLI)

```bash
# Prédiction en mode texte
python scripts/demo_predict.py "https://www.example.com/login"

# Prédiction structurée JSON pour intégration API
python scripts/demo_predict.py "https://www.example.com/login" --json
```

---

## 5. Arborescence du Dépôt

```
PROJ-IASD01/
├── AUDIT.md                        ← Audit complet initial et matrice de notation
├── MEMBER_TASKS.md                 ← Cahier des charges exclusif membre IASD15
├── COMMIT_PLAN.md                  ← Plan d'exécution des commits atomiques
├── requirements.txt                ← Dépendances Python certifiées
├── .gitignore                      ← Règles d'exclusion des artéfacts et logs
├── README.md                       ← Documentation principale
│
├── docs/
│   ├── Rapport_Phishing_IASD01.pdf ← Rapport scientifique officiel (17 pages)
│   ├── Rapport_Projet.md           ← Source markdown du rapport scientifique
│   └── EXPERIMENT_*.md             ← Protocoles méthodologiques détaillés
│
├── src/
│   ├── config.py                   ← Paramètres, seuils et taxonomie des features
│   ├── data_loader.py              ← Chargement et déduplication (425 doublons)
│   ├── split.py                    ← GroupShuffleSplit & GroupKFold par hôte
│   ├── preprocessing.py            ← ColumnTransformer + Pipeline scellé et joblib
│   ├── build_features.py           ← Extraction et persistance URL_ONLY
│   ├── url_features.py             ← Descripteurs lexicaux temps réel et mots suspects
│   ├── train.py                    ← Entraînement et validation croisée
│   ├── evaluate.py                 ← Métriques (F1, PR-AUC, ROC-AUC, confusion)
│   └── predict.py                  ← Moteur d'inférence sécurisé hors-ligne
│
├── models/
│   ├── best_model_A.joblib         ← Pipeline complet Exp. A (50 features + TLD)
│   ├── best_model_B.joblib         ← Pipeline Exp. B (44 features + TLD)
│   ├── best_model_B_prime.joblib   ← Pipeline Exp. B' (38 features sans TLD)
│   ├── best_model_C.joblib         ← Pipeline Champion Exp. C (18 URL_ONLY)
│   ├── best_model_D.joblib         ← Pipeline Exp. D (Host features)
│   └── model_metadata_*.json       ← Métadonnées complètes et hyperparamètres
│
├── reports/
│   ├── figures/                    ← 10 figures EDA + 10 courbes PR/ROC
│   └── tables/                     ← 56 tests d'hypothèses, métriques, CSV
│
├── scripts/
│   └── demo_predict.py             ← CLI interactif de démonstration
│
├── delivery_ml/
│   └── MANIFEST_SHA256.csv         ← Manifeste d'intégrité cryptographique
│
├── notebooks/
│   ├── 01_inspection.ipynb         ← Inspection et nettoyage du dataset
│   └── 02_eda_statistiques.ipynb   ← EDA approfondie et tests statistiques
│
├── tests/
│   ├── test_data_leakage.py        ← Suite de validation anti-fuite de données
│   ├── test_build_features.py      ← Tests du générateur de features
│   ├── test_url_features.py        ← Tests des descripteurs lexicaux
│   ├── test_delivery_ml.py         ← Tests d'intégrité de la livraison
│   └── ...                         ← Tests unitaires modulaires
│
└── webapp/
    ├── webapp/
    │   ├── app.py                  ← Serveur Flask avec les 4 volets et API
    │   ├── services.py             ← Couche logique et traitement par lot CSV
    │   ├── templates/              ← Templates Jinja2 des volets interactifs
    │   └── static/                 ← Assets locaux autonomes (CSS, Chart.js)
    └── tests/
        └── test_webapp.py          ← Suite de 57 tests de l'application web
```

---

## 6. Synthèse des Expériences Scientifiques

| Expérience | Features | Autonomie Réseau | F1-Score | PR-AUC | Modèle Retenu | Statut de Déploiement |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp. A** | 50 vars + TLD | Non (Requiert HTML) | 1.0000 | 1.0000 | RandomForest (100 arbres) | Référence théorique |
| **Exp. B** | 44 vars + TLD | Non (Sans DERIVED) | 0.9999 | 1.0000 | RandomForest (200 arbres) | Analyse de sensibilité |
| **Exp. B′** | 38 vars (sans TLD) | Non (Purified HTML) | 0.9994 | 0.9999 | RandomForest (max_depth=20) | Robustesse TLD |
| **Exp. C** | **18 vars (URL_ONLY)** | **Oui (100% Hors-ligne)** | **0.9936** | **0.9979** | **RandomForest (100 arbres)** | **Champion Déployable** |
| **Exp. D** | 8 vars (Host_ONLY) | Oui (Host uniquement) | 0.7408 | 0.8585 | RandomForest (haute précision) | Analyse de domaine |

---

## 7. Rigueur Méthodologique & Protection Anti-Fuite

- **Hermétisme strict :** Toutes les transformations (standardisation `StandardScaler`, regroupement `TLDGrouper`, encodage `OneHotEncoder`, transformation `log1p`) sont isolées **à l'intérieur du pipeline scikit-learn** et ajustées exclusivement sur le partitionnement d'entraînement.
- **Découpage par Hôte :** Aucun domaine ou nom d'hôte n'est partagé entre train et test (`GroupShuffleSplit` et `GroupKFold` par hôte), garantissant une évaluation exempte de mémorisation de domaine.
- **Validation Statistique :** Rejet formel de la normalité via Shapiro-Wilk ($p < 0.001$), confirmant la nécessité de modélisations non linéaires et d'arbres de décision.
