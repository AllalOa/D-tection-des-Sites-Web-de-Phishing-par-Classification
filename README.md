# PROJ-IASD01 — Détection de phishing par classification supervisée

Pipeline ML complet (CRISP-DM) pour la détection de sites de phishing à partir du dataset PhiUSIIL (UCI id 967).

**Python :** 3.14.2 | **scikit-learn :** 1.8.0 | **pandas :** 2.3.3

---

## Installation

```bash
# 1. Cloner ou copier le dossier PROJ-IASD01/
# 2. Installer les dépendances (environnement Python 3.14.2)
pip install -r requirements.txt

# 3. Placer le fichier CSV dans data/raw/
# Le fichier attendu : data/raw/PhiUSIIL_Phishing_URL_Dataset.csv
```

---

## Ordre d'exécution

### 1. Tests unitaires (sans entraînement)

```bash
# Depuis le répertoire PROJ-IASD01/
pytest tests/ -v
```

### 2. Entraînement des modèles

```bash
# Expérience C : URL_ONLY (modèle déployable — recommandé pour commencer)
python -m src.train --experiment C

# Expérience A : toutes les 50 variables numériques + TLD
python -m src.train --experiment A

# Expérience B : A sans les variables DERIVED
python -m src.train --experiment B
```

Les modèles sont sauvegardés dans `models/`.  
Le meilleur modèle de l'expérience C est sauvegardé dans `models/best_model.joblib`.

### 3. Évaluation finale (une seule fois sur le test)

```bash
python -m src.evaluate --experiment C
python -m src.evaluate --experiment A
python -m src.evaluate --experiment B
```

Les figures (`reports/figures/`) et tableaux (`reports/tables/`) sont générés automatiquement.

### 4. Prédiction sur une URL

```bash
python -m src.predict "https://www.example.com/login?user=test"
```

Retourne la classe prédite (phishing/légitime) et le score brut du modèle.

### 5. Notebooks (optionnel, pour l'EDA et les rapports)

Ouvrir dans l'ordre :
1. `notebooks/01_inspection.ipynb` — inspection et vérification du dataset
2. `notebooks/02_eda_statistiques.ipynb` — EDA, distributions, corrélations, tests statistiques
3. `notebooks/03_modelisation.ipynb` — modélisation et validation croisée
4. `notebooks/04_evaluation.ipynb` — évaluation finale et robustesse

---

## Arborescence

```
PROJ-IASD01/
├── data/
│   ├── raw/                         ← Placer le CSV ici (non versionné)
│   └── processed/                   ← Train/test splitts (non versionnés)
├── notebooks/
│   ├── 01_inspection.ipynb
│   ├── 02_eda_statistiques.ipynb
│   ├── 03_modelisation.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── config.py                    ← Chemins, constantes, familles de variables
│   ├── data_loader.py               ← Chargement et nettoyage du CSV
│   ├── split.py                     ← Découpage train/test par Domain
│   ├── preprocessing.py             ← Pipelines sklearn
│   ├── url_features.py              ← Calcul features URL_ONLY (sans réseau)
│   ├── train.py                     ← Entraînement + GridSearchCV
│   ├── evaluate.py                  ← Évaluation finale et robustesse
│   └── predict.py                   ← Prédiction réutilisable
├── models/
│   ├── best_model.joblib            ← Meilleur modèle exp. C (non versionné)
│   └── model_metadata.json          ← Métadonnées du meilleur modèle
├── tests/
│   ├── test_config.py
│   ├── test_data_loader.py
│   ├── test_split.py
│   ├── test_preprocessing.py
│   ├── test_url_features.py
│   └── test_predict.py
├── reports/
│   ├── figures/                     ← Courbes PR, ROC, learning curves (png)
│   └── tables/                      ← Métriques, erreurs, features (csv)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Expériences

| Expérience | Variables | Description |
|-----------|-----------|-------------|
| A | 50 numériques + TLD | Référence complète |
| B | 44 numériques + TLD (sans DERIVED) | Sans variables dérivées |
| C | 18 URL_ONLY | **Modèle déployable** (aucun accès page) |

---

## Règles importantes

- **Pas de fuite de données** : toutes les transformations sont apprises sur le train uniquement (inside Pipeline).
- **Découpage par Domain** : GroupShuffleSplit + GroupKFold pour éviter toute fuite de domaines.
- **Le test est utilisé UNE SEULE FOIS** pour l'évaluation finale.
- **random_state=42** partout pour la reproductibilité.
- La variable `model_score` de `predict_url()` n'est **pas** une probabilité calibrée.

---

## Dataset

- **Source** : PhiUSIIL Phishing URL Dataset (UCI id 967, CC BY 4.0)
- **Taille** : 235 795 lignes × 56 colonnes
- **Cible** : `label` recodé → `phishing` = 1 (phishing), 0 (légitime)
- **Classe positive** : phishing (~42.8%)
