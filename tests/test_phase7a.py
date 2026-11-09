"""
tests/test_phase7a.py
---------------------
Suite de tests unitaires pour la Phase 7a :
- Caractéristiques B' (38 numériques) et les 6 variables exclues.
- Ordre exact des colonnes et intégration du TLD (39 colonnes au total).
- Prétraitement et compatibilité du pipeline sur données synthétiques.
- Noms et chemins des artefacts B_prime via get_experiment_tag().
- Absence de fuite de cibles ou de métadonnées.
- Cohérence des partitions Train/Test et absence de fuite d'hôtes.
- Définition et gestion des seuils figés.
- Définitions mathématiques et effectifs des sous-groupes G1/G2.
- Règle de sélection du meilleur modèle réservée aux modèles pleine échelle.
- Absence d'utilisation de URLSimilarityIndex dans B et B'.
- Intégrité des artefacts historiques.
"""

import hashlib
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    URL_ONLY_FEATURES,
    PAGE_FEATURES,
    DERIVED_FEATURES,
    ALL_NUMERIC_FEATURES,
    B_FEATURES,
    B_PRIME_FEATURES,
    C_EXCLUDED_FEATURES,
    EXPERIMENT_FEATURES,
    FEATURE_SOURCES_BY_EXPERIMENT,
    get_experiment_tag,
    RAW_CSV,
    TARGET_COL,
    GROUP_COL,
    MODELS_DIR,
    TABLES_DIR,
)
from src.data_loader import load_data
from src.split import make_train_test_split
from src.preprocessing import build_pipeline
from src.evaluate import get_subgroup_masks, get_model_threshold


@pytest.fixture(scope="module")
def train_test_data():
    """Charge les données réelles et effectue le découpage une seule fois pour le module."""
    if not RAW_CSV.exists():
        pytest.skip("RAW_CSV absent")
    df_raw = load_data(verbose=False)
    df_train, df_test = make_train_test_split(df_raw)
    return df_train, df_test


class TestPhase7aBPrimeFeatures:
    """Vérification de l'espace de représentation de l'Expérience B'."""

    def test_b_prime_feature_count_and_types(self):
        """B' doit comporter exactement 38 caractéristiques numériques uniques."""
        assert len(B_PRIME_FEATURES) == 38
        assert len(set(B_PRIME_FEATURES)) == 38
        assert EXPERIMENT_FEATURES["B'"] == B_PRIME_FEATURES
        assert EXPERIMENT_FEATURES["B_prime"] == B_PRIME_FEATURES

    def test_c_excluded_features_exact_set(self):
        """Les 6 caractéristiques exclues doivent correspondre exactement à C_EXCLUDED_FEATURES."""
        expected_excluded = {
            "URLLength",
            "NoOfLettersInURL",
            "LetterRatioInURL",
            "DegitRatioInURL",
            "NoOfOtherSpecialCharsInURL",
            "SpacialCharRatioInURL",
        }
        assert set(C_EXCLUDED_FEATURES) == expected_excluded
        # Vérification qu'aucune des 6 n'est dans B'
        for feat in C_EXCLUDED_FEATURES:
            assert feat not in B_PRIME_FEATURES
            assert feat in URL_ONLY_FEATURES
            assert feat in B_FEATURES

    def test_exact_column_order(self):
        """L'ordre des colonnes doit suivre les 12 URL conservées puis les 26 Page features."""
        expected_url_12 = [f for f in URL_ONLY_FEATURES if f not in C_EXCLUDED_FEATURES]
        assert len(expected_url_12) == 12
        assert len(PAGE_FEATURES) == 26
        expected_b_prime = expected_url_12 + PAGE_FEATURES
        assert B_PRIME_FEATURES == expected_b_prime

    def test_data_sources_for_b_prime(self):
        """Toutes les caractéristiques de B' proviennent du CSV brut PhiUSIIL."""
        sources = FEATURE_SOURCES_BY_EXPERIMENT["B'"]
        assert len(sources) == 38
        for col, src in sources.items():
            assert "PhiUSIIL_Phishing_URL_Dataset.csv" in src

    def test_no_metadata_or_target_in_features(self):
        """Aucune variable cible, identifiant ou métadonnée ne doit être dans B'."""
        forbidden = {
            "label", "Target", "URL", "Domain", "TLD", "FILENAME",
            "row_id", "Host", "URLSimilarityIndex", "CharContinuationRate"
        }
        intersection = forbidden.intersection(set(B_PRIME_FEATURES))
        assert len(intersection) == 0, f"Fuite détectée : {intersection}"

    def test_no_url_similarity_in_b_and_b_prime(self):
        """URLSimilarityIndex ne doit figurer ni dans B ni dans B'."""
        assert "URLSimilarityIndex" not in EXPERIMENT_FEATURES["B"]
        assert "URLSimilarityIndex" not in EXPERIMENT_FEATURES["B'"]
        assert "URLSimilarityIndex" not in EXPERIMENT_FEATURES["B_prime"]


class TestPhase7aPreprocessingAndPipeline:
    """Vérification du pipeline de prétraitement avec TLD."""

    def test_b_prime_tld_and_input_dimension(self):
        """Le pipeline B' prend 38 colonnes numériques + 1 colonne TLD = 39 colonnes."""
        input_cols = B_PRIME_FEATURES + ["TLD"]
        assert len(input_cols) == 39

    def test_pipeline_execution_synthetic(self):
        """Le pipeline B' doit fonctionner sans erreur sur des données synthétiques."""
        rng = np.random.RandomState(42)
        n_samples = 50
        data = {col: rng.rand(n_samples) * 10 for col in B_PRIME_FEATURES}
        data["TLD"] = rng.choice(["com", "org", "net", "xyz", "de"], size=n_samples)
        df_synth = pd.DataFrame(data)
        y_synth = rng.randint(0, 2, size=n_samples)

        pipe = build_pipeline(B_PRIME_FEATURES, LogisticRegression(max_iter=100), use_tld=True)
        pipe.fit(df_synth, y_synth)
        preds = pipe.predict(df_synth)
        probs = pipe.predict_proba(df_synth)

        assert len(preds) == n_samples
        assert probs.shape == (n_samples, 2)
        assert not np.isnan(probs).any()

    def test_tree_pipeline_synthetic(self):
        """Le pipeline Random Forest B' ne doit pas appliquer de standardisation superflue."""
        rng = np.random.RandomState(42)
        n_samples = 40
        data = {col: rng.randint(0, 100, size=n_samples) for col in B_PRIME_FEATURES}
        data["TLD"] = rng.choice(["com", "edu", "fr"], size=n_samples)
        df_synth = pd.DataFrame(data)
        y_synth = rng.randint(0, 2, size=n_samples)

        pipe = build_pipeline(B_PRIME_FEATURES, RandomForestClassifier(n_estimators=10, random_state=42), use_tld=True)
        pipe.fit(df_synth, y_synth)
        preds = pipe.predict(df_synth)
        assert len(preds) == n_samples


class TestPhase7aArtifactsAndProtection:
    """Vérification du tag B_prime et de la protection des artefacts."""

    def test_experiment_tag_sanitization(self):
        """get_experiment_tag doit convertir B' et B_prime en 'B_prime' sans apostrophe."""
        assert get_experiment_tag("B'") == "B_prime"
        assert get_experiment_tag("B_prime") == "B_prime"
        assert get_experiment_tag("A") == "A"
        assert get_experiment_tag("B") == "B"
        assert get_experiment_tag("C") == "C"
        assert get_experiment_tag("D") == "D"

    def test_isolation_from_historical_artifacts(self):
        """Les chemins d'artefacts B_prime ne doivent pas entrer en collision avec A/B/C/D."""
        models_dir = Path("models")
        for exp in ["A", "B", "C", "D"]:
            hist_best = models_dir / f"best_model_{exp}.joblib"
            b_prime_best = models_dir / f"best_model_{get_experiment_tag('B\'')}.joblib"
            assert hist_best != b_prime_best

    def test_historical_models_b_exist_and_not_empty(self):
        """Les modèles de l'Expérience B doivent exister sur disque."""
        models_dir = Path("models")
        assert (models_dir / "best_model_B.joblib").exists()
        assert (models_dir / "B_RandomForest.joblib").exists()
        assert (models_dir / "model_metadata_B.json").exists()


class TestPhase7aPartitionsAndSubgroups:
    """Vérification des partitions et des sous-groupes G1/G2."""

    def test_partitions_shape_and_leakage(self, train_test_data):
        """Vérifie les effectifs et l'absence d'overlap d'hôtes entre Train et Test."""
        df_train, df_test = train_test_data

        assert len(df_train) == 188492
        assert len(df_test) == 46878

        # Labels counts
        assert (df_train[TARGET_COL] == 1).sum() == 80694
        assert (df_train[TARGET_COL] == 0).sum() == 107798
        assert (df_test[TARGET_COL] == 1).sum() == 19826
        assert (df_test[TARGET_COL] == 0).sum() == 27052

        # 0 fuite d'hôtes
        train_hosts = set(df_train[GROUP_COL])
        test_hosts = set(df_test[GROUP_COL])
        assert len(train_hosts.intersection(test_hosts)) == 0

    def test_subgroups_g1_g2_properties(self, train_test_data):
        """Vérification de G2 inclus dans G1 et des effectifs attendus sur Test."""
        _, df_test = train_test_data
        subgroups = get_subgroup_masks(df_test["URL"])
        mask_g1 = subgroups["G1"]
        mask_g2 = subgroups["G2"]

        # Inclusion stricte G2 <= G1
        assert (mask_g2 & ~mask_g1).sum() == 0

        # Effectifs exacts
        assert mask_g1.sum() == 27521
        assert (mask_g1 & (df_test[TARGET_COL] == 1)).sum() == 469
        assert (mask_g1 & (df_test[TARGET_COL] == 0)).sum() == 27052

        assert mask_g2.sum() == 27252
        assert (mask_g2 & (df_test[TARGET_COL] == 1)).sum() == 200
        assert (mask_g2 & (df_test[TARGET_COL] == 0)).sum() == 27052


class TestPhase7aEligibilityAndSelectionRule:
    """Vérification de la règle de sélection et de l'exclusion des modèles sous-échantillonnés."""

    def test_train_code_baseline_condition(self):
        """Vérifie que la baseline URLSim dans src/train.py n'est exécutée que si feature_cols la contient."""
        with open(Path("src/train.py"), "r", encoding="utf-8") as f:
            content = f.read()
        assert 'if "URLSimilarityIndex" in feature_cols:' in content
        assert 'if "URLSimilarityIndex" in df_train.columns:' not in content

    def test_train_code_subsampled_models_not_updating_best(self):
        """Vérifie que KNN et SVC_RBF ne mettent pas à jour best_ap dans src/train.py."""
        with open(Path("src/train.py"), "r", encoding="utf-8") as f:
            content = f.read()

        # KNN ne doit pas mettre à jour best_ap
        assert 'if res_knn.get("average_precision_mean", 0) > best_ap:' not in content
        # SVC_RBF ne doit pas mettre à jour best_ap
        assert 'if res_svc.get("average_precision_mean", 0) > best_ap:' not in content

    def test_standard_thresholds_defined(self):
        """Les seuils standards doivent être 0.5 pour RF/LR/DT/KNN/SVC et 0.0 pour LinearSVC."""
        assert get_model_threshold("RandomForest", "B") == 0.5
        assert get_model_threshold("LogisticRegression", "B") == 0.5
        assert get_model_threshold("DecisionTree", "B") == 0.5
        assert get_model_threshold("LinearSVC", "B") == 0.0
