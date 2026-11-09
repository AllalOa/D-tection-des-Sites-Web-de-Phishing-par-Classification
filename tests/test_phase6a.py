"""
test_phase6a.py - Tests unitaires de validation et de préparation pour la Phase 6a (Expérience A).

Couverture des exigences de préparation de l'expérience A :
1. Définition stricte et ordre des 50 caractéristiques numériques de A.
2. Exactitude des 51 colonnes en entrée du pipeline (50 numériques + 1 catégorielle TLD).
3. Source des données (data/raw/PhiUSIIL_Phishing_URL_Dataset.csv) sans mélange avec C ou D.
4. Absence totale de fuite de métadonnées ou de labels.
5. Cohérence absolue du découpage Train (188 492) / Test (46 878) et identité des row_id avec C et D.
6. Validation du prétraitement (log1p, scaling, TLDGrouper, OneHotEncoder) sur données synthétiques.
7. Isolation absolue des seuils (rejet du seuil 0.9352 de D pour A).
8. Protection et étanchéité des chemins d'artefacts pour A.
9. Intégrité cryptographique SHA256 des 22 artefacts historiques de C et D.
"""

import hashlib
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    RAW_CSV,
    URL_FEATURES_CSV,
    HOST_FEATURES_CSV,
    EXPERIMENT_FEATURES,
    URL_ONLY_FEATURES,
    PAGE_FEATURES,
    DERIVED_FEATURES,
    ALL_NUMERIC_FEATURES,
    FEATURE_SOURCES_BY_EXPERIMENT,
    MODELS_DIR,
    TABLES_DIR,
    TARGET_COL,
    GROUP_COL,
    get_experiment_tag,
)
from src.data_loader import load_data
from src.split import make_train_test_split, make_cv_folds, get_cv_groups
from src.preprocessing import build_pipeline, TLDGrouper
from src.evaluate import get_model_threshold


# =====================================================================
# 1. Audit des caractéristiques et dimensions de A
# =====================================================================
class TestExperimentAFeatures:
    def test_numeric_feature_count_is_exactly_50(self):
        feat_a = EXPERIMENT_FEATURES["A"]
        assert len(feat_a) == 50
        assert feat_a == ALL_NUMERIC_FEATURES

    def test_composition_18_url_26_page_6_derived(self):
        feat_a = EXPERIMENT_FEATURES["A"]
        url_part = [c for c in feat_a if c in URL_ONLY_FEATURES]
        page_part = [c for c in feat_a if c in PAGE_FEATURES]
        derived_part = [c for c in feat_a if c in DERIVED_FEATURES]

        assert len(url_part) == 18
        assert len(page_part) == 26
        assert len(derived_part) == 6
        assert len(url_part) + len(page_part) + len(derived_part) == 50

    def test_pipeline_input_is_51_columns_with_tld(self):
        feat_a = EXPERIMENT_FEATURES["A"]
        input_cols = feat_a + ["TLD"]
        assert len(input_cols) == 51
        assert "TLD" not in feat_a
        assert input_cols[-1] == "TLD"

    def test_no_forbidden_metadata_or_target(self):
        forbidden = {"URL", "Host", "Domain", "row_id", "phishing", "label", "FILENAME"}
        feat_a = EXPERIMENT_FEATURES["A"]
        leak = set(feat_a) & forbidden
        assert len(leak) == 0, f"Colonnes interdites détectées dans A : {leak}"


# =====================================================================
# 2. Source des données et absence de mélange
# =====================================================================
class TestExperimentASources:
    def test_source_is_raw_csv_exclusively(self):
        sources = FEATURE_SOURCES_BY_EXPERIMENT["A"]
        for col, src in sources.items():
            assert "PhiUSIIL_Phishing_URL_Dataset.csv" in src
            assert "url_features_all.csv" not in src
            assert "host_features_all.csv" not in src


# =====================================================================
# 3. Cohérence du partitionnement et identité des row_id
# =====================================================================
class TestExperimentAPartitions:
    def test_train_test_split_counts_and_zero_host_overlap(self):
        df_raw = load_data(verbose=False)
        train_df, test_df = make_train_test_split(df_raw)

        assert len(train_df) == 188492
        assert len(test_df) == 46878

        # Distribution exacte des classes
        train_counts = train_df[TARGET_COL].value_counts().to_dict()
        test_counts = test_df[TARGET_COL].value_counts().to_dict()
        assert train_counts[0] == 107798
        assert train_counts[1] == 80694
        assert test_counts[0] == 27052
        assert test_counts[1] == 19826

        # Étanchéité absolue des hôtes
        train_hosts = set(train_df[GROUP_COL])
        test_hosts = set(test_df[GROUP_COL])
        assert len(train_hosts & test_hosts) == 0

    def test_row_id_strictly_identical_to_c_and_d(self):
        df_raw = load_data(verbose=False)
        train_a, test_a = make_train_test_split(df_raw)

        df_c = pd.read_csv(URL_FEATURES_CSV)
        train_c, test_c = make_train_test_split(df_c)
        assert (train_a["row_id"].values == train_c["row_id"].values).all()
        assert (test_a["row_id"].values == test_c["row_id"].values).all()

        df_d = pd.read_csv(HOST_FEATURES_CSV)
        train_d, test_d = make_train_test_split(df_d)
        assert (train_a["row_id"].values == train_d["row_id"].values).all()
        assert (test_a["row_id"].values == test_d["row_id"].values).all()


# =====================================================================
# 4. Prétraitement et compatibilité des pipelines sur données synthétiques
# =====================================================================
class TestExperimentAPipelineSynthetic:
    @pytest.mark.parametrize("model_name,clf", [
        ("DummyClassifier", DummyClassifier(strategy="most_frequent")),
        ("LogisticRegression", LogisticRegression(C=1.0)),
        ("DecisionTree", DecisionTreeClassifier(max_depth=5)),
        ("RandomForest", RandomForestClassifier(n_estimators=10, max_depth=5)),
    ])
    def test_pipeline_fits_and_predicts_synthetic_data(self, model_name, clf):
        feat_a = EXPERIMENT_FEATURES["A"]
        pipe = build_pipeline(feat_a, clf, use_tld=True, model_name=model_name)

        # Génération de 10 observations synthétiques (50 numériques + TLD)
        np.random.seed(42)
        syn_data = {col: np.random.uniform(0, 100, size=10) for col in feat_a}
        syn_data["TLD"] = ["com", "org", "net", "xyz", "biz", "info", "online", "top", "site", "unknown_tld"]
        df_syn = pd.DataFrame(syn_data)
        y_syn = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        # Entraînement et prédiction sur données synthétiques
        pipe.fit(df_syn, y_syn)
        preds = pipe.predict(df_syn)
        assert len(preds) == 10

    def test_tld_grouper_and_unknown_handling(self):
        grouper = TLDGrouper(top_n=3, other_label="autre")
        train_tlds = pd.Series(["com", "com", "com", "org", "org", "net", "biz", "info"])
        grouper.fit(train_tlds)
        assert grouper.top_tlds_ == {"com", "org", "net"}

        test_tlds = pd.Series(["com", "biz", "xyz", "org"])
        transformed = grouper.transform(test_tlds).ravel()
        assert list(transformed) == ["com", "autre", "autre", "org"]


# =====================================================================
# 5. Isolation des seuils et non-propagation
# =====================================================================
class TestExperimentAThresholdPolicy:
    def test_standard_threshold_for_a_is_0_5(self):
        thresh = get_model_threshold("RandomForest", "A", policy="standard")
        assert thresh == 0.5

    def test_high_precision_threshold_raises_for_a(self):
        with pytest.raises(ValueError, match="Aucun seuil haute précision"):
            get_model_threshold("RandomForest", "A", policy="high_precision")


# =====================================================================
# 6. Protection des fichiers et non-écrasement
# =====================================================================
class TestExperimentAArtifactProtection:
    def test_a_artifact_paths_are_distinct(self):
        for model in ["RandomForest", "LogisticRegression", "DecisionTree", "KNN", "LinearSVC", "SVC_RBF"]:
            a_path = MODELS_DIR / f"A_{model}.joblib"
            c_path = MODELS_DIR / f"C_{model}.joblib"
            d_path = MODELS_DIR / f"D_{model}.joblib"
            assert a_path != c_path
            assert a_path != d_path

        assert (MODELS_DIR / "best_model_A.joblib") != (MODELS_DIR / "best_model_C.joblib")
        assert (MODELS_DIR / "best_model_A.joblib") != (MODELS_DIR / "best_model_D.joblib")
        assert (TABLES_DIR / "cv_results_A.csv") != (TABLES_DIR / "cv_results_C.csv")
        assert (TABLES_DIR / "test_metrics_A.csv") != (TABLES_DIR / "test_metrics_C.csv")

    def test_all_22_historical_c_and_d_hashes_intact(self):
        hash_file = TABLES_DIR / "phase5c_artifact_hashes.csv"
        assert hash_file.exists()
        df_hashes = pd.read_csv(hash_file)
        for _, row in df_hashes.iterrows():
            p = Path(row["file_path"])
            assert p.exists(), f"Artefact historique introuvable : {p}"
            current_h = hashlib.sha256(p.read_bytes()).hexdigest().upper()
            assert current_h == row["sha256"].upper(), f"Altération détectée pour {p} !"
