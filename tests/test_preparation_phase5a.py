"""
test_preparation_phase5a.py - Tests de préparation et d'audit pour l'Expérience C (Phase 5a).

Vérifications :
1. Sélection de la source C (url_features_all.csv).
2. Présence et validité des 18 caractéristiques URL_ONLY.
3. Ordre strict des caractéristiques.
4. Cohérence absolue des row_id et labels avec l'Expérience D.
5. Découpage identique par Host (Train/Test et CV).
6. Non-modification et intégrité des artefacts de l'Expérience D.
7. Chargement simulé et dispatch des prédictions pour C et D.
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.config import (
    URL_FEATURES_CSV,
    HOST_FEATURES_CSV,
    EXPERIMENT_FEATURES,
    URL_ONLY_FEATURES,
    FEATURE_SOURCES_BY_EXPERIMENT,
    GROUP_COL,
    TARGET_COL,
    MODELS_DIR,
    TABLES_DIR,
)
from src.split import make_train_test_split, make_cv_folds, get_cv_groups
from src.predict import _load_model_and_metadata, predict_url


# =====================================================================
# 1. Sélection de la source et intégrité du CSV C
# =====================================================================
class TestCSourceSelection:
    """Vérifie la source de données configurée pour l'expérience C."""

    def test_c_source_points_to_processed_url_csv(self):
        """Toutes les features de C doivent pointer vers url_features_all.csv."""
        sources = FEATURE_SOURCES_BY_EXPERIMENT["C"]
        for feat, src_path in sources.items():
            assert "url_features_all.csv" in src_path, f"Source inattendue pour {feat}: {src_path}"

    def test_url_features_csv_exists_and_shape(self):
        """Le fichier url_features_all.csv doit exister et contenir exactement 235 370 lignes."""
        assert URL_FEATURES_CSV.exists(), f"Fichier introuvable: {URL_FEATURES_CSV}"
        df = pd.read_csv(URL_FEATURES_CSV)
        assert len(df) == 235370, f"Nombre de lignes inattendu: {len(df)}"
        assert df["row_id"].nunique() == 235370, "Les row_id ne sont pas uniques !"


# =====================================================================
# 2. Vérification des 18 caractéristiques C
# =====================================================================
class TestCFeaturesAudit:
    """Vérifie les caractéristiques de l'expérience C."""

    def test_c_feature_count_and_names(self):
        """L'expérience C doit comporter exactement les 18 caractéristiques URL_ONLY."""
        c_feats = EXPERIMENT_FEATURES["C"]
        assert len(c_feats) == 18, f"Attendu 18 features, obtenu {len(c_feats)}"
        assert c_feats == URL_ONLY_FEATURES, "EXPERIMENT_FEATURES['C'] ne correspond pas à URL_ONLY_FEATURES"

    def test_all_18_features_in_csv_no_nans(self):
        """Toutes les 18 features doivent être dans le CSV sans NaN ni Inf."""
        df = pd.read_csv(URL_FEATURES_CSV)
        c_feats = EXPERIMENT_FEATURES["C"]
        for feat in c_feats:
            assert feat in df.columns, f"Feature absente du CSV: {feat}"

        nan_count = df[c_feats].isna().sum().sum()
        assert nan_count == 0, f"Présence de {nan_count} NaNs dans les features C !"

        inf_count = np.isinf(df[c_feats].select_dtypes(include=[np.number])).sum().sum()
        assert inf_count == 0, f"Présence de {inf_count} Infs dans les features C !"

    def test_no_leak_of_metadata_in_c_features(self):
        """Les métadonnées (URL, Host, Domain, TLD, row_id, raw_index, phishing) ne doivent pas être des features explicatives."""
        c_feats = set(EXPERIMENT_FEATURES["C"])
        forbidden = {"URL", "Host", "Domain", "TLD", "row_id", "raw_index", "phishing", "label"}
        leaked = c_feats & forbidden
        assert len(leaked) == 0, f"Fuite de colonnes interdites dans les features C: {leaked}"


# =====================================================================
# 3. Ordre des caractéristiques
# =====================================================================
class TestCFeaturesOrder:
    """Vérifie l'ordre strict des variables."""

    def test_c_features_order_matches_url_only_features(self):
        """L'ordre dans EXPERIMENT_FEATURES['C'] doit être strictement identique à URL_ONLY_FEATURES."""
        c_feats = EXPERIMENT_FEATURES["C"]
        for idx, (f_c, f_orig) in enumerate(zip(c_feats, URL_ONLY_FEATURES)):
            assert f_c == f_orig, f"Désaccord d'ordre à l'index {idx}: {f_c} != {f_orig}"

    def test_csv_features_order(self):
        """Les colonnes features dans url_features_all.csv doivent respecter l'ordre officiel."""
        df = pd.read_csv(URL_FEATURES_CSV, nrows=2)
        c_feats = EXPERIMENT_FEATURES["C"]
        csv_feat_order = [c for c in df.columns if c in c_feats]
        assert csv_feat_order == c_feats, "L'ordre des colonnes features dans le CSV ne respecte pas l'ordre officiel !"


# =====================================================================
# 4. Cohérence absolue avec l'Expérience D (row_id et labels)
# =====================================================================
class TestConsistencyWithD:
    """Vérifie l'alignement strict entre C et D."""

    def test_same_row_ids_and_labels(self):
        """url_features_all.csv et host_features_all.csv doivent avoir les mêmes row_id et labels."""
        df_c = pd.read_csv(URL_FEATURES_CSV, usecols=["row_id", TARGET_COL, GROUP_COL])
        df_d = pd.read_csv(HOST_FEATURES_CSV, usecols=["row_id", TARGET_COL, GROUP_COL])

        assert len(df_c) == len(df_d) == 235370
        assert (df_c["row_id"].values == df_d["row_id"].values).all(), "Désaccord sur les row_id entre C et D !"
        assert (df_c[TARGET_COL].values == df_d[TARGET_COL].values).all(), "Désaccord sur les labels entre C et D !"
        assert (df_c[GROUP_COL].values == df_d[GROUP_COL].values).all(), "Désaccord sur les groupes Host entre C et D !"

    def test_class_balance_exact(self):
        """Les effectifs par classe doivent être exactement 134 850 (0) et 100 520 (1)."""
        df_c = pd.read_csv(URL_FEATURES_CSV, usecols=[TARGET_COL])
        counts = df_c[TARGET_COL].value_counts().to_dict()
        assert counts[0] == 134850, f"Attendu 134850 légitimes, obtenu {counts[0]}"
        assert counts[1] == 100520, f"Attendu 100520 phishing, obtenu {counts[1]}"


# =====================================================================
# 5. Découpage identique par Host (Train/Test et CV)
# =====================================================================
class TestSplitConsistencyByHost:
    """Vérifie que make_train_test_split produit des partitions identiques pour C et D."""

    def test_split_produces_identical_splits(self):
        """Train et Test doivent contenir exactement les mêmes row_id et observations pour C et D."""
        df_c = pd.read_csv(URL_FEATURES_CSV, usecols=["row_id", GROUP_COL, TARGET_COL])
        df_d = pd.read_csv(HOST_FEATURES_CSV, usecols=["row_id", GROUP_COL, TARGET_COL])

        train_c, test_c = make_train_test_split(df_c)
        train_d, test_d = make_train_test_split(df_d)

        assert len(train_c) == len(train_d) == 188492
        assert len(test_c) == len(test_d) == 46878

        assert (train_c["row_id"].values == train_d["row_id"].values).all()
        assert (test_c["row_id"].values == test_d["row_id"].values).all()

        # Aucun Host partagé
        train_hosts = set(train_c[GROUP_COL])
        test_hosts = set(test_c[GROUP_COL])
        assert len(train_hosts & test_hosts) == 0, "Fuite de Host entre Train et Test !"

    def test_cv_folds_no_host_overlap(self):
        """GroupKFold ne doit avoir aucun Host partagé entre train et val sur chaque pli."""
        df_c = pd.read_csv(URL_FEATURES_CSV, usecols=["row_id", GROUP_COL, TARGET_COL])
        train_c, _ = make_train_test_split(df_c)

        gkf = make_cv_folds(train_c)
        groups = get_cv_groups(train_c)

        for fold, (t_idx, v_idx) in enumerate(gkf.split(train_c, train_c[TARGET_COL], groups=groups)):
            t_hosts = set(train_c.iloc[t_idx][GROUP_COL])
            v_hosts = set(train_c.iloc[v_idx][GROUP_COL])
            assert len(t_hosts & v_hosts) == 0, f"Pli {fold} présente un chevauchement d'hôtes !"


# =====================================================================
# 6. Protection et intégrité des artefacts de D
# =====================================================================
class TestDArtifactsIntegrity:
    """Vérifie que les artefacts de l'expérience D existent et ne risquent pas d'être écrasés."""

    def test_d_artifacts_present(self):
        """Les fichiers clés de D doivent être présents."""
        d_files = [
            MODELS_DIR / "best_model_D.joblib",
            MODELS_DIR / "model_metadata_D.json",
            TABLES_DIR / "test_metrics_D.csv",
            TABLES_DIR / "cv_results_D.csv",
        ]
        for f in d_files:
            assert f.exists(), f"Artefact D manquant: {f}"

    def test_c_and_d_target_paths_distinct(self):
        """Les chemins cibles de C et D doivent être strictement distincts."""
        for exp in ["C", "D"]:
            best_model_path = MODELS_DIR / f"best_model_{exp}.joblib"
            meta_path = MODELS_DIR / f"model_metadata_{exp}.json"
            cv_path = TABLES_DIR / f"cv_results_{exp}.csv"
            metrics_path = TABLES_DIR / f"test_metrics_{exp}.csv"

            # Les chemins de C ne doivent pas être ceux de D
            if exp == "C":
                assert best_model_path != MODELS_DIR / "best_model_D.joblib"
                assert meta_path != MODELS_DIR / "model_metadata_D.json"
                assert cv_path != TABLES_DIR / "cv_results_D.csv"
                assert metrics_path != TABLES_DIR / "test_metrics_D.csv"


# =====================================================================
# 7. Chargement simulé et dispatch dans predict.py
# =====================================================================
class TestPredictDispatchMocked:
    """Vérifie la logique de prédiction pour C et D avec mocks sans accès réseau ni entraînement."""

    def test_load_model_and_metadata_targets_correct_paths(self):
        """_load_model_and_metadata cible best_model_<exp>.joblib et model_metadata_<exp>.json."""
        with patch("joblib.load") as mock_load, patch("builtins.open") as mock_open, patch("json.load") as mock_json:
            mock_json.return_value = {"experiment": "C", "model_name": "RandomForest", "feature_cols": URL_ONLY_FEATURES}
            with patch.object(Path, "exists", return_value=True):
                pipe, meta = _load_model_and_metadata("C")
                assert meta["experiment"] == "C"

            mock_json.return_value = {"experiment": "D", "model_name": "RandomForest", "feature_cols": ["HostLength"]}
            with patch.object(Path, "exists", return_value=True):
                pipe_d, meta_d = _load_model_and_metadata("D")
                assert meta_d["experiment"] == "D"

    def test_predict_url_c_uses_compute_url_features(self):
        """predict_url avec experiment='C' appelle compute_url_features et passe les 18 colonnes ordonnées."""
        mock_pipe = MagicMock()
        mock_pipe.predict.return_value = np.array([0])
        mock_pipe.predict_proba.return_value = np.array([[0.8, 0.2]])

        mock_meta = {
            "experiment": "C",
            "model_name": "RandomForest",
            "feature_cols": URL_ONLY_FEATURES,
        }

        with patch("src.predict._load_model_and_metadata", return_value=(mock_pipe, mock_meta)):
            with patch("src.predict.compute_url_features", wraps=lambda u: {col: 1 for col in URL_ONLY_FEATURES}) as mock_feat:
                res = predict_url("https://www.example.com/test", experiment="C")
                assert mock_feat.called
                assert res["experiment"] == "C"
                assert res["threshold_used"] == 0.5
                assert res["predicted_class"] == 0

                # Vérifier les colonnes passées au pipeline
                passed_df = mock_pipe.predict.call_args[0][0]
                assert list(passed_df.columns) == URL_ONLY_FEATURES

    def test_predict_url_c_rejects_high_precision_threshold(self):
        """predict_url pour C avec threshold='high_precision' doit lever ValueError."""
        mock_pipe = MagicMock()
        mock_meta = {
            "experiment": "C",
            "model_name": "RandomForest",
            "feature_cols": URL_ONLY_FEATURES,
        }
        with patch("src.predict._load_model_and_metadata", return_value=(mock_pipe, mock_meta)):
            with pytest.raises(ValueError, match="Aucun seuil haute précision n'est documenté"):
                predict_url("https://www.example.com", experiment="C", threshold="high_precision")
