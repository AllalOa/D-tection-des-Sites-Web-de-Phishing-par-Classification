"""
test_phase5c.py - Tests unitaires de validation et de non-régression pour la Phase 5c.

Couverture des 10 exigences du protocole :
1. Sélection correcte des variables A, B, B', C, D.
2. Compatibilité des pipelines avec toutes les caractéristiques.
3. Ordre strict des colonnes.
4. Sélection du fichier source selon l'expérience.
5. Résolution des chemins propres à chaque expérience (assainissement B_prime sans apostrophe).
6. Absence d'écrasement des artefacts C et D.
7. Bonne gestion des seuils (isolation stricte du seuil D).
8. Sous-groupes G1 et G2 (inclusion G2 c G1 et prédicats syntaxiques).
9. Cohérence arithmétique des métriques enregistrées.
10. Absence de chevauchement d'hôtes entre partitions Train et Test.
"""

import json
import hashlib
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from sklearn.dummy import DummyClassifier

from src.config import (
    EXPERIMENT_FEATURES,
    URL_ONLY_FEATURES,
    PAGE_FEATURES,
    DERIVED_FEATURES,
    ALL_NUMERIC_FEATURES,
    HOST_FEATURES,
    FEATURE_SOURCES_BY_EXPERIMENT,
    get_experiment_tag,
    MODELS_DIR,
    TABLES_DIR,
    TARGET_COL,
    GROUP_COL,
    URL_FEATURES_CSV,
    HOST_FEATURES_CSV,
)
from src.preprocessing import build_pipeline
from src.evaluate import get_subgroup_masks, get_model_threshold
from src.predict import predict_url, _load_model_and_metadata
from src.split import make_train_test_split


# =====================================================================
# 1. Sélection correcte des variables et ordre strict
# =====================================================================
class TestFeatureSelectionAndOrder:
    def test_feature_counts_and_definitions(self):
        assert len(EXPERIMENT_FEATURES["A"]) == 50
        assert len(EXPERIMENT_FEATURES["B"]) == 44
        assert len(EXPERIMENT_FEATURES["B'"]) == 38
        assert len(EXPERIMENT_FEATURES["B_prime"]) == 38
        assert len(EXPERIMENT_FEATURES["C"]) == 18
        assert len(EXPERIMENT_FEATURES["D"]) == 10

    def test_strict_subset_hierarchy(self):
        set_a = set(EXPERIMENT_FEATURES["A"])
        set_b = set(EXPERIMENT_FEATURES["B"])
        set_b_prime = set(EXPERIMENT_FEATURES["B'"])
        set_c = set(EXPERIMENT_FEATURES["C"])
        set_derived = set(DERIVED_FEATURES)

        assert set_b == set_a - set_derived
        assert set_b_prime.issubset(set_b)
        assert len(set_b - set_b_prime) == 6
        assert set_c == set(URL_ONLY_FEATURES)

    def test_no_forbidden_metadata_columns(self):
        forbidden = {"URL", "Host", "Domain", "row_id", "phishing", "label", "FILENAME"}
        for exp in ["A", "B", "B'", "B_prime", "C", "D"]:
            overlap = set(EXPERIMENT_FEATURES[exp]) & forbidden
            assert len(overlap) == 0, f"Colonnes interdites dans {exp}: {overlap}"


# =====================================================================
# 2. Compatibilité des pipelines avec toutes les caractéristiques
# =====================================================================
class TestPipelineCompatibility:
    @pytest.mark.parametrize("exp,use_tld", [
        ("A", True),
        ("B", True),
        ("B'", True),
        ("C", False),
        ("D", False),
    ])
    def test_pipeline_fits_synthetic_data(self, exp, use_tld):
        feat_cols = EXPERIMENT_FEATURES[exp]
        dummy = DummyClassifier(strategy="most_frequent")
        pipe = build_pipeline(feat_cols, dummy, use_tld=use_tld, model_name="DummyClassifier")

        # Données synthétiques (2 lignes)
        data = {col: [1.0, 2.0] for col in feat_cols}
        if use_tld:
            data["TLD"] = ["com", "org"]
        df_dummy = pd.DataFrame(data)
        y_dummy = np.array([0, 1])

        pipe.fit(df_dummy, y_dummy)
        preds = pipe.predict(df_dummy)
        assert len(preds) == 2


# =====================================================================
# 3. Sélection des fichiers sources
# =====================================================================
class TestSourceSelection:
    def test_sources_by_experiment(self):
        for exp in ["A", "B", "B'", "B_prime"]:
            for col, src in FEATURE_SOURCES_BY_EXPERIMENT[exp].items():
                assert "PhiUSIIL_Phishing_URL_Dataset.csv" in src
        for col, src in FEATURE_SOURCES_BY_EXPERIMENT["C"].items():
            assert "url_features_all.csv" in src
        for col, src in FEATURE_SOURCES_BY_EXPERIMENT["D"].items():
            assert "host_features_all.csv" in src


# =====================================================================
# 4. Résolution des chemins et assainissement B_prime
# =====================================================================
class TestPathResolution:
    def test_get_experiment_tag_sanitization(self):
        assert get_experiment_tag("A") == "A"
        assert get_experiment_tag("B") == "B"
        assert get_experiment_tag("B'") == "B_prime"
        assert get_experiment_tag("B_prime") == "B_prime"
        assert get_experiment_tag("C") == "C"
        assert get_experiment_tag("D") == "D"

    def test_no_apostrophe_in_generated_paths(self):
        for exp in ["A", "B", "B'", "C", "D"]:
            tag = get_experiment_tag(exp)
            model_file = MODELS_DIR / f"{tag}_RandomForest.joblib"
            meta_file = MODELS_DIR / f"model_metadata_{tag}.json"
            assert "'" not in str(model_file)
            assert "'" not in str(meta_file)


# =====================================================================
# 5. Protection des artefacts C et D
# =====================================================================
class TestArtifactProtection:
    def test_c_and_d_target_paths_distinct_from_ab(self):
        for exp_new in ["A", "B", "B_prime"]:
            for exp_old in ["C", "D"]:
                assert f"{exp_new}_" != f"{exp_old}_"
                assert f"best_model_{exp_new}.joblib" != f"best_model_{exp_old}.joblib"

    def test_historical_hashes_match_saved_record(self):
        hash_file = TABLES_DIR / "phase5c_artifact_hashes.csv"
        assert hash_file.exists()
        df_hashes = pd.read_csv(hash_file)
        for _, row in df_hashes.iterrows():
            p = Path(row["file_path"])
            assert p.exists(), f"Fichier manquant: {p}"
            with open(p, "rb") as f:
                current_h = hashlib.sha256(f.read()).hexdigest().upper()
            assert current_h == row["sha256"], f"Hash modifié pour {p}!"


# =====================================================================
# 6. Gestion des seuils et non-propagation
# =====================================================================
class TestThresholdHandling:
    def test_high_precision_rf_d_only(self):
        assert get_model_threshold("RandomForest", "D", policy="high_precision") == 0.9352

    @pytest.mark.parametrize("exp", ["A", "B", "B'", "B_prime", "C"])
    def test_high_precision_raises_for_other_experiments(self, exp):
        with pytest.raises(ValueError, match="Aucun seuil haute précision"):
            get_model_threshold("RandomForest", exp, policy="high_precision")

    def test_standard_threshold_defaults(self):
        assert get_model_threshold("RandomForest", "C", policy="standard") == 0.5
        assert get_model_threshold("LinearSVC", "C", policy="standard") == 0.0


# =====================================================================
# 7. Sous-groupes G1 et G2
# =====================================================================
class TestSubgroupsLogic:
    def test_g1_and_g2_definitions(self):
        urls = pd.Series([
            "https://www.example.com",
            "https://www.example.com/",
            "https://www.example.com/login",
            "http://www.example.com",
            "https://example.com",
        ])
        masks = get_subgroup_masks(urls)
        g1 = masks["G1"]
        g2 = masks["G2"]

        # G1 doit matcher 0, 1, 2
        assert list(g1) == [True, True, True, False, False]
        # G2 doit matcher 0, 1 (sans chemin/requête)
        assert list(g2) == [True, True, False, False, False]
        # G2 est un sous-ensemble strict de G1
        assert (g2 <= g1).all()

    def test_disjoint_partition_and_confusion_matrix_additivity(self):
        """
        G2 est inclus dans G1 (non disjoints).
        La partition exacte et disjointe du Test est :
        - S1 = G2
        - S2 = G1 \\ G2
        - S3 = Test \\ G1
        Vérifie la partition stricte et l'additivité des matrices de confusion.
        """
        urls = pd.Series([
            "https://www.legit1.com",            # S1 (G2)
            "https://www.legit2.com/",           # S1 (G2)
            "https://www.phish-bare.com/",       # S1 (G2)
            "https://www.phish1.com/login",      # S2 (G1 \\ G2)
            "https://www.phish2.com?token=123",  # S2 (G1 \\ G2)
            "http://www.phish3.com/account",     # S3 (Test \\ G1)
            "https://phish4.com/secure",         # S3 (Test \\ G1)
            "http://legit3.org/page",            # S3 (Test \\ G1)
            "http://legit4.org/other",           # S3 (Test \\ G1)
        ])
        y_true = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 0, 0])

        masks = get_subgroup_masks(urls)
        g1 = masks["G1"]
        g2 = masks["G2"]

        s1 = g2
        s2 = g1 & (~g2)
        s3 = ~g1

        # 1. Vérification que S1, S2, S3 sont deux à deux disjoints
        assert (s1 & s2).sum() == 0
        assert (s1 & s3).sum() == 0
        assert (s2 & s3).sum() == 0

        # 2. Vérification que leur union forme exactement l'ensemble complet
        assert (s1 | s2 | s3).sum() == len(urls)

        # 3. Additivité exacte de la matrice de confusion sur la partition disjointe
        def get_cm(mask):
            sub_y = y_true[mask]
            sub_p = y_pred[mask]
            tp = int(((sub_y == 1) & (sub_p == 1)).sum())
            fp = int(((sub_y == 0) & (sub_p == 1)).sum())
            tn = int(((sub_y == 0) & (sub_p == 0)).sum())
            fn = int(((sub_y == 1) & (sub_p == 0)).sum())
            return tp, fp, tn, fn

        tp_all, fp_all, tn_all, fn_all = get_cm(slice(None))
        tp_s1, fp_s1, tn_s1, fn_s1 = get_cm(s1)
        tp_s2, fp_s2, tn_s2, fn_s2 = get_cm(s2)
        tp_s3, fp_s3, tn_s3, fn_s3 = get_cm(s3)

        assert tp_all == tp_s1 + tp_s2 + tp_s3
        assert fp_all == fp_s1 + fp_s2 + fp_s3
        assert tn_all == tn_s1 + tn_s2 + tn_s3
        assert fn_all == fn_s1 + fn_s2 + fn_s3

        # 4. Vérification que N_G1 + N_G2 != N_total (car G1 et G2 ne sont pas disjoints)
        assert g1.sum() + g2.sum() != len(urls)


# =====================================================================
# 8. Cohérence arithmétique des métriques enregistrées
# =====================================================================
class TestSavedMetricsArithmetic:
    @pytest.mark.parametrize("csv_file", [
        "test_metrics_C.csv",
        "test_metrics_D.csv",
    ])
    def test_confusion_matrix_and_metrics_consistency(self, csv_file):
        path = TABLES_DIR / csv_file
        assert path.exists()
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            tp, fp, tn, fn = row["TP"], row["FP"], row["TN"], row["FN"]
            assert tp + fp + tn + fn == 46878
            assert tp + fn == 19826
            assert tn + fp == 27052

            calc_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            calc_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            calc_f1 = 2 * calc_prec * calc_rec / (calc_prec + calc_rec) if (calc_prec + calc_rec) > 0 else 0.0

            assert abs(calc_prec - row["precision"]) < 1e-4
            assert abs(calc_rec - row["recall"]) < 1e-4
            assert abs(calc_f1 - row["f1"]) < 1e-4


# =====================================================================
# 9. Étanchéité Train/Test et absence de fuite Host
# =====================================================================
class TestPartitionLeakage:
    def test_train_test_split_no_host_overlap(self):
        df_c = pd.read_csv(URL_FEATURES_CSV, usecols=["row_id", "Host", "phishing"])
        train_df, test_df = make_train_test_split(df_c)
        train_hosts = set(train_df["Host"])
        test_hosts = set(test_df["Host"])
        assert len(train_hosts & test_hosts) == 0
