"""
test_phase6c.py - Tests unitaires de validation et de non-régression pour la Phase 6c-bis (Expérience B).

Couverture des exigences de préparation et de validation :
1. Définitions exactes de G1/G2 via get_subgroup_masks().
2. Inclusion stricte G2 subset G1.
3. Décomposition du jeu en trois ensembles mutuellement disjoints.
4. Les 44 variables numériques de B (URL_ONLY + PAGE).
5. Les 6 variables dérivées exclues de B (DERIVED_FEATURES).
6. L'ajout correct du TLD (45 colonnes en entrée du pipeline).
7. Détection des incohérences de schéma d'entrée (colonnes manquantes).
8. Chemins conformes des artefacts B (subgroups_evaluation_B.csv, best_model_metrics_B.csv, etc.).
9. Absence de propagation du seuil haute précision de D vers B.
10. Calcul effectif de l'Average Precision pour LinearSVC via decision_function.
11. Intégrité cryptographique SHA256 des 22 artefacts historiques de C et D.
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
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score

from src.config import (
    RAW_CSV,
    URL_FEATURES_CSV,
    HOST_FEATURES_CSV,
    EXPERIMENT_FEATURES,
    B_FEATURES,
    URL_ONLY_FEATURES,
    PAGE_FEATURES,
    DERIVED_FEATURES,
    ALL_NUMERIC_FEATURES,
    FEATURE_SOURCES_BY_EXPERIMENT,
    MODELS_DIR,
    TABLES_DIR,
    REPORTS_DIR,
    TARGET_COL,
    GROUP_COL,
    get_experiment_tag,
)
from src.data_loader import load_data
from src.split import make_train_test_split, make_cv_folds, get_cv_groups
from src.preprocessing import build_pipeline, TLDGrouper
from src.evaluate import get_subgroup_masks, get_model_threshold


# =====================================================================
# 1. Audit des caractéristiques et dimensions de B
# =====================================================================
class TestExperimentBFeatures:
    def test_numeric_feature_count_is_exactly_44(self):
        feat_b = EXPERIMENT_FEATURES["B"]
        assert len(feat_b) == 44
        assert feat_b == B_FEATURES

    def test_composition_18_url_and_26_page(self):
        feat_b = EXPERIMENT_FEATURES["B"]
        url_part = [c for c in feat_b if c in URL_ONLY_FEATURES]
        page_part = [c for c in feat_b if c in PAGE_FEATURES]

        assert len(url_part) == 18
        assert len(page_part) == 26
        assert len(url_part) + len(page_part) == 44

    def test_strict_exclusion_of_derived_features(self):
        feat_b = EXPERIMENT_FEATURES["B"]
        derived_in_b = set(feat_b) & set(DERIVED_FEATURES)
        assert len(derived_in_b) == 0, f"Caractéristiques dérivées détectées dans B : {derived_in_b}"
        for col in [
            "URLSimilarityIndex",
            "CharContinuationRate",
            "TLDLegitimateProb",
            "URLCharProb",
            "DomainTitleMatchScore",
            "URLTitleMatchScore",
        ]:
            assert col not in feat_b

    def test_pipeline_input_is_45_columns_with_tld(self):
        feat_b = EXPERIMENT_FEATURES["B"]
        input_cols = feat_b + ["TLD"]
        assert len(input_cols) == 45
        assert "TLD" not in feat_b
        assert input_cols[-1] == "TLD"

    def test_missing_input_column_raises_error(self):
        feat_b = EXPERIMENT_FEATURES["B"]
        lsvc = LinearSVC(random_state=42)
        pipe = build_pipeline(feat_b, lsvc, use_tld=True, model_name="LinearSVC")

        # Jeu avec une colonne manquante
        syn_data = {col: np.random.randn(5) for col in feat_b if col != "URLLength"}
        syn_data["TLD"] = ["com"] * 5
        df_incomplete = pd.DataFrame(syn_data)
        y = np.array([0, 1, 0, 1, 0])

        with pytest.raises((ValueError, KeyError)):
            pipe.fit(df_incomplete, y)

    def test_no_forbidden_metadata_or_target(self):
        forbidden = {"URL", "Host", "Domain", "row_id", "phishing", "label", "FILENAME"}
        feat_b = EXPERIMENT_FEATURES["B"]
        leak = set(feat_b) & forbidden
        assert len(leak) == 0, f"Colonnes interdites détectées dans B : {leak}"


# =====================================================================
# 2. Source des données et étanchéité
# =====================================================================
class TestExperimentBSources:
    def test_source_is_raw_csv_exclusively(self):
        sources = FEATURE_SOURCES_BY_EXPERIMENT["B"]
        assert len(sources) == 44
        for col, src in sources.items():
            assert "PhiUSIIL_Phishing_URL_Dataset.csv" in src
            assert "url_features_all.csv" not in src
            assert "host_features_all.csv" not in src


# =====================================================================
# 3. Cohérence du partitionnement et identité des row_id
# =====================================================================
class TestExperimentBPartitions:
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

    def test_row_id_strictly_identical_to_a_c_and_d(self):
        df_raw = load_data(verbose=False)
        train_b, test_b = make_train_test_split(df_raw)

        df_c = pd.read_csv(URL_FEATURES_CSV)
        train_c, test_c = make_train_test_split(df_c)
        assert (train_b["row_id"].values == train_c["row_id"].values).all()
        assert (test_b["row_id"].values == test_c["row_id"].values).all()

        df_d = pd.read_csv(HOST_FEATURES_CSV)
        train_d, test_d = make_train_test_split(df_d)
        assert (train_b["row_id"].values == train_d["row_id"].values).all()
        assert (test_b["row_id"].values == test_d["row_id"].values).all()


# =====================================================================
# 4. Prétraitement et compatibilité des pipelines sur données synthétiques
# =====================================================================
class TestExperimentBPipelineSynthetic:
    @pytest.mark.parametrize("model_name,clf", [
        ("DummyClassifier", DummyClassifier(strategy="most_frequent")),
        ("LogisticRegression", LogisticRegression(C=1.0, max_iter=200)),
        ("DecisionTree", DecisionTreeClassifier(max_depth=5)),
        ("RandomForest", RandomForestClassifier(n_estimators=10, max_depth=5)),
    ])
    def test_pipeline_fits_and_predicts_synthetic_data(self, model_name, clf):
        feat_b = EXPERIMENT_FEATURES["B"]
        pipe = build_pipeline(feat_b, clf, use_tld=True, model_name=model_name)

        # 12 observations synthétiques (44 numériques + TLD)
        np.random.seed(42)
        syn_data = {col: np.random.uniform(0, 100, size=12) for col in feat_b}
        syn_data["TLD"] = ["com", "org", "net", "xyz", "biz", "info", "online", "top", "site", "unknown_tld", "io", "co"]
        df_syn = pd.DataFrame(syn_data)
        y_syn = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        pipe.fit(df_syn, y_syn)
        preds = pipe.predict(df_syn)
        assert len(preds) == 12

    def test_tld_grouper_and_unknown_handling(self):
        grouper = TLDGrouper(top_n=3, other_label="autre")
        train_tlds = pd.Series(["com", "com", "com", "org", "org", "net", "biz", "info"])
        grouper.fit(train_tlds)
        assert grouper.top_tlds_ == {"com", "org", "net"}

        test_tlds = pd.Series(["com", "biz", "xyz", "org"])
        transformed = grouper.transform(test_tlds).ravel()
        assert list(transformed) == ["com", "autre", "autre", "org"]


# =====================================================================
# 5. Audit de LinearSVC et calcul de l'AP sur decision_function
# =====================================================================
class TestLinearSVCDecisionFunctionAP:
    def test_linearsvc_decision_function_computes_valid_average_precision(self):
        feat_b = EXPERIMENT_FEATURES["B"]
        lsvc = LinearSVC(random_state=42, max_iter=1000)
        pipe = build_pipeline(feat_b, lsvc, use_tld=True, model_name="LinearSVC")

        np.random.seed(42)
        syn_data = {col: np.random.normal(loc=0.0, scale=1.0, size=20) for col in feat_b}
        syn_data["TLD"] = ["com", "org"] * 10
        df_syn = pd.DataFrame(syn_data)
        y_syn = np.array([0, 1] * 10)

        pipe.fit(df_syn, y_syn)

        assert hasattr(pipe, "decision_function")
        dec_scores = pipe.decision_function(df_syn)
        assert dec_scores.ndim == 1
        assert len(dec_scores) == 20
        assert np.all(np.isfinite(dec_scores))

        ap = average_precision_score(y_syn, dec_scores)
        assert 0.0 <= ap <= 1.0

    def test_linearsvc_cross_val_score_average_precision_native_support(self):
        feat_b = EXPERIMENT_FEATURES["B"]
        lsvc = LinearSVC(random_state=42, max_iter=1000)
        pipe = build_pipeline(feat_b, lsvc, use_tld=True, model_name="LinearSVC")

        np.random.seed(42)
        syn_data = {col: np.random.normal(0, 1, size=20) for col in feat_b}
        syn_data["TLD"] = ["com", "org"] * 10
        df_syn = pd.DataFrame(syn_data)
        y_syn = np.array([0, 1] * 10)

        scores = cross_val_score(pipe, df_syn, y_syn, scoring="average_precision", cv=2)
        assert len(scores) == 2
        for s in scores:
            assert 0.0 <= s <= 1.0


# =====================================================================
# 6. Isolation des seuils et non-propagation
# =====================================================================
class TestExperimentBThresholdPolicy:
    def test_standard_threshold_for_b_is_0_5(self):
        thresh = get_model_threshold("RandomForest", "B", policy="standard")
        assert thresh == 0.5

    def test_standard_threshold_for_linearsvc_is_0_0(self):
        thresh = get_model_threshold("LinearSVC", "B", policy="standard")
        assert thresh == 0.0

    def test_high_precision_threshold_raises_for_b(self):
        with pytest.raises(ValueError, match="Aucun seuil haute précision"):
            get_model_threshold("RandomForest", "B", policy="high_precision")


# =====================================================================
# 7. Chemins et isolation des artefacts de B
# =====================================================================
class TestExperimentBArtifactProtection:
    def test_b_artifact_paths_are_distinct_from_a_c_d(self):
        for model in ["RandomForest", "LogisticRegression", "DecisionTree", "KNN", "LinearSVC", "SVC_RBF"]:
            b_path = MODELS_DIR / f"B_{model}.joblib"
            a_path = MODELS_DIR / f"A_{model}.joblib"
            c_path = MODELS_DIR / f"C_{model}.joblib"
            d_path = MODELS_DIR / f"D_{model}.joblib"
            assert b_path != a_path
            assert b_path != c_path
            assert b_path != d_path

        assert (MODELS_DIR / "best_model_B.joblib") != (MODELS_DIR / "best_model_A.joblib")
        assert (MODELS_DIR / "best_model_B.joblib") != (MODELS_DIR / "best_model_C.joblib")
        assert (MODELS_DIR / "best_model_B.joblib") != (MODELS_DIR / "best_model_D.joblib")
        assert (TABLES_DIR / "cv_results_B.csv") != (TABLES_DIR / "cv_results_A.csv")
        assert (TABLES_DIR / "cv_results_B.csv") != (TABLES_DIR / "cv_results_C.csv")
        assert (TABLES_DIR / "test_metrics_B.csv") != (TABLES_DIR / "test_metrics_A.csv")

    def test_subgroups_evaluation_b_path_exact_name(self):
        expected_csv = TABLES_DIR / "subgroups_evaluation_B.csv"
        assert expected_csv.name == "subgroups_evaluation_B.csv"
        assert expected_csv.name != "subgroup_metrics_B.csv"

    def test_all_22_historical_c_and_d_hashes_intact_against_audit_csv(self):
        audit_file = TABLES_DIR / "phase6cbis_integrity_audit.csv"
        assert audit_file.exists()
        df_audit = pd.read_csv(audit_file)
        assert len(df_audit) == 22
        for _, row in df_audit.iterrows():
            p = Path(row["relative_path"])
            assert p.exists(), f"Artefact historique introuvable : {p}"
            current_h = hashlib.sha256(p.read_bytes()).hexdigest().upper()
            assert current_h == row["expected_sha256"].upper(), f"Altération détectée pour {p} !"
            assert current_h == row["actual_sha256"].upper()
            assert row["match"] is True or row["match"] == 1


# =====================================================================
# 8. Audit rigoureux des définitions de sous-groupes G1/G2
# =====================================================================
class TestSubgroupsExactLogicAndPartitions:
    def test_get_subgroup_masks_definitions_on_synthetic_urls(self):
        urls = pd.Series([
            "https://www.example.com",           # G1: True, G2: True (racine vide)
            "https://www.example.com/",          # G1: True, G2: True (slash isolé)
            "https://www.example.com/path",      # G1: True, G2: False (chemin non vide)
            "https://www.example.com/?q=1",      # G1: True, G2: False (requête présente)
            "https://www.example.com/#frag",     # G1: True, G2: False (fragment présent)
            "http://www.example.com",            # G1: False, G2: False (pas https)
            "https://example.com",               # G1: False, G2: False (pas www.)
            "ftp://www.example.com",             # G1: False, G2: False
        ])

        masks = get_subgroup_masks(urls)
        g1 = masks["G1"]
        g2 = masks["G2"]

        assert list(g1) == [True, True, True, True, True, False, False, False]
        assert list(g2) == [True, True, False, False, False, False, False, False]

    def test_inclusion_g2_subset_of_g1_always_holds(self):
        # Sur un large éventail d'URLs
        urls = pd.Series([
            "https://www.google.com/",
            "https://www.google.com/search?q=test",
            "https://www.bank.org",
            "http://www.phish.com",
            "https://phish.com/login",
            "https://www.phish-login.com#section",
            "https://www.phish-login.com/?token=123",
        ])
        masks = get_subgroup_masks(urls)
        # Partout où G2 est True, G1 DOIT être True
        assert (masks["G2"] & ~masks["G1"]).sum() == 0

    def test_three_disjoint_sets_partition(self):
        urls = pd.Series([
            "https://www.site1.com/",            # set 1: G2
            "https://www.site2.com/page",        # set 2: G1 \ G2
            "http://site3.com/test",             # set 3: Test \ G1
            "https://site4.com/",                # set 3: Test \ G1
        ])
        masks = get_subgroup_masks(urls)
        g1 = masks["G1"]
        g2 = masks["G2"]

        set1 = g2
        set2 = g1 & ~g2
        set3 = ~g1

        # Vérifier que les trois ensembles sont disjoints
        assert (set1 & set2).sum() == 0
        assert (set1 & set3).sum() == 0
        assert (set2 & set3).sum() == 0

        # Vérifier que leur union couvre 100% du jeu
        assert (set1 | set2 | set3).sum() == len(urls)
