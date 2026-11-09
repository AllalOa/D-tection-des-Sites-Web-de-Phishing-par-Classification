"""
test_config.py - Tests de la cohérence de src/config.py.

Vérifie que :
- Les trois familles (URL_ONLY, PAGE, DERIVED) partitionnent exactement les 50 colonnes
- Les longueurs sont correctes (18 + 26 + 6 = 50)
- Les familles sont disjointes (pas de doublon)
- Les listes sont non vides
"""

import pytest
from src.config import (
    URL_ONLY_FEATURES,
    PAGE_FEATURES,
    DERIVED_FEATURES,
    ALL_NUMERIC_FEATURES,
    RANDOM_STATE,
    N_FOLDS,
    TEST_SIZE,
    TOP_N_TLD,
)


class TestFeatureFamilies:
    """Tests sur la partition des familles de features."""

    def test_url_only_count(self):
        """URL_ONLY doit contenir exactement 18 variables."""
        assert len(URL_ONLY_FEATURES) == 18, (
            f"URL_ONLY_FEATURES a {len(URL_ONLY_FEATURES)} variables, attendu 18"
        )

    def test_page_count(self):
        """PAGE doit contenir exactement 26 variables."""
        assert len(PAGE_FEATURES) == 26, (
            f"PAGE_FEATURES a {len(PAGE_FEATURES)} variables, attendu 26"
        )

    def test_derived_count(self):
        """DERIVED doit contenir exactement 6 variables."""
        assert len(DERIVED_FEATURES) == 6, (
            f"DERIVED_FEATURES a {len(DERIVED_FEATURES)} variables, attendu 6"
        )

    def test_total_count(self):
        """La réunion des trois familles doit faire exactement 50 variables."""
        assert len(ALL_NUMERIC_FEATURES) == 50, (
            f"ALL_NUMERIC_FEATURES a {len(ALL_NUMERIC_FEATURES)} variables, attendu 50"
        )

    def test_no_overlap_url_only_page(self):
        """URL_ONLY et PAGE doivent être disjoints."""
        overlap = set(URL_ONLY_FEATURES) & set(PAGE_FEATURES)
        assert len(overlap) == 0, f"Variables en commun entre URL_ONLY et PAGE : {overlap}"

    def test_no_overlap_url_only_derived(self):
        """URL_ONLY et DERIVED doivent être disjoints."""
        overlap = set(URL_ONLY_FEATURES) & set(DERIVED_FEATURES)
        assert len(overlap) == 0, f"Variables en commun entre URL_ONLY et DERIVED : {overlap}"

    def test_no_overlap_page_derived(self):
        """PAGE et DERIVED doivent être disjoints."""
        overlap = set(PAGE_FEATURES) & set(DERIVED_FEATURES)
        assert len(overlap) == 0, f"Variables en commun entre PAGE et DERIVED : {overlap}"

    def test_partition_covers_all(self):
        """
        Les trois familles réunies doivent être exactement l'ensemble complet.
        Il ne doit pas y avoir de variables dans ALL_NUMERIC non couvertes.
        """
        union = set(URL_ONLY_FEATURES) | set(PAGE_FEATURES) | set(DERIVED_FEATURES)
        assert union == set(ALL_NUMERIC_FEATURES), (
            f"La réunion des familles ne correspond pas à ALL_NUMERIC_FEATURES.\n"
            f"Différence : {union.symmetric_difference(set(ALL_NUMERIC_FEATURES))}"
        )

    def test_no_duplicates_in_families(self):
        """Chaque famille ne doit pas contenir de doublons internes."""
        assert len(URL_ONLY_FEATURES) == len(set(URL_ONLY_FEATURES)), "Doublons dans URL_ONLY"
        assert len(PAGE_FEATURES) == len(set(PAGE_FEATURES)), "Doublons dans PAGE"
        assert len(DERIVED_FEATURES) == len(set(DERIVED_FEATURES)), "Doublons dans DERIVED"


class TestConfigParams:
    """Tests des paramètres généraux."""

    def test_random_state(self):
        """random_state doit être 42."""
        assert RANDOM_STATE == 42

    def test_n_folds(self):
        """K doit être >= 5."""
        assert N_FOLDS >= 5, f"N_FOLDS={N_FOLDS}, doit être >= 5"

    def test_test_size(self):
        """test_size doit être entre 0 et 1."""
        assert 0 < TEST_SIZE < 1

    def test_top_n_tld(self):
        """TOP_N_TLD doit être 20."""
        assert TOP_N_TLD == 20


class TestSpecificFeatures:
    """Tests que des features clés sont bien dans les bonnes familles."""

    def test_ishttps_in_url_only(self):
        assert "IsHTTPS" in URL_ONLY_FEATURES

    def test_url_length_in_url_only(self):
        assert "URLLength" in URL_ONLY_FEATURES

    def test_url_similarity_in_derived(self):
        assert "URLSimilarityIndex" in DERIVED_FEATURES

    def test_line_of_code_in_page(self):
        assert "LineOfCode" in PAGE_FEATURES

    def test_no_label_in_features(self):
        """La colonne 'label' ne doit PAS être dans les features."""
        all_feats = set(ALL_NUMERIC_FEATURES)
        assert "label" not in all_feats
        assert "phishing" not in all_feats


class TestExperimentFeatures:
    """Tests des définitions d'expériences A, B, B' et C."""

    def test_experiment_feature_counts(self):
        from src.config import EXPERIMENT_FEATURES
        assert len(EXPERIMENT_FEATURES["A"]) == 50
        assert len(EXPERIMENT_FEATURES["B"]) == 44
        assert len(EXPERIMENT_FEATURES["B'"]) == 38
        assert len(EXPERIMENT_FEATURES["C"]) == 18
        assert len(EXPERIMENT_FEATURES["D"]) == 10

    def test_experiment_sources_documented(self):
        from src.config import FEATURE_SOURCES_BY_EXPERIMENT, EXPERIMENT_FEATURES
        for exp in ["A", "B", "B'", "C", "D"]:
            assert exp in FEATURE_SOURCES_BY_EXPERIMENT
            assert len(FEATURE_SOURCES_BY_EXPERIMENT[exp]) == len(EXPERIMENT_FEATURES[exp])
