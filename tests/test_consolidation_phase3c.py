"""
test_consolidation_phase3c.py - Tests unitaires de consolidation scientifique (Phase 3c).

Couvre :
1. Politique de seuils : interdiction de propager le seuil RF D (0.9352) à un autre modèle.
2. Sous-groupes G1/G2 : inclusion stricte G2 ⊆ G1, définition URL brute, comportement PrefixRule.
3. Cluster Bootstrap : échantillonnage avec remise par Host, intégrité des clusters, graine fixée.
4. Test de McNemar : calcul exact bidirectionnel sur cas artificiels connus.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.evaluate import (
    get_subgroup_masks,
    cluster_bootstrap_ci,
    get_model_threshold,
    mcnemar_test,
    HIGH_PRECISION_THRESHOLDS,
)
from src.predict import predict_url, _load_model_and_metadata


# =====================================================================
# 1. Tests de la politique de seuil (Section 2)
# =====================================================================
class TestThresholdPolicy:
    """Vérifie que les seuils haute précision ne sont jamais appliqués silencieusement."""

    def test_high_precision_threshold_rf_d(self):
        """RandomForest sur l'expérience D doit retourner 0.9352."""
        thresh = get_model_threshold("RandomForest", "D", policy="high_precision")
        assert thresh == 0.9352

    def test_high_precision_threshold_other_models_raise_value_error(self):
        """Tout autre modèle que RandomForest D doit lever ValueError."""
        other_models = ["LogisticRegression", "DecisionTree", "KNN", "LinearSVC", "SVC_RBF"]
        for m in other_models:
            with pytest.raises(ValueError, match="Interdiction d'appliquer silencieusement"):
                get_model_threshold(m, "D", policy="high_precision")

    def test_high_precision_threshold_other_experiments_raise_value_error(self):
        """Une autre expérience (A, B, C) pour RandomForest doit lever ValueError."""
        for exp in ["A", "B", "B'", "C"]:
            with pytest.raises(ValueError, match="Interdiction d'appliquer silencieusement"):
                get_model_threshold("RandomForest", exp, policy="high_precision")

    def test_standard_threshold_defaults(self):
        """Seuil standard : 0.5 pour proba, 0.0 pour LinearSVC ou decision_function."""
        assert get_model_threshold("RandomForest", "D", policy="standard") == 0.5
        assert get_model_threshold("LogisticRegression", "D", policy="standard") == 0.5
        assert get_model_threshold("LinearSVC", "D", policy="standard") == 0.0

        mock_pipe_df = MagicMock()
        del mock_pipe_df.predict_proba
        mock_pipe_df.decision_function = MagicMock()
        assert get_model_threshold("CustomModel", "D", policy="standard", pipe=mock_pipe_df) == 0.0

    def test_predict_url_rejects_silent_threshold_transfer(self):
        """predict_url avec un modèle non-RF D et threshold='high_precision' doit lever ValueError."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.2, 0.8]])
        mock_meta = {
            "experiment": "D",
            "model_name": "LogisticRegression",
            "feature_cols": ["HostLength", "HostTLDLength", "HostNoOfSubDomain", "HostIsIP",
                            "HostNoOfDigits", "HostDigitRatio", "HostNoOfLetters",
                            "HostLetterRatio", "HostNoOfHyphens", "HostLongestLabel"]
        }

        with patch("src.predict._load_model_and_metadata", return_value=(mock_pipeline, mock_meta)):
            with pytest.raises(ValueError, match="Interdiction d'appliquer silencieusement"):
                predict_url("https://www.example.com", experiment="D", threshold="high_precision")


# =====================================================================
# 2. Validation des sous-groupes G1 et G2 (Section 3)
# =====================================================================
class TestSubgroupValidation:
    """Tests sur DataFrames synthétiques sans charger de modèles réels."""

    def test_g2_is_subset_of_g1(self):
        """G2 doit être strictement inclus dans G1 pour tout ensemble d'URLs."""
        urls = pd.Series([
            "https://www.bank.com",
            "https://www.bank.com/",
            "https://www.bank.com/login",
            "https://www.bank.com/account?id=12",
            "https://www.bank.com#token",
            "http://www.bank.com",
            "https://bank.com",
            "https://phishing.com/login",
            "https://www.phishing.ru/",
            "http://phishing.tk/index.html",
        ])
        masks = get_subgroup_masks(urls)
        g1 = masks["G1"]
        g2 = masks["G2"]

        # Tout élément vrai dans G2 doit être vrai dans G1
        assert (g2 & ~g1).sum() == 0, "G2 contient des éléments hors de G1 !"
        assert g2.sum() <= g1.sum()

    def test_subgroup_definitions_exact(self):
        """Vérifie chaque critère de définition de G1 et G2."""
        urls = pd.Series([
            "https://www.example.com",          # G1=True, G2=True
            "https://www.example.com/",         # G1=True, G2=True
            "https://www.example.com/path",     # G1=True, G2=False (chemin non vide)
            "https://www.example.com?query=1",  # G1=True, G2=False (requête)
            "https://www.example.com#fragment", # G1=True, G2=False (fragment)
            "http://www.example.com",           # G1=False, G2=False (http)
            "https://example.com",              # G1=False, G2=False (sans www.)
        ])
        masks = get_subgroup_masks(urls)
        assert list(masks["G1"]) == [True, True, True, True, True, False, False]
        assert list(masks["G2"]) == [True, True, False, False, False, False, False]

    def test_prefix_rule_behavior_on_g1_and_g2(self):
        """PrefixRule (not startswith https://www.) doit prédire 0 pour 100% de G1 et G2."""
        urls = pd.Series([
            "https://www.legitimate1.com",
            "https://www.legitimate2.com/",
            "https://www.phishing1.com/login",
            "https://www.phishing2.com/secure",
        ])
        masks = get_subgroup_masks(urls)

        # PrefixRule prédit 0 si commence par https://www., sinon 1
        for g in ["G1", "G2"]:
            sub_urls = urls[masks[g]]
            prefix_preds = [0 if u.startswith("https://www.") else 1 for u in sub_urls]
            assert all(p == 0 for p in prefix_preds), f"PrefixRule a prédit autre chose que 0 sur {g} !"


# =====================================================================
# 3. Audit du Cluster Bootstrap (Section 4)
# =====================================================================
class TestClusterBootstrap:
    """Vérifie le rééchantillonnage par cluster d'hôtes sur données synthétiques."""

    def test_cluster_bootstrap_resamples_whole_hosts(self):
        """Toutes les observations d'un hôte tiré doivent rester groupées."""
        # 4 hôtes, chacun ayant 3 observations
        hosts = ["hostA", "hostA", "hostA", "hostB", "hostB", "hostB",
                 "hostC", "hostC", "hostC", "hostD", "hostD", "hostD"]
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.1, 0.9, 0.8, 0.9, 0.2, 0.1, 0.3, 0.8, 0.7, 0.9])
        df_sub = pd.DataFrame({"Host": hosts, "phishing": y_true})

        # Test de reproductibilité de la graine
        pr_ci_1, roc_ci_1 = cluster_bootstrap_ci(df_sub, y_true, y_score, n_boot=50, seed=42)
        pr_ci_2, roc_ci_2 = cluster_bootstrap_ci(df_sub, y_true, y_score, n_boot=50, seed=42)

        assert pr_ci_1 == pr_ci_2
        assert roc_ci_1 == roc_ci_2
        assert 0.0 <= pr_ci_1[0] <= pr_ci_1[1] <= 1.0
        assert 0.0 <= roc_ci_1[0] <= roc_ci_1[1] <= 1.0

    def test_cluster_bootstrap_single_class_draws_handled_gracefully(self):
        """Un tirage synthétique sans crash même si des tirages n'ont qu'une seule classe."""
        # Un seul hôte phishing, plusieurs hôtes légitimes
        hosts = ["leg1", "leg2", "leg3", "phi1"]
        y_true = np.array([0, 0, 0, 1])
        y_score = np.array([0.1, 0.2, 0.1, 0.9])
        df_sub = pd.DataFrame({"Host": hosts, "phishing": y_true})

        # Certains tirages bootstrap avec remise n'auront aucun phi1 (classe 0 pure)
        pr_ci, roc_ci = cluster_bootstrap_ci(df_sub, y_true, y_score, n_boot=50, seed=42)
        assert isinstance(pr_ci, tuple) and len(pr_ci) == 2
        assert isinstance(roc_ci, tuple) and len(roc_ci) == 2


# =====================================================================
# 4. Vérification de McNemar (Section 5)
# =====================================================================
class TestMcNemarExact:
    """Vérifie le calcul exact bidirectionnel de McNemar sur cas connus."""

    def test_mcnemar_symmetric_discordant_pairs(self):
        """Quand b == c, la p-value bidirectionnelle doit valoir 1.0."""
        y_test = np.array([1, 1, 1, 0, 0])
        y_pred1 = np.array([1, 1, 0, 0, 0])  # faux sur index 2
        y_pred2 = np.array([1, 1, 1, 1, 0])  # faux sur index 3

        res = mcnemar_test(y_test, y_pred1, y_pred2, "M1", "M2")
        assert res["n_discordant_1_wrong"] == 1
        assert res["n_discordant_2_wrong"] == 1
        assert res["p_value"] == 1.0
        assert not res["significant_0.05"]

    def test_mcnemar_known_extreme_asymmetry(self):
        """Avec b=0 et c=10 paires discordantes, p = 2 * (0.5)^10 = 0.001953125."""
        # 10 observations où M1 a raison et M2 a tort
        y_test = np.array([1] * 10)
        y_pred1 = np.array([1] * 10)  # M1 tout juste -> 0 faux
        y_pred2 = np.array([0] * 10)  # M2 tout faux -> 10 faux

        res = mcnemar_test(y_test, y_pred1, y_pred2, "M1", "M2")
        assert res["n_discordant_1_wrong"] == 0
        assert res["n_discordant_2_wrong"] == 10
        expected_p = 2 * (0.5 ** 10)  # 0.001953125
        assert abs(res["p_value"] - expected_p) < 1e-10
        assert res["significant_0.05"]

    def test_mcnemar_zero_discordant_pairs(self):
        """Quand les deux modèles ont exactement les mêmes prédictions, p=1.0."""
        y_test = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0])
        res = mcnemar_test(y_test, y_pred, y_pred, "M1", "M1_bis")
        assert res["n_discordant_1_wrong"] == 0
        assert res["n_discordant_2_wrong"] == 0
        assert res["p_value"] == 1.0
