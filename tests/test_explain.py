"""
tests/test_explain.py — Tests unitaires de l'explicabilité locale SHAP.
Auteur : Mohamed Makhloufi (IASD15 / @mohamedmakhloufi2004)
Projet : PROJ-IASD01 — Détection des sites de phishing par classification
"""

import time
import pytest
from src.explain import PhishExplainer, get_explainer, explain_url
from src.predict import InvalidURLError


@pytest.fixture(scope="module")
def explainer_c():
    """Initialise une fois l'explainer pour l'expérience C."""
    return get_explainer("C")


class TestPhishExplainerInit:
    def test_explainer_initialization_c(self, explainer_c):
        """Vérifie l'initialisation correcte de TreeExplainer sur l'expérience C."""
        assert explainer_c.experiment == "C"
        assert len(explainer_c.feature_names) == 18
        assert explainer_c.classifier is not None
        assert explainer_c.explainer is not None
        assert 0.0 <= explainer_c.base_value <= 1.0

    def test_explainer_initialization_d(self):
        """Vérifie l'initialisation sur l'expérience D."""
        exp_d = get_explainer("D")
        assert exp_d.experiment == "D"
        assert len(exp_d.feature_names) == 10
        assert 0.0 <= exp_d.base_value <= 1.0


class TestShapAdditivityAndValues:
    @pytest.mark.parametrize("url", [
        "https://www.google.com/",
        "https://www.service-login.com/",
        "https://secure-banking-alert.account-update.xyz/login.php",
    ])
    def test_exact_local_additivity(self, explainer_c, url):
        """
        Vérifie la propriété fondamentale de Shapley (Efficience / Additivité locale) :
        phi_0 + sum(phi_i) == f(x) (erreur < 1e-4)
        """
        res = explainer_c.explain(url)
        assert res["success"] is True
        assert res["additivity_valid"] is True
        assert res["additivity_error"] < 1e-4
        assert abs(res["predicted_proba"] - res["reconstructed_proba"]) < 1e-4

    def test_top_factors_ordering(self, explainer_c):
        """Vérifie que les facteurs de risque sont correctement triés par amplitude."""
        res = explainer_c.explain("https://www.service-login.com/")
        
        # Facteurs phishing triés par shap_value décroissante
        phish_vals = [f["shap_value"] for f in res["top_phishing_factors"]]
        assert phish_vals == sorted(phish_vals, reverse=True)
        
        # Facteurs légitimes triés par shap_value croissante (les plus négatifs d'abord)
        legit_vals = [f["shap_value"] for f in res["top_legitimate_factors"]]
        assert legit_vals == sorted(legit_vals)


class TestInputValidationAndPerformance:
    @pytest.mark.parametrize("invalid_url", [
        "",
        "   ",
        "not_a_valid_url",
        "ftp://invalid-scheme.com",
        "http://",
        "https://",
    ])
    def test_invalid_url_raises(self, explainer_c, invalid_url):
        """Vérifie que les entrées malformées déclenchent InvalidURLError ou ValueError."""
        with pytest.raises((InvalidURLError, ValueError)):
            explainer_c.explain(invalid_url)

    def test_computation_latency_under_300ms(self, explainer_c):
        """Vérifie que l'explication locale s'exécute en moins de 300 ms en mode chaud."""
        # Appel à chaud
        url = "https://www.service-login.com/"
        _ = explainer_c.explain(url)

        t0 = time.perf_counter()
        res = explainer_c.explain(url)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < 300.0, f"Temps d'exécution excessif : {elapsed_ms:.2f} ms"
        assert res["latency_ms"] < 300.0


class TestVisualizationAndHelper:
    def test_waterfall_plot_base64_generation(self, explainer_c):
        """Vérifie que la visualisation base64 est générée sans erreur."""
        b64 = explainer_c.generate_waterfall_plot_base64("https://www.service-login.com/")
        assert isinstance(b64, str)
        assert b64.startswith("data:image/png;base64,")
        assert len(b64) > 1000  # Image PNG valide et non vide

    def test_explain_url_helper_function(self):
        """Vérifie l'utilitaire top-level explain_url."""
        res = explain_url("https://www.google.com/")
        assert res["success"] is True
        assert res["verdict"] in ("phishing", "legitime")
        assert 0.0 <= res["predicted_proba"] <= 1.0
        assert "contributions" in res
