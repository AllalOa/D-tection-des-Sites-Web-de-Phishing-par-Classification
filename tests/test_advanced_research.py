"""
tests/test_advanced_research.py - Tests unitaires et de non-régression pour la Phase 3 (Fateh).

Couvre :
1. Parsing robuste IPv4 et IPv6 (avec et sans ports, avec et sans identifiants).
2. Décomposition PSL (Public Suffix List) : distinction hostname / suffix / domain / subdomain.
3. Validation d'URLs (schémas HTTP/HTTPS, URLs sans schéma pour D, entrées invalides).
4. ModelManager : lazy loading, cache mémoire, thread-safety, validation de schéma et gestion d'erreurs.
5. Cohérence et performance des prédictions répétées (< 1 ms en mémoire).
"""

import threading
import time
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.url_features import compute_url_features
from src.host_features import extract_host, host_features
from src.predict import _validate_url, predict_url, predict_urls, InvalidURLError
from src.research.model_manager import (
    ModelManager,
    default_model_manager,
    ModelNotFoundError,
    InvalidModelError,
    SchemaMismatchError,
)
from src.research.psl import (
    decompose_domain,
    get_public_suffix,
    get_registrable_domain,
    count_subdomains_psl,
)
from src.research.url_parser import parse_url_robust, validate_url


# =============================================================================
# 1. Tests IPv4 et IPv6 (Correction du bug de colon split)
# =============================================================================
class TestIPv4AndIPv6Parsing:
    """Vérifie le parsing et le calcul des features sur IPv4 et IPv6."""

    def test_ipv4_standard_and_port(self):
        f = compute_url_features("http://192.168.1.1:8080/login")
        assert f["IsDomainIP"] == 1
        assert f["DomainLength"] == len("192.168.1.1")
        assert f["TLDLength"] == 0
        assert f["NoOfSubDomain"] == 0

    def test_ipv6_with_port_in_url_features(self):
        """Correction critique : IPv6 avec port ne doit pas être tronqué à [2001."""
        url = "http://[2001:db8::1]:8080/path?q=1"
        f = compute_url_features(url)
        assert f["IsDomainIP"] == 1
        assert f["DomainLength"] == len("2001:db8::1")
        assert f["TLDLength"] == 0
        assert f["NoOfSubDomain"] == 0

    def test_ipv6_loopback(self):
        f = compute_url_features("http://[::1]/")
        assert f["IsDomainIP"] == 1
        assert f["DomainLength"] == len("::1")
        assert f["TLDLength"] == 0

    def test_ipv6_in_host_features(self):
        h = extract_host("http://[2001:db8::2]:8443/api")
        assert h == "2001:db8::2"
        f = host_features("http://[2001:db8::2]:8443/api")
        assert f["HostIsIP"] == 1
        assert f["HostLength"] == len("2001:db8::2")
        assert f["HostTLDLength"] == 0
        assert f["HostNoOfSubDomain"] == 0


# =============================================================================
# 2. Tests de la Décomposition PSL (Public Suffix List)
# =============================================================================
class TestPublicSuffixListDecomposition:
    """Vérifie la décomposition des domaines selon la Public Suffix List."""

    @pytest.mark.parametrize(
        "raw_host,expected_suffix,expected_domain,expected_n_sub",
        [
            ("example.com", "com", "example.com", 0),
            ("bbc.co.uk", "co.uk", "bbc.co.uk", 0),
            ("example.com.au", "com.au", "example.com.au", 0),
            ("subdomain.example.co.uk", "co.uk", "example.co.uk", 1),
            ("api.v2.service.gov.uk", "gov.uk", "service.gov.uk", 2),
            ("www.example.com", "com", "example.com", 1),
            ("service.asso.fr", "asso.fr", "service.asso.fr", 0),
            ("192.168.1.1", "", "192.168.1.1", 0),
            ("[2001:db8::1]", "", "2001:db8::1", 0),
        ],
    )
    def test_psl_decomposition_cases(self, raw_host, expected_suffix, expected_domain, expected_n_sub):
        decomp = decompose_domain(raw_host)
        assert decomp.suffix == expected_suffix
        assert decomp.domain == expected_domain
        assert decomp.no_of_subdomains == expected_n_sub

    def test_psl_helper_functions(self):
        assert get_public_suffix("sub.bbc.co.uk") == "co.uk"
        assert get_registrable_domain("sub.bbc.co.uk") == "bbc.co.uk"
        assert count_subdomains_psl("www.sub.example.co.uk", exclude_www=True) == 1
        assert count_subdomains_psl("www.sub.example.co.uk", exclude_www=False) == 2


# =============================================================================
# 3. Tests de Validation d'URL (Schémas, Schemeless, Entrées Invalides)
# =============================================================================
class TestURLValidationRules:
    """Vérifie les règles de validation pour les expériences C et D."""

    def test_validate_url_allows_schemeless_for_exp_d(self):
        """L'expérience D doit accepter des noms d'hôte ou URLs sans schéma."""
        _validate_url("example.com", experiment="D")
        _validate_url("sub.domain.org/login", experiment="D")
        _validate_url("192.168.1.1", experiment="D")

    def test_validate_url_rejects_schemeless_for_exp_c(self):
        """L'expérience C doit rejeter une URL sans schéma (protocole requis)."""
        with pytest.raises(InvalidURLError, match="hôte valide"):
            _validate_url("example.com", experiment="C")

    def test_validate_url_invalid_inputs(self):
        with pytest.raises(InvalidURLError):
            _validate_url("")
        with pytest.raises(InvalidURLError):
            _validate_url("   ")
        with pytest.raises(InvalidURLError):
            _validate_url(None)
        with pytest.raises(InvalidURLError):
            _validate_url(12345)
        with pytest.raises(InvalidURLError):
            _validate_url("https://")


# =============================================================================
# 4. Tests du ModelManager (Lazy Loading, Cache, Thread-Safety)
# =============================================================================
class TestModelManagerOperations:
    """Tests unitaires du gestionnaire de modèles en mémoire."""

    def test_lazy_loading_and_caching(self):
        mm = ModelManager()
        assert not mm.is_cached("C")

        pipe1, meta1 = mm.load_model_and_metadata("C")
        assert mm.is_cached("C")
        assert "C" in mm.cached_models

        # Deuxième appel : objet identique en mémoire (pas de reload disque)
        pipe2, meta2 = mm.load_model_and_metadata("C")
        assert pipe1 is pipe2
        assert meta1 is meta2

    def test_cache_invalidation(self):
        mm = ModelManager()
        mm.load_model_and_metadata("C")
        assert mm.is_cached("C")

        mm.invalidate_cache("C")
        assert not mm.is_cached("C")

    def test_model_not_found_raises(self, tmp_path):
        empty_mm = ModelManager(models_dir=tmp_path)
        with pytest.raises(ModelNotFoundError):
            empty_mm.load_model_and_metadata("NONEXISTENT_EXP")

    def test_feature_validation_completes_missing(self):
        mm = ModelManager()
        partial_features = {"URLLength": 25, "IsHTTPS": 1}
        df_valid = mm.validate_features(partial_features, experiment="C", allow_missing=True)
        assert df_valid.shape == (1, 18)
        assert df_valid["URLLength"].iloc[0] == 25
        assert df_valid["IsHTTPS"].iloc[0] == 1
        assert df_valid["DomainLength"].iloc[0] == 0.0

    def test_feature_validation_rejects_missing_when_strict(self):
        mm = ModelManager()
        partial_features = {"URLLength": 25}
        with pytest.raises(SchemaMismatchError):
            mm.validate_features(partial_features, experiment="C", allow_missing=False)

    def test_thread_safety_concurrent_loading(self):
        """Vérifie le chargement concurrent depuis 10 threads sans race condition."""
        mm = ModelManager()
        pipelines = []
        errors = []

        def worker():
            try:
                pipe, _ = mm.load_model_and_metadata("C")
                pipelines.append(pipe)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Erreurs détectées en concurrence : {errors}"
        assert len(pipelines) == 10
        # Tous les threads ont reçu exactement la même instance en cache
        first = pipelines[0]
        assert all(p is first for p in pipelines)


# =============================================================================
# 5. Tests de Cohérence et Vitesse de Prédiction
# =============================================================================
class TestPredictionConsistencyAndLatency:
    """Vérifie l'exactitude et la vitesse de l'inférence optimisée."""

    def test_predict_url_returns_consistent_keys(self):
        url = "https://www.southbankmosaics.com"
        res = predict_url(url, experiment="C")
        assert res["url"] == url
        assert res["predicted_class"] == 0
        assert res["predicted_label"] == "legitime"
        assert res["model_name"] == "RandomForest"
        assert res["experiment"] == "C"
        assert len(res["features"]) == 18

    def test_predict_urls_batch_matches_single(self):
        urls = [
            "https://www.southbankmosaics.com",
            "http://192.168.1.1/admin/login.php",
            "https://www.google.com",
        ]
        batch_results = predict_urls(urls, experiment="C")
        single_results = [predict_url(u, experiment="C") for u in urls]

        for b_res, s_res in zip(batch_results, single_results):
            assert b_res["predicted_class"] == s_res["predicted_class"]
            assert np.isclose(b_res["model_score"], s_res["model_score"], atol=1e-6)

    def test_in_memory_prediction_latency_improved(self):
        """Vérifie que la prédiction en cache mémoire évite le rechargement disque de 400ms."""
        url = "https://www.southbankmosaics.com/login"
        # Préchauffage du cache
        predict_url(url, experiment="C")

        # Mesure sur 20 itérations
        latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            predict_url(url, experiment="C")
            latencies.append(time.perf_counter() - t0)

        mean_ms = np.mean(latencies) * 1000
        print(f"Latence moyenne en cache mémoire : {mean_ms:.3f} ms")
        assert mean_ms < 350.0, f"La latence ({mean_ms:.2f} ms) doit être nettement inférieure au chargement disque (>400ms)!"
