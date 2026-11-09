"""
test_host_features.py - Tests de src/host_features.py.

Vérifie :
- Cas de bords pour extract_host (schémas http/https/sans schéma, www, port, credentials @, IP v4, majuscules, point final, hôte vide)
- Invariance de host_features sur différentes variantes d'une même URL
- Cohérence unitaire / lot (batch) sur 1 000 URLs réelles
- Structure, colonnes et types des 11 variables HOST_FEATURES
"""

import pytest
import pandas as pd
import numpy as np

from src.host_features import (
    extract_host,
    host_features,
    host_features_batch,
    HOST_FEATURES,
)
from src.predict import InvalidURLError
from src.data_loader import load_data


class TestExtractHostEdgeCases:
    """Tests des cas de bords pour extract_host."""

    def test_http_scheme(self):
        """Schéma http:// correctement extrait."""
        assert extract_host("http://example.com") == "example.com"

    def test_https_scheme(self):
        """Schéma https:// correctement extrait."""
        assert extract_host("https://example.com") == "example.com"

    def test_no_scheme_accepted(self):
        """URLs sans schéma acceptées pour les tests."""
        assert extract_host("example.com") == "example.com"
        assert extract_host("sub.example.com/login?u=1") == "sub.example.com"

    def test_www_initial_stripped(self):
        """'www.' initial supprimé, mais pas au milieu du domaine."""
        assert extract_host("https://www.example.com") == "example.com"
        assert extract_host("http://www.sub.example.com") == "sub.example.com"
        # www au milieu d'un nom
        assert extract_host("https://awww.example.com") == "awww.example.com"

    def test_port_stripped(self):
        """Port explicite (:80, :8080) retiré de l'hôte."""
        assert extract_host("http://example.com:8080") == "example.com"
        assert extract_host("https://example.com:443/test") == "example.com"

    def test_credentials_with_at_stripped(self):
        """Identifiants utilisateur avec '@' correctement retirés."""
        assert extract_host("http://user:pass@example.com/path") == "example.com"
        assert extract_host("https://admin@malicious-host.org:8080") == "malicious-host.org"

    def test_ipv4_host(self):
        """Hôte en adresse IPv4 valide préservé sans altération."""
        assert extract_host("http://192.168.1.1:8000/api") == "192.168.1.1"
        assert extract_host("https://10.0.0.1") == "10.0.0.1"

    def test_uppercase_converted(self):
        """Majuscules converties en minuscules."""
        assert extract_host("HTTPS://WWW.EXAMPLE.COM/Path") == "example.com"
        assert extract_host("HTTP://X.COM:8080/#F") == "x.com"

    def test_trailing_dot_stripped(self):
        """Point final de FQDN (ex: example.com.) supprimé."""
        assert extract_host("http://example.com.") == "example.com"
        assert extract_host("https://www.example.com./") == "example.com"

    def test_empty_or_invalid_host_raises(self):
        """Hôte vide, invalide ou chaîne d'espaces lève InvalidURLError."""
        with pytest.raises(InvalidURLError):
            extract_host("")
        with pytest.raises(InvalidURLError):
            extract_host("   ")
        with pytest.raises(InvalidURLError):
            extract_host(None)
        with pytest.raises(InvalidURLError):
            extract_host(123)
        with pytest.raises(InvalidURLError):
            extract_host("https://")
        with pytest.raises(InvalidURLError):
            extract_host("http://www.")


class TestHostFeaturesInvariance:
    """Test d'invariance requis : host_features est identique pour différentes représentations."""

    def test_invariance_across_variations(self):
        """host_features est strictement identique pour http://x.com, https://www.x.com/a?b=1 et https://X.COM:8080/#f."""
        f1 = host_features("http://x.com")
        f2 = host_features("https://www.x.com/a?b=1")
        f3 = host_features("https://X.COM:8080/#f")

        assert f1 == f2, f"Divergence entre f1 et f2: {f1} != {f2}"
        assert f2 == f3, f"Divergence entre f2 et f3: {f2} != {f3}"
        assert f1["HostLength"] == 5  # 'x.com'
        assert f1["HostTLDLength"] == 3  # 'com'
        assert f1["HostNoOfSubDomain"] == 0
        assert f1["HostIsIP"] == 0
        assert f1["HostNoOfDigits"] == 0
        assert f1["HostNoOfLetters"] == 4  # 'xcom'
        assert f1["HostNoOfHyphens"] == 0
        assert f1["HostLongestLabel"] == 3  # 'com'


class TestHostFeaturesUnitVsBatch:
    """Cohérence unitaire / lot et validation du schéma de données."""

    def test_batch_matches_unit_on_1000_urls(self):
        """Vérifie que host_features_batch produit les mêmes résultats que host_features sur 1 000 URLs réelles."""
        df = load_data()
        sample_urls = df["URL"].iloc[:1000].tolist()

        # Calcul unitaire
        unit_records = [host_features(u) for u in sample_urls]
        unit_df = pd.DataFrame.from_records(unit_records)[HOST_FEATURES]

        # Calcul par lot
        batch_df = host_features_batch(sample_urls)

        pd.testing.assert_frame_equal(unit_df, batch_df)

    def test_host_features_structure_and_no_missing(self):
        """Vérifie que les 10 colonnes sont présentes, ordonnées et sans valeurs manquantes."""
        res = host_features("https://www.sub-domain.example.co.uk/test")
        assert list(res.keys()) == HOST_FEATURES
        assert len(res) == 10
        for k, v in res.items():
            assert not np.isnan(v), f"NaN détecté dans {k}"
            assert isinstance(v, (int, float, np.integer, np.floating))


class TestHostSplitConsistency:
    """Vérifie l'isolation par Host et la cohérence des splits entre A, B, B', C et D."""

    def test_no_host_shared_between_train_and_test(self):
        """Aucun Host ne doit être commun entre train et test."""
        from src.split import make_train_test_split
        df_loaded = load_data(verbose=False)
        train_df, test_df = make_train_test_split(df_loaded)

        train_hosts = set(train_df["Host"])
        test_hosts = set(test_df["Host"])
        overlap = train_hosts & test_hosts
        assert len(overlap) == 0, f"{len(overlap)} hôtes partagés entre train et test !"

    def test_row_ids_identical_across_all_experiments(self):
        """Les row_id train et test doivent être strictement identiques pour A, B, B', C et D."""
        from src.split import make_train_test_split
        from src.config import URL_FEATURES_CSV, HOST_FEATURES_CSV

        df_raw = load_data(verbose=False)
        df_c = pd.read_csv(URL_FEATURES_CSV)
        df_d = pd.read_csv(HOST_FEATURES_CSV)

        train_raw, test_raw = make_train_test_split(df_raw)
        train_c, test_c = make_train_test_split(df_c)
        train_d, test_d = make_train_test_split(df_d)

        # Ensembles identiques
        assert set(train_raw["row_id"]) == set(train_c["row_id"]) == set(train_d["row_id"]), (
            "Désalignement des row_id train entre expériences !"
        )
        assert set(test_raw["row_id"]) == set(test_c["row_id"]) == set(test_d["row_id"]), (
            "Désalignement des row_id test entre expériences !"
        )

        # Ordre et valeurs identiques
        assert (train_raw["row_id"].values == train_c["row_id"].values).all()
        assert (train_raw["row_id"].values == train_d["row_id"].values).all()
        assert (test_raw["row_id"].values == test_c["row_id"].values).all()
        assert (test_raw["row_id"].values == test_d["row_id"].values).all()
