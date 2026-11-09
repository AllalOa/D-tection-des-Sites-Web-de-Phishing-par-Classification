"""
test_url_features.py - Tests de src/url_features.py.

Vérifie :
- Calcul correct des 18 features URL_ONLY sur des URLs connues
- Gestion des cas limites (URLs vides, IP, sans schéma)
- Présence de toutes les colonnes attendues dans compute_url_features_batch()
"""

import pytest
import numpy as np
import pandas as pd

from src.url_features import compute_url_features, compute_url_features_batch
from src.config import URL_ONLY_FEATURES


class TestComputeUrlFeatures:
    """Tests unitaires sur des URLs connues."""

    def test_returns_all_18_features(self):
        """compute_url_features doit retourner exactement les 18 features."""
        result = compute_url_features("https://www.example.com/path?q=1")
        assert set(result.keys()) == set(URL_ONLY_FEATURES), (
            f"Features manquantes ou en trop : "
            f"{set(URL_ONLY_FEATURES).symmetric_difference(set(result.keys()))}"
        )

    def test_ishttps_true_for_https(self):
        """IsHTTPS doit être 1 pour une URL https."""
        result = compute_url_features("https://www.example.com")
        assert result["IsHTTPS"] == 1

    def test_ishttps_false_for_http(self):
        """IsHTTPS doit être 0 pour une URL http."""
        result = compute_url_features("http://www.example.com")
        assert result["IsHTTPS"] == 0

    def test_url_length(self):
        """URLLength doit être la longueur de la chaîne URL."""
        url = "https://example.com/abc"
        result = compute_url_features(url)
        assert result["URLLength"] == len(url)

    def test_is_domain_ip_true(self):
        """IsDomainIP doit être 1 pour une URL avec adresse IP."""
        result = compute_url_features("http://192.168.1.1/login")
        assert result["IsDomainIP"] == 1

    def test_is_domain_ip_false(self):
        """IsDomainIP doit être 0 pour un domaine normal."""
        result = compute_url_features("https://www.google.com")
        assert result["IsDomainIP"] == 0

    def test_tld_length_correct(self):
        """TLDLength doit correspondre à la longueur du TLD."""
        result = compute_url_features("https://www.example.com")
        assert result["TLDLength"] == len("com")

    def test_tld_length_org(self):
        """TLDLength pour .org."""
        result = compute_url_features("https://www.example.org")
        assert result["TLDLength"] == len("org")

    def test_no_of_subdomain(self):
        """NoOfSubDomain : www.sub.example.com a 2 sous-domaines."""
        result = compute_url_features("https://www.sub.example.com")
        assert result["NoOfSubDomain"] == 2

    def test_no_subdomain(self):
        """NoOfSubDomain : example.com a 0 sous-domaines."""
        result = compute_url_features("https://example.com")
        assert result["NoOfSubDomain"] == 0

    def test_has_obfuscation_percent(self):
        """HasObfuscation doit être 1 si l'URL contient %xx."""
        result = compute_url_features("http://example.com/%20path%3D")
        assert result["HasObfuscation"] == 1
        assert result["NoOfObfuscatedChar"] == 2

    def test_no_obfuscation(self):
        """HasObfuscation doit être 0 pour une URL propre."""
        result = compute_url_features("https://www.example.com/normal")
        assert result["HasObfuscation"] == 0
        assert result["NoOfObfuscatedChar"] == 0

    def test_no_of_equals(self):
        """NoOfEqualsInURL : compte les '=' dans l'URL."""
        result = compute_url_features("http://example.com/p?a=1&b=2")
        assert result["NoOfEqualsInURL"] == 2

    def test_no_of_ampersand(self):
        """NoOfAmpersandInURL : compte les '&' dans l'URL."""
        result = compute_url_features("http://example.com/p?a=1&b=2&c=3")
        assert result["NoOfAmpersandInURL"] == 2

    def test_no_of_qmark(self):
        """NoOfQMarkInURL : compte les '?' dans l'URL."""
        result = compute_url_features("http://example.com/p?a=1?b=2")
        assert result["NoOfQMarkInURL"] == 2

    def test_letter_ratio_in_range(self):
        """LetterRatioInURL doit être entre 0 et 1."""
        result = compute_url_features("https://www.example.com/path123")
        assert 0.0 <= result["LetterRatioInURL"] <= 1.0

    def test_digit_ratio_in_range(self):
        """DegitRatioInURL doit être entre 0 et 1."""
        result = compute_url_features("https://www.example.com/path123")
        assert 0.0 <= result["DegitRatioInURL"] <= 1.0

    def test_obfuscation_ratio_in_range(self):
        """ObfuscationRatio doit être entre 0 et 1."""
        result = compute_url_features("http://example.com/%20path")
        assert 0.0 <= result["ObfuscationRatio"] <= 1.0

    def test_spacial_char_ratio_in_range(self):
        """SpacialCharRatioInURL doit être entre 0 et 1."""
        result = compute_url_features("http://example.com/path@#!")
        assert 0.0 <= result["SpacialCharRatioInURL"] <= 1.0

    def test_all_values_numeric(self):
        """Toutes les valeurs doivent être numériques (int ou float)."""
        result = compute_url_features("https://www.phishing-site.net/login?id=1&pass=2")
        for key, val in result.items():
            assert isinstance(val, (int, float)), (
                f"La valeur de '{key}' n'est pas numérique : {type(val)}"
            )

    def test_empty_url_does_not_crash(self):
        """Une URL vide ne doit pas lever d'exception."""
        result = compute_url_features("")
        assert result["URLLength"] == 0
        assert result["IsHTTPS"] == 0

    def test_non_string_input_converted(self):
        """Une entrée non-string (ex: None, int) doit être convertie."""
        result = compute_url_features(None)
        assert isinstance(result["URLLength"], int)


class TestComputeUrlFeaturesBatch:
    """Tests de la fonction batch."""

    def test_returns_dataframe(self):
        """compute_url_features_batch doit retourner un DataFrame."""
        urls = ["https://a.com", "http://b.org"]
        result = compute_url_features_batch(urls)
        assert isinstance(result, pd.DataFrame)

    def test_correct_shape(self):
        """Le DataFrame doit avoir n lignes et 18 colonnes."""
        urls = ["https://a.com", "http://b.org", "ftp://c.net"]
        result = compute_url_features_batch(urls)
        assert result.shape == (3, 18)

    def test_all_columns_present(self):
        """Toutes les colonnes URL_ONLY_FEATURES doivent être présentes."""
        urls = ["https://www.example.com"]
        result = compute_url_features_batch(urls)
        for col in URL_ONLY_FEATURES:
            assert col in result.columns, f"Colonne manquante : {col}"

    def test_column_order(self):
        """Les colonnes doivent être dans l'ordre de URL_ONLY_FEATURES."""
        urls = ["https://www.example.com"]
        result = compute_url_features_batch(urls)
        assert list(result.columns) == URL_ONLY_FEATURES

    def test_series_input(self):
        """compute_url_features_batch doit accepter une pd.Series."""
        urls = pd.Series(["https://a.com", "http://b.org"])
        result = compute_url_features_batch(urls)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
