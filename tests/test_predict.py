"""
test_predict.py - Tests de src/predict.py.

Vérifie :
- Validation des entrées (vide, trop longue, sans hôte, type invalide)
- Aucun appel réseau (monkeypatch du socket)
- Structure du résultat retourné
- Gestion des fichiers manquants (modèle, métadonnées)

NOTE : les tests nécessitant le modèle réel (modèle entraîné) utilisent
un monkeypatch qui simule joblib.load et json.load.
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.predict import _validate_url, predict_url, InvalidURLError
from src.config import MAX_URL_LENGTH, BEST_MODEL_PATH, MODEL_METADATA_PATH


class TestValidateUrl:
    """Tests de la fonction _validate_url."""

    def test_valid_url_no_exception(self):
        """Une URL valide ne doit pas lever d'exception."""
        _validate_url("https://www.example.com/login")  # Ne doit pas lever

    def test_empty_string_raises(self):
        """Une chaîne vide doit lever ValueError."""
        with pytest.raises(ValueError, match="vide"):
            _validate_url("")

    def test_whitespace_only_raises(self):
        """Une chaîne d'espaces doit lever ValueError."""
        with pytest.raises(ValueError, match="vide"):
            _validate_url("   ")

    def test_too_long_url_raises(self):
        """Une URL dépassant MAX_URL_LENGTH doit lever ValueError."""
        long_url = "https://example.com/" + "a" * (MAX_URL_LENGTH + 10)
        with pytest.raises(ValueError, match="longueur maximale"):
            _validate_url(long_url)

    def test_url_exactly_max_length_valid(self):
        """Une URL exactement à MAX_URL_LENGTH ne doit pas lever d'exception."""
        # Construire une URL valide de longueur exacte MAX_URL_LENGTH
        prefix = "https://x.co/"
        padding = "a" * (MAX_URL_LENGTH - len(prefix))
        url = prefix + padding
        assert len(url) == MAX_URL_LENGTH
        _validate_url(url)  # Ne doit pas lever

    def test_no_host_raises(self):
        """Une URL sans hôte (netloc vide) doit lever ValueError."""
        with pytest.raises(ValueError, match="hôte"):
            _validate_url("https://")

    def test_non_string_raises_invalid_url_error(self):
        """Un entier doit lever InvalidURLError (héritant de ValueError)."""
        with pytest.raises(InvalidURLError, match="chaîne"):
            _validate_url(42)

    def test_none_raises_invalid_url_error(self):
        """None doit lever InvalidURLError."""
        with pytest.raises(InvalidURLError, match="chaîne"):
            _validate_url(None)

    def test_list_raises_invalid_url_error(self):
        """Une liste doit lever InvalidURLError."""
        with pytest.raises(InvalidURLError, match="chaîne"):
            _validate_url(["https://example.com"])


class TestPredictUrl:
    """Tests de predict_url avec mock du modèle."""

    @pytest.fixture
    def mock_pipeline(self):
        """Crée un mock pipeline sklearn avec predict et predict_proba."""
        pipeline = MagicMock()
        pipeline.predict.return_value = np.array([1])  # phishing
        pipeline.predict_proba.return_value = np.array([[0.1, 0.9]])
        pipeline.decision_function.return_value = np.array([0.8])
        return pipeline

    @pytest.fixture
    def mock_metadata(self):
        """Métadonnées simulées du modèle."""
        return {
            "experiment": "C",
            "model_name": "RandomForest",
            "feature_cols": [
                "URLLength", "DomainLength", "IsDomainIP", "TLDLength",
                "NoOfSubDomain", "HasObfuscation", "NoOfObfuscatedChar",
                "ObfuscationRatio", "NoOfLettersInURL", "LetterRatioInURL",
                "NoOfDegitsInURL", "DegitRatioInURL", "NoOfEqualsInURL",
                "NoOfQMarkInURL", "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL",
                "SpacialCharRatioInURL", "IsHTTPS"
            ],
            "random_state": 42,
        }

    def test_predict_returns_dict(self, mock_pipeline, mock_metadata):
        """predict_url doit retourner un dict."""
        with patch("src.predict.BEST_MODEL_PATH", Path("fake/model.joblib")):
            with patch("src.predict.MODEL_METADATA_PATH", Path("fake/meta.json")):
                with patch("src.predict.BEST_MODEL_PATH") as mock_path:
                    mock_path.exists.return_value = True
                    with patch("src.predict.MODEL_METADATA_PATH") as mock_meta_path:
                        mock_meta_path.exists.return_value = True
                        with patch("joblib.load", return_value=mock_pipeline):
                            with patch("builtins.open", MagicMock()):
                                with patch("json.load", return_value=mock_metadata):
                                    result = predict_url("https://www.example.com")
        assert isinstance(result, dict)

    def test_predict_result_keys(self, mock_pipeline, mock_metadata):
        """predict_url doit contenir les clés attendues."""
        with patch("src.predict.BEST_MODEL_PATH") as mock_path:
            mock_path.exists.return_value = True
            with patch("src.predict.MODEL_METADATA_PATH") as mock_meta_path:
                mock_meta_path.exists.return_value = True
                with patch("joblib.load", return_value=mock_pipeline):
                    with patch("builtins.open", MagicMock()):
                        with patch("json.load", return_value=mock_metadata):
                            result = predict_url("https://www.example.com")
        required_keys = ["url", "predicted_class", "predicted_label", "model_score", "features"]
        for key in required_keys:
            assert key in result, f"Clé manquante : '{key}'"

    def test_predict_class_is_int(self, mock_pipeline, mock_metadata):
        """predicted_class doit être un int."""
        with patch("src.predict.BEST_MODEL_PATH") as mock_path:
            mock_path.exists.return_value = True
            with patch("src.predict.MODEL_METADATA_PATH") as mock_meta_path:
                mock_meta_path.exists.return_value = True
                with patch("joblib.load", return_value=mock_pipeline):
                    with patch("builtins.open", MagicMock()):
                        with patch("json.load", return_value=mock_metadata):
                            result = predict_url("https://www.example.com")
        assert isinstance(result["predicted_class"], int)

    def test_predict_label_values(self, mock_pipeline, mock_metadata):
        """predicted_label doit être 'phishing' ou 'legitime'."""
        with patch("src.predict.BEST_MODEL_PATH") as mock_path:
            mock_path.exists.return_value = True
            with patch("src.predict.MODEL_METADATA_PATH") as mock_meta_path:
                mock_meta_path.exists.return_value = True
                with patch("joblib.load", return_value=mock_pipeline):
                    with patch("builtins.open", MagicMock()):
                        with patch("json.load", return_value=mock_metadata):
                            result = predict_url("https://www.example.com")
        assert result["predicted_label"] in ("phishing", "legitime")

    def test_missing_model_raises(self):
        """FileNotFoundError si best_model.joblib est absent."""
        with patch.object(Path, "exists", return_value=False):
            with patch("src.predict.MODEL_METADATA_PATH") as mock_meta_path:
                mock_meta_path.exists.return_value = True
                with pytest.raises(FileNotFoundError, match="Modèle introuvable"):
                    predict_url("https://www.example.com")

    def test_missing_metadata_raises(self):
        """FileNotFoundError si model_metadata.json est absent."""
        with patch("src.predict.BEST_MODEL_PATH") as mock_path:
            mock_path.exists.return_value = True
            with patch("src.predict.MODEL_METADATA_PATH") as mock_meta_path:
                mock_meta_path.exists.return_value = False
                with pytest.raises(FileNotFoundError, match="Métadonnées introuvables"):
                    predict_url("https://www.example.com")

    def test_invalid_url_raises_before_network(self):
        """Les erreurs de validation doivent se produire avant tout accès réseau."""
        # Monkeypatch socket pour garantir qu'aucun appel réseau n'a lieu
        import socket
        original_getaddrinfo = socket.getaddrinfo

        calls_to_network = []

        def mock_getaddrinfo(*args, **kwargs):
            calls_to_network.append(args)
            return original_getaddrinfo(*args, **kwargs)

        with patch("socket.getaddrinfo", side_effect=mock_getaddrinfo):
            with pytest.raises((ValueError, TypeError)):
                predict_url("")  # URL invalide

        # Aucun appel réseau ne doit avoir été effectué
        assert len(calls_to_network) == 0, (
            "Des appels réseau ont été effectués lors de la validation de l'URL !"
        )

    def test_no_network_call_on_valid_url(self, mock_pipeline, mock_metadata):
        """predict_url ne doit effectuer aucun appel réseau, même pour une URL valide."""
        import socket

        network_calls = []

        def mock_connect(*args, **kwargs):
            network_calls.append(args)
            raise ConnectionRefusedError("Appel réseau interdit dans les tests !")

        with patch("socket.socket.connect", side_effect=mock_connect):
            with patch("src.predict.BEST_MODEL_PATH") as mock_path:
                mock_path.exists.return_value = True
                with patch("src.predict.MODEL_METADATA_PATH") as mock_meta_path:
                    mock_meta_path.exists.return_value = True
                    with patch("joblib.load", return_value=mock_pipeline):
                        with patch("builtins.open", MagicMock()):
                            with patch("json.load", return_value=mock_metadata):
                                # Cette ligne ne doit pas déclencher socket.connect
                                result = predict_url("https://www.example.com")

        assert len(network_calls) == 0, (
            "Des appels réseau (socket.connect) ont été détectés dans predict_url !"
        )

    def test_internal_error_not_converted_to_invalid_url(self, mock_metadata):
        """Une vraie erreur interne (ex: ValueError dans le pipeline) ne doit pas être interceptée comme InvalidURLError."""
        mock_pipeline = MagicMock()
        mock_pipeline.predict.side_effect = ValueError("Erreur interne scikit-learn non liée à l'URL")

        with patch("src.predict.BEST_MODEL_PATH") as mock_path:
            mock_path.exists.return_value = True
            with patch("src.predict.MODEL_METADATA_PATH") as mock_meta_path:
                mock_meta_path.exists.return_value = True
                with patch("joblib.load", return_value=mock_pipeline):
                    with patch("builtins.open", MagicMock()):
                        with patch("json.load", return_value=mock_metadata):
                            with pytest.raises(ValueError) as excinfo:
                                predict_url("https://www.example.com")
                            # Doit être le ValueError interne et pas InvalidURLError
                            assert not isinstance(excinfo.value, InvalidURLError)
                            assert "Erreur interne scikit-learn" in str(excinfo.value)

    def test_feature_cols_and_order_from_metadata(self, mock_pipeline):
        """Vérifie que les variables passées au pipeline proviennent des métadonnées et respectent leur ordre exact."""
        custom_cols = ["IsHTTPS", "TLDLength", "DomainLength", "NoOfSubDomain", "HasObfuscation"]
        custom_metadata = {
            "experiment": "C",
            "model_name": "TestModel",
            "feature_cols": custom_cols,
        }
        with patch("src.predict.BEST_MODEL_PATH") as mock_path:
            mock_path.exists.return_value = True
            with patch("src.predict.MODEL_METADATA_PATH") as mock_meta_path:
                mock_meta_path.exists.return_value = True
                with patch("joblib.load", return_value=mock_pipeline):
                    with patch("builtins.open", MagicMock()):
                        with patch("json.load", return_value=custom_metadata):
                            predict_url("https://www.example.com")
                            args, _ = mock_pipeline.predict.call_args
                            df_passed = args[0]
                            assert list(df_passed.columns) == custom_cols

    def test_missing_model_clear_error_no_crash(self):
        """Sans modèle présent, _load_model_and_metadata lève FileNotFoundError avec un message explicite."""
        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Modèle introuvable"):
                predict_url("https://www.example.com")
