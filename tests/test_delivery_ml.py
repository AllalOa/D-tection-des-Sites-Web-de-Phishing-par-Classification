"""
tests/test_delivery_ml.py - Tests de validation du paquet de livraison ML (Phases 8b / 8b-bis).

Vérifie :
1. L'existence et la chargeabilité des 5 modèles (.joblib).
2. La liste exacte des caractéristiques de B' et l'absence de TLD.
3. La configuration réelle des transformateurs TLD pour A et B (top_n=20, 21 colonnes OHE).
4. La cohérence des métadonnées (schémas, hyperparamètres, random_state).
5. L'inférence du modèle C et la correspondance exacte des clés avec le guide d'intégration.
6. L'absence absolue d'appels réseau lors de l'inférence (isolation socket).
7. La conformité des métriques documentées (F1 B' = 0.999420).
8. La conformité intégrale du fichier MANIFEST_SHA256.csv.
"""

import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.config import (
    ALL_NUMERIC_FEATURES,
    B_FEATURES,
    B_PRIME_FEATURES,
    HOST_FEATURES,
    PAGE_FEATURES,
    ROOT_DIR,
    URL_ONLY_FEATURES,
)
from src.predict import InvalidURLError, predict_url

MODELS_DIR = ROOT_DIR / "models"
REPORTS_TABLES_DIR = ROOT_DIR / "reports" / "tables"
DELIVERY_DIR = ROOT_DIR / "delivery_ml"
SCRIPTS_DIR = ROOT_DIR / "scripts"


def test_models_exist_and_loadable():
    """Vérifie que les 5 modèles .joblib existent et se chargent sans erreur."""
    tags = ["A", "B", "B_prime", "C", "D"]
    for tag in tags:
        model_path = MODELS_DIR / f"best_model_{tag}.joblib"
        assert model_path.exists(), f"Modèle manquant: {model_path}"
        model = joblib.load(model_path)
        assert hasattr(model, "predict"), f"L'objet chargé {tag} n'a pas de méthode predict"


def test_b_prime_exact_features_and_no_tld():
    """Vérifie que B' utilise exactement 38 caractéristiques numériques sans TLD."""
    model_b_prime = joblib.load(MODELS_DIR / "best_model_B_prime.joblib")
    preproc = model_b_prime.named_steps["preprocessor"]

    feature_names = list(preproc.feature_names_in_)
    assert len(feature_names) == 38, f"Attendu 38 caractéristiques pour B', obtenu {len(feature_names)}"
    assert feature_names == B_PRIME_FEATURES, "L'ordre des colonnes de B' ne correspond pas à B_PRIME_FEATURES"
    assert "TLD" not in feature_names, "TLD ne doit pas être présent dans B'"

    # Vérification des métadonnées
    with open(MODELS_DIR / "model_metadata_B_prime.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["use_tld"] is False
    assert len(meta["feature_cols"]) == 38
    assert meta["feature_cols"] == B_PRIME_FEATURES


def test_tld_transformers_a_and_b():
    """Vérifie la configuration réelle des transformateurs TLD (top_n=20, 21 catégories)."""
    for tag in ["A", "B"]:
        model = joblib.load(MODELS_DIR / f"best_model_{tag}.joblib")
        preproc = model.named_steps["preprocessor"]

        tld_step = [trans for name, trans, cols in preproc.transformers_ if "tld" in name.lower()][0]
        grouper = tld_step.named_steps["grouper"]
        encoder = tld_step.named_steps["encoder"]

        assert grouper.top_n == 20, f"Exp {tag}: attendu top_n=20, obtenu {grouper.top_n}"
        assert grouper.other_label == "autre"
        assert len(grouper.top_tlds_) == 20
        assert len(encoder.categories_[0]) == 21, f"Exp {tag}: attendu 21 catégories OHE, obtenu {len(encoder.categories_[0])}"

    # Vérification du nombre total de caractéristiques en entrée du classifier
    model_a = joblib.load(MODELS_DIR / "best_model_A.joblib")
    assert model_a.named_steps["classifier"].n_features_in_ == 71, "A doit avoir 71 features (50 num + 21 TLD)"

    model_b = joblib.load(MODELS_DIR / "best_model_B.joblib")
    assert model_b.named_steps["classifier"].n_features_in_ == 65, "B doit avoir 65 features (44 num + 21 TLD)"


def test_metadata_consistency():
    """Vérifie la cohérence des 5 fichiers de métadonnées JSON."""
    tags = ["A", "B", "B_prime", "C", "D"]
    for tag in tags:
        meta_path = MODELS_DIR / f"model_metadata_{tag}.json"
        assert meta_path.exists(), f"Métadonnées manquantes: {meta_path}"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["model_name"] == "RandomForest"
        assert meta["random_state"] == 42


def test_predict_url_fields_and_mapping():
    """Vérifie que predict_url renvoie exactement les clés et types documentés."""
    url = "https://www.southbankmosaics.com"
    res = predict_url(url, threshold=0.5)

    expected_keys = {
        "url",
        "predicted_class",
        "predicted_label",
        "model_score",
        "threshold_used",
        "features",
        "model_name",
        "experiment",
    }
    assert set(res.keys()) == expected_keys, f"Clés inattendues : {set(res.keys()) ^ expected_keys}"

    assert res["predicted_class"] == 0
    assert res["predicted_label"] == "legitime"
    assert isinstance(res["model_score"], float)
    assert 0.0 <= res["model_score"] <= 1.0
    assert res["threshold_used"] == 0.5
    assert len(res["features"]) == 18
    assert res["model_name"] == "RandomForest"
    assert res["experiment"] == "C"


def test_predict_url_no_network_calls(monkeypatch):
    """Vérifie qu'aucun appel réseau (socket.connect) n'est effectué durant l'inférence."""
    network_calls = []

    def block_connect(*args, **kwargs):
        network_calls.append(args)
        raise RuntimeError("Tentative d'accès réseau non autorisée durant l'inférence!")

    monkeypatch.setattr(socket.socket, "connect", block_connect)

    url = "https://www.southbankmosaics.com"
    res = predict_url(url)
    assert res["predicted_class"] == 0
    assert len(network_calls) == 0


def test_predict_url_invalid_inputs():
    """Vérifie la gestion d'erreurs sur entrées invalides."""
    with pytest.raises((InvalidURLError, ValueError)):
        predict_url("")

    with pytest.raises((InvalidURLError, ValueError)):
        predict_url("not_a_valid_url")

    with pytest.raises((InvalidURLError, ValueError)):
        predict_url(12345)


def test_documented_metrics_consistency():
    """Vérifie la stricte conformité des métriques F1 certifiées dans les CSV."""
    df_bp = pd.read_csv(REPORTS_TABLES_DIR / "test_metrics_B_prime.csv")
    rf_bp = df_bp[df_bp["model"] == "RandomForest"].iloc[0]
    # F1 doit être ~ 0.999420, et NON 0.9998
    assert round(rf_bp["f1"], 5) == 0.99942, f"F1 B' inattendu: {rf_bp['f1']}"
    assert rf_bp["FP"] == 13
    assert rf_bp["FN"] == 10

    df_c = pd.read_csv(REPORTS_TABLES_DIR / "test_metrics_C.csv")
    rf_c = df_c[df_c["model"] == "RandomForest"].iloc[0]
    assert round(rf_c["f1"], 4) == 0.9936
    assert rf_c["FP"] == 62
    assert rf_c["FN"] == 189


def test_demo_predict_script_cli():
    """Vérifie l'exécution du script CLI demo_predict.py en mode texte et JSON."""
    script_path = SCRIPTS_DIR / "demo_predict.py"
    assert script_path.exists()

    # Test mode texte
    cmd_text = [sys.executable, str(script_path), "https://www.southbankmosaics.com"]
    proc = subprocess.run(cmd_text, capture_output=True, text=True)
    assert proc.returncode == 0, f"Erreur script demo text : {proc.stderr}"
    assert "LEGITIME" in proc.stdout

    # Test mode JSON
    cmd_json = [sys.executable, str(script_path), "https://www.southbankmosaics.com", "--json"]
    proc_json = subprocess.run(cmd_json, capture_output=True, text=True)
    assert proc_json.returncode == 0, f"Erreur script demo json : {proc_json.stderr}"
    data = json.loads(proc_json.stdout)
    assert data["verdict"] == "LEGITIME"
    assert data["is_phishing"] is False
    assert "features" in data
    assert len(data["features"]) == 18


def test_manifest_sha256_integrity():
    """Vérifie que tous les fichiers du manifest existent et correspondent à leur hash."""
    manifest_path = DELIVERY_DIR / "MANIFEST_SHA256.csv"
    assert manifest_path.exists(), "delivery_ml/MANIFEST_SHA256.csv manquant"

    df_manifest = pd.read_csv(manifest_path)
    col = "relative_path" if "relative_path" in df_manifest.columns else "file_path"
    assert "sha256" in df_manifest.columns

    for _, row in df_manifest.iterrows():
        rel_path = row[col]
        expected_sha = row["sha256"].strip()
        full_path = ROOT_DIR / rel_path
        assert full_path.exists(), f"Fichier du manifest inexistant: {rel_path}"

        hasher = hashlib.sha256()
        with open(full_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        actual_sha = hasher.hexdigest()
        assert actual_sha == expected_sha, (
            f"Hash divergent pour {rel_path} : attendu {expected_sha}, obtenu {actual_sha}"
        )
