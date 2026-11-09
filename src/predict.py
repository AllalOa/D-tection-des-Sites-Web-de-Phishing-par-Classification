"""
predict.py - Fonction de prédiction réutilisable pour la détection de phishing.

Usage programmatique :
    from src.predict import predict_url
    result = predict_url("https://example.com/login")

Usage ligne de commande :
    python -m src.predict "https://example.com/login"

La fonction charge models/best_model.joblib et model_metadata.json.
Aucun accès réseau : seules les features URL_ONLY sont calculées.

ATTENTION : model_score N'EST PAS une "confiance" au sens probabiliste strict.
C'est le score brut du modèle (predict_proba[:,1] ou decision_function).
"""

import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import joblib

from src.config import (
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    MAX_URL_LENGTH,
    URL_ONLY_FEATURES,
    get_experiment_tag,
)
from src.url_features import compute_url_features


class InvalidURLError(ValueError):
    """Exception levée exclusivement pour une entrée URL invalide."""
    pass


def _validate_url(url: str) -> None:
    """
    Valide l'URL en entrée. Lève InvalidURLError si invalide.

    Validations :
        - Type str non vide
        - Longueur <= MAX_URL_LENGTH
        - Hôte (netloc) présent

    Parameters
    ----------
    url : str

    Raises
    ------
    InvalidURLError
        Si l'URL ne passe pas les validations.
    """
    if not isinstance(url, str):
        raise InvalidURLError(f"L'URL doit être une chaîne de caractères, reçu : {type(url)}")

    if not url.strip():
        raise InvalidURLError("L'URL ne peut pas être vide ou contenir uniquement des espaces.")

    if len(url) > MAX_URL_LENGTH:
        raise InvalidURLError(
            f"L'URL dépasse la longueur maximale autorisée ({MAX_URL_LENGTH} caractères). "
            f"Longueur reçue : {len(url)}."
        )

    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            raise InvalidURLError(
                f"L'URL '{url}' ne contient pas d'hôte valide (netloc absent). "
                "Vérifiez le schéma (ex: 'https://')."
            )
    except InvalidURLError:
        raise
    except Exception as e:
        raise InvalidURLError(f"Impossible de parser l'URL '{url}' : {e}")


def _load_model_and_metadata(experiment: str = None) -> tuple:
    """
    Charge le pipeline et les métadonnées depuis models/.
    Si experiment est spécifié (ex: 'C', 'D'), charge models/best_model_<experiment>.joblib et models/model_metadata_<experiment>.json.
    Sinon charge BEST_MODEL_PATH et MODEL_METADATA_PATH (ou models/best_model_C.joblib si présent).

    Returns
    -------
    tuple (pipeline, dict)
        (pipeline sklearn, métadonnées)

    Raises
    ------
    FileNotFoundError
        Si le modèle ou les métadonnées sont introuvables.
    """
    if experiment:
        exp_tag = get_experiment_tag(experiment)
        model_path = MODELS_DIR / f"best_model_{exp_tag}.joblib"
        meta_path = MODELS_DIR / f"model_metadata_{exp_tag}.json"
    elif BEST_MODEL_PATH.exists():
        model_path = BEST_MODEL_PATH
        meta_path = MODEL_METADATA_PATH
    elif (MODELS_DIR / "best_model_C.joblib").exists():
        model_path = MODELS_DIR / "best_model_C.joblib"
        meta_path = MODELS_DIR / "model_metadata_C.json"
    else:
        model_path = BEST_MODEL_PATH
        meta_path = MODEL_METADATA_PATH

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}\n"
            "Aucun modèle entraîné n'est présent dans models/. "
            "Veuillez d'abord entraîner le modèle avec : python -m src.train"
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Métadonnées introuvables : {meta_path}\n"
            "Aucun fichier model_metadata.json présent dans models/. "
            "Veuillez d'abord entraîner le modèle avec : python -m src.train"
        )

    pipeline = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return pipeline, metadata


def predict_url(url: str, experiment: str = None, threshold: str | float = "standard") -> dict:
    """
    Prédit si une URL est du phishing en utilisant le modèle sauvegardé.

    Charge models/best_model.joblib ou best_model_<experiment>.joblib.
    Calcule les features (host_features pour D, compute_url_features sinon)
    depuis la chaîne d'URL uniquement. Aucun accès réseau.

    Politique de seuil :
        - 'standard' : 0.5 pour modèles probabilistes, 0.0 pour decision_function
        - 'high_precision' : seuil documenté optimisé sur TRAIN (ex: 0.9352 pour RF D).
          Lève ValueError pour tout autre modèle.
        - float : seuil explicite fourni par l'utilisateur.

    Parameters
    ----------
    url : str
        L'URL à analyser (max 2048 caractères, hôte requis).
    experiment : str, optional
        Expérience ciblée ('D', 'C', etc.). Par défaut autodétectée.
    threshold : str or float, default='standard'
        Seuil ou politique de décision ('standard', 'high_precision', ou valeur float).

    Returns
    -------
    dict avec les clés :
        - url (str) : l'URL analysée
        - predicted_class (int) : 1 = phishing, 0 = légitime
        - predicted_label (str) : "phishing" ou "legitime"
        - model_score (float) : score brut du modèle (NON une "confiance")
        - threshold_used (float) : seuil effectif appliqué
        - features (dict) : valeurs des features calculées
        - model_name (str) : nom du modèle utilisé
        - experiment (str) : expérience du modèle

    Raises
    ------
    InvalidURLError
        Si l'URL est vide, trop longue, sans hôte, ou non-chaîne.
    FileNotFoundError
        Si le modèle ou les métadonnées sont absents.
    ValueError
        Si la politique de seuil haute précision n'est pas documentée pour ce modèle.
    """
    # ── 1. Validation de l'entrée ──────────────────────────────────────
    _validate_url(url)

    # ── 2. Chargement du modèle ────────────────────────────────────────
    pipeline, metadata = _load_model_and_metadata(experiment=experiment)

    # La liste des variables et leur ordre viennent des métadonnées
    feature_cols = metadata.get("feature_cols", None)
    if not feature_cols:
        raise ValueError("La liste 'feature_cols' est absente des métadonnées.")

    exp = metadata.get("experiment", experiment if experiment else "unknown")

    # ── 3. Calcul des features appropriées ─────────────────────────────
    if exp == "D":
        from src.host_features import host_features
        features = host_features(url)
    else:
        features = compute_url_features(url)

    # Construire le DataFrame avec les colonnes dans l'ordre EXACT de l'entraînement
    for col in feature_cols:
        if col not in features:
            features[col] = 0

    df_input = pd.DataFrame([[features[col] for col in feature_cols]], columns=feature_cols)

    # ── 4. Prédiction et score brut ───────────────────────────────────
    predicted_class_default = int(pipeline.predict(df_input)[0])

    if hasattr(pipeline, "predict_proba"):
        try:
            model_score = float(pipeline.predict_proba(df_input)[0][1])
        except Exception:
            model_score = float(pipeline.decision_function(df_input)[0])
    elif hasattr(pipeline, "decision_function"):
        model_score = float(pipeline.decision_function(df_input)[0])
    else:
        model_score = float(predicted_class_default)

    # ── 5. Résolution de la politique de seuil ─────────────────────────
    model_name = metadata.get("model_name", "RandomForest")
    if threshold == "standard":
        eff_threshold = 0.0 if not hasattr(pipeline, "predict_proba") and hasattr(pipeline, "decision_function") else 0.5
        predicted_class = predicted_class_default
    elif threshold == "high_precision":
        if exp == "D" and model_name == "RandomForest":
            eff_threshold = 0.9352
            predicted_class = 1 if model_score >= eff_threshold else 0
        else:
            raise ValueError(
                f"Aucun seuil haute précision n'est documenté pour {exp}/{model_name}. "
                "Interdiction d'appliquer silencieusement le seuil 0.9352 propre à D/RandomForest."
            )
    elif isinstance(threshold, (int, float)):
        eff_threshold = float(threshold)
        predicted_class = 1 if model_score >= eff_threshold else 0
    else:
        raise ValueError(f"Politique de seuil inconnue ou invalide : {threshold}")

    return {
        "url": url,
        "predicted_class": predicted_class,
        "predicted_label": "phishing" if predicted_class == 1 else "legitime",
        "model_score": model_score,
        "threshold_used": eff_threshold,
        # AVERTISSEMENT : model_score n'est PAS une probabilité calibrée.
        # C'est le score brut de predict_proba[:,1] ou decision_function.
        "features": features,
        "model_name": model_name,
        "experiment": exp,
    }


def main():
    """Point d'entrée ligne de commande."""
    if len(sys.argv) < 2:
        print("Usage : python -m src.predict <URL>")
        print("Exemple : python -m src.predict 'https://example.com'")
        sys.exit(1)

    url = sys.argv[1]

    try:
        result = predict_url(url)
        sep_line = "-" * 50
        print(f"\n{sep_line}")
        print(f"URL analysee    : {result['url']}")
        print(f"Classe predite  : {result['predicted_class']} ({result['predicted_label'].upper()})")
        print(f"Score du modele : {result['model_score']:.6f}  [NON une probabilite calibree]")
        print(f"Modele          : {result['model_name']} (experience {result['experiment']})")
        print(f"\nFeatures calculees :")
        for feat, val in result["features"].items():
            print(f"  {feat:35s}: {val}")
        print(f"{sep_line}\n")
    except InvalidURLError as e:
        print(f"[ERREUR] Entree invalide : {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"[ERREUR] Fichier manquant : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
