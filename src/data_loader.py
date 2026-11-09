"""
data_loader.py - Chargement et validation du dataset PhiUSIIL.

Responsabilités :
- Lecture du CSV brut depuis RAW_CSV défini dans config.py
- Vérification du schéma (colonnes attendues, types)
- Suppression des URL dupliquées (garde la première occurrence)
- Exclusion de FILENAME
- Recodage de la cible : phishing = 1 - label

Usage :
    from src.data_loader import load_data
    df = load_data()
"""

import pandas as pd
import numpy as np

from src.config import (
    RAW_CSV,
    ORIGINAL_TARGET_COL,
    TARGET_COL,
    ALL_NUMERIC_FEATURES,
    EXCLUDE_COLS,
)
from src.host_features import extract_host

# Colonnes obligatoires attendues dans le CSV
REQUIRED_COLS = ["FILENAME", "URL", "Domain", "TLD", "Title", ORIGINAL_TARGET_COL]


def load_data(csv_path=None, verbose: bool = True) -> pd.DataFrame:
    """
    Charge le dataset PhiUSIIL depuis le CSV brut et applique les
    transformations initiales (déduplication, recodage cible).

    Parameters
    ----------
    csv_path : path-like, optional
        Chemin vers le CSV. Si None, utilise RAW_CSV de config.py.
    verbose : bool
        Si True, affiche des informations de chargement.

    Returns
    -------
    pd.DataFrame
        DataFrame nettoyé avec la colonne 'phishing' ajoutée.

    Raises
    ------
    FileNotFoundError
        Si le fichier CSV n'existe pas au chemin spécifié.
    ValueError
        Si des colonnes obligatoires sont absentes ou si les valeurs
        de la cible sont inattendues.
    """
    path = csv_path if csv_path is not None else RAW_CSV

    # ── 1. Vérification de l'existence du fichier ──────────────────────
    if not path.exists() if hasattr(path, "exists") else not __import__("os").path.exists(path):
        raise FileNotFoundError(
            f"Fichier CSV introuvable : {path}\n"
            "Placez le fichier dans data/raw/ ou passez le chemin en argument."
        )

    # ── 2. Chargement ──────────────────────────────────────────────────
    if verbose:
        print(f"[data_loader] Chargement de : {path}")
    df = pd.read_csv(path)

    if verbose:
        print(f"[data_loader] Shape initial : {df.shape}")

    # ── 3. Vérification du schéma ──────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Colonnes obligatoires absentes du CSV : {missing_cols}\n"
            f"Colonnes présentes : {list(df.columns)}"
        )

    missing_numeric = [c for c in ALL_NUMERIC_FEATURES if c not in df.columns]
    if missing_numeric:
        raise ValueError(
            f"Variables numériques manquantes : {missing_numeric}"
        )

    # ── 4. Vérification des valeurs de la cible ────────────────────────
    target_values = set(df[ORIGINAL_TARGET_COL].unique())
    expected_values = {0, 1}
    if not target_values.issubset(expected_values):
        raise ValueError(
            f"Valeurs inattendues dans '{ORIGINAL_TARGET_COL}' : {target_values}. "
            f"Attendu : {expected_values}"
        )

    # ── 5. Suppression des URL dupliquées (garde la première occurrence) ─
    n_before = len(df)
    df = df.drop_duplicates(subset=["URL"], keep="first").reset_index(drop=True)
    n_removed = n_before - len(df)
    if verbose and n_removed > 0:
        print(f"[data_loader] {n_removed} lignes supprimées (URL dupliquées).")

    # ── 6. Recodage de la cible : phishing = 1 - label ─────────────────
    # Après recodage : phishing=1 (classe positive), légitime=0
    df[TARGET_COL] = 1 - df[ORIGINAL_TARGET_COL]

    # ── 7. Ajout de row_id et Host (nom d'hôte normalisé) ─────────────
    df["row_id"] = df.index
    df["Host"] = df["URL"].apply(extract_host)

    # ── 8. Exclusion de FILENAME (identifiant inutile) ──────────────────
    if "FILENAME" in df.columns:
        df = df.drop(columns=["FILENAME"])

    if verbose:
        print(f"[data_loader] Shape final : {df.shape}")
        print(f"[data_loader] Répartition '{TARGET_COL}' :")
        dist = df[TARGET_COL].value_counts()
        print(f"  phishing=1 : {dist.get(1, 0)} ({dist.get(1, 0)/len(df)*100:.2f}%)")
        print(f"  légitime=0 : {dist.get(0, 0)} ({dist.get(0, 0)/len(df)*100:.2f}%)")

    return df


def get_X_y(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Extrait la matrice de features X et le vecteur cible y.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame issu de load_data().
    feature_cols : list
        Liste des colonnes à utiliser comme features.

    Returns
    -------
    tuple (pd.DataFrame, pd.Series)
        (X, y) où y = colonne 'phishing'.

    Raises
    ------
    ValueError
        Si des colonnes de feature_cols sont absentes.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes de features absentes : {missing}")

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    return X, y
