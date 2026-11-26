"""
src/build_features.py — Extraction et persistance des caractéristiques URL_ONLY.

Permet de générer le fichier de caractéristiques dérivées pour l'ensemble du
dataset PhiUSIIL (ou un échantillon) à partir du fichier brut, de façon idempotente.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    RAW_CSV,
    PROCESSED_DIR,
    URL_FEATURES_CSV,
    URL_ONLY_FEATURES,
    TARGET_COL,
    RANDOM_STATE,
)
from src.data_loader import load_data
from src.url_features import compute_url_features_batch

log = logging.getLogger("phishguard.build_features")


def build_features(
    raw_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Extrait les 18 descripteurs URL_ONLY depuis les URLs brutes du dataset dédupliqué.

    Parameters
    ----------
    raw_path : Path, optional
        Chemin vers le CSV brut PhiUSIIL. Si None, utilise config.RAW_CSV.
    output_path : Path, optional
        Chemin de sauvegarde du CSV extrait. Si None, utilise config.URL_FEATURES_CSV.
    verbose : bool
        Affiche la progression du traitement si True.

    Returns
    -------
    pd.DataFrame
        DataFrame avec les colonnes méta (row_id, raw_index, Domain, URL, phishing)
        et les 18 descripteurs URL_ONLY_FEATURES.
    """
    raw_path = Path(raw_path) if raw_path else RAW_CSV
    output_path = Path(output_path) if output_path else URL_FEATURES_CSV

    if not raw_path.exists():
        raise FileNotFoundError(f"Fichier de données brutes introuvable : {raw_path}")

    if verbose:
        print(f"[build_features] Chargement des données brutes depuis {raw_path}...")

    # Chargement et déduplication via le data_loader éprouvé
    df = load_data(raw_path, verbose=verbose)

    if verbose:
        print(f"[build_features] Calcul des descripteurs URL_ONLY pour {len(df)} URLs...")

    # Extraction par lot des descripteurs
    features_df = compute_url_features_batch(df["URL"], verbose=verbose)

    # Fusion des métadonnées essentielles et des descripteurs calculés
    meta_cols = ["row_id", "raw_index", "URL", "Domain", TARGET_COL]
    meta_cols_present = [col for col in meta_cols if col in df.columns]

    final_df = pd.concat([df[meta_cols_present], features_df], axis=1)

    # Sauvegarde du résultat
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    if verbose:
        print(f"[build_features] Caractéristiques sauvegardées avec succès dans {output_path} ({final_df.shape}).")

    return final_df


if __name__ == "__main__":
    import sys
    verbose_flag = "-v" in sys.argv or "--verbose" in sys.argv
    build_features(verbose=verbose_flag)
