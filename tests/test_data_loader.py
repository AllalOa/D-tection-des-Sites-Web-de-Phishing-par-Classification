"""
test_data_loader.py - Tests de src/data_loader.py.

Vérifie :
- Absence de doublons d'URL après chargement
- Recodage correct de la cible (phishing = 1 - label)
- Absence de la colonne FILENAME dans X
- Gestion des erreurs (fichier manquant, colonnes manquantes)
- Shape et types cohérents
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import tempfile
import os

from src.data_loader import load_data, get_X_y
from src.config import (
    RAW_CSV,
    TARGET_COL,
    ORIGINAL_TARGET_COL,
    ALL_NUMERIC_FEATURES,
)


@pytest.fixture(scope="module")
def loaded_df():
    """
    Charge le dataset réel une seule fois pour tous les tests.
    Skippé si le fichier CSV n'existe pas.
    """
    if not RAW_CSV.exists():
        pytest.skip(f"Fichier CSV introuvable : {RAW_CSV}. Tests data_loader ignorés.")
    return load_data(verbose=False)


class TestDataLoaderRealData:
    """Tests sur le dataset réel (si disponible)."""

    def test_no_url_duplicates(self, loaded_df):
        """Aucune URL dupliquée après chargement."""
        n_dups = loaded_df["URL"].duplicated().sum()
        assert n_dups == 0, f"{n_dups} URLs dupliquées trouvées après load_data()."

    def test_target_recoding(self, loaded_df):
        """
        phishing = 1 - label.
        Vérifie que pour chaque ligne, phishing + label_original = 1.
        """
        # On doit pouvoir recalculer : phishing = 1 - label_original
        # Mais attention : 'label' est conservé dans le df issu de load_data
        # uniquement si on ne l'a pas supprimé (il n'est pas dans EXCLUDE_COLS par défaut).
        if ORIGINAL_TARGET_COL in loaded_df.columns:
            check = (1 - loaded_df[ORIGINAL_TARGET_COL]) == loaded_df[TARGET_COL]
            assert check.all(), "Recodage phishing = 1 - label incorrect !"
        else:
            # Vérifie au moins que les valeurs sont 0 ou 1
            unique_vals = set(loaded_df[TARGET_COL].unique())
            assert unique_vals.issubset({0, 1}), f"Valeurs inattendues dans {TARGET_COL}: {unique_vals}"

    def test_filename_absent(self, loaded_df):
        """La colonne FILENAME ne doit pas être présente dans le DataFrame chargé."""
        assert "FILENAME" not in loaded_df.columns, "FILENAME est encore présent après load_data()."

    def test_target_col_present(self, loaded_df):
        """La colonne 'phishing' doit être présente."""
        assert TARGET_COL in loaded_df.columns, f"Colonne '{TARGET_COL}' absente."

    def test_target_values(self, loaded_df):
        """Les valeurs de 'phishing' doivent être 0 ou 1."""
        vals = set(loaded_df[TARGET_COL].unique())
        assert vals.issubset({0, 1}), f"Valeurs inattendues : {vals}"

    def test_all_numeric_features_present(self, loaded_df):
        """Toutes les 50 variables numériques doivent être présentes."""
        missing = [c for c in ALL_NUMERIC_FEATURES if c not in loaded_df.columns]
        assert len(missing) == 0, f"Variables numériques manquantes : {missing}"

    def test_no_missing_values(self, loaded_df):
        """Aucune valeur manquante dans les columns numériques."""
        missing_counts = loaded_df[ALL_NUMERIC_FEATURES].isnull().sum()
        cols_with_na = missing_counts[missing_counts > 0]
        assert len(cols_with_na) == 0, f"Valeurs manquantes trouvées : {cols_with_na.to_dict()}"

    def test_shape_plausible(self, loaded_df):
        """La shape doit être proche de (235 795 - 425, 56+)."""
        n_rows = len(loaded_df)
        # Après suppression des 425 doublons URL, on attend ~235 370 lignes
        assert 234_000 < n_rows < 236_000, (
            f"Nombre de lignes inattendu : {n_rows}. Attendu ~235 370."
        )

    def test_phishing_prevalence(self, loaded_df):
        """Le taux de phishing (classe positive) devrait être ~42.8%."""
        rate = loaded_df[TARGET_COL].mean()
        # Après recodage, phishing=1 correspond à label=0 (~42.8%)
        assert 0.40 < rate < 0.46, (
            f"Taux de phishing inattendu : {rate:.4f}. Attendu ~0.428."
        )


class TestDataLoaderErrors:
    """Tests des cas d'erreur."""

    def test_missing_file_raises(self):
        """FileNotFoundError si le CSV n'existe pas."""
        fake_path = Path("/chemin/inexistant/fake.csv")
        with pytest.raises(FileNotFoundError, match="Fichier CSV introuvable"):
            load_data(csv_path=fake_path, verbose=False)

    def test_missing_columns_raises(self, tmp_path):
        """ValueError si des colonnes obligatoires sont absentes."""
        # Créer un CSV minimal sans les colonnes requises
        df_bad = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        bad_csv = tmp_path / "bad.csv"
        df_bad.to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="Colonnes obligatoires absentes"):
            load_data(csv_path=bad_csv, verbose=False)

    def test_bad_target_values_raises(self, tmp_path):
        """ValueError si la cible contient des valeurs non binaires."""
        # Créer un CSV avec des valeurs de cible invalides
        df_bad = pd.DataFrame({
            "FILENAME": ["f1"],
            "URL": ["http://x.com"],
            "Domain": ["x.com"],
            "TLD": ["com"],
            "Title": ["Title"],
            "label": [99],  # Valeur invalide
        })
        # Ajouter les colonnes numériques avec des valeurs factices
        from src.config import ALL_NUMERIC_FEATURES
        for col in ALL_NUMERIC_FEATURES:
            df_bad[col] = 0
        bad_csv = tmp_path / "bad_target.csv"
        df_bad.to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="Valeurs inattendues"):
            load_data(csv_path=bad_csv, verbose=False)


class TestGetXY:
    """Tests de get_X_y()."""

    def test_get_x_y_basic(self, loaded_df):
        """get_X_y doit retourner (X, y) avec les bonnes formes."""
        X, y = get_X_y(loaded_df, ALL_NUMERIC_FEATURES)
        assert X.shape[1] == len(ALL_NUMERIC_FEATURES)
        assert len(y) == len(X)
        assert set(y.unique()).issubset({0, 1})

    def test_get_x_y_missing_col_raises(self, loaded_df):
        """get_X_y doit lever ValueError si une colonnes est absente."""
        with pytest.raises(ValueError, match="Colonnes de features absentes"):
            get_X_y(loaded_df, ["colonne_inexistante"])
