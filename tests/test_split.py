"""
test_split.py - Tests de src/split.py.

Vérifie :
- Aucun domaine commun entre train et test
- Les tailles du découpage sont raisonnables (~80/20)
- Les plis GroupKFold n'ont pas de fuite de domaines
- leave_one_tld_out isole correctement le TLD
"""

import pytest
import numpy as np
import pandas as pd

from src.config import RAW_CSV, TARGET_COL, GROUP_COL, TEST_SIZE, N_FOLDS
from src.data_loader import load_data
from src.split import (
    make_train_test_split,
    make_cv_folds,
    get_cv_groups,
    leave_one_tld_out,
    get_top_tlds,
)


@pytest.fixture(scope="module")
def df():
    """Charge le dataset réel (skippé si absent)."""
    if not RAW_CSV.exists():
        pytest.skip(f"Fichier CSV introuvable : {RAW_CSV}")
    return load_data(verbose=False)


@pytest.fixture(scope="module")
def train_test_split(df):
    """Découpe train/test une seule fois pour tous les tests."""
    return make_train_test_split(df)


class TestTrainTestSplit:
    """Tests du découpage train/test."""

    def test_no_domain_overlap(self, train_test_split):
        """Aucun domaine commun entre train et test."""
        df_train, df_test = train_test_split
        train_domains = set(df_train[GROUP_COL])
        test_domains = set(df_test[GROUP_COL])
        overlap = train_domains & test_domains
        assert len(overlap) == 0, (
            f"{len(overlap)} domaines en commun entre train et test : "
            f"{list(overlap)[:5]}..."
        )

    def test_test_size_approximate(self, df, train_test_split):
        """Le jeu de test doit représenter environ TEST_SIZE du total."""
        df_train, df_test = train_test_split
        total = len(df_train) + len(df_test)
        actual_test_size = len(df_test) / total
        # Tolérance large car GroupShuffleSplit n'est pas exact
        assert 0.15 < actual_test_size < 0.30, (
            f"Taille du test : {actual_test_size:.2f}, "
            f"attendu proche de {TEST_SIZE}"
        )

    def test_train_is_majority(self, train_test_split):
        """Le train doit être plus grand que le test."""
        df_train, df_test = train_test_split
        assert len(df_train) > len(df_test)

    def test_total_rows_preserved(self, df, train_test_split):
        """La somme train + test doit égaler le total après deduplication."""
        df_train, df_test = train_test_split
        assert len(df_train) + len(df_test) == len(df)

    def test_target_present_train(self, train_test_split):
        """La colonne TARGET_COL doit être dans df_train."""
        df_train, _ = train_test_split
        assert TARGET_COL in df_train.columns

    def test_both_classes_in_train(self, train_test_split):
        """Les deux classes doivent être représentées dans le train."""
        df_train, _ = train_test_split
        vals = set(df_train[TARGET_COL].unique())
        assert vals == {0, 1}, f"Classes manquantes dans le train : {vals}"

    def test_both_classes_in_test(self, train_test_split):
        """Les deux classes doivent être représentées dans le test."""
        _, df_test = train_test_split
        vals = set(df_test[TARGET_COL].unique())
        assert vals == {0, 1}, f"Classes manquantes dans le test : {vals}"


class TestCVFolds:
    """Tests des plis de validation croisée."""

    def test_n_folds(self, train_test_split):
        """GroupKFold doit avoir N_FOLDS plis."""
        df_train, _ = train_test_split
        cv = make_cv_folds(df_train)
        assert cv.get_n_splits() == N_FOLDS

    def test_no_domain_leak_in_folds(self, train_test_split):
        """Aucun domaine en commun entre train et validation dans chaque pli."""
        df_train, _ = train_test_split
        cv = make_cv_folds(df_train)
        groups = get_cv_groups(df_train)
        X = df_train.iloc[:, :3]  # Juste quelques colonnes pour le split

        for fold_idx, (train_idx, val_idx) in enumerate(
            cv.split(X, df_train[TARGET_COL], groups=groups)
        ):
            fold_train_domains = set(groups[train_idx])
            fold_val_domains = set(groups[val_idx])
            overlap = fold_train_domains & fold_val_domains
            assert len(overlap) == 0, (
                f"Pli {fold_idx}: {len(overlap)} domaines en commun "
                f"entre train et validation."
            )


class TestLeaveTLDOut:
    """Tests de leave_one_tld_out."""

    def test_isolates_correct_tld(self, df):
        """Le pli de validation ne doit contenir que le TLD spécifié."""
        top_tld = get_top_tlds(df, n=1)[0]
        df_train, df_val = leave_one_tld_out(df, top_tld)
        assert (df_val["TLD"] == top_tld).all()

    def test_train_excludes_tld(self, df):
        """Le pli d'entraînement ne doit pas contenir le TLD isolé."""
        top_tld = get_top_tlds(df, n=1)[0]
        df_train, _ = leave_one_tld_out(df, top_tld)
        assert (df_train["TLD"] != top_tld).all()

    def test_total_preserved(self, df):
        """Total train + val = total df."""
        top_tld = get_top_tlds(df, n=1)[0]
        df_train, df_val = leave_one_tld_out(df, top_tld)
        assert len(df_train) + len(df_val) == len(df)

    def test_invalid_tld_raises(self, df):
        """ValueError si le TLD n'existe pas."""
        with pytest.raises(ValueError, match="introuvable"):
            leave_one_tld_out(df, "tld_inventé_inexistant_99999")

    def test_missing_tld_column_raises(self, df):
        """ValueError si la colonne TLD est absente."""
        df_no_tld = df.drop(columns=["TLD"])
        with pytest.raises(ValueError, match="absente"):
            leave_one_tld_out(df_no_tld, "com")
