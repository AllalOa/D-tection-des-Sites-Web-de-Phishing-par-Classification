"""
split.py - Découpage train/test et plis de validation croisée.

Stratégie :
- Découpage test 20 % par Domain (GroupShuffleSplit) pour éviter
  toute fuite de domaines entre train et test.
- Plis GroupKFold(5) par Domain sur le train.
- Fonction leave-one-TLD-out pour l'analyse de robustesse.

Le jeu de test NE SERT QU'À L'ÉVALUATION FINALE.
random_state=42 pour la reproductibilité.

Usage :
    from src.split import make_train_test_split, make_cv_folds, leave_one_tld_out
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

from src.config import (
    GROUP_COL,
    TARGET_COL,
    RANDOM_STATE,
    TEST_SIZE,
    N_FOLDS,
)


def make_train_test_split(df: pd.DataFrame) -> tuple:
    """
    Découpe le DataFrame en train et test en respectant les groupes de Domain.

    Aucun domaine n'apparaît à la fois dans le train et dans le test.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame complet issu de load_data().

    Returns
    -------
    tuple (pd.DataFrame, pd.DataFrame)
        (df_train, df_test)

    Raises
    ------
    ValueError
        Si la colonne de groupe est absente.
    """
    if GROUP_COL not in df.columns:
        raise ValueError(
            f"Colonne de groupe '{GROUP_COL}' absente du DataFrame."
        )

    groups = df[GROUP_COL].values

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(splitter.split(df, df[TARGET_COL], groups=groups))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Sanity check : aucun domaine commun
    train_domains = set(df_train[GROUP_COL])
    test_domains = set(df_test[GROUP_COL])
    overlap = train_domains & test_domains
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} domaines en commun entre train et test ! "
            "Vérifiez GroupShuffleSplit."
        )

    return df_train, df_test


def make_cv_folds(df_train: pd.DataFrame) -> GroupKFold:
    """
    Crée un objet GroupKFold(K=5) pour la validation croisée sur le train.

    Les groupes utilisés sont les domaines (GROUP_COL).
    Utilisation typique :
        gkf = make_cv_folds(df_train)
        for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups):
            ...

    Parameters
    ----------
    df_train : pd.DataFrame
        DataFrame d'entraînement.

    Returns
    -------
    GroupKFold
        Objet configuré avec N_FOLDS plis.
    """
    return GroupKFold(n_splits=N_FOLDS)


def get_cv_groups(df_train: pd.DataFrame) -> np.ndarray:
    """
    Retourne le vecteur de groupes (Domain) pour df_train.

    Parameters
    ----------
    df_train : pd.DataFrame

    Returns
    -------
    np.ndarray
        Vecteur des domaines.
    """
    if GROUP_COL not in df_train.columns:
        raise ValueError(f"Colonne '{GROUP_COL}' absente.")
    return df_train[GROUP_COL].values


def leave_one_tld_out(df: pd.DataFrame, tld: str) -> tuple:
    """
    Découpe le DataFrame en laissant un TLD unique comme jeu de validation.

    Utile pour l'analyse de robustesse leave-one-TLD-out.
    NE sert PAS à la sélection de modèle.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame (train ou complet).
    tld : str
        Valeur du TLD à isoler comme jeu de validation.

    Returns
    -------
    tuple (pd.DataFrame, pd.DataFrame)
        (df_autres_tld, df_tld) = (train, validation)

    Raises
    ------
    ValueError
        Si le TLD n'existe pas dans le DataFrame.
    """
    if "TLD" not in df.columns:
        raise ValueError("Colonne 'TLD' absente du DataFrame.")

    if tld not in df["TLD"].values:
        raise ValueError(
            f"TLD '{tld}' introuvable dans le DataFrame. "
            f"TLD disponibles (top 10) : {list(df['TLD'].value_counts().head(10).index)}"
        )

    mask_tld = df["TLD"] == tld
    df_val = df[mask_tld].reset_index(drop=True)
    df_train = df[~mask_tld].reset_index(drop=True)

    return df_train, df_val


def get_top_tlds(df: pd.DataFrame, n: int = 5) -> list:
    """
    Retourne les n TLD les plus fréquents du DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    n : int

    Returns
    -------
    list of str
    """
    return list(df["TLD"].value_counts().head(n).index)
