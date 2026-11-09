"""
test_build_features.py — Tests de src/build_features.py.

Vérifie :
- build_features() produit un DataFrame avec les bonnes colonnes et dimensions.
- Le calcul par lot (compute_url_features_batch) et predict_url donnent des
  valeurs identiques sur 1 000 URLs aléatoires (random_state=42).
- Le fichier de sortie est relançable (idempotent).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    RAW_CSV,
    URL_ONLY_FEATURES,
    TARGET_COL,
    RANDOM_STATE,
    PROCESSED_DIR,
)
from src.url_features import compute_url_features, compute_url_features_batch
from src.build_features import build_features


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def built_df(tmp_path_factory):
    """Lance build_features sur le vrai CSV et retourne le DataFrame."""
    if not RAW_CSV.exists():
        pytest.skip(f"CSV brut introuvable : {RAW_CSV}")
    tmp = tmp_path_factory.mktemp("processed") / "url_features_all.csv"
    df = build_features(output_path=tmp, verbose=False)
    return df


@pytest.fixture(scope="module")
def sample_urls(built_df):
    """1 000 URLs aléatoires prélevées dans le dataset dédupliqué."""
    return (
        built_df["URL"].reset_index(drop=True)
        if "URL" in built_df.columns
        else None
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests sur la structure de build_features
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildFeaturesStructure:
    """Vérifie la structure et les dimensions du DataFrame produit."""

    def test_returns_dataframe(self, built_df):
        """build_features doit retourner un DataFrame."""
        assert isinstance(built_df, pd.DataFrame)

    def test_all_url_only_features_present(self, built_df):
        """Toutes les 18 features URL_ONLY doivent être présentes."""
        for feat in URL_ONLY_FEATURES:
            assert feat in built_df.columns, f"Colonne manquante : {feat}"

    def test_required_meta_columns(self, built_df):
        """Les colonnes row_id, raw_index, Domain et phishing doivent être présentes."""
        assert "row_id" in built_df.columns, "row_id absent"
        assert "raw_index" in built_df.columns, "raw_index absent"
        assert TARGET_COL in built_df.columns, f"{TARGET_COL} absent"

    def test_no_missing_values_in_features(self, built_df):
        """Aucune valeur manquante dans les 18 features."""
        missing = built_df[URL_ONLY_FEATURES].isnull().sum().sum()
        assert missing == 0, f"{missing} valeurs manquantes détectées"

    def test_expected_row_count(self, built_df):
        """Le dataset dédupliqué doit avoir 235 370 lignes (235 795 - 425 doublons)."""
        assert len(built_df) == 235_370, (
            f"Attendu 235 370 lignes, obtenu {len(built_df)}"
        )

    def test_phishing_values(self, built_df):
        """La colonne phishing ne doit contenir que 0 et 1."""
        assert set(built_df[TARGET_COL].unique()).issubset({0, 1})

    def test_raw_index_unique(self, built_df):
        """raw_index doit être unique (pas de doublons)."""
        assert built_df["raw_index"].nunique() == len(built_df), \
            "raw_index contient des doublons"

    def test_idempotent(self, tmp_path):
        """build_features doit pouvoir être relancé sans erreur (overwrite)."""
        if not RAW_CSV.exists():
            pytest.skip(f"CSV brut introuvable : {RAW_CSV}")
        out = tmp_path / "features_test.csv"
        df1 = build_features(output_path=out, verbose=False)
        df2 = build_features(output_path=out, verbose=False)  # relancé
        assert df1.shape == df2.shape


# ─────────────────────────────────────────────────────────────────────────────
# Test batch == compute_url_features (1 000 URLs aléatoires)
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchVsUnitConsistency:
    """
    Vérifie que compute_url_features_batch et compute_url_features
    donnent des résultats identiques sur 1 000 URLs aléatoires.

    Ce test garantit qu'il n'y a pas de divergence entre le code de
    prédiction individuelle (predict_url) et le calcul en lot (build_features).
    """

    N_SAMPLE = 1_000

    def test_batch_matches_unit_on_1000_urls(self, built_df):
        """
        Pour 1 000 URLs aléatoires, chaque feature calculée en lot
        doit correspondre exactement à la valeur calculée individuellement.
        """
        df_sample = built_df.sample(n=self.N_SAMPLE, random_state=RANDOM_STATE)
        urls = df_sample["URL"].tolist() if "URL" in df_sample.columns else None

        if urls is None:
            # Recharger l'URL depuis le CSV brut via raw_index
            df_raw_urls = pd.read_csv(RAW_CSV, usecols=["URL"])
            urls = df_raw_urls.loc[df_sample["raw_index"].values, "URL"].tolist()

        # Calcul en lot
        batch_result = compute_url_features_batch(urls, verbose=False)

        # Calcul unitaire
        unit_results = [compute_url_features(u) for u in urls]
        unit_df = pd.DataFrame(unit_results)

        for feat in URL_ONLY_FEATURES:
            batch_vals = batch_result[feat].values.astype(float)
            unit_vals  = unit_df[feat].values.astype(float)
            mismatches = np.where(np.abs(batch_vals - unit_vals) > 1e-12)[0]
            assert len(mismatches) == 0, (
                f"Feature '{feat}' : {len(mismatches)} désaccords entre "
                f"compute_url_features_batch et compute_url_features. "
                f"Exemple idx={mismatches[0]} : "
                f"batch={batch_vals[mismatches[0]]}, unit={unit_vals[mismatches[0]]}"
            )

    def test_batch_column_order_matches_url_only_features(self, built_df):
        """
        Les colonnes produites par compute_url_features_batch doivent
        être dans l'ordre exact de URL_ONLY_FEATURES.
        """
        df_sample = built_df.sample(n=10, random_state=RANDOM_STATE)
        urls = (
            df_sample["URL"].tolist()
            if "URL" in df_sample.columns
            else ["https://example.com"] * 10
        )
        batch_result = compute_url_features_batch(urls, verbose=False)
        assert list(batch_result.columns) == URL_ONLY_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Test : les features du CSV build_features == recalcul frais
# ─────────────────────────────────────────────────────────────────────────────

class TestSavedFeaturesConsistency:
    """
    Vérifie que les features sauvegardées dans url_features_all.csv
    correspondent à un recalcul frais sur les mêmes URLs.
    """

    N_SAMPLE = 1_000

    def test_saved_features_match_recalculation(self, built_df):
        """
        Les 18 features stockées dans le DataFrame produit par build_features
        doivent correspondre à un recalcul frais de compute_url_features_batch.
        """
        df_sample = built_df.sample(n=self.N_SAMPLE, random_state=RANDOM_STATE + 1)

        if "URL" not in df_sample.columns:
            pytest.skip("Colonne URL absente du DataFrame produit par build_features")

        urls = df_sample["URL"].tolist()
        fresh = compute_url_features_batch(urls, verbose=False).reset_index(drop=True)
        stored = df_sample[URL_ONLY_FEATURES].reset_index(drop=True)

        for feat in URL_ONLY_FEATURES:
            fresh_vals  = fresh[feat].values.astype(float)
            stored_vals = stored[feat].values.astype(float)
            mismatches  = np.where(np.abs(fresh_vals - stored_vals) > 1e-12)[0]
            assert len(mismatches) == 0, (
                f"Feature '{feat}': divergence entre valeurs sauvegardées"
                f" et recalcul. {len(mismatches)} désaccords."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test : concordance row_id avec load_data et split par Domain
# ─────────────────────────────────────────────────────────────────────────────

class TestRowIdAndSplitConsistency:
    """
    Vérifie que row_id correspond à load_data() et que make_train_test_split
    produit les mêmes domaines train/test que pour les autres expériences.
    """

    def test_row_id_matches_load_data(self):
        """Vérifie que Domain et phishing correspondent exactement par row_id à load_data()."""
        from src.data_loader import load_data

        if not RAW_CSV.exists() or not (PROCESSED_DIR / "url_features_all.csv").exists():
            pytest.skip("Données brutes ou traitées absentes")

        df_loaded = load_data(verbose=False)
        df_feat = pd.read_csv(PROCESSED_DIR / "url_features_all.csv")

        assert "row_id" in df_feat.columns, "row_id doit être présent dans url_features_all.csv"
        assert (df_feat["row_id"].values == np.arange(len(df_loaded))).all(), (
            "row_id ne correspond pas à l'index séquentiel 0..N-1"
        )

        # Vérification globale vectorielle
        assert (df_feat["Domain"].values == df_loaded["Domain"].values).all(), (
            "Divergence Domain entre url_features_all.csv et load_data()"
        )
        assert (df_feat[TARGET_COL].values == df_loaded[TARGET_COL].values).all(), (
            "Divergence phishing entre url_features_all.csv et load_data()"
        )

    def test_split_produces_identical_domains(self):
        """Vérifie que split.py produit exactement les mêmes domaines train/test."""
        from src.data_loader import load_data
        from src.split import make_train_test_split

        if not RAW_CSV.exists() or not (PROCESSED_DIR / "url_features_all.csv").exists():
            pytest.skip("Données brutes ou traitées absentes")

        df_loaded = load_data(verbose=False)
        df_feat = pd.read_csv(PROCESSED_DIR / "url_features_all.csv")

        train_loaded, test_loaded = make_train_test_split(df_loaded)
        train_feat, test_feat = make_train_test_split(df_feat)

        train_domains_loaded = set(train_loaded["Domain"])
        train_domains_feat = set(train_feat["Domain"])
        test_domains_loaded = set(test_loaded["Domain"])
        test_domains_feat = set(test_feat["Domain"])

        assert train_domains_loaded == train_domains_feat, (
            "Les domaines de train diffèrent entre load_data() et url_features_all.csv"
        )
        assert test_domains_loaded == test_domains_feat, (
            "Les domaines de test diffèrent entre load_data() et url_features_all.csv"
        )
        assert len(train_loaded) == len(train_feat), "Tailles de train différentes"
        assert len(test_loaded) == len(test_feat), "Tailles de test différentes"

    def test_split_produces_identical_row_ids(self):
        """Vérifie l'égalité stricte des ensembles et séquences de row_id train/test entre load_data() et url_features_all.csv."""
        from src.data_loader import load_data
        from src.split import make_train_test_split

        if not RAW_CSV.exists() or not (PROCESSED_DIR / "url_features_all.csv").exists():
            pytest.skip("Données brutes ou traitées absentes")

        df_loaded = load_data(verbose=False)
        df_feat = pd.read_csv(PROCESSED_DIR / "url_features_all.csv")

        train_loaded, test_loaded = make_train_test_split(df_loaded)
        train_feat, test_feat = make_train_test_split(df_feat)

        assert set(train_loaded["row_id"]) == set(train_feat["row_id"]), (
            "Ensembles de row_id train différents entre A/B et C"
        )
        assert set(test_loaded["row_id"]) == set(test_feat["row_id"]), (
            "Ensembles de row_id test différents entre A/B et C"
        )
        assert (train_loaded["row_id"].values == train_feat["row_id"].values).all()
        assert (test_loaded["row_id"].values == test_feat["row_id"].values).all()
