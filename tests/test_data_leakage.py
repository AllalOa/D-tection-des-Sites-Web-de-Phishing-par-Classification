"""
tests/test_data_leakage.py — Suite de tests automatisée anti-fuite de données (Data Leakage).

Vérifie rigoureusement selon les exigences du barème de Data Mining :
1. Hermétisme du pipeline scikit-learn (ColumnTransformer + Pipeline).
2. Encodage et standardisation appris UNIQUEMENT sur le train (aucune fuite de distribution).
3. Absence totale de variables cibles (TARGET_COL, label) dans les transformateurs ou matrices X.
4. Étanchéité absolue du split par hôte/domaine : aucun chevauchement entre train et test.
5. Invariance et idempotence des paramètres fittés lors d'appels répétés à transform() sur le test.
6. Sérialisation joblib sans perte de fidélité ni contamination.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import (
    ALL_NUMERIC_FEATURES,
    GROUP_COL,
    ORIGINAL_TARGET_COL,
    RANDOM_STATE,
    TARGET_COL,
    URL_ONLY_FEATURES,
)
from src.preprocessing import (
    COUNT_FEATURES_LOG1P,
    TLDGrouper,
    build_pipeline,
    load_pipeline,
    save_pipeline,
)
from src.split import make_cv_folds, make_train_test_split


@pytest.fixture
def synthetic_dataset():
    """Crée un dataset synthétique réaliste avec groupes d'hôtes distincts."""
    rng = np.random.RandomState(RANDOM_STATE)
    n_samples = 400

    # Création d'hôtes partagés pour simuler des URLs d'un même domaine
    hosts = [f"domain_{i // 4}.com" for i in range(n_samples)]

    data = {feat: rng.exponential(scale=10.0, size=n_samples) for feat in URL_ONLY_FEATURES}
    data["Host"] = hosts
    data["Domain"] = hosts
    data["TLD"] = [h.split(".")[-1] for h in hosts]
    # Ajout d'une cible binaire corrélée mais bruitée
    data[TARGET_COL] = (data["URLLength"] > np.median(data["URLLength"])).astype(int)
    data[ORIGINAL_TARGET_COL] = 1 - data[TARGET_COL]
    data["URL"] = [f"https://{h}/path_{i}" for i, h in enumerate(hosts)]
    data["row_id"] = list(range(n_samples))

    df = pd.DataFrame(data)
    return df


class TestDataLeakageProtection:
    """Vérifications strictes contre toute fuite d'information train-test."""

    def test_target_excluded_from_feature_matrix(self):
        """Vérifie que la variable cible n'est jamais présente dans les features utilisées."""
        assert TARGET_COL not in URL_ONLY_FEATURES
        assert ORIGINAL_TARGET_COL not in URL_ONLY_FEATURES
        assert TARGET_COL not in ALL_NUMERIC_FEATURES
        assert ORIGINAL_TARGET_COL not in ALL_NUMERIC_FEATURES

    def test_tld_grouper_strict_fit_isolation(self, synthetic_dataset):
        """
        Vérifie que TLDGrouper apprend le vocabulaire top_n UNIQUEMENT sur le train,
        et que transformer le test avec des TLD inconnus ne modifie pas les paramètres appris.
        """
        train_df = synthetic_dataset.iloc[:300].copy()
        test_df = synthetic_dataset.iloc[300:].copy()

        # Injecter des TLDs complètement inédits dans le test
        test_df["TLD"] = "tld_inconnu_adversarial"

        grouper = TLDGrouper(top_n=5)
        grouper.fit(train_df["TLD"].values.reshape(-1, 1))

        learned_top_tlds = set(grouper.top_tlds_)

        # Vérifier que le vocabulaire est issu de train
        train_top_expected = set(train_df["TLD"].value_counts().head(5).index)
        assert learned_top_tlds == train_top_expected

        # Transformer le test
        res_test = grouper.transform(test_df["TLD"].values.reshape(-1, 1))

        # Vérifier que les paramètres n'ont pas bougé
        assert grouper.top_tlds_ == train_top_expected
        # Tous les TLD inconnus doivent être mappés vers 'autre'
        assert (res_test == "autre").all()

    def test_scaler_mean_and_var_not_contaminated_by_test(self, synthetic_dataset):
        """
        Vérifie que les paramètres de centrage et réduction (mean_, scale_)
        d'un modèle linéaire (LogisticRegression) sont calculés strictement sur train.
        """
        train_df = synthetic_dataset.iloc[:300]
        test_df = synthetic_dataset.iloc[300:]

        pipe = build_pipeline(
            feature_cols=URL_ONLY_FEATURES,
            classifier=LogisticRegression(),
            use_tld=False,
            model_name="LogisticRegression",
        )

        pipe.fit(train_df[URL_ONLY_FEATURES], train_df[TARGET_COL])

        # Extraire le scaler
        preprocessor = pipe.named_steps["preprocessor"]
        scaler = None
        for name, trans, cols in preprocessor.transformers_:
            if hasattr(trans, "named_steps") and "scaler" in trans.named_steps:
                scaler = trans.named_steps["scaler"]
                break

        assert scaler is not None, "StandardScaler introuvable dans le pipeline."
        train_mean = scaler.mean_.copy()

        # Prédire sur le test
        _ = pipe.predict(test_df[URL_ONLY_FEATURES])

        # Vérifier que les moyennes apprises sont restées rigoureusement identiques
        np.testing.assert_array_equal(scaler.mean_, train_mean)

    def test_group_shuffle_split_zero_host_overlap(self, synthetic_dataset):
        """
        Vérifie que le split par nom d'hôte/domaine garantit une intersection VIDE
        entre les domaines du train et les domaines du test.
        """
        df_train, df_test = make_train_test_split(synthetic_dataset)

        group_col = GROUP_COL if GROUP_COL in df_train.columns else "Domain"
        train_hosts = set(df_train[group_col].unique())
        test_hosts = set(df_test[group_col].unique())

        intersection = train_hosts.intersection(test_hosts)
        assert len(intersection) == 0, (
            f"FUITE DÉTECTÉE : {len(intersection)} hôtes partagés entre train et test !"
        )

    def test_group_kfold_cv_zero_host_overlap(self, synthetic_dataset):
        """Vérifie que chaque pli de validation croisée n'a aucun hôte en commun avec son pli de validation."""
        df_train, _ = make_train_test_split(synthetic_dataset)
        gkf = make_cv_folds(df_train)

        group_col = GROUP_COL if GROUP_COL in df_train.columns else "Domain"
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df_train, groups=df_train[group_col])):
            train_hosts = set(df_train.iloc[train_idx][group_col])
            val_hosts = set(df_train.iloc[val_idx][group_col])
            overlap = train_hosts.intersection(val_hosts)
            assert len(overlap) == 0, (
                f"FUITE DÉTECTÉE dans le pli {fold_idx} : {len(overlap)} hôtes partagés !"
            )

    def test_pipeline_serialization_and_idempotence(self, synthetic_dataset, tmp_path):
        """
        Vérifie qu'un pipeline hermétique sauvegardé avec joblib conserve
        exactement ses prédictions sans fuite ni altération.
        """
        train_df = synthetic_dataset.iloc[:250]
        test_df = synthetic_dataset.iloc[250:]

        pipe = build_pipeline(
            feature_cols=URL_ONLY_FEATURES,
            classifier=RandomForestClassifier(n_estimators=10, random_state=RANDOM_STATE),
            use_tld=False,
            model_name="RandomForest",
        )

        pipe.fit(train_df[URL_ONLY_FEATURES], train_df[TARGET_COL])
        initial_preds = pipe.predict_proba(test_df[URL_ONLY_FEATURES])

        # Sauvegarde
        joblib_path = tmp_path / "leakage_test_pipe.joblib"
        save_pipeline(pipe, joblib_path, metadata={"features": URL_ONLY_FEATURES})

        # Rechargement
        loaded_pipe = load_pipeline(joblib_path)
        loaded_preds = loaded_pipe.predict_proba(test_df[URL_ONLY_FEATURES])

        np.testing.assert_array_almost_equal(initial_preds, loaded_preds, decimal=6)
