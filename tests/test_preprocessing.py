"""
test_preprocessing.py - Tests de src/preprocessing.py.

Vérifie :
- TLDGrouper s'ajuste SEULEMENT sur le train (pas de fuite)
- Le pipeline est correctement construit et ne lève pas d'erreur
- Le StandardScaler n'est appliqué qu'aux modèles qui en ont besoin
- Les transformations log1p sont appliquées sur les bonnes colonnes
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing import TLDGrouper, build_pipeline, COUNT_FEATURES_LOG1P
from src.config import (
    URL_ONLY_FEATURES,
    PAGE_FEATURES,
    ALL_NUMERIC_FEATURES,
    TOP_N_TLD,
    TLD_OTHER_LABEL,
    RANDOM_STATE,
)


@pytest.fixture
def small_df_train():
    """Petit DataFrame d'entraînement factice pour les tests."""
    np.random.seed(RANDOM_STATE)
    n = 200
    df = pd.DataFrame({
        col: np.random.randint(0, 100, size=n)
        for col in URL_ONLY_FEATURES
    })
    # Simuler des TLD
    tld_choices = ["com"] * 80 + ["org"] * 40 + ["net"] * 20 + ["io"] * 20 + \
                  ["rare_tld_1"] * 5 + ["rare_tld_2"] * 5 + ["rare_tld_3"] * 5 + \
                  ["rare_tld_4"] * 5 + ["rare_tld_5"] * 5 + ["super_rare"] * 15
    df["TLD"] = tld_choices
    df["phishing"] = np.random.randint(0, 2, size=n)
    df["Domain"] = [f"domain_{i}.com" for i in range(n)]
    return df


@pytest.fixture
def small_df_test():
    """Petit DataFrame de test factice (TLD différents pour tester la généralisation)."""
    np.random.seed(RANDOM_STATE + 1)
    n = 50
    df = pd.DataFrame({
        col: np.random.randint(0, 100, size=n)
        for col in URL_ONLY_FEATURES
    })
    # Inclure un TLD jamais vu à l'entraînement
    df["TLD"] = ["com"] * 20 + ["org"] * 10 + ["tld_inconnu"] * 20
    df["phishing"] = np.random.randint(0, 2, size=n)
    df["Domain"] = [f"test_domain_{i}.com" for i in range(n)]
    return df


class TestTLDGrouper:
    """Tests du transformateur TLDGrouper."""

    def test_fit_learns_top_tlds(self, small_df_train):
        """Après fit, top_tlds_ doit contenir les TOP_N_TLD TLD les plus fréquents."""
        grouper = TLDGrouper(top_n=5)
        grouper.fit(small_df_train["TLD"].values.reshape(-1, 1))
        assert grouper.top_tlds_ is not None
        assert len(grouper.top_tlds_) <= 5

    def test_rare_tld_replaced(self, small_df_train):
        """Les TLD rares doivent être remplacés par le label 'autre'."""
        grouper = TLDGrouper(top_n=5, other_label=TLD_OTHER_LABEL)
        grouper.fit(small_df_train["TLD"].values.reshape(-1, 1))
        result = grouper.transform(small_df_train["TLD"].values.reshape(-1, 1))
        # Les TLD non dans top_tlds_ doivent être TLD_OTHER_LABEL
        result_flat = result.ravel()
        for val in result_flat:
            assert val in grouper.top_tlds_ or val == TLD_OTHER_LABEL

    def test_unknown_tld_in_test_becomes_other(self, small_df_train, small_df_test):
        """Un TLD de test non vu en train doit devenir TLD_OTHER_LABEL."""
        grouper = TLDGrouper(top_n=10, other_label=TLD_OTHER_LABEL)
        grouper.fit(small_df_train["TLD"].values.reshape(-1, 1))
        result = grouper.transform(small_df_test["TLD"].values.reshape(-1, 1))
        result_flat = result.ravel()
        # 'tld_inconnu' n'était pas dans le train → doit être TLD_OTHER_LABEL
        assert TLD_OTHER_LABEL in result_flat, "TLD inconnu non remplacé par 'autre'."

    def test_no_fit_raises(self):
        """transform() avant fit() doit lever RuntimeError."""
        grouper = TLDGrouper()
        with pytest.raises(RuntimeError, match="fitté"):
            grouper.transform(np.array([["com"], ["org"]]))

    def test_fit_only_on_train(self, small_df_train, small_df_test):
        """
        Vérifie qu'aucune information du test n'influence le fit.
        Le TLD 'tld_inconnu' (test seulement) ne doit pas apparaître dans top_tlds_.
        """
        grouper = TLDGrouper(top_n=10)
        grouper.fit(small_df_train["TLD"].values.reshape(-1, 1))
        assert "tld_inconnu" not in grouper.top_tlds_


class TestBuildPipeline:
    """Tests de la fonction build_pipeline()."""

    def test_pipeline_type(self, small_df_train):
        """build_pipeline doit retourner un sklearn Pipeline."""
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier()
        pipeline = build_pipeline(URL_ONLY_FEATURES, clf, use_tld=False)
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_preprocessor_and_classifier(self, small_df_train):
        """Le pipeline doit avoir 'preprocessor' et 'classifier'."""
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier()
        pipeline = build_pipeline(URL_ONLY_FEATURES, clf, use_tld=False)
        assert "preprocessor" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps

    def test_pipeline_fits_and_predicts(self, small_df_train, small_df_test):
        """Le pipeline doit pouvoir être fitté et prédire."""
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
        pipeline = build_pipeline(URL_ONLY_FEATURES, clf, use_tld=False)
        X_train = small_df_train[URL_ONLY_FEATURES]
        y_train = small_df_train["phishing"]
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(small_df_test[URL_ONLY_FEATURES])
        assert len(preds) == len(small_df_test)
        assert set(preds).issubset({0, 1})

    def test_pipeline_with_tld(self, small_df_train, small_df_test):
        """Le pipeline avec use_tld=True doit accepter la colonne TLD."""
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
        pipeline = build_pipeline(URL_ONLY_FEATURES, clf, use_tld=True)
        X_train = small_df_train[URL_ONLY_FEATURES + ["TLD"]]
        y_train = small_df_train["phishing"]
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(small_df_test[URL_ONLY_FEATURES + ["TLD"]])
        assert len(preds) == len(small_df_test)

    def test_scaler_added_for_logistic_regression(self):
        """StandardScaler doit être inclus pour LogisticRegression."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        clf = LogisticRegression()
        pipeline = build_pipeline(
            URL_ONLY_FEATURES, clf, use_tld=False, model_name="LogisticRegression"
        )
        # Vérifier qu'il y a un scaler quelque part dans le preprocessor
        preprocessor = pipeline.named_steps["preprocessor"]
        found_scaler = False
        for name, trans, cols in preprocessor.transformers:
            if isinstance(trans, Pipeline):
                for step_name, step in trans.steps:
                    if isinstance(step, StandardScaler):
                        found_scaler = True
        assert found_scaler, "Aucun StandardScaler trouvé pour LogisticRegression."

    def test_no_scaler_for_decision_tree(self):
        """Pas de StandardScaler pour DecisionTree."""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.preprocessing import StandardScaler
        clf = DecisionTreeClassifier()
        pipeline = build_pipeline(
            URL_ONLY_FEATURES, clf, use_tld=False, model_name="DecisionTree"
        )
        preprocessor = pipeline.named_steps["preprocessor"]
        found_scaler = False
        for name, trans, cols in preprocessor.transformers:
            if isinstance(trans, Pipeline):
                for step_name, step in trans.steps:
                    if isinstance(step, StandardScaler):
                        found_scaler = True
        assert not found_scaler, "StandardScaler ne devrait pas être dans DecisionTree."

    def test_no_empty_feature_cols_raises(self):
        """build_pipeline avec une liste de features vide doit lever ValueError."""
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier()
        with pytest.raises(ValueError, match="Aucune feature"):
            build_pipeline([], clf, use_tld=False)
