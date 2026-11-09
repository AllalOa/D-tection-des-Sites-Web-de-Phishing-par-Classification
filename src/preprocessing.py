"""
preprocessing.py - Pipelines de prétraitement sklearn.

Responsabilités :
- Regroupement des TLD rares (top 20 appris sur le train → SEULEMENT le train)
- Encodage one-hot des TLD
- Transformation log1p sur les compteurs à queue lourde
- Mise à l'échelle StandardScaler UNIQUEMENT pour Logistic Regression, KNN, SVM
- Tout encapsulé dans sklearn Pipeline + ColumnTransformer (pas de fuite de données)

IMPORTANT : Les transformers sont AJUSTÉS uniquement sur les données d'entraînement.
Les valeurs aberrantes sont documentées mais NON supprimées.

Usage :
    from src.preprocessing import build_pipeline
    pipeline = build_pipeline(feature_cols, model, use_tld=True, scale=False)
"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer,
)
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    TOP_N_TLD,
    TLD_OTHER_LABEL,
    RANDOM_STATE,
)

# Variables compteurs à queue lourde (asymétrie forte)
# identifiées lors de l'EDA (valeurs larges mais jamais négatives)
COUNT_FEATURES_LOG1P = [
    "NoOfSubDomain",
    "NoOfObfuscatedChar",
    "NoOfLettersInURL",
    "NoOfDegitsInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "LineOfCode",
    "LargestLineLength",
    "NoOfPopup",
    "NoOfiFrame",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "URLLength",
    "DomainLength",
    "TLDLength",
]

# Modèles nécessitant une mise à l'échelle
MODELS_NEEDING_SCALE = {"LogisticRegression", "KNN", "LinearSVC", "SVC"}


class TLDGrouper(BaseEstimator, TransformerMixin):
    """
    Transformateur qui regroupe les TLD rares en 'autre'.

    Le top N TLD est appris sur les données d'entraînement lors de fit().
    Les TLD absents du top N sont remplacés par TLD_OTHER_LABEL.

    Parameters
    ----------
    top_n : int
        Nombre de TLD à conserver explicitement.
    other_label : str
        Étiquette pour les TLD rares.
    """

    def __init__(self, top_n: int = TOP_N_TLD, other_label: str = TLD_OTHER_LABEL):
        self.top_n = top_n
        self.other_label = other_label
        self.top_tlds_ = None

    def fit(self, X, y=None):
        """Apprend les top_n TLD sur X (Series ou DataFrame 1-colonne)."""
        tld_series = self._to_series(X)
        self.top_tlds_ = set(
            tld_series.value_counts().head(self.top_n).index.tolist()
        )
        return self

    def transform(self, X):
        """Remplace les TLD non présents dans top_tlds_ par other_label."""
        if self.top_tlds_ is None:
            raise RuntimeError("TLDGrouper doit être fitté avant transform().")
        tld_series = self._to_series(X)
        grouped = tld_series.where(
            tld_series.isin(self.top_tlds_), other=self.other_label
        )
        return grouped.values.reshape(-1, 1)

    @staticmethod
    def _to_series(X) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0]
        if isinstance(X, pd.Series):
            return X
        return pd.Series(X.ravel())


def _log1p_transform(X):
    """Applique numpy.log1p sur les valeurs (compteurs ≥ 0)."""
    return np.log1p(np.abs(X))


def build_pipeline(
    feature_cols: list,
    classifier,
    use_tld: bool = False,
    model_name: str = "",
) -> Pipeline:
    """
    Construit un sklearn Pipeline complet pour l'expérience donnée.

    Parameters
    ----------
    feature_cols : list
        Liste des colonnes numériques features (ex: URL_ONLY_FEATURES).
    classifier : estimator sklearn
        Le classifieur à intégrer dans le pipeline.
    use_tld : bool
        Si True, ajoute la feature TLD regroupée + one-hot encodée.
    model_name : str
        Nom du modèle (ex: 'LogisticRegression') pour activer la mise à l'échelle.

    Returns
    -------
    Pipeline
        Pipeline complet prêt à être fitté.
    """
    # Séparer les features selon qu'elles nécessitent log1p ou non
    log_cols = [c for c in feature_cols if c in COUNT_FEATURES_LOG1P]
    passthrough_cols = [c for c in feature_cols if c not in COUNT_FEATURES_LOG1P]

    # Mise à l'échelle : uniquement pour LR, KNN, SVM
    scale = model_name in MODELS_NEEDING_SCALE

    # ── Construction des transformers ──────────────────────────────────
    transformers = []

    if log_cols:
        log_steps = [("log1p", FunctionTransformer(_log1p_transform))]
        if scale:
            log_steps.append(("scaler", StandardScaler()))
        transformers.append(("log_features", Pipeline(log_steps), log_cols))

    if passthrough_cols:
        passthrough_steps = []
        if scale:
            passthrough_steps.append(("scaler", StandardScaler()))
        if passthrough_steps:
            transformers.append(
                ("pass_features", Pipeline(passthrough_steps), passthrough_cols)
            )
        else:
            transformers.append(("pass_features", "passthrough", passthrough_cols))

    if use_tld:
        tld_pipeline = Pipeline([
            ("grouper", TLDGrouper(top_n=TOP_N_TLD, other_label=TLD_OTHER_LABEL)),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            )),
        ])
        transformers.append(("tld_features", tld_pipeline, ["TLD"]))

    if not transformers:
        raise ValueError("Aucune feature fournie pour construire le pipeline.")

    col_transformer = ColumnTransformer(transformers=transformers)

    # ── Pipeline final ─────────────────────────────────────────────────
    pipeline = Pipeline([
        ("preprocessor", col_transformer),
        ("classifier", classifier),
    ])

    return pipeline


def describe_outliers(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Décrit les valeurs aberrantes (> Q3 + 1.5*IQR) pour chaque feature numérique.

    NOTE : les aberrantes NE SONT PAS supprimées (elles peuvent être un signal).

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list

    Returns
    -------
    pd.DataFrame
        Tableau avec n_outliers, pct_outliers, min, max, Q75, upper_fence
        par colonne.
    """
    records = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        q75 = df[col].quantile(0.75)
        q25 = df[col].quantile(0.25)
        iqr = q75 - q25
        upper_fence = q75 + 1.5 * iqr
        mask = df[col] > upper_fence
        records.append({
            "feature": col,
            "n_outliers": int(mask.sum()),
            "pct_outliers": round(mask.mean() * 100, 2),
            "min": df[col].min(),
            "max": df[col].max(),
            "Q75": q75,
            "upper_fence": upper_fence,
        })
    return pd.DataFrame(records).sort_values("pct_outliers", ascending=False)
