"""
train.py - Entraînement des modèles de classification pour la détection de phishing.

Usage (ligne de commande) :
    python -m src.train --experiment A
    python -m src.train --experiment B
    python -m src.train --experiment C

Expériences :
    A : toutes les 50 variables numériques + TLD regroupé
    B : A moins DERIVED (44 variables + TLD)
    C : URL_ONLY (18 variables, modèle déployable)

Modèles entraînés par expérience :
    - DummyClassifier (baseline classe majoritaire)
    - DecisionTree max_depth=1 sur URLSimilarityIndex (baseline règle simple)
    - LogisticRegression (GridSearchCV)
    - DecisionTree (GridSearchCV)
    - RandomForest (GridSearchCV)
    - KNN : sur sous-échantillon <= 40 000 lignes
    - LinearSVC : sur tout le train
    - SVC RBF : sur sous-échantillon de 30 000 lignes (modèle différent, documenté)

Résultats sauvegardés dans :
    - models/<experiment>_<model>.joblib (pipeline complet)
    - models/best_model_<experiment>.joblib (meilleur modèle de l'expérience)
    - models/model_metadata_<experiment>.json (métadonnées du best_model)
    - reports/tables/cv_results_<experiment>.csv (métriques de CV)
"""

import argparse
import json
import warnings
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline

import sklearn
import scipy
import sys

from src.config import (
    RAW_CSV,
    URL_FEATURES_CSV,
    HOST_FEATURES_CSV,
    MODELS_DIR,
    TABLES_DIR,
    TARGET_COL,
    GROUP_COL,
    RANDOM_STATE,
    N_FOLDS,
    EXPERIMENT_FEATURES,
    PARAM_GRIDS,
    KNN_SUBSAMPLE_SIZE,
    SVC_RBF_SUBSAMPLE_SIZE,
    REFIT_METRIC,
    SCORING_METRICS,
    BEST_MODEL_PATH,
    MODEL_METADATA_PATH,
    get_experiment_tag,
)
from src.data_loader import load_data
from src.split import make_train_test_split, make_cv_folds, get_cv_groups
from src.preprocessing import build_pipeline

warnings.filterwarnings("ignore")


def subsample_train(df_train: pd.DataFrame, max_size: int, stratify_col: str = None) -> pd.DataFrame:
    """
    Sous-échantillonne df_train de façon stratifiée (si stratify_col fourni).

    Parameters
    ----------
    df_train : pd.DataFrame
    max_size : int
    stratify_col : str, optional

    Returns
    -------
    pd.DataFrame
    """
    if len(df_train) <= max_size:
        return df_train

    if stratify_col:
        # Stratification proportionnelle
        rng = np.random.RandomState(RANDOM_STATE)
        parts = []
        for val, grp in df_train.groupby(stratify_col):
            n = max(1, int(max_size * len(grp) / len(df_train)))
            idx = rng.choice(len(grp), size=min(n, len(grp)), replace=False)
            parts.append(grp.iloc[idx])
        sub = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).head(max_size)
    else:
        sub = df_train.sample(n=max_size, random_state=RANDOM_STATE)

    return sub.reset_index(drop=True)


def run_cv(pipeline, X, y, groups, cv, scoring, name: str) -> dict:
    """
    Exécute une validation croisée GroupKFold et retourne les métriques moyennes.

    Parameters
    ----------
    pipeline : sklearn Pipeline
    X, y, groups : données et groupes
    cv : GroupKFold
    scoring : list of str
    name : str

    Returns
    -------
    dict avec les métriques moyennes
    """
    print(f"  [CV] {name} - début...")
    scores = cross_validate(
        pipeline, X, y, groups=groups, cv=cv,
        scoring=scoring, n_jobs=-1, return_train_score=False,
    )
    result = {"model": name}
    for metric in scoring:
        key = f"test_{metric}"
        if key in scores:
            vals = scores[key]
            result[f"{metric}_mean"] = float(np.mean(vals))
            result[f"{metric}_std"] = float(np.std(vals))
    _ap = result.get("average_precision_mean", None)
    _ap_str = f"{_ap:.4f}" if _ap is not None else "N/A"
    _ap_std = result.get("average_precision_std", 0)
    _std_str = f"{_ap_std:.4f}" if _ap is not None else "N/A"
    print(f"  [CV] {name} - AP={_ap_str} ± {_std_str}")
    return result


def run_gridsearch(pipeline, X, y, groups, cv, param_grid: dict, model_name: str) -> tuple:
    """
    Exécute un GridSearchCV avec GroupKFold par Domain.

    Parameters
    ----------
    pipeline : sklearn Pipeline
    X, y, groups
    cv : GroupKFold
    param_grid : dict
    model_name : str

    Returns
    -------
    tuple (best_pipeline, cv_results_dict)
    """
    print(f"  [GridSearch] {model_name} - grille: {param_grid}")
    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=SCORING_METRICS,
        refit=REFIT_METRIC,
        n_jobs=-1,
        verbose=0,
        return_train_score=False,
    )
    gs.fit(X, y, **{"groups": groups} if cv is not None else {})
    best = gs.best_estimator_
    best_idx = gs.best_index_
    results = {
        "model": model_name,
        "best_params": gs.best_params_,
    }
    for metric in SCORING_METRICS:
        key = f"mean_test_{metric}"
        std_key = f"std_test_{metric}"
        if key in gs.cv_results_:
            results[f"{metric}_mean"] = float(gs.cv_results_[key][best_idx])
            results[f"{metric}_std"] = float(gs.cv_results_[std_key][best_idx])
    print(f"  [GridSearch] {model_name} - meilleur AP={results.get('average_precision_mean', 'N/A'):.4f}, "
          f"params={gs.best_params_}")
    return best, results


def train_experiment(experiment: str) -> dict:
    """
    Entraîne tous les modèles pour une expérience donnée.

    Parameters
    ----------
    experiment : str
        'A', 'B', 'B'' (ou 'B_prime'), 'C' ou 'D'

    Returns
    -------
    dict
        Résultats de CV par modèle.
    """
    exp_tag = get_experiment_tag(experiment)

    print(f"\n{'='*60}")
    print(f"EXPÉRIENCE {experiment}")
    print(f"{'='*60}")

    # ── Chargement des données ──────────────────────────────────────
    if experiment == "C":
        if not URL_FEATURES_CSV.exists():
            raise FileNotFoundError(
                f"Fichier de features introuvable : {URL_FEATURES_CSV}\n"
                "Veuillez générer les features avec : python -m src.build_features --set url"
            )
        print(f"[data_loader] Chargement de : {URL_FEATURES_CSV}")
        df = pd.read_csv(URL_FEATURES_CSV)
    elif experiment == "D":
        if not HOST_FEATURES_CSV.exists():
            raise FileNotFoundError(
                f"Fichier de features introuvable : {HOST_FEATURES_CSV}\n"
                "Veuillez générer les features avec : python -m src.build_features --set host"
            )
        print(f"[data_loader] Chargement de : {HOST_FEATURES_CSV}")
        df = pd.read_csv(HOST_FEATURES_CSV)
    else:
        df = load_data(verbose=True)

    df_train, df_test = make_train_test_split(df)
    print(f"[train] Train: {len(df_train)} lignes | Test: {len(df_test)} lignes")

    feature_cols = EXPERIMENT_FEATURES[experiment].copy()
    use_tld = experiment in ("A", "B", "B'")  # TLD catégoriel pour A, B, B' ; aucun pour C et D

    # Pour C, on n'utilise que URL_ONLY, pas de TLD séparé
    # (la colonne TLD n'est pas dans feature_cols mais peut être ajoutée)

    # Vérifier que les colonnes existent
    missing = [c for c in feature_cols if c not in df_train.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans df_train : {missing}")

    y_train = df_train[TARGET_COL].values
    groups_train = get_cv_groups(df_train)
    cv = make_cv_folds(df_train)

    # Colonnes à fournir à X (features + TLD si use_tld)
    input_cols = feature_cols + (["TLD"] if use_tld else [])
    X_train = df_train[input_cols]

    all_results = []
    best_ap = -1.0
    best_pipeline = None
    best_model_name = None

    # ── 1. Baseline : DummyClassifier ──────────────────────────────
    t_start = time.time()
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy_pipe = Pipeline([("classifier", dummy)])
    res = run_cv(dummy_pipe, X_train[feature_cols] if not use_tld else X_train,
                 y_train, groups_train, cv, ["average_precision", "roc_auc", "f1"],
                 "DummyClassifier")
    res["train_time_sec"] = round(time.time() - t_start, 2)
    print(f"  [time] DummyClassifier : {res['train_time_sec']}s")
    all_results.append(res)

    # ── 2. Baseline : DecisionTree max_depth=1 sur URLSimilarityIndex ─
    # Exécuté uniquement si la caractéristique fait partie de l'expérience (ex: Expérience A)
    if "URLSimilarityIndex" in feature_cols:
        dt_base = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)
        dt_base_pipe = Pipeline([("classifier", dt_base)])
        res_dt_base = run_cv(
            dt_base_pipe,
            df_train[["URLSimilarityIndex"]],
            y_train, groups_train, cv,
            ["average_precision", "roc_auc", "f1"],
            "DecisionTree_depth1_URLSimilarityIndex"
        )
        all_results.append(res_dt_base)

    # ── 3. LogisticRegression ──────────────────────────────────────
    t_start = time.time()
    lr = LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, solver="saga"
    )
    lr_pipe = build_pipeline(feature_cols, lr, use_tld=use_tld, model_name="LogisticRegression")
    param_grid_lr = {k.replace("LogisticRegression", "classifier"): v
                     for k, v in {"classifier__C": [0.1, 1.0, 10.0]}.items()}
    # Correction de la clé
    param_grid_lr = {"classifier__C": [0.1, 1.0, 10.0]}
    best_lr, res_lr = run_gridsearch(lr_pipe, X_train, y_train, groups_train, cv,
                                      param_grid_lr, "LogisticRegression")
    res_lr["type"] = "gridsearch"
    res_lr["train_time_sec"] = round(time.time() - t_start, 2)
    print(f"  [time] LogisticRegression : {res_lr['train_time_sec']}s")
    all_results.append(res_lr)
    if res_lr.get("average_precision_mean", 0) > best_ap:
        best_ap = res_lr["average_precision_mean"]
        best_pipeline = best_lr
        best_model_name = "LogisticRegression"

    # Sauvegarder le modèle
    model_path = MODELS_DIR / f"{exp_tag}_LogisticRegression.joblib"
    joblib.dump(best_lr, model_path)
    print(f"  [save] -> {model_path}")

    # ── 4. DecisionTree ────────────────────────────────────────────
    t_start = time.time()
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt_pipe = build_pipeline(feature_cols, dt, use_tld=use_tld, model_name="DecisionTree")
    param_grid_dt = {"classifier__max_depth": [5, 10, 20, None]}
    best_dt, res_dt = run_gridsearch(dt_pipe, X_train, y_train, groups_train, cv,
                                      param_grid_dt, "DecisionTree")
    res_dt["type"] = "gridsearch"
    res_dt["train_time_sec"] = round(time.time() - t_start, 2)
    print(f"  [time] DecisionTree : {res_dt['train_time_sec']}s")
    all_results.append(res_dt)
    if res_dt.get("average_precision_mean", 0) > best_ap:
        best_ap = res_dt["average_precision_mean"]
        best_pipeline = best_dt
        best_model_name = "DecisionTree"

    joblib.dump(best_dt, MODELS_DIR / f"{exp_tag}_DecisionTree.joblib")
    print(f"  [save] -> {MODELS_DIR / f'{exp_tag}_DecisionTree.joblib'}")

    # ── 5. RandomForest ────────────────────────────────────
    t_start = time.time()
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf_pipe = build_pipeline(feature_cols, rf, use_tld=use_tld, model_name="RandomForest")
    param_grid_rf = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 20],
    }
    best_rf, res_rf = run_gridsearch(rf_pipe, X_train, y_train, groups_train, cv,
                                      param_grid_rf, "RandomForest")
    res_rf["type"] = "gridsearch"
    res_rf["train_time_sec"] = round(time.time() - t_start, 2)
    print(f"  [time] RandomForest : {res_rf['train_time_sec']}s")
    all_results.append(res_rf)
    if res_rf.get("average_precision_mean", 0) > best_ap:
        best_ap = res_rf["average_precision_mean"]
        best_pipeline = best_rf
        best_model_name = "RandomForest"

    joblib.dump(best_rf, MODELS_DIR / f"{exp_tag}_RandomForest.joblib")
    print(f"  [save] -> {MODELS_DIR / f'{exp_tag}_RandomForest.joblib'}")

    # ── 6. KNN (sous-échantillon <= 40 000) ────────────────────────
    t_start = time.time()
    df_sub_knn = subsample_train(df_train, KNN_SUBSAMPLE_SIZE, stratify_col=TARGET_COL)
    print(f"  [KNN] sous-échantillon: {len(df_sub_knn)} lignes (max {KNN_SUBSAMPLE_SIZE})")
    X_sub_knn = df_sub_knn[input_cols]
    y_sub_knn = df_sub_knn[TARGET_COL].values
    groups_sub_knn = df_sub_knn[GROUP_COL].values
    cv_sub = make_cv_folds(df_sub_knn)

    knn = KNeighborsClassifier(n_jobs=-1)
    knn_pipe = build_pipeline(feature_cols, knn, use_tld=use_tld, model_name="KNN")
    param_grid_knn = {"classifier__n_neighbors": [5, 15]}
    best_knn, res_knn = run_gridsearch(knn_pipe, X_sub_knn, y_sub_knn, groups_sub_knn,
                                        cv_sub, param_grid_knn, "KNN")
    res_knn["type"] = "gridsearch_subsample"
    res_knn["subsample_size"] = len(df_sub_knn)
    res_knn["note"] = f"Entraîné sur sous-échantillon de {len(df_sub_knn)} lignes (max {KNN_SUBSAMPLE_SIZE})"
    res_knn["train_time_sec"] = round(time.time() - t_start, 2)
    print(f"  [time] KNN : {res_knn['train_time_sec']}s")
    all_results.append(res_knn)
    # Règle d'éligibilité formelle : seuls les modèles entraînés sur le Train complet (188 492 lignes)
    # sont éligibles au best_model. Les modèles sous-échantillonnés (KNN <= 40k, SVC_RBF <= 30k)
    # sont des sondes exploratoires et ne concourent pas pour le best_model global.

    joblib.dump(best_knn, MODELS_DIR / f"{exp_tag}_KNN.joblib")

    # ── 7. LinearSVC (tout le train) ───────────────────────────────
    t_start = time.time()
    lsvc = LinearSVC(random_state=RANDOM_STATE, max_iter=2000)
    lsvc_pipe = build_pipeline(feature_cols, lsvc, use_tld=use_tld, model_name="LinearSVC")
    # LinearSVC n'a pas predict_proba mais fournit decision_function -> compatible average_precision
    res_lsvc = run_cv(lsvc_pipe, X_train, y_train, groups_train, cv,
                       ["average_precision", "roc_auc", "f1"], "LinearSVC")
    res_lsvc["type"] = "cv_decision_function"
    res_lsvc["note"] = "LinearSVC évalué via decision_function pour AP et ROC-AUC"

    lsvc_pipe.fit(X_train, y_train)
    joblib.dump(lsvc_pipe, MODELS_DIR / f"{exp_tag}_LinearSVC.joblib")
    res_lsvc["train_time_sec"] = round(time.time() - t_start, 2)
    print(f"  [time] LinearSVC : {res_lsvc['train_time_sec']}s")
    all_results.append(res_lsvc)
    if res_lsvc.get("average_precision_mean", 0) > best_ap:
        best_ap = res_lsvc["average_precision_mean"]
        best_pipeline = lsvc_pipe
        best_model_name = "LinearSVC"

    # ── 8. SVC RBF (30 000 lignes − modèle DIFFÉRENT) ─────────────
    t_start = time.time()
    print(f"  [SVC-RBF] ATTENTION : modèle différent, entraîné sur {SVC_RBF_SUBSAMPLE_SIZE} lignes")
    df_sub_svc = subsample_train(df_train, SVC_RBF_SUBSAMPLE_SIZE, stratify_col=TARGET_COL)
    X_sub_svc = df_sub_svc[input_cols]
    y_sub_svc = df_sub_svc[TARGET_COL].values
    groups_sub_svc = df_sub_svc[GROUP_COL].values
    cv_sub_svc = make_cv_folds(df_sub_svc)

    svc = SVC(kernel="rbf", random_state=RANDOM_STATE, probability=True)
    svc_pipe = build_pipeline(feature_cols, svc, use_tld=use_tld, model_name="SVC")
    res_svc = run_cv(svc_pipe, X_sub_svc, y_sub_svc, groups_sub_svc, cv_sub_svc,
                      ["average_precision", "roc_auc", "f1"], "SVC_RBF")
    res_svc["type"] = "cv_subsample"
    res_svc["subsample_size"] = len(df_sub_svc)
    res_svc["note"] = (
        f"SVC RBF entraîné sur {len(df_sub_svc)} lignes. "
        "C'est un modèle DIFFÉRENT du RandomForest/LR entraîné sur tout le train."
    )

    svc_pipe.fit(X_sub_svc, y_sub_svc)
    joblib.dump(svc_pipe, MODELS_DIR / f"{exp_tag}_SVC_RBF.joblib")
    res_svc["train_time_sec"] = round(time.time() - t_start, 2)
    print(f"  [time] SVC_RBF : {res_svc['train_time_sec']}s")
    all_results.append(res_svc)

    # ── Sauvegarde des résultats CV ─────────────────────────────────
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_path = TABLES_DIR / f"cv_results_{exp_tag}.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8")
    print(f"\n[train] Résultats CV sauvegardés : {results_path}")

    # ── Sauvegarde du meilleur modèle ──────────────────────────────
    if best_pipeline is not None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        best_model_path = MODELS_DIR / f"best_model_{exp_tag}.joblib"
        best_meta_path = MODELS_DIR / f"model_metadata_{exp_tag}.json"
        joblib.dump(best_pipeline, best_model_path)
        print(f"[train] Meilleur modèle sauvegardé : {best_model_path}")
        print(f"[train] Meilleur modèle : {best_model_name} (AP={best_ap:.4f})")

        # Métadonnées du meilleur modèle
        # Extraire les hyperparamètres du best_model
        classifier_params = {}
        if hasattr(best_pipeline, "named_steps"):
            clf = best_pipeline.named_steps.get("classifier", None)
            if clf is not None:
                classifier_params = clf.get_params()

        metadata = {
            "experiment": experiment,
            "model_name": best_model_name,
            "feature_cols": feature_cols,
            "use_tld": use_tld,
            "best_params": classifier_params,
            "cv_average_precision": best_ap,
            "sklearn_version": sklearn.__version__,
            "scipy_version": scipy.__version__,
            "numpy_version": np.__version__,
            "python_version": sys.version,
            "random_state": RANDOM_STATE,
            "n_folds": N_FOLDS,
            "training_size": len(df_train),
            "date": datetime.now().isoformat(),
        }
        with open(best_meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"[train] Métadonnées sauvegardées : {best_meta_path}")

    return {"results": all_results, "best_model": best_model_name, "best_ap": best_ap}


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Entraîne les modèles pour une expérience PROJ-IASD01."
    )
    parser.add_argument(
        "--experiment",
        choices=["A", "B", "B'", "B_prime", "C", "D"],
        default="C",
        help="Expérience à entraîner : A (toutes vars), B (sans DERIVED), B' (sans 6 vars divergentes), C (URL_ONLY), D (HOST_FEATURES).",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    result = train_experiment(args.experiment)

    print(f"\n[TERMINÉ] Expérience {args.experiment}")
    print(f"  Meilleur modèle : {result['best_model']} "
          f"(AP moyen CV = {result['best_ap']:.4f})")


if __name__ == "__main__":
    main()
