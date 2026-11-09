"""
evaluate.py - Évaluation finale sur le jeu de test et analyse de robustesse.

Usage :
    python -m src.evaluate --experiment C
    python -m src.evaluate --experiment A --experiment B --experiment C

Évaluations réalisées :
    1. Métriques finales sur le test (precision, recall, F1, ROC-AUC, PR-AUC)
    2. Matrice de confusion avec FN et FP
    3. Courbes PR et ROC (sauvegardées dans reports/figures/)
    4. Seuil optimal choisi sur le train (recall max avec precision >= 0.99)
    5. Test de McNemar entre les 2 meilleurs modèles
    6. Courbes d'apprentissage (1%, 5%, 20%, 100% du train)
    7. Robustesse leave-TLD-out sur les 5 TLD les plus fréquents
    8. Analyse d'erreurs : tableau CSV des FN et FP du test
    9. Importances de permutation sur le test (meilleur modèle de C)
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Pas d'affichage interactif
import seaborn as sns
import joblib
import scipy.stats

from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, precision_recall_curve,
    roc_curve,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, LearningCurveDisplay

import warnings
warnings.filterwarnings("ignore")

from src.config import (
    TARGET_COL, GROUP_COL, MODELS_DIR, FIGURES_DIR, TABLES_DIR,
    RANDOM_STATE, N_FOLDS, EXPERIMENT_FEATURES, URL_FEATURES_CSV, HOST_FEATURES_CSV, get_experiment_tag,
)
from src.data_loader import load_data
from src.split import make_train_test_split, make_cv_folds, get_cv_groups, leave_one_tld_out, get_top_tlds


def evaluate_prefix_rule(df_test: pd.DataFrame) -> dict:
    """
    Évalue la règle baseline PrefixRule (URL ne commençant pas par 'https://www.' => phishing)
    sur le jeu de test.

    Parameters
    ----------
    df_test : pd.DataFrame
        DataFrame de test contenant au moins 'URL' (ou lié par row_id) et TARGET_COL.

    Returns
    -------
    dict
        Métriques de performance (précision, rappel, F1, PR-AUC, ROC-AUC, TP, FP, FN, TN).
    """
    if "URL" not in df_test.columns:
        df_raw = load_data(verbose=False)
        urls = df_raw.loc[df_test["row_id"].values, "URL"].values
    else:
        urls = df_test["URL"].values

    y_test = df_test[TARGET_COL].values
    y_pred = np.array([0 if str(u).startswith("https://www.") else 1 for u in urls])
    y_score = y_pred.astype(float)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model": "PrefixRule",
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "pr_auc": average_precision_score(y_test, y_score),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }
    return metrics


def get_scores(pipeline, X, y) -> tuple:
    """
    Retourne (y_pred, y_score) en utilisant predict_proba ou decision_function.

    Parameters
    ----------
    pipeline : sklearn Pipeline
    X : features
    y : vraies étiquettes

    Returns
    -------
    tuple (np.ndarray, np.ndarray)
        (y_pred, y_score) - y_score = probabilités ou scores de décision.
    """
    y_pred = pipeline.predict(X)
    if hasattr(pipeline, "predict_proba"):
        try:
            y_score = pipeline.predict_proba(X)[:, 1]
        except Exception:
            y_score = pipeline.decision_function(X)
    elif hasattr(pipeline, "decision_function"):
        y_score = pipeline.decision_function(X)
    else:
        y_score = y_pred.astype(float)
    return y_pred, y_score


def find_optimal_threshold(pipeline, X_train, y_train, min_precision: float = 0.99) -> float:
    """
    Choisit le seuil maximisant le recall avec précision >= min_precision.
    Calculé sur le train (via prédictions out-of-fold simulées par cross_val_predict).

    Parameters
    ----------
    pipeline : sklearn Pipeline
    X_train, y_train : données d'entraînement
    min_precision : float

    Returns
    -------
    float
        Seuil optimal.
    """
    from sklearn.model_selection import cross_val_predict
    # Seulement si predict_proba disponible
    if not hasattr(pipeline, "predict_proba"):
        return 0.0 if hasattr(pipeline, "decision_function") else 0.5

    try:
        # On utilise les scores de proba hors-pli sur le train
        y_score_oof = cross_val_predict(
            pipeline, X_train, y_train,
            cv=GroupKFold(n_splits=N_FOLDS),
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
    except Exception:
        return 0.5

    precisions, recalls, thresholds = precision_recall_curve(y_train, y_score_oof)

    # Chercher le seuil maximum recall avec precision >= min_precision
    mask = precisions[:-1] >= min_precision
    if mask.any():
        threshold = thresholds[mask][np.argmax(recalls[:-1][mask])]
    else:
        threshold = 0.5

    return float(threshold)


def evaluate_on_test(pipeline, X_test, y_test, threshold: float = 0.5) -> dict:
    """
    Calcule toutes les métriques finales sur le jeu de test.

    Parameters
    ----------
    pipeline : sklearn Pipeline
    X_test, y_test : données de test
    threshold : float

    Returns
    -------
    dict de métriques
    """
    _, y_score = get_scores(pipeline, X_test, y_test)

    # Appliquer le seuil
    y_pred_thresh = (y_score >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred_thresh)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": threshold,
        "precision": precision_score(y_test, y_pred_thresh, zero_division=0),
        "recall": recall_score(y_test, y_pred_thresh, zero_division=0),
        "f1": f1_score(y_test, y_pred_thresh, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "pr_auc": average_precision_score(y_test, y_score),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "FN_note": "Phishing classé comme légitime (dangereux)",
        "FP_note": "Légitime classé comme phishing (faux alarme)",
    }
    return metrics


def plot_pr_roc_curves(pipeline, X_test, y_test, model_name: str, experiment: str):
    """
    Trace et sauvegarde les courbes PR et ROC.
    """
    _, y_score = get_scores(pipeline, X_test, y_test)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Expérience {experiment} - {model_name}", fontsize=13)

    # Courbe PR
    prec, rec, _ = precision_recall_curve(y_test, y_score)
    ap = average_precision_score(y_test, y_score)
    axes[0].plot(rec, prec, color="steelblue", lw=2, label=f"AP = {ap:.4f}")
    axes[0].set_xlabel("Recall (Phishing)")
    axes[0].set_ylabel("Precision (Phishing)")
    axes[0].set_title("Courbe Precision-Recall")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    axes[1].plot(fpr, tpr, color="tomato", lw=2, label=f"AUC = {auc:.4f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("Taux de Faux Positifs")
    axes[1].set_ylabel("Taux de Vrais Positifs")
    axes[1].set_title("Courbe ROC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    exp_tag = get_experiment_tag(experiment)
    out_path = FIGURES_DIR / f"pr_roc_{exp_tag}_{model_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [figure] -> {out_path}")


def mcnemar_test(y_test, y_pred1, y_pred2, name1: str, name2: str) -> dict:
    """
    Test de McNemar entre deux classifieurs sur le même jeu de test.

    Parameters
    ----------
    y_test, y_pred1, y_pred2 : arrays
    name1, name2 : str

    Returns
    -------
    dict avec discordantes et p-value.
    """
    # Paires discordantes
    both_wrong_1 = (y_pred1 != y_test) & (y_pred2 == y_test)  # 1 faux, 2 correct
    both_wrong_2 = (y_pred1 == y_test) & (y_pred2 != y_test)  # 1 correct, 2 faux
    b = int(both_wrong_1.sum())
    c = int(both_wrong_2.sum())

    # Test exact de McNemar via scipy.stats.binomtest
    n_discordant = b + c
    if n_discordant == 0:
        p_value = 1.0
    else:
        result = scipy.stats.binomtest(b, n=n_discordant, p=0.5, alternative="two-sided")
        p_value = result.pvalue

    return {
        "model_1": name1,
        "model_2": name2,
        "n_discordant_1_wrong": b,
        "n_discordant_2_wrong": c,
        "p_value": p_value,
        "significant_0.05": p_value < 0.05,
    }


def learning_curves(pipeline, X_train, y_train, groups, experiment: str, model_name: str):
    """
    Trace les courbes d'apprentissage via GroupKFold sur le train.
    Entraîne sur 1%, 5%, 20%, 100% du pli d'entraînement.
    """
    from sklearn.model_selection import learning_curve

    train_sizes_abs = [0.01, 0.05, 0.20, 1.0]
    cv = GroupKFold(n_splits=N_FOLDS)

    try:
        train_sizes, train_scores, val_scores = learning_curve(
            pipeline, X_train, y_train,
            train_sizes=train_sizes_abs,
            cv=cv,
            groups=groups,
            scoring="average_precision",
            n_jobs=-1,
            shuffle=False,
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_sizes, np.mean(train_scores, axis=1), "o-",
                color="steelblue", label="Train")
        ax.fill_between(train_sizes,
                         np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                         np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                         alpha=0.15, color="steelblue")
        ax.plot(train_sizes, np.mean(val_scores, axis=1), "o-",
                color="tomato", label="Validation")
        ax.fill_between(train_sizes,
                         np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                         np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
                         alpha=0.15, color="tomato")
        ax.set_xlabel("Taille du train")
        ax.set_ylabel("Average Precision")
        ax.set_title(f"Courbes d'apprentissage - {experiment} {model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        exp_tag = get_experiment_tag(experiment)
        out_path = FIGURES_DIR / f"learning_curve_{exp_tag}_{model_name}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  [figure] -> {out_path}")

    except Exception as e:
        print(f"  [warning] Courbe d'apprentissage échouée pour {model_name}: {e}")


def leave_tld_out_analysis(df_train, pipeline, feature_cols, use_tld: bool,
                            experiment: str, n_top: int = 5):
    """
    Analyse robustesse leave-one-TLD-out sur les n_top TLD les plus fréquents du train.
    """
    top_tlds = get_top_tlds(df_train, n=n_top)
    records = []
    input_cols = feature_cols + (["TLD"] if use_tld else [])

    for tld in top_tlds:
        df_rest, df_val = leave_one_tld_out(df_train, tld)

        if len(df_val) == 0:
            continue

        X_val_tld = df_val[input_cols]
        y_val_tld = df_val[TARGET_COL].values

        # Vérifier si le TLD a plusieurs classes
        unique_classes = np.unique(y_val_tld)
        has_both_classes = len(unique_classes) > 1

        try:
            _, y_score_tld = get_scores(pipeline, X_val_tld, y_val_tld)
            y_pred_tld = (y_score_tld >= 0.5).astype(int)

            cm = confusion_matrix(y_val_tld, y_pred_tld, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

            rec = recall_score(y_val_tld, y_pred_tld, zero_division=0)
            roc = roc_auc_score(y_val_tld, y_score_tld) if has_both_classes else None

            row = {
                "TLD": tld,
                "n_samples": len(df_val),
                "n_phishing": int((y_val_tld == 1).sum()),
                "n_legitime": int((y_val_tld == 0).sum()),
                "recall": rec,
                "FN": int(fn),
                "FP": int(fp),
                "roc_auc": roc if roc is not None else "N/A (1 seule classe)",
            }
        except Exception as e:
            row = {"TLD": tld, "erreur": str(e)}

        records.append(row)

    result_df = pd.DataFrame(records)
    exp_tag = get_experiment_tag(experiment)
    out_path = TABLES_DIR / f"leave_tld_out_{exp_tag}.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[evaluate] Leave-TLD-out -> {out_path}")
    print(result_df.to_string(index=False))
    return result_df


def error_analysis(pipeline, X_test, y_test, df_test_full, experiment: str,
                   feature_cols: list, threshold: float = 0.5):
    """
    Crée le tableau d'analyse d'erreurs (FN et FP) sur le test.
    """
    _, y_score = get_scores(pipeline, X_test, y_test)
    y_pred = (y_score >= threshold).astype(int)

    mask_fn = (y_test == 1) & (y_pred == 0)  # Phishing classé légitime
    mask_fp = (y_test == 0) & (y_pred == 1)  # Légitime classé phishing

    error_mask = mask_fn | mask_fp
    error_df = df_test_full[error_mask].copy()
    error_df["y_true"] = y_test[error_mask]
    error_df["y_pred"] = y_pred[error_mask]
    error_df["model_score"] = y_score[error_mask]
    error_df["error_type"] = np.where(mask_fn[error_mask], "FN (phishing manqué)", "FP (faux alarme)")

    # Colonnes d'intérêt
    cols_out = []
    for c in ["URL", "Domain", "TLD", "y_true", "y_pred", "model_score", "error_type"]:
        if c in error_df.columns:
            cols_out.append(c)

    exp_tag = get_experiment_tag(experiment)
    out_path = TABLES_DIR / f"errors_{exp_tag}.csv"
    error_df[cols_out].to_csv(out_path, index=False, encoding="utf-8")
    print(f"[evaluate] Erreurs -> {out_path} ({len(error_df)} erreurs)")

    # Domaines présents dans les 2 classes
    cross_domain_domains = df_test_full.groupby("Domain")[TARGET_COL].nunique()
    cross_domains_2classes = cross_domain_domains[cross_domain_domains > 1].index
    print(f"[evaluate] {len(cross_domains_2classes)} domaines dans les 2 classes dans le test")
    return error_df


def get_subgroup_masks(urls: pd.Series) -> dict:
    """
    Définit les sous-groupes d'évaluation depuis la colonne URL brute :
    - G1 : URL commençant par 'https://www.'
    - G2 : G1 ET sans chemin non vide (hors '/' final), sans requête ni fragment.
    """
    import urllib.parse

    def is_g1(u):
        return str(u).startswith("https://www.")

    def is_g2(u):
        if not is_g1(u):
            return False
        parsed = urllib.parse.urlsplit(str(u))
        return (parsed.path in ("", "/")) and (not parsed.query) and (not parsed.fragment)

    return {
        "G1": urls.apply(is_g1),
        "G2": urls.apply(is_g2),
    }


def cluster_bootstrap_ci(df_sub: pd.DataFrame, y_true: np.ndarray, y_score: np.ndarray,
                         n_boot: int = 1000, seed: int = 42) -> tuple:
    """
    Calcule l'intervalle de confiance à 95 % par rééchantillonnage de 'Host' (1000 tirages, graine 42).
    """
    rng = np.random.RandomState(seed)
    unique_hosts = df_sub[GROUP_COL].unique()
    host_to_indices = df_sub.groupby(GROUP_COL).indices

    pr_aucs = []
    roc_aucs = []

    for _ in range(n_boot):
        sampled_hosts = rng.choice(unique_hosts, size=len(unique_hosts), replace=True)
        idx = np.concatenate([host_to_indices[h] for h in sampled_hosts])
        y_b = y_true[idx]
        s_b = y_score[idx]

        if len(np.unique(y_b)) > 1:
            try:
                pr_aucs.append(average_precision_score(y_b, s_b))
            except Exception:
                pass
            try:
                roc_aucs.append(roc_auc_score(y_b, s_b))
            except Exception:
                pass

    pr_ci = (
        float(np.percentile(pr_aucs, 2.5)) if pr_aucs else np.nan,
        float(np.percentile(pr_aucs, 97.5)) if pr_aucs else np.nan,
    )
    roc_ci = (
        float(np.percentile(roc_aucs, 2.5)) if roc_aucs else np.nan,
        float(np.percentile(roc_aucs, 97.5)) if roc_aucs else np.nan,
    )
    return pr_ci, roc_ci


# Seuils haute précision documentés depuis le TRAIN (validation croisée / OOF)
# Seul RandomForest D dispose d'un seuil haute précision calculé sur le TRAIN (0.9352)
HIGH_PRECISION_THRESHOLDS = {
    ("D", "RandomForest"): 0.9352,
}


def get_model_threshold(model_name: str, experiment: str, policy: str = "standard", pipe=None) -> float:
    """
    Politique de seuil explicite par modèle et expérience :
    - standard : 0.5 pour modèles avec predict_proba, 0.0 pour decision_function
    - high_precision : seuil OOF optimisé sur TRAIN pour précision >= 0.99.
      Lève ValueError si non documenté pour le couple (experiment, model_name).
    """
    if policy == "standard":
        if pipe is not None and not hasattr(pipe, "predict_proba") and hasattr(pipe, "decision_function"):
            return 0.0
        if model_name == "LinearSVC":
            return 0.0
        return 0.5
    elif policy == "high_precision":
        key = (experiment, model_name)
        if key in HIGH_PRECISION_THRESHOLDS:
            return HIGH_PRECISION_THRESHOLDS[key]
        raise ValueError(
            f"Aucun seuil haute précision n'a été calculé et documenté sur le TRAIN "
            f"pour l'expérience '{experiment}' et le modèle '{model_name}'. "
            f"Interdiction d'appliquer silencieusement le seuil d'un autre modèle."
        )
    else:
        raise ValueError(f"Politique de seuil inconnue : {policy}")


def evaluate_subgroups(experiment: str, optimal_threshold: float = None) -> pd.DataFrame:
    """
    Évalue tous les modèles sauvegardés et PrefixRule sur les sous-groupes G1 et G2.
    Fonction réutilisable pour toutes les expériences (A, B, B', C, D).
    """
    if experiment == "C":
        df = pd.read_csv(URL_FEATURES_CSV)
    elif experiment == "D":
        df = pd.read_csv(HOST_FEATURES_CSV)
    else:
        df = load_data(verbose=False)

    if "URL" not in df.columns:
        df_raw = load_data(verbose=False)
        df["URL"] = df_raw.loc[df["row_id"].values, "URL"].values

    df_train, df_test = make_train_test_split(df)
    masks = get_subgroup_masks(df_test["URL"])

    exp_tag = get_experiment_tag(experiment)
    candidate_models = ["RandomForest", "LogisticRegression", "DecisionTree", "KNN", "LinearSVC", "SVC_RBF"]

    loaded_models = {}
    for m in candidate_models:
        path = MODELS_DIR / f"{exp_tag}_{m}.joblib"
        if path.exists():
            loaded_models[m] = joblib.load(path)

    results = []

    for g_name, mask in masks.items():
        df_sub = df_test[mask].copy().reset_index(drop=True)
        y_true = df_sub[TARGET_COL].values
        n_total = len(df_sub)
        n_phi = int((y_true == 1).sum())
        n_leg = int((y_true == 0).sum())
        prev = n_phi / n_total if n_total > 0 else 0.0
        less_than_100_phishing = bool(n_phi < 100)

        use_tld = experiment in ("A", "B", "B'", "B_prime")
        input_cols = EXPERIMENT_FEATURES[experiment] + (["TLD"] if use_tld else [])
        # 1. Modèles ML
        for m_name, pipe in loaded_models.items():
            X_sub = df_sub[input_cols]

            if hasattr(pipe, "predict_proba"):
                y_score = pipe.predict_proba(X_sub)[:, 1]
            elif hasattr(pipe, "decision_function"):
                y_score = pipe.decision_function(X_sub)
            else:
                y_score = pipe.predict(X_sub).astype(float)

            pr_auc = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan
            roc_auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan

            th_std = get_model_threshold(m_name, experiment, policy="standard", pipe=pipe)
            pred_05 = (y_score >= th_std).astype(int)
            cm_05 = confusion_matrix(y_true, pred_05, labels=[0, 1])
            tn_05, fp_05, fn_05, tp_05 = cm_05.ravel()

            try:
                th_opt = optimal_threshold if (optimal_threshold is not None and m_name == "RandomForest" and experiment == "D") else get_model_threshold(m_name, experiment, policy="high_precision", pipe=pipe)
                pred_opt = (y_score >= th_opt).astype(int)
                cm_opt = confusion_matrix(y_true, pred_opt, labels=[0, 1])
                tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel()
                th_opt_val = th_opt
                fp_opt_val = int(fp_opt)
                fn_opt_val = int(fn_opt)
            except ValueError:
                th_opt_val = "N/A"
                fp_opt_val = "N/A"
                fn_opt_val = "N/A"

            pr_ci, roc_ci = cluster_bootstrap_ci(df_sub, y_true, y_score)

            results.append({
                "subgroup": g_name,
                "model": m_name,
                "n_total": n_total,
                "n_legitime": n_leg,
                "n_phishing": n_phi,
                "phishing_prevalence": prev,
                "less_than_100_phishing": less_than_100_phishing,
                "pr_auc": pr_auc,
                "pr_auc_ci_95": f"[{pr_ci[0]:.4f}, {pr_ci[1]:.4f}]",
                "roc_auc": roc_auc,
                "roc_auc_ci_95": f"[{roc_ci[0]:.4f}, {roc_ci[1]:.4f}]",
                "threshold_0.5": 0.5 if hasattr(pipe, "predict_proba") else 0.0,
                "FP_at_0.5": int(fp_05),
                "FN_at_0.5": int(fn_05),
                "optimal_threshold": th_opt_val,
                "FP_at_opt": fp_opt_val,
                "FN_at_opt": fn_opt_val,
            })

        # 2. PrefixRule
        y_score_prefix = np.zeros(n_total, dtype=float)
        pred_prefix = np.zeros(n_total, dtype=int)

        pr_prefix = float(average_precision_score(y_true, y_score_prefix)) if len(np.unique(y_true)) > 1 else prev
        roc_prefix = 0.5

        cm_pref = confusion_matrix(y_true, pred_prefix, labels=[0, 1])
        tn_pref, fp_pref, fn_pref, tp_pref = cm_pref.ravel()

        pr_ci_pref, roc_ci_pref = cluster_bootstrap_ci(df_sub, y_true, y_score_prefix)

        results.append({
            "subgroup": g_name,
            "model": "PrefixRule",
            "n_total": n_total,
            "n_legitime": n_leg,
            "n_phishing": n_phi,
            "phishing_prevalence": prev,
            "less_than_100_phishing": less_than_100_phishing,
            "pr_auc": pr_prefix,
            "pr_auc_ci_95": f"[{pr_ci_pref[0]:.4f}, {pr_ci_pref[1]:.4f}]",
            "roc_auc": roc_prefix,
            "roc_auc_ci_95": f"[{roc_ci_pref[0]:.4f}, {roc_ci_pref[1]:.4f}]",
            "threshold_0.5": "N/A",
            "FP_at_0.5": int(fp_pref),
            "FN_at_0.5": int(fn_pref),
            "optimal_threshold": "N/A",
            "FP_at_opt": int(fp_pref),
            "FN_at_opt": int(fn_pref),
        })

    res_df = pd.DataFrame(results)
    out_csv = TABLES_DIR / f"subgroups_evaluation_{exp_tag}.csv"
    res_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n[evaluate] Résultats sous-groupes sauvegardés -> {out_csv}")
    return res_df


def run_evaluation(experiment: str):
    """
    Évaluation complète pour une expérience.
    """
    exp_tag = get_experiment_tag(experiment)
    print(f"\n{'='*60}")
    print(f"ÉVALUATION - Expérience {experiment}")
    print(f"{'='*60}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Charger et découper les données
    if experiment == "C":
        if not URL_FEATURES_CSV.exists():
            raise FileNotFoundError(
                f"Fichier de features introuvable : {URL_FEATURES_CSV}\n"
                "Veuillez générer les features avec : python -m src.build_features --set url"
            )
        df = pd.read_csv(URL_FEATURES_CSV)
    elif experiment == "D":
        if not HOST_FEATURES_CSV.exists():
            raise FileNotFoundError(
                f"Fichier de features introuvable : {HOST_FEATURES_CSV}\n"
                "Veuillez générer les features avec : python -m src.build_features --set host"
            )
        df = pd.read_csv(HOST_FEATURES_CSV)
    else:
        df = load_data(verbose=False)

    df_train, df_test = make_train_test_split(df)

    feature_cols = EXPERIMENT_FEATURES[experiment].copy()
    use_tld = experiment in ("A", "B", "B'", "B_prime")
    input_cols = feature_cols + (["TLD"] if use_tld else [])

    X_train = df_train[input_cols]
    y_train = df_train[TARGET_COL].values
    groups_train = get_cv_groups(df_train)

    X_test = df_test[input_cols]
    y_test = df_test[TARGET_COL].values

    # Modèles candidats pour l'expérience
    candidate_models = ["RandomForest", "LogisticRegression", "DecisionTree", "KNN", "LinearSVC", "SVC_RBF"]
    test_results = []
    trained_models = {}

    print("\n--- Évaluation de tous les modèles sur le jeu de test ---")
    for m_name in candidate_models:
        model_file = MODELS_DIR / f"{exp_tag}_{m_name}.joblib"
        if not model_file.exists():
            continue
        try:
            pipe = joblib.load(model_file)
            trained_models[m_name] = pipe
            # Seuil calculé sur le train
            thresh = find_optimal_threshold(pipe, X_train, y_train, min_precision=0.99)
            m_res = evaluate_on_test(pipe, X_test, y_test, threshold=thresh)
            m_res["model"] = m_name
            m_res["experiment"] = experiment
            test_results.append(m_res)
            print(f"  Modèle {m_name:<20}: Thresh={thresh:.4f} | PR-AUC={m_res['pr_auc']:.4f} | ROC-AUC={m_res['roc_auc']:.4f} | F1={m_res['f1']:.4f} | Prec={m_res['precision']:.4f} | Rec={m_res['recall']:.4f}")
        except Exception as e:
            print(f"  [warning] Échec évaluation {m_name} : {e}")

    # Évaluation de la baseline DummyClassifier
    y_dummy_pred = np.zeros(len(y_test), dtype=int)
    y_dummy_score = np.full(len(y_test), fill_value=float(np.mean(y_train)))
    cm_dummy = confusion_matrix(y_test, y_dummy_pred, labels=[0, 1])
    tn_d, fp_d, fn_d, tp_d = cm_dummy.ravel()
    dummy_res = {
        "threshold": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "roc_auc": 0.5,
        "pr_auc": float(average_precision_score(y_test, y_dummy_score)),
        "TP": int(tp_d),
        "FP": int(fp_d),
        "TN": int(tn_d),
        "FN": int(fn_d),
        "FN_note": "Phishing classé comme légitime (dangereux)",
        "FP_note": "Légitime classé comme phishing (faux alarme)",
        "model": "DummyClassifier",
        "experiment": experiment,
    }
    test_results.append(dummy_res)
    print(f"  Modèle {'DummyClassifier':<20}: Thresh=0.5000 | PR-AUC={dummy_res['pr_auc']:.4f} | ROC-AUC={dummy_res['roc_auc']:.4f} | F1={dummy_res['f1']:.4f} | Prec={dummy_res['precision']:.4f} | Rec={dummy_res['recall']:.4f}")

    # Évaluation de la baseline PrefixRule
    prefix_res = evaluate_prefix_rule(df_test)
    prefix_res["threshold"] = "N/A"
    prefix_res["experiment"] = experiment
    test_results.append(prefix_res)
    print(f"  Modèle {'PrefixRule':<20}: Thresh=N/A    | PR-AUC={prefix_res['pr_auc']:.4f} | ROC-AUC={prefix_res['roc_auc']:.4f} | F1={prefix_res['f1']:.4f} | Prec={prefix_res['precision']:.4f} | Rec={prefix_res['recall']:.4f}")

    # Tableau global sauvegardé
    metrics_df = pd.DataFrame(test_results)
    out_path = TABLES_DIR / f"test_metrics_{exp_tag}.csv"
    metrics_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[evaluate] Métriques test complètes -> {out_path}")

    # Sélection du meilleur modèle pour les tracés et l'analyse d'erreur
    best_d_file = MODELS_DIR / f"best_model_{exp_tag}.joblib"
    if best_d_file.exists():
        pipeline = joblib.load(best_d_file)
        # Déterminer son nom
        model_name = "BestModel"
        meta_file = MODELS_DIR / f"model_metadata_{exp_tag}.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                model_name = json.load(f).get("model_name", "BestModel")
    elif "RandomForest" in trained_models:
        pipeline = trained_models["RandomForest"]
        model_name = "RandomForest"
    elif trained_models:
        model_name = list(trained_models.keys())[0]
        pipeline = trained_models[model_name]
    else:
        print("[evaluate] Aucun modèle entraîné trouvé.")
        return test_results

    threshold = find_optimal_threshold(pipeline, X_train, y_train, min_precision=0.99)
    print(f"[evaluate] Seuil optimal : {threshold:.4f}")

    # -- 2. Métriques finales sur le test -------------------------------
    print("[evaluate] Évaluation finale sur le test...")
    metrics = evaluate_on_test(pipeline, X_test, y_test, threshold=threshold)
    print(f"\n{'-'*40}")
    print(f"Résultats finaux - {experiment} - {model_name}")
    for k, v in metrics.items():
        if not k.endswith("_note"):
            print(f"  {k:30s}: {v}")

    # Sauvegarder
    best_out_path = TABLES_DIR / f"best_model_metrics_{exp_tag}.csv"
    best_metrics_df = pd.DataFrame([{**metrics, "experiment": experiment, "model": model_name}])
    best_metrics_df.to_csv(best_out_path, index=False, encoding="utf-8")
    print(f"[evaluate] Métriques meilleur modèle -> {best_out_path}")

    # -- 3. Courbes PR et ROC --------------------------------------------
    plot_pr_roc_curves(pipeline, X_test, y_test, model_name, experiment)

    # -- 4. Courbes d'apprentissage -------------------------------------
    print("[evaluate] Courbes d'apprentissage...")
    learning_curves(pipeline, X_train, y_train, groups_train, experiment, model_name)

    # -- 5. Robustesse leave-TLD-out ------------------------------------
    print("[evaluate] Analyse leave-TLD-out...")
    leave_tld_out_analysis(df_train, pipeline, feature_cols, use_tld, experiment)

    # -- 6. Analyse d'erreurs --------------------------------------------
    print("[evaluate] Analyse des erreurs...")
    error_analysis(pipeline, X_test, y_test, df_test, experiment, feature_cols, threshold)

    # -- 7. Importances de permutation (si expérience C) --------------
    if experiment == "C":
        print("[evaluate] Importances de permutation...")
        try:
            perm_imp = permutation_importance(
                pipeline, X_test, y_test,
                n_repeats=10, random_state=RANDOM_STATE,
                scoring="average_precision", n_jobs=-1,
            )
            imp_df = pd.DataFrame({
                "feature": input_cols,
                "importance_mean": perm_imp.importances_mean,
                "importance_std": perm_imp.importances_std,
            }).sort_values("importance_mean", ascending=False)
            imp_path = TABLES_DIR / f"permutation_importance_{exp_tag}.csv"
            imp_df.to_csv(imp_path, index=False, encoding="utf-8")
            print(f"[evaluate] Importances de permutation -> {imp_path}")
            print(imp_df.head(10).to_string(index=False))
        except Exception as e:
            print(f"[warning] Importances de permutation échouées : {e}")

    # -- 8. Evaluation sous-groupes G1 et G2 --
    print("\n[evaluate] Evaluation des sous-groupes G1 et G2...")
    evaluate_subgroups(experiment)

    print(f"\n[TERMINE] Evaluation {experiment}")
    return metrics


def main():
    """Point d'entrée de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Évaluation PROJ-IASD01")
    parser.add_argument("--experiment", choices=["A", "B", "B'", "B_prime", "C", "D"], default="C")
    args = parser.parse_args()
    run_evaluation(args.experiment)


if __name__ == "__main__":
    main()
