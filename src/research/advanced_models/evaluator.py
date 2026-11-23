"""
Evaluator for advanced models on URL_ONLY feature space (Exp C).
Uniform evaluation with PR-AUC, ROC-AUC, F1, Precision, Recall, confusion matrix,
and rigorous subgroup analysis (G1: https://www., G2: bare hosts under https://www., G1_strict).
Includes cluster bootstrap confidence intervals on Host groups.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import urllib.parse
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

GROUP_COL = "Host"
TARGET_COL = "phishing"


def get_subgroup_masks(urls: pd.Series) -> Dict[str, pd.Series]:
    """
    Subgroups:
    - G1: Starts with 'https://www.'
    - G2: G1 AND bare host (path in ('', '/'), no query, no fragment)
    - G1_strict: G1 but NOT G2 (disjoint population)
    """
    def is_g1(u: str) -> bool:
        return str(u).startswith("https://www.")

    def is_g2(u: str) -> bool:
        if not is_g1(u):
            return False
        parsed = urllib.parse.urlsplit(str(u))
        return (parsed.path in ("", "/")) and (not parsed.query) and (not parsed.fragment)

    g1_mask = urls.apply(is_g1)
    g2_mask = urls.apply(is_g2)
    g1_strict_mask = g1_mask & (~g2_mask)

    return {
        "G1": g1_mask,
        "G2": g2_mask,
        "G1_strict": g1_strict_mask,
    }


def cluster_bootstrap_ci(
    df_sub: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Computes 95% bootstrap confidence interval by resampling distinct Hosts.
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


def evaluate_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes standard classification metrics at a given threshold.
    """
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    pr_auc = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan
    roc_auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }
