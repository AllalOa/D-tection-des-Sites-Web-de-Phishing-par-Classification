"""
services.py — couche « données » de l'application PhishGuard (PROJ-IASD01).

Rôle : séparer la logique (chargement / validation des résultats, règles de
fiabilité des prédictions) de la couche HTTP (app.py), pour pouvoir la tester.

Principe clé : les métriques affichées par l'application proviennent des
artefacts du projet (reports/tables/*.csv). Les valeurs documentées
(Documentation PROJ-IASD01, v1.0, §6) ne servent que de REPLI lorsque le CSV
est absent ou incohérent, et l'interface indique toujours la source utilisée.
"""

from __future__ import annotations

import copy
import logging
import math
import re
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

log = logging.getLogger("phishguard")

# ══════════════════════════════════════════════════════════════════════════════
#  Constantes du protocole (identiques à la documentation §2.1 et §8.4)
# ══════════════════════════════════════════════════════════════════════════════

TEST_N_PHISHING = 19_826      # phishing dans le jeu de test indépendant
TEST_N_LEGIT = 27_052         # légitimes dans le jeu de test indépendant
TEST_SIZE = TEST_N_PHISHING + TEST_N_LEGIT  # 46 878

EXP_ORDER = ["A", "B", "B_prime", "C", "D"]

#: Expériences dont les variables sont calculables depuis la seule chaîne d'URL.
#: A, B et B′ exigent des variables de page (LineOfCode, HasPasswordField…) que
#: l'application ne peut pas obtenir : leur prédiction interactive serait fausse
#: (variables remplacées par 0 — cf. documentation §8.5).
PREDICTABLE = ("C", "D")
DEFAULT_EXPERIMENT = "C"

#: Seuil « high_precision » : documenté uniquement pour D / RandomForest (§8.4).
HIGH_PRECISION_THRESHOLD = 0.9352
STANDARD_THRESHOLD = 0.5

MAX_URL_LENGTH = 2048

DATASET_INFO = {
    "name": "PhiUSIIL Phishing URL Dataset (UCI id 967)",
    "raw_rows": 235_795,
    "total_rows": 235_370,       # après suppression de 425 doublons exacts
    "duplicates_removed": 425,
    "total_cols": 56,
    "train_size": 188_492,
    "test_size": TEST_SIZE,
    "n_phishing": 100_520,
    "n_legitimate": 134_850,
    "phishing_pct": 42.71,
    "legitimate_pct": 57.29,
    "distinct_hosts": 219_094,
}

# ══════════════════════════════════════════════════════════════════════════════
#  Valeurs documentées (REPLI) — Documentation PROJ-IASD01 §6.1 / §6.3
# ══════════════════════════════════════════════════════════════════════════════

_FALLBACK_EXPERIMENTS = {
    "A": {
        "tag": "A", "name": "Exp. A — Full Features",
        "short": "A (50 vars)", "color": "#4d8dff",
        "n_features": 50, "feature_type": "50 numériques + TLD",
        "description": "Référence complète : toutes les variables URL, page et dérivées",
        "needs_network": True, "best_model": "RandomForest",
        "hyperparams": "max_depth=None, n_estimators=100",
        "metrics": {"pr_auc": 1.0000, "roc_auc": 1.0000,
                    "tp": 19826, "fp": 0, "tn": 27052, "fn": 0},
        "cv_ap": "1.0000 ± 0.0000", "train_time": "356 s",
    },
    "B": {
        "tag": "B", "name": "Exp. B — Sans DERIVED",
        "short": "B (44 vars)", "color": "#8b5cf6",
        "n_features": 44, "feature_type": "44 numériques + TLD",
        "description": "A sans les 6 variables dérivées (URLSimilarityIndex, etc.)",
        "needs_network": True, "best_model": "RandomForest",
        "hyperparams": "max_depth=None, n_estimators=200",
        "metrics": {"pr_auc": 1.0000, "roc_auc": 1.0000,
                    "tp": 19824, "fp": 2, "tn": 27050, "fn": 2},
        "cv_ap": "1.0000 ± 0.0000", "train_time": "558 s",
    },
    "B_prime": {
        "tag": "B_prime", "name": "Exp. B′ — Purified Content",
        "short": "B′ (38 vars)", "color": "#22d3ee",
        "n_features": 38, "feature_type": "38 numériques (sans TLD)",
        "description": "B sans les 6 variables URL non reproductibles ni les dérivées",
        "needs_network": True, "best_model": "RandomForest",
        "hyperparams": "max_depth=20, n_estimators=200",
        "metrics": {"pr_auc": 0.999996, "roc_auc": 0.999997,
                    "tp": 19816, "fp": 13, "tn": 27039, "fn": 10},
        "cv_ap": "0.999994", "train_time": "323 s",
    },
    "C": {
        "tag": "C", "name": "Exp. C — URL-Only (Déployable)",
        "short": "C (18 vars)", "color": "#34e0a1",
        "n_features": 18, "feature_type": "18 URL recalculées",
        "description": "Modèle déployable — variables extraites uniquement de la chaîne URL",
        "needs_network": False, "best_model": "RandomForest",
        "hyperparams": "max_depth=20, n_estimators=200",
        "metrics": {"pr_auc": 0.9979, "roc_auc": 0.9974,
                    "tp": 19637, "fp": 62, "tn": 26990, "fn": 189},
        "cv_ap": "0.9981 ± 0.0002", "train_time": "224 s",
    },
    "D": {
        "tag": "D", "name": "Exp. D — Host-Only",
        "short": "D (10 vars)", "color": "#ff9f43",
        "n_features": 10, "feature_type": "10 variables du nom d'hôte",
        "description": "Variables extraites uniquement du nom d'hôte normalisé",
        "needs_network": False, "best_model": "RandomForest",
        "hyperparams": "max_depth=20, n_estimators=200",
        "metrics": {"pr_auc": 0.8585, "roc_auc": 0.8473,
                    "tp": 12624, "fp": 1649, "tn": 25403, "fn": 7202},
        "cv_ap": "0.8544 ± 0.0069", "train_time": "≈ 210 s",
    },
}

#: Sous-groupes structurels (Documentation §7.3) — repli si le CSV est absent.
SUBGROUPS_DOC = [
    # exp, G1 PR-AUC [IC95], G2 PR-AUC [IC95], G2 FP, G2 FN
    ("A", 1.0000, (1.0000, 1.0000), 1.0000, (1.0000, 1.0000), 0, 0),
    ("B", 0.9999, (1.0000, 1.0000), 0.9999, (0.9998, 1.0000), 2, 0),
    ("B_prime", 0.9983, (0.9951, 0.9999), 0.9998, (0.9992, 1.0000), 13, 0),
    ("C", 0.8426, (0.8069, 0.8724), 0.5521, (0.4823, 0.6213), 62, 86),
    ("D", 0.2727, (0.2231, 0.3302), 0.2254, (0.1624, 0.2958), 1649, 108),
]

#: Répartition des préfixes d'URL par classe (Documentation §7.1, figure 5), en %.
PREFIX_BIAS_DOC = {
    "labels": ["http:// (sans www)", "http://www.", "https:// (sans www)", "https://www."],
    "legitimate": [0.0, 0.0, 0.0, 100.0],
    "phishing": [12.3, 39.1, 46.2, 2.4],
}

#: Références de comparaison pour l'expérience C (Documentation §6.3).
BASELINES_DOC = [
    ("RandomForest (retenu)", 0.9969, 0.9905, 0.9936, 0.9979),
    ("Règle de préfixe https://www.", 1.0000, 0.9763, 0.9880, 0.9863),
    ("DummyClassifier", 0.0, 0.0, 0.0, 0.4229),
]


# ══════════════════════════════════════════════════════════════════════════════
#  Utilitaires génériques
# ══════════════════════════════════════════════════════════════════════════════

def to_jsonable(obj):
    """Convertit récursivement numpy / NaN en types JSON valides."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if np is not None:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            obj = float(obj)
        if isinstance(obj, np.ndarray):
            return to_jsonable(obj.tolist())
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj


def _num(value):
    """Retourne un float fini ou None."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def read_table(tables_dir: Path, filename: str):
    """Lit un CSV de reports/tables/ ; colonnes en minuscules ; None si absent/illisible."""
    path = Path(tables_dir) / filename
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # fichier corrompu
        log.warning("CSV illisible %s : %s", path, exc)
        return None
    if df.empty:
        return None
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def table_records(tables_dir: Path, filename: str):
    """CSV → liste de dicts JSON-safe (NaN → None), clés en minuscules."""
    df = read_table(tables_dir, filename)
    if df is None:
        return []
    return to_jsonable(df.to_dict("records"))


# ══════════════════════════════════════════════════════════════════════════════
#  Chargement des expériences (CSV d'abord, valeurs documentées en repli)
# ══════════════════════════════════════════════════════════════════════════════

def confusion_metrics(tp, fp, tn, fn):
    """Précision / rappel / F1 recalculés depuis la matrice de confusion."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def valid_confusion(tp, fp, tn, fn) -> bool:
    """La matrice doit correspondre exactement au jeu de test du protocole."""
    return (
        all(isinstance(v, int) and v >= 0 for v in (tp, fp, tn, fn))
        and tp + fn == TEST_N_PHISHING
        and tn + fp == TEST_N_LEGIT
    )


def _pick_best_model_row(df: pd.DataFrame, best_model: str):
    """Ligne du meilleur modèle (RandomForest) dans test_metrics_<exp>.csv."""
    if "model" not in df.columns:
        return None
    names = df["model"].astype(str).str.lower().str.replace(r"[\s_\-]", "", regex=True)
    target = best_model.lower().replace(" ", "")
    hit = df[names == target]
    if hit.empty:
        hit = df[names.str.contains(target, regex=False)]
    return None if hit.empty else hit.iloc[0]


def _metrics_from_csv(tables_dir, exp):
    """Métriques de test de l'expérience depuis le CSV ; None si invalide."""
    df = read_table(tables_dir, f"test_metrics_{exp['tag']}.csv")
    if df is None:
        return None
    row = _pick_best_model_row(df, exp["best_model"])
    if row is None:
        return None
    try:
        tp, fp, tn, fn = (int(row[c]) for c in ("tp", "fp", "tn", "fn"))
    except (KeyError, TypeError, ValueError):
        return None
    if not valid_confusion(tp, fp, tn, fn):
        log.warning("test_metrics_%s.csv : matrice de confusion incohérente avec le "
                    "protocole (%s/%s/%s/%s) — repli sur les valeurs documentées.",
                    exp["tag"], tp, fp, tn, fn)
        return None
    metrics = {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
               **confusion_metrics(tp, fp, tn, fn)}
    for key in ("pr_auc", "roc_auc"):
        value = _num(row.get(key)) if key in row.index else None
        metrics[key] = value if value is not None else exp["metrics"][key]
    return metrics


def _cv_ap_from_csv(tables_dir, exp):
    df = read_table(tables_dir, f"cv_results_{exp['tag']}.csv")
    if df is None or "average_precision_mean" not in df.columns:
        return None
    row = _pick_best_model_row(df, exp["best_model"])
    if row is None:
        return None
    mean = _num(row["average_precision_mean"])
    std = _num(row.get("average_precision_std")) if "average_precision_std" in row.index else None
    if mean is None:
        return None
    return f"{mean:.4f} ± {std:.4f}" if std is not None else f"{mean:.4f}"


def load_experiments(tables_dir: Path):
    """
    Retourne (experiments, sources).

    experiments[tag]["metrics"] contient précision / rappel / F1 TOUJOURS
    recalculés depuis la matrice de confusion (cohérence arithmétique).
    sources[tag] ∈ {"csv", "documentation"} indique l'origine des chiffres.
    """
    experiments, sources = {}, {}
    for tag in EXP_ORDER:
        exp = copy.deepcopy(_FALLBACK_EXPERIMENTS[tag])
        csv_metrics = _metrics_from_csv(tables_dir, exp)
        if csv_metrics is not None:
            exp["metrics"] = csv_metrics
            cv_ap = _cv_ap_from_csv(tables_dir, exp)
            if cv_ap:
                exp["cv_ap"] = cv_ap
            sources[tag] = "csv"
        else:
            m = exp["metrics"]
            m.update(confusion_metrics(m["tp"], m["fp"], m["tn"], m["fn"]))
            sources[tag] = "documentation"
        exp["predictable"] = tag in PREDICTABLE
        exp["n_errors"] = exp["metrics"]["fp"] + exp["metrics"]["fn"]
        experiments[tag] = exp
    return experiments, sources


# ══════════════════════════════════════════════════════════════════════════════
#  Fiabilité d'une prédiction (biais du jeu de données — Documentation §7)
# ══════════════════════════════════════════════════════════════════════════════

def url_subgroup(url: str):
    """
    'G2' : https://www. + aucun chemin / requête / fragment (ex. https://www.x.com/)
    'G1' : commence par https://www. (mais possède un chemin, une requête…)
    None : autre.
    Cette définition suit exactement la documentation §7.3 (sensible à la casse).
    """
    if not isinstance(url, str) or not url.startswith("https://www."):
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return "G1"
    if parts.path in ("", "/") and not parts.query and not parts.fragment:
        return "G2"
    return "G1"


def reliability_notes(url: str, experiment: str, experiments=None):
    """Avertissements scientifiques à afficher à côté d'une prédiction."""
    notes = []
    group = url_subgroup(url)
    if experiment == "C" and group == "G2":
        notes.append({
            "level": "warning", "code": "G2",
            "text": ("URL nue de type « https://www.domaine/ » (sous-groupe G2). Dans le jeu de "
                     "test, le modèle C n'y atteint que PR-AUC = 0,55 : la prédiction est peu "
                     "fiable (doc. §7.3)."),
        })
    elif group is not None:
        notes.append({
            "level": "info", "code": "G1",
            "text": ("Le préfixe « https://www. » est présent chez 100 % des URLs légitimes du "
                     "corpus et seulement 2,4 % des phishing : le modèle peut s'appuyer sur ce "
                     "raccourci (doc. §7.1)."),
        })
    if experiment == "D":
        notes.append({
            "level": "warning", "code": "D",
            "text": "Le nom d'hôte seul manque 36,3 % des attaques (rappel 0,64) — usage indicatif.",
        })
    return notes


def classify_label(predicted_class) -> str:
    return "phishing" if int(predicted_class) == 1 else "legitimate"
