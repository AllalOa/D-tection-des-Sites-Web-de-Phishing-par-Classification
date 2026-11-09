"""
app.py — Application web Flask pour PROJ-IASD01.

Lance le serveur avec :
    python webapp/app.py
    ou
    flask --app webapp/app run --debug

Pages :
    /               — Tableau de bord (dashboard)
    /experiments    — Comparaison des 5 expériences
    /evaluation     — Évaluation détaillée par expérience
    /predict        — Tester le modèle sur une URL
"""

import sys
import json
from pathlib import Path

# ── Chemin projet ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import (
    Flask, render_template, request, jsonify, send_from_directory, abort,
)
import pandas as pd
import numpy as np

# ── Config projet ──────────────────────────────────────────────────────────────
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR  = REPORTS_DIR / "tables"
MODELS_DIR  = ROOT / "models"

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)
app.config["SECRET_KEY"] = "proj-iasd01-2026"

# ══════════════════════════════════════════════════════════════════════════════
#  Données des expériences (extraites de final_experiments_comparison.csv)
# ══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = {
    "A": {
        "tag": "A", "name": "Exp. A — Full Features",
        "short": "A (50 vars)", "color": "#0d6efd",
        "n_features": 50, "feature_type": "50 numériques + TLD",
        "description": "Référence complète : toutes les variables URL, page et dérivées",
        "needs_network": True, "best_model": "RandomForest",
        "hyperparams": "max_depth=None, n_estimators=100",
        "metrics": {
            "pr_auc": 1.0000, "roc_auc": 1.0000,
            "precision": 1.0000, "recall": 1.0000, "f1": 1.0000,
            "tp": 19826, "fp": 0, "tn": 27052, "fn": 0,
        },
        "cv_ap": "1.0000 ± 0.0000",
        "train_time": "356s",
    },
    "B": {
        "tag": "B", "name": "Exp. B — Sans DERIVED",
        "short": "B (44 vars)", "color": "#6610f2",
        "n_features": 44, "feature_type": "44 numériques + TLD",
        "description": "A sans les 6 variables dérivées (URLSimilarityIndex, etc.)",
        "needs_network": True, "best_model": "RandomForest",
        "hyperparams": "max_depth=None, n_estimators=200",
        "metrics": {
            "pr_auc": 1.0000, "roc_auc": 1.0000,
            "precision": 0.9999, "recall": 0.9999, "f1": 0.9999,
            "tp": 19824, "fp": 2, "tn": 27050, "fn": 2,
        },
        "cv_ap": "1.0000 ± 0.0000",
        "train_time": "558s",
    },
    "B_prime": {
        "tag": "B_prime", "name": "Exp. B′ — Purified Content",
        "short": "B′ (38 vars)", "color": "#0dcaf0",
        "n_features": 38, "feature_type": "38 numériques (sans TLD)",
        "description": "B sans les 6 variables URL non reproductibles ni les dérivées",
        "needs_network": True, "best_model": "RandomForest",
        "hyperparams": "max_depth=20, n_estimators=200",
        "metrics": {
            "pr_auc": 1.0000, "roc_auc": 1.0000,
            "precision": 0.9993, "recall": 0.9995, "f1": 0.9994,
            "tp": 19816, "fp": 13, "tn": 27039, "fn": 10,
        },
        "cv_ap": "1.0000 ± 0.0000",
        "train_time": "323s",
    },
    "C": {
        "tag": "C", "name": "Exp. C — URL-Only (Déployable)",
        "short": "C (18 vars)", "color": "#198754",
        "n_features": 18, "feature_type": "18 URL recalculées",
        "description": "Modèle déployable — features extraites uniquement de la chaîne URL",
        "needs_network": False, "best_model": "RandomForest",
        "hyperparams": "max_depth=20, n_estimators=200",
        "metrics": {
            "pr_auc": 0.9979, "roc_auc": 0.9974,
            "precision": 0.9969, "recall": 0.9905, "f1": 0.9936,
            "tp": 19637, "fp": 62, "tn": 26990, "fn": 189,
        },
        "cv_ap": "0.9981 ± 0.0002",
        "train_time": "224s",
    },
    "D": {
        "tag": "D", "name": "Exp. D — Host-Only",
        "short": "D (10 vars)", "color": "#fd7e14",
        "n_features": 10, "feature_type": "10 Host recalculées",
        "description": "Features extraites uniquement du nom d'hôte normalisé",
        "needs_network": False, "best_model": "RandomForest",
        "hyperparams": "max_depth=20, n_estimators=200",
        "metrics": {
            "pr_auc": 0.8585, "roc_auc": 0.8473,
            "precision": 0.8845, "recall": 0.6367, "f1": 0.7404,
            "tp": 12624, "fp": 1649, "tn": 25403, "fn": 7202,
        },
        "cv_ap": "0.8544 ± 0.0069",
        "train_time": "~210s",
    },
}

EXP_ORDER = ["A", "B", "B_prime", "C", "D"]

# ── Données du dataset ─────────────────────────────────────────────────────────
DATASET_INFO = {
    "name": "PhiUSIIL Phishing URL Dataset (UCI id 967)",
    "total_rows": 235795,
    "total_cols": 56,
    "train_size": 188492,
    "test_size": 46878,
    "phishing_pct": 42.8,
    "legitimate_pct": 57.2,
}


# ══════════════════════════════════════════════════════════════════════════════
#  Fonctions utilitaires
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(filename):
    """Charge un CSV depuis reports/tables/ (retourne None si absent)."""
    path = TABLES_DIR / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def get_figures_for_experiment(exp_tag):
    """Retourne les noms de fichiers de figures disponibles pour une expérience."""
    figures = {}
    for fig_type, prefix in [
        ("pr_roc", "pr_roc_"),
        ("learning_curve", "learning_curve_"),
    ]:
        for f in FIGURES_DIR.glob(f"{prefix}{exp_tag}_*.png"):
            figures[fig_type] = f.name
    return figures


def load_feature_importance():
    """Charge l'importance des features par permutation (expérience C)."""
    df = load_csv("permutation_importance_C.csv")
    if df is not None:
        return df.to_dict("records")
    return []


def load_model_metadata(exp_tag):
    """Charge les métadonnées d'un modèle."""
    path = MODELS_DIR / f"model_metadata_{exp_tag}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ══════════════════════════════════════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    """Page d'accueil — tableau de bord."""
    return render_template(
        "home.html",
        experiments=EXPERIMENTS,
        exp_order=EXP_ORDER,
        dataset=DATASET_INFO,
    )


@app.route("/experiments")
def experiments():
    """Comparaison détaillée des expériences."""
    # Charger les résultats CV pour chaque expérience
    cv_results = {}
    for exp in EXP_ORDER:
        cv_df = load_csv(f"cv_results_{exp}.csv")
        if cv_df is not None:
            cv_results[exp] = cv_df.to_dict("records")

    return render_template(
        "experiments.html",
        experiments=EXPERIMENTS,
        exp_order=EXP_ORDER,
        cv_results=cv_results,
        dataset=DATASET_INFO,
    )


@app.route("/evaluation")
@app.route("/evaluation/<experiment>")
def evaluation(experiment="C"):
    """Évaluation détaillée d'une expérience."""
    if experiment not in EXPERIMENTS:
        abort(404)

    exp = EXPERIMENTS[experiment]
    figures = get_figures_for_experiment(exp["tag"])

    # Charger les métriques de test par modèle
    test_metrics = load_csv(f"test_metrics_{exp['tag']}.csv")
    test_metrics_list = test_metrics.to_dict("records") if test_metrics is not None else []

    # Charger leave-TLD-out
    tld_out = load_csv(f"leave_tld_out_{exp['tag']}.csv")
    tld_out_list = tld_out.to_dict("records") if tld_out is not None else []

    # Charger subgroup analysis
    subgroups = load_csv(f"subgroups_evaluation_{exp['tag']}.csv")
    subgroups_list = subgroups.to_dict("records") if subgroups is not None else []

    # Feature importance (C uniquement)
    feat_importance = load_feature_importance() if experiment == "C" else []

    # Métadonnées du modèle
    metadata = load_model_metadata(exp["tag"])

    return render_template(
        "evaluation.html",
        exp=exp,
        experiment=experiment,
        experiments=EXPERIMENTS,
        exp_order=EXP_ORDER,
        figures=figures,
        test_metrics=test_metrics_list,
        tld_out=tld_out_list,
        subgroups=subgroups_list,
        feat_importance=feat_importance,
        metadata=metadata,
    )


@app.route("/predict")
def predict_page():
    """Interface de prédiction d'URL."""
    return render_template(
        "predict.html",
        experiments=EXPERIMENTS,
        exp_order=EXP_ORDER,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API de prédiction — retourne JSON."""
    from src.predict import predict_url, InvalidURLError

    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    experiment = data.get("experiment", None)

    if not url:
        return jsonify({"error": "Veuillez fournir une URL."}), 400

    try:
        result = predict_url(url, experiment=experiment)
        return jsonify({
            "success": True,
            "url": result["url"],
            "predicted_class": result["predicted_class"],
            "predicted_label": result["predicted_label"],
            "model_score": round(result["model_score"], 6),
            "threshold_used": result["threshold_used"],
            "model_name": result["model_name"],
            "experiment": result["experiment"],
            "features": {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in result["features"].items()
            },
        })
    except InvalidURLError as e:
        return jsonify({"error": f"URL invalide : {e}"}), 400
    except FileNotFoundError as e:
        return jsonify({"error": f"Modèle introuvable : {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Erreur serveur : {e}"}), 500


@app.route("/figures/<path:filename>")
def serve_figure(filename):
    """Sert les images de reports/figures/."""
    return send_from_directory(FIGURES_DIR, filename)


@app.route("/eda_figures/<path:filename>")
def serve_eda_figure(filename):
    """Sert les figures EDA de reports/figures/."""
    return send_from_directory(FIGURES_DIR, filename)


# ══════════════════════════════════════════════════════════════════════════════
#  Filtres Jinja2
# ══════════════════════════════════════════════════════════════════════════════

@app.template_filter("pct")
def pct_filter(value):
    """Formate un float en pourcentage."""
    try:
        return f"{float(value) * 100:.2f}%"
    except (ValueError, TypeError):
        return str(value)


@app.template_filter("fmt4")
def fmt4_filter(value):
    """Formate un float avec 4 décimales."""
    try:
        return f"{float(value):.4f}"
    except (ValueError, TypeError):
        return str(value)


@app.template_filter("fmt_int")
def fmt_int_filter(value):
    """Formate un entier avec séparateurs de milliers."""
    try:
        return f"{int(value):,}".replace(",", " ")
    except (ValueError, TypeError):
        return str(value)


# ══════════════════════════════════════════════════════════════════════════════
#  Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  PROJ-IASD01 — Détection de Phishing")
    print(f"  Serveur web : http://127.0.0.1:5000")
    print(f"{'='*60}\n")
    app.run(debug=True, host="127.0.0.1", port=5000)
