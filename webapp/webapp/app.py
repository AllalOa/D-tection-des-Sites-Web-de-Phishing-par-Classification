"""
app.py — Application web Flask (PhishGuard) pour PROJ-IASD01.

Lancement (depuis la racine du projet) :
    python webapp/app.py
    flask --app webapp/app run --debug

Variables d'environnement (toutes optionnelles) :
    PHISHGUARD_SECRET_KEY   clé secrète Flask (sinon : clé aléatoire à chaque démarrage)
    PHISHGUARD_DEBUG        "1" pour activer le mode debug (défaut : désactivé)
    PHISHGUARD_HOST / PHISHGUARD_PORT   (défaut : 127.0.0.1 / 5000)

Pages :
    /                       Tableau de bord
    /experiments            Comparaison des 5 expériences
    /evaluation/<exp>       Évaluation détaillée (A, B, B_prime, C, D)
    /limites                Biais du jeu de données et limites (doc. §7, §11)
    /predict                Tester un modèle sur une URL

API :
    POST /api/predict       {url, experiment?, threshold?} -> JSON
    GET  /api/experiments   métriques + origine des chiffres (csv / documentation)
    GET  /health            état des artefacts (modèles, tables, figures)
"""

from __future__ import annotations

import inspect
import logging
import os
import secrets
import sys
from pathlib import Path

# ── Chemin projet ──────────────────────────────────────────────────────────────
def _find_project_root() -> Path:
    """Trouve la racine du projet en remontant depuis __file__."""
    cur = Path(__file__).resolve().parent
    for _ in range(5):
        if (cur / "models").is_dir() and (cur / "src").is_dir() and (cur / "reports").is_dir():
            return cur
        cur = cur.parent
    # Repli si la structure est standard (webapp/webapp/app.py -> 3 niveaux au-dessus)
    fallback = Path(__file__).resolve().parent.parent.parent
    if (fallback / "models").is_dir():
        return fallback
    return Path(__file__).resolve().parent.parent

ROOT = _find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

try:                      # lancé comme paquet : flask --app webapp/app
    from . import services as svc
except ImportError:       # lancé comme script : python webapp/app.py
    import services as svc

# ── Chemins des artefacts du projet ────────────────────────────────────────────
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
MODELS_DIR = ROOT / "models"

ALLOWED_FIGURE_EXT = {".png", ".jpg", ".jpeg", ".svg"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("phishguard")

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = os.environ.get("PHISHGUARD_SECRET_KEY") or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024          # une requête de prédiction est minuscule
app.json.sort_keys = False

EXP_ORDER = svc.EXP_ORDER


# ══════════════════════════════════════════════════════════════════════════════
#  Accès aux données (avec cache invalidé si un CSV change)
# ══════════════════════════════════════════════════════════════════════════════

_cache = {"key": None, "value": None}


def _tables_signature():
    if not TABLES_DIR.is_dir():
        return ()
    return tuple(sorted((p.name, p.stat().st_mtime_ns) for p in TABLES_DIR.glob("*.csv")))


def get_experiments():
    """(experiments, sources) — métriques lues dans reports/tables, repli documenté."""
    key = (str(TABLES_DIR), _tables_signature())
    if _cache["key"] != key or _cache["value"] is None:
        _cache["value"] = svc.load_experiments(TABLES_DIR)
        _cache["key"] = key
        for tag, source in _cache["value"][1].items():
            if source != "csv":
                log.info("Expérience %s : métriques issues de la documentation "
                         "(test_metrics_%s.csv absent ou invalide).", tag, tag)
    return _cache["value"]


def get_figures_for_experiment(exp_tag):
    """Noms de fichiers de figures disponibles pour une expérience."""
    figures = {}
    if not FIGURES_DIR.is_dir():
        return figures
    for fig_type, prefix in (("pr_roc", "pr_roc_"), ("learning_curve", "learning_curve_")):
        matches = sorted(FIGURES_DIR.glob(f"{prefix}{exp_tag}_*.png"))
        if matches:
            figures[fig_type] = matches[-1].name
    return figures


def load_model_metadata(exp_tag):
    """Métadonnées JSON du modèle (dict vide si absentes ou illisibles)."""
    import json
    path = MODELS_DIR / f"model_metadata_{exp_tag}.json"
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        log.warning("Métadonnées illisibles : %s", path)
        return {}


def artifacts_status():
    """État des artefacts nécessaires à l'application (pour /health et l'interface)."""
    models = {tag: any(MODELS_DIR.glob(f"best_model_{tag}.joblib")) if MODELS_DIR.is_dir() else False
              for tag in EXP_ORDER}
    n_csv = len(list(TABLES_DIR.glob("*.csv"))) if TABLES_DIR.is_dir() else 0
    n_png = len(list(FIGURES_DIR.glob("*.png"))) if FIGURES_DIR.is_dir() else 0
    try:
        import importlib.util
        predictor = importlib.util.find_spec("src.predict") is not None
    except Exception:
        predictor = False
    return {
        "models": models,
        "tables_csv": n_csv,
        "figures_png": n_png,
        "predictor_module": predictor,
        "ready_for_prediction": predictor and all(models.get(t) for t in svc.PREDICTABLE),
    }


@app.context_processor
def inject_globals():
    experiments, sources = get_experiments()
    return {
        "experiments": experiments,
        "exp_order": EXP_ORDER,
        "data_sources": sources,
        "uses_documented_values": any(s != "csv" for s in sources.values()),
        "predictable": svc.PREDICTABLE,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Pages
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    """Tableau de bord."""
    return render_template("home.html", dataset=svc.DATASET_INFO,
                           baselines=svc.BASELINES_DOC)


@app.route("/experiments")
def experiments():
    """Comparaison détaillée des expériences."""
    cv_results = {}
    for tag in EXP_ORDER:
        rows = svc.table_records(TABLES_DIR, f"cv_results_{tag}.csv")
        if rows:
            cv_results[tag] = rows
    return render_template("experiments.html", cv_results=cv_results, dataset=svc.DATASET_INFO)


@app.route("/evaluation")
@app.route("/evaluation/<experiment>")
def evaluation(experiment="C"):
    """Évaluation détaillée d'une expérience."""
    experiments_, _ = get_experiments()
    if experiment not in experiments_:
        abort(404)
    exp = experiments_[experiment]
    tag = exp["tag"]

    return render_template(
        "evaluation.html",
        exp=exp,
        experiment=experiment,
        figures=get_figures_for_experiment(tag),
        test_metrics=svc.table_records(TABLES_DIR, f"test_metrics_{tag}.csv"),
        tld_out=svc.table_records(TABLES_DIR, f"leave_tld_out_{tag}.csv"),
        subgroups=svc.table_records(TABLES_DIR, f"subgroups_evaluation_{tag}.csv"),
        feat_importance=(svc.table_records(TABLES_DIR, "permutation_importance_C.csv")
                         if experiment == "C" else []),
        metadata=load_model_metadata(tag),
    )


@app.route("/limites")
def limits():
    """Biais du jeu de données, sous-groupes G1/G2 et limites connues (doc. §7, §11)."""
    return render_template(
        "limits.html",
        prefix_bias=svc.PREFIX_BIAS_DOC,
        subgroups=svc.SUBGROUPS_DOC,
        baselines=svc.BASELINES_DOC,
        dataset=svc.DATASET_INFO,
    )


# ── Volet 1 : Contrôle Qualité ───────────────────────────────────────────────
@app.route("/quality")
def quality():
    """Volet 1 du barème académique : Contrôle qualité des données."""
    qc_data = svc.load_quality_control(TABLES_DIR)
    return render_template("quality.html", **qc_data)


# ── Volet 2 : EDA Interactive ────────────────────────────────────────────────
@app.route("/eda")
def eda():
    """Volet 2 du barème académique : Galerie exploratoire interactive (>= 10 figures)."""
    figures = svc.load_eda_figures(FIGURES_DIR)
    return render_template("eda.html", figures=figures, dataset=svc.DATASET_INFO)


# ── Volet 3 : Validation Statistique ─────────────────────────────────────────
@app.route("/statistiques")
def statistics():
    """Volet 3 du barème académique : Validation statistique (>= 4 tests formels)."""
    tests_list = svc.load_statistical_tests(TABLES_DIR)
    return render_template("statistics.html", tests=tests_list)


# ── Volet 4 : Benchmark & Inférence par lot ──────────────────────────────────
@app.route("/benchmark", methods=["GET", "POST"])
def benchmark():
    """Volet 4 du barème académique : Benchmark de modèles et inférence par lot CSV."""
    import io
    import pandas as pd
    benchmarks = svc.load_model_benchmarks(TABLES_DIR)
    batch_results = None
    batch_error = None

    if request.method == "POST":
        file = request.files.get("csv_file")
        if not file or not file.filename:
            batch_error = "Veuillez sélectionner un fichier CSV valide."
        else:
            try:
                # Lecture tolérante du CSV
                content = file.read().decode("utf-8", errors="ignore")
                df_upload = pd.read_csv(io.StringIO(content))
                if df_upload.empty:
                    batch_error = "Le fichier CSV fourni est vide."
                else:
                    batch_results = svc.process_batch_inference_csv(df_upload, experiment="C", max_rows=500)
            except Exception as e:
                batch_error = f"Erreur lors du traitement du fichier CSV : {e}"

    return render_template(
        "benchmark.html",
        benchmarks=benchmarks,
        batch_results=batch_results,
        batch_error=batch_error,
    )


@app.route("/predict")
def predict_page():
    """Interface de prédiction d'URL."""
    return render_template("predict.html", default_experiment=svc.DEFAULT_EXPERIMENT,
                           status=artifacts_status(),
                           high_precision=svc.HIGH_PRECISION_THRESHOLD)


@app.route("/figures/<path:filename>")
def serve_figure(filename):
    """Sert les images de reports/figures/ (extensions image uniquement)."""
    if Path(filename).suffix.lower() not in ALLOWED_FIGURE_EXT:
        abort(404)
    return send_from_directory(FIGURES_DIR, filename)


# ══════════════════════════════════════════════════════════════════════════════
#  API
# ══════════════════════════════════════════════════════════════════════════════

def load_predictor():
    """Importe src.predict à la demande (remplaçable dans les tests)."""
    from src.predict import InvalidURLError, predict_url
    return predict_url, InvalidURLError


def _threshold_kwarg(predict_url):
    """Nom du paramètre de seuil accepté par predict_url, ou None."""
    try:
        params = inspect.signature(predict_url).parameters
    except (TypeError, ValueError):
        return None
    for name in ("threshold", "threshold_policy", "policy"):
        if name in params:
            return name
    return None


def _error(message, status, **extra):
    return jsonify({"success": False, "error": message, **extra}), status


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Prédiction sur une URL. Corps JSON : {url, experiment?, threshold?}."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _error("Corps JSON invalide : attendu {\"url\": \"...\"}.", 400)

    url = data.get("url")
    if not isinstance(url, str) or not url.strip():
        return _error("Veuillez fournir une URL.", 400)
    url = url.strip()
    if len(url) > svc.MAX_URL_LENGTH:
        return _error(f"URL trop longue (maximum {svc.MAX_URL_LENGTH} caractères).", 400)

    experiments_, _ = get_experiments()
    experiment = str(data.get("experiment") or svc.DEFAULT_EXPERIMENT)
    if experiment not in experiments_:
        return _error(f"Expérience inconnue : {experiment!r}.", 400,
                      allowed=list(svc.PREDICTABLE))
    if experiment not in svc.PREDICTABLE:
        return _error(
            f"L'expérience {experiment} exige des variables de page (LineOfCode, "
            "HasPasswordField…) impossibles à obtenir depuis la seule URL : la prédiction "
            "serait fausse. Utilisez C (URL seule) ou D (hôte seul).", 422,
            allowed=list(svc.PREDICTABLE))

    # Politique de seuil (doc. §8.4) : standard (0,5) | high_precision (D) | nombre
    threshold = data.get("threshold")
    if threshold in ("", "standard"):
        threshold = None
    if threshold is not None:
        if threshold == "high_precision":
            if experiment != "D":
                return _error("Le seuil « high_precision » n'est documenté que pour l'expérience D.", 400)
        elif isinstance(threshold, bool) or not isinstance(threshold, (int, float)) \
                or not 0.0 < float(threshold) < 1.0:
            return _error("Seuil invalide : « standard », « high_precision » ou un nombre "
                          "strictement compris entre 0 et 1.", 400)

    try:
        predict_url, InvalidURLError = load_predictor()
    except ImportError:
        log.exception("src.predict introuvable")
        return _error("Module de prédiction indisponible (src/predict.py introuvable).", 503)

    kwargs = {"experiment": experiment}
    if threshold is not None:
        kw = _threshold_kwarg(predict_url)
        if kw is None:
            return _error("Cette version de src.predict ne gère pas de seuil personnalisé.", 400)
        kwargs[kw] = threshold

    try:
        result = predict_url(url, **kwargs)
    except InvalidURLError as exc:
        return _error(f"URL invalide : {exc}", 400)
    except FileNotFoundError:
        log.exception("Modèle introuvable pour l'expérience %s", experiment)
        return _error(f"Modèle de l'expérience {experiment} introuvable : entraînez-le avec "
                      f"`python -m src.train --experiment {experiment}`.", 503)
    except Exception:  # ne jamais exposer la trace au client
        log.exception("Erreur de prédiction")
        return _error("Erreur interne lors de la prédiction.", 500)

    try:
        predicted_class = int(result["predicted_class"])
        payload = {
            "success": True,
            "url": result.get("url", url),
            "predicted_class": predicted_class,
            "predicted_label": result.get("predicted_label",
                                          svc.classify_label(predicted_class)),
            "model_score": round(float(result["model_score"]), 6),
            "threshold_used": float(result["threshold_used"]),
            "model_name": result.get("model_name", "RandomForest"),
            "experiment": result.get("experiment", experiment),
            "subgroup": svc.url_subgroup(url),
            "notes": svc.reliability_notes(url, experiment),
            "features": {k: (round(v, 6) if isinstance(v, float) else v)
                         for k, v in dict(result.get("features", {})).items()},
        }
    except (KeyError, TypeError, ValueError):
        log.exception("Résultat de prédiction inattendu : %r", result)
        return _error("Réponse inattendue du module de prédiction.", 500)
    return jsonify(svc.to_jsonable(payload))


@app.route("/api/experiments")
def api_experiments():
    """Métriques des expériences + origine des chiffres."""
    experiments_, sources = get_experiments()
    return jsonify(svc.to_jsonable({
        "test_size": svc.TEST_SIZE,
        "experiments": {t: {"name": e["name"], "n_features": e["n_features"],
                            "needs_network": e["needs_network"],
                            "predictable": e["predictable"],
                            "metrics": e["metrics"], "source": sources[t]}
                        for t, e in experiments_.items()},
    }))


@app.route("/health")
def health():
    """État de l'application et des artefacts."""
    status = artifacts_status()
    _, sources = get_experiments()
    return jsonify(svc.to_jsonable({"status": "ok", "artifacts": status,
                                    "metrics_source": sources}))


# ══════════════════════════════════════════════════════════════════════════════
#  En-têtes de sécurité & gestion d'erreurs
# ══════════════════════════════════════════════════════════════════════════════

CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
       "script-src 'self' 'unsafe-inline'; font-src 'self'; connect-src 'self'; "
       "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'")


@app.after_request
def set_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Content-Security-Policy", CSP)
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def _wants_json():
    return request.path.startswith("/api/")


def _render_error(code, title, message):
    if _wants_json():
        return _error(message, code)
    return render_template("error.html", code=code, title=title, message=message), code


@app.errorhandler(404)
def not_found(_e):
    return _render_error(404, "Page introuvable", "La ressource demandée n'existe pas.")


@app.errorhandler(405)
def method_not_allowed(_e):
    return _render_error(405, "Méthode non autorisée", "Cette route n'accepte pas cette méthode HTTP.")


@app.errorhandler(413)
def too_large(_e):
    return _render_error(413, "Requête trop volumineuse", "Le corps de la requête dépasse la limite autorisée.")


@app.errorhandler(500)
def server_error(_e):
    return _render_error(500, "Erreur interne", "Une erreur inattendue s'est produite.")


# ══════════════════════════════════════════════════════════════════════════════
#  Filtres Jinja2
# ══════════════════════════════════════════════════════════════════════════════

@app.template_filter("pct")
def pct_filter(value, digits=2):
    """0.9936 -> '99.36 %'."""
    try:
        return f"{float(value) * 100:.{int(digits)}f}\u00a0%"
    except (ValueError, TypeError):
        return "—"


@app.template_filter("fmt4")
def fmt4_filter(value):
    """Float avec 4 décimales."""
    try:
        return f"{float(value):.4f}"
    except (ValueError, TypeError):
        return "—"


@app.template_filter("extract")
def extract_filter(key, mapping):
    """'C' | extract(experiments) -> experiments['C'] (utilisable avec map())."""
    return mapping[key]


@app.template_filter("fmt_int")
def fmt_int_filter(value):
    """Entier avec séparateur de milliers (espace)."""
    try:
        return f"{int(value):,}".replace(",", " ")
    except (ValueError, TypeError):
        return "—"


# ══════════════════════════════════════════════════════════════════════════════
#  Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    host = os.environ.get("PHISHGUARD_HOST", "127.0.0.1")
    port = int(os.environ.get("PHISHGUARD_PORT", "5000"))
    debug = os.environ.get("PHISHGUARD_DEBUG", "0") == "1"
    print(f"\n{'=' * 60}\n  PROJ-IASD01 — Détection de Phishing\n"
          f"  Serveur web : http://{host}:{port}\n{'=' * 60}\n")
    app.run(debug=debug, host=host, port=port)
