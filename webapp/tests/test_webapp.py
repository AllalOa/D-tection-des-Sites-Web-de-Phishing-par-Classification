"""
test_webapp.py — tests automatisés de l'application Flask PhishGuard.

Ne nécessitent NI modèles entraînés NI jeu de données : le prédicteur est simulé
et les CSV sont écrits dans un dossier temporaire. À placer dans tests/ :
    pytest tests/test_webapp.py -v
"""

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")
pytest.importorskip("flask")

from webapp import app as webapp_module          # noqa: E402
from webapp import services as svc               # noqa: E402

PAGES = ["/", "/experiments", "/limites", "/predict",
         "/evaluation", "/evaluation/A", "/evaluation/B", "/evaluation/B_prime",
         "/evaluation/C", "/evaluation/D"]


class InvalidURLError(ValueError):
    pass


def fake_predict_url(url, experiment="C", threshold=None):
    """Prédicteur simulé : renvoie volontairement des types numpy."""
    if "invalid" in url:
        raise InvalidURLError("hôte manquant")
    if "nomodel" in url:
        raise FileNotFoundError("/secret/path/best_model_C.joblib")
    if "boom" in url:
        raise RuntimeError("détail interne confidentiel")
    return {"url": url, "predicted_class": np.int64(1), "predicted_label": "phishing",
            "model_score": np.float64(0.97), "threshold_used": 0.5,
            "model_name": "RandomForest", "experiment": experiment,
            "features": {"IsHTTPS": np.int64(0), "URLLength": np.float64(31.0)}}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Client Flask sans artefacts (tables vides) et avec prédicteur simulé."""
    monkeypatch.setattr(webapp_module, "TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(webapp_module, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(webapp_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(webapp_module, "load_predictor",
                        lambda: (fake_predict_url, InvalidURLError))
    webapp_module.app.config["TESTING"] = True
    return webapp_module.app.test_client()


def post(client, payload):
    return client.post("/api/predict", json=payload)


# ── Pages ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", PAGES)
def test_pages_render_without_artifacts(client, path):
    """Toutes les pages s'affichent même sans aucun CSV / modèle (états vides)."""
    assert client.get(path).status_code == 200


def test_unknown_experiment_is_404(client):
    assert client.get("/evaluation/Z").status_code == 404


def test_404_is_json_for_api_and_html_otherwise(client):
    assert client.get("/api/nope").get_json()["success"] is False
    assert b"Page introuvable" in client.get("/nope").data


def test_pages_have_no_external_dependency(client):
    """Application utilisable hors ligne : aucune ressource CDN."""
    for path in PAGES:
        html = client.get(path).get_data(as_text=True)
        assert not re.search(r'(?:src|href)="https?://', html), path


def test_security_headers(client):
    r = client.get("/")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert "default-src 'self'" in r.headers["Content-Security-Policy"]
    assert client.get("/health").status_code == 200


def test_secret_key_not_hardcoded():
    assert webapp_module.app.config["SECRET_KEY"] != "proj-iasd01-2026"


def test_figures_route_serves_images_only(client, tmp_path):
    fig = tmp_path / "figures"
    fig.mkdir()
    (fig / "a.png").write_bytes(b"\x89PNG")
    (fig / "secret.csv").write_text("x")
    assert client.get("/figures/a.png").status_code == 200
    assert client.get("/figures/secret.csv").status_code == 404
    assert client.get("/figures/../../etc/passwd").status_code == 404


# ── Cohérence des données affichées ───────────────────────────────────────────

def test_dataset_numbers_are_consistent():
    d = svc.DATASET_INFO
    assert d["train_size"] + d["test_size"] == d["total_rows"]
    assert d["n_phishing"] + d["n_legitimate"] == d["total_rows"]
    assert d["raw_rows"] - d["duplicates_removed"] == d["total_rows"]
    assert round(100 * d["n_phishing"] / d["total_rows"], 2) == d["phishing_pct"]
    assert round(100 * d["n_legitimate"] / d["total_rows"], 2) == d["legitimate_pct"]


def test_fallback_metrics_match_protocol(tmp_path):
    experiments, sources = svc.load_experiments(tmp_path)
    assert set(sources.values()) == {"documentation"}
    for tag, exp in experiments.items():
        m = exp["metrics"]
        assert m["tp"] + m["fn"] == svc.TEST_N_PHISHING, tag
        assert m["tn"] + m["fp"] == svc.TEST_N_LEGIT, tag
    # F1 documentés (§6.1)
    expected = {"A": 1.0, "B": 0.9999, "B_prime": 0.9994, "C": 0.9936, "D": 0.7404}
    for tag, f1 in expected.items():
        assert experiments[tag]["metrics"]["f1"] == pytest.approx(f1, abs=5e-5), tag
    assert experiments["C"]["n_errors"] == 251     # 62 FP + 189 FN (§6.2)


def _write_test_metrics(tables, tag, tp, fp, tn, fn, model="RandomForest"):
    tables.mkdir(exist_ok=True)
    pd.DataFrame([{"model": model, "pr_auc": 0.99, "roc_auc": 0.98,
                   "TP": tp, "FP": fp, "TN": tn, "FN": fn}]
                 ).to_csv(tables / f"test_metrics_{tag}.csv", index=False)


def test_csv_metrics_override_fallback_and_are_recomputed(tmp_path):
    _write_test_metrics(tmp_path, "C", 19700, 30, 27022, 126)
    exp, src = svc.load_experiments(tmp_path)
    assert src["C"] == "csv" and src["D"] == "documentation"
    m = exp["C"]["metrics"]
    assert m["fp"] == 30 and m["precision"] == pytest.approx(19700 / 19730)


def test_inconsistent_csv_is_rejected(tmp_path):
    _write_test_metrics(tmp_path, "C", 5, 5, 5, 5)       # n'a pas la taille du jeu de test
    _, src = svc.load_experiments(tmp_path)
    assert src["C"] == "documentation"


def test_banner_shown_when_fallback_used(client):
    assert "Métriques de repli" in client.get("/").get_data(as_text=True)


def test_html_is_escaped_in_csv_content(client, tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    pd.DataFrame([{"tld": "<script>alert(1)</script>", "n_samples": 1, "recall": 0.9}]
                 ).to_csv(tables / "leave_tld_out_C.csv", index=False)
    pd.DataFrame([{"feature": "</script><script>alert(2)", "importance_mean": 0.1,
                   "importance_std": 0.0}]).to_csv(tables / "permutation_importance_C.csv", index=False)
    html = client.get("/evaluation/C").get_data(as_text=True)
    assert "<script>alert(1)" not in html
    assert "</script><script>alert(2)" not in html


# ── API de prédiction ─────────────────────────────────────────────────────────

def test_predict_success_serialises_numpy_types(client):
    r = post(client, {"url": "http://evil.example/login"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] and body["predicted_class"] == 1
    assert body["experiment"] == "C"                # C par défaut (jamais A/B/B′)
    assert body["features"]["IsHTTPS"] == 0


@pytest.mark.parametrize("payload", [{}, {"url": ""}, {"url": "   "}, {"url": 5}, {"url": None}])
def test_predict_rejects_missing_url(client, payload):
    assert post(client, payload).status_code == 400


def test_predict_rejects_non_object_body(client):
    assert client.post("/api/predict", data="x", content_type="text/plain").status_code == 400
    assert post(client, [1, 2]).status_code == 400


def test_predict_rejects_too_long_url(client):
    r = post(client, {"url": "http://a.com/" + "x" * 2100})
    assert r.status_code == 400


@pytest.mark.parametrize("exp", ["A", "B", "B_prime"])
def test_page_based_experiments_are_refused(client, exp):
    """A/B/B′ : prédiction impossible depuis la seule URL (doc. §8.5) -> 422."""
    r = post(client, {"url": "http://a.com", "experiment": exp})
    assert r.status_code == 422 and "variables de page" in r.get_json()["error"]


def test_unknown_experiment(client):
    assert post(client, {"url": "http://a.com", "experiment": "ZZ"}).status_code == 400


@pytest.mark.parametrize("thr, code", [("high_precision", 400), (0, 400), (1.5, 400),
                                       ("abc", 400), (True, 400)])
def test_threshold_validation_on_experiment_c(client, thr, code):
    assert post(client, {"url": "http://a.com", "threshold": thr}).status_code == code


def test_threshold_policy_is_forwarded_to_predictor(client, monkeypatch):
    seen = {}

    def recording_predict(url, experiment="C", threshold=None):
        seen.update(experiment=experiment, threshold=threshold)
        return fake_predict_url(url, experiment=experiment)

    monkeypatch.setattr(webapp_module, "load_predictor",
                        lambda: (recording_predict, InvalidURLError))
    r = post(client, {"url": "http://a.com", "experiment": "D", "threshold": "high_precision"})
    assert r.status_code == 200
    assert seen == {"experiment": "D", "threshold": "high_precision"}


def test_threshold_refused_if_predictor_does_not_support_it(client, monkeypatch):
    monkeypatch.setattr(webapp_module, "load_predictor",
                        lambda: (lambda url, experiment="C": fake_predict_url(url), InvalidURLError))
    r = post(client, {"url": "http://a.com", "experiment": "D", "threshold": "high_precision"})
    assert r.status_code == 400


def test_error_mapping_does_not_leak_internals(client):
    assert post(client, {"url": "http://invalid"}).status_code == 400
    r = post(client, {"url": "http://nomodel"})
    assert r.status_code == 503 and "/secret/path" not in r.get_data(as_text=True)
    r = post(client, {"url": "http://boom"})
    assert r.status_code == 500 and "confidentiel" not in r.get_data(as_text=True)


def test_predictor_module_missing_gives_503(client, monkeypatch):
    def broken():
        raise ImportError("no module")
    monkeypatch.setattr(webapp_module, "load_predictor", broken)
    assert post(client, {"url": "http://a.com"}).status_code == 503


def test_api_experiments_and_health(client):
    data = client.get("/api/experiments").get_json()
    assert data["test_size"] == 46878 and set(data["experiments"]) == set(svc.EXP_ORDER)
    assert data["experiments"]["C"]["predictable"] is True
    assert data["experiments"]["A"]["predictable"] is False
    assert client.get("/health").get_json()["status"] == "ok"


# ── Règles de fiabilité (biais §7) ────────────────────────────────────────────

@pytest.mark.parametrize("url, group", [
    ("https://www.service-login.com/", "G2"),
    ("https://www.service-login.com", "G2"),
    ("https://www.example.com/login", "G1"),
    ("https://www.example.com/?a=1", "G1"),
    ("https://example.com/", None),
    ("http://www.example.com/", None),
    ("HTTPS://www.example.com/", None),       # sensible à la casse (doc. §7.1)
])
def test_url_subgroup(url, group):
    assert svc.url_subgroup(url) == group


def test_g2_warning_is_returned_by_api(client):
    notes = post(client, {"url": "https://www.service-login.com/"}).get_json()["notes"]
    assert any(n["code"] == "G2" and n["level"] == "warning" for n in notes)


def test_to_jsonable_handles_numpy_and_nan():
    out = svc.to_jsonable({"a": np.int64(3), "b": np.float64("nan"), "c": [np.bool_(True)]})
    assert out == {"a": 3, "b": None, "c": [True]}
