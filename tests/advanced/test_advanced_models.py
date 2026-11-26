import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from src.research.advanced_models.evaluator import get_subgroup_masks, evaluate_predictions, cluster_bootstrap_ci
from src.research.advanced_models.calibrator import AdvancedCalibrator, compute_brier_metrics
from src.research.advanced_models.benchmark import benchmark_model_inference
from src.config import URL_ONLY_FEATURES

def test_subgroup_masks_disjointness():
    urls = pd.Series([
        "https://www.google.com",
        "https://www.google.com/",
        "https://www.google.com/search?q=test",
        "http://www.google.com",
        "https://paypal-update.com/login",
    ])
    masks = get_subgroup_masks(urls)
    assert masks["G1"][0] == True
    assert masks["G2"][0] == True  # bare host
    assert masks["G1_strict"][0] == False

    assert masks["G1"][2] == True
    assert masks["G2"][2] == False # has path and query
    assert masks["G1_strict"][2] == True

    # Check G1 = G2 + G1_strict
    assert (masks["G1"] == (masks["G2"] | masks["G1_strict"])).all()

def test_evaluator_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.6, 0.9])
    res = evaluate_predictions(y_true, y_score, threshold=0.5)
    assert res["TP"] == 2
    assert res["TN"] == 2
    assert res["FP"] == 0
    assert res["FN"] == 0
    assert res["f1"] == 1.0
    assert res["pr_auc"] == 1.0

def test_calibrator_platt_and_isotonic():
    rng = np.random.RandomState(42)
    scores = rng.uniform(0.1, 0.9, size=100)
    y = (scores > 0.5).astype(int)
    cal = AdvancedCalibrator().fit(scores, y)
    platt_preds = cal.predict_platt(scores)
    iso_preds = cal.predict_isotonic(scores)
    assert len(platt_preds) == 100
    assert len(iso_preds) == 100
    assert (platt_preds >= 0).all() and (platt_preds <= 1).all()
    assert (iso_preds >= 0).all() and (iso_preds <= 1).all()

def test_advanced_models_artifacts_exist():
    adv_dir = Path("models/advanced")
    target_file = adv_dir / "randomforest_baseline_url_only.joblib"
    if not target_file.exists():
        pytest.skip(f"Artefact {target_file} absent de l'environnement local.")
    assert (adv_dir / "randomforest_baseline_url_only.joblib").exists()
    assert (adv_dir / "xgboost_url_only.joblib").exists()
    assert (adv_dir / "lightgbm_url_only.joblib").exists()
    assert (adv_dir / "catboost_url_only.joblib").exists()
    assert (adv_dir / "calibration" / "xgboost_calibrator.joblib").exists()
