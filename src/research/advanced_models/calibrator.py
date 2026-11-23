"""
Calibration and threshold optimization module.
Out-of-fold calibration (Platt scaling & Isotonic regression) strictly preserving GroupKFold on Host.
Computes Brier Score before and after calibration and extracts calibration curves.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_recall_curve


class AdvancedCalibrator:
    """
    Fits and manages Platt Scaling and Isotonic Regression calibrators
    trained strictly on out-of-fold predictions.
    """

    def __init__(self):
        self.platt_model = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
        self.isotonic_model = IsotonicRegression(out_of_bounds="clip")
        self.fitted_ = False
        self.optimal_threshold_raw_ = 0.5
        self.optimal_threshold_calibrated_ = 0.5

    def fit(self, oof_scores: np.ndarray, y_train: np.ndarray) -> "AdvancedCalibrator":
        """
        Fits calibrators strictly on OOF scores and true labels from Train CV.
        """
        oof_scores = np.asarray(oof_scores, dtype=float)
        y_train = np.asarray(y_train, dtype=int)

        # Platt scaling (logistic on log-odds / logits or raw probs)
        # To avoid numerical issues, clip probabilities slightly away from 0 and 1
        clipped = np.clip(oof_scores, 1e-7, 1.0 - 1e-7)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        self.platt_model.fit(logits, y_train)

        # Isotonic regression on raw probabilities
        self.isotonic_model.fit(oof_scores, y_train)

        # Select optimal threshold on OOF to maximize F1-score
        thresholds = np.linspace(0.01, 0.99, 99)
        f1s_raw = [f1_score(y_train, (oof_scores >= t).astype(int), zero_division=0) for t in thresholds]
        self.optimal_threshold_raw_ = float(thresholds[np.argmax(f1s_raw)])

        cal_oof = self.predict_platt(oof_scores)
        f1s_cal = [f1_score(y_train, (cal_oof >= t).astype(int), zero_division=0) for t in thresholds]
        self.optimal_threshold_calibrated_ = float(thresholds[np.argmax(f1s_cal)])

        self.fitted_ = True
        return self

    def predict_platt(self, scores: np.ndarray) -> np.ndarray:
        """Transforms scores using Platt scaling."""
        scores = np.asarray(scores, dtype=float)
        clipped = np.clip(scores, 1e-7, 1.0 - 1e-7)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return self.platt_model.predict_proba(logits)[:, 1]

    def predict_isotonic(self, scores: np.ndarray) -> np.ndarray:
        """Transforms scores using Isotonic regression."""
        scores = np.asarray(scores, dtype=float)
        return self.isotonic_model.predict(scores)


def compute_brier_metrics(
    y_true: np.ndarray,
    scores_raw: np.ndarray,
    scores_platt: np.ndarray,
    scores_iso: np.ndarray,
) -> Dict[str, float]:
    """Computes Brier Score Loss for raw, Platt, and Isotonic probabilities."""
    return {
        "brier_raw": float(brier_score_loss(y_true, scores_raw)),
        "brier_platt": float(brier_score_loss(y_true, scores_platt)),
        "brier_isotonic": float(brier_score_loss(y_true, scores_iso)),
    }
