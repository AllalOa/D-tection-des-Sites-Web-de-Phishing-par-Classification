"""
Unified Advanced Models Pipeline Runner:
1. Loads URL_ONLY dataset with frozen GroupShuffleSplit on Host (Train: 188,492 / Test: 46,878).
2. Performs 5-Fold GroupKFold on Host to obtain strictly out-of-fold predictions.
3. Fits Baseline C (Random Forest), XGBoost, LightGBM, and CatBoost on the exact same folds.
4. Fits Calibrators (Platt & Isotonic) strictly on OOF scores and freezes optimal decision thresholds.
5. Trains final models on the entire Train partition (188,492 obs) with column transformer.
6. Evaluates on independent Test partition (46,878 obs) + Subgroups (G1, G2, G1_strict).
7. Profiles latency and memory with benchmark_model_inference.
8. Persists all artifacts to models/advanced/ and reports/advanced/.
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Core ML
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

# Boosting libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Project modules
from src.config import (
    URL_FEATURES_CSV,
    URL_ONLY_FEATURES,
    GROUP_COL,
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE,
    N_FOLDS,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.host_features import extract_host
from src.research.advanced_models.evaluator import (
    get_subgroup_masks,
    cluster_bootstrap_ci,
    evaluate_predictions,
)
from src.research.advanced_models.calibrator import (
    AdvancedCalibrator,
    compute_brier_metrics,
)
from src.research.advanced_models.benchmark import benchmark_model_inference

# Define 11 heavy-tailed count columns for log1p transform (historical Exp C pipeline)
HEAVY_TAIL_COLS = [
    "URLLength",
    "DomainLength",
    "TLDLength",
    "NoOfSubDomain",
    "NoOfObfuscatedChar",
    "NoOfLettersInURL",
    "NoOfDegitsInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
]
PASSTHROUGH_COLS = [c for c in URL_ONLY_FEATURES if c not in HEAVY_TAIL_COLS]


def build_preprocessor() -> ColumnTransformer:
    """Builds historical ColumnTransformer with log1p on heavy-tailed count features."""
    return ColumnTransformer(
        transformers=[
            ("log1p", FunctionTransformer(np.log1p, validate=True), HEAVY_TAIL_COLS),
            ("passthrough", "passthrough", PASSTHROUGH_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def load_dataset_and_split():
    """Loads url_features_all.csv and creates strict GroupShuffleSplit by Host."""
    print(f"[pipeline] Loading dataset from: {URL_FEATURES_CSV}")
    df = pd.read_csv(URL_FEATURES_CSV)
    print(f"[pipeline] Raw shape: {df.shape}")

    # Ensure Host and row_id exist
    if "Host" not in df.columns:
        if "URL" in df.columns:
            df["Host"] = df["URL"].apply(extract_host)
        else:
            raise ValueError("Neither Host nor URL column found in dataset.")

    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df))

    # Split
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(splitter.split(df, df[TARGET_COL], groups=df[GROUP_COL].values))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    print(f"[pipeline] Train partition: {df_train.shape} (phishing={df_train[TARGET_COL].sum()})")
    print(f"[pipeline] Test partition:  {df_test.shape} (phishing={df_test[TARGET_COL].sum()})")

    overlap = set(df_train[GROUP_COL]) & set(df_test[GROUP_COL])
    assert len(overlap) == 0, f"Host overlap detected: {len(overlap)} hosts!"
    print(f"[pipeline] Host leak check passed: 0 hosts in common between train and test.")

    return df_train, df_test


def get_model_specs():
    """Returns candidate estimators with competitive hyperparameter configurations."""
    return {
        "RandomForest_Baseline": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method="hist",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300,
            num_leaves=63,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=350,
            depth=7,
            learning_rate=0.08,
            random_seed=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
            thread_count=-1,
        ),
    }


def run_experiment():
    """Executes the complete Phase 4 comparative experiment."""
    df_train, df_test = load_dataset_and_split()

    X_train_df = df_train[URL_ONLY_FEATURES]
    y_train = df_train[TARGET_COL].values
    groups_train = df_train[GROUP_COL].values

    X_test_df = df_test[URL_ONLY_FEATURES]
    y_test = df_test[TARGET_COL].values

    # Preprocessing
    preprocessor = build_preprocessor()
    print("[pipeline] Fitting ColumnTransformer on Train...")
    X_train_proc = preprocessor.fit_transform(X_train_df)
    X_test_proc = preprocessor.transform(X_test_df)

    gkf = GroupKFold(n_splits=N_FOLDS)
    model_specs = get_model_specs()

    output_dir_models = Path("models/advanced")
    output_dir_cal = output_dir_models / "calibration"
    output_dir_models.mkdir(parents=True, exist_ok=True)
    output_dir_cal.mkdir(parents=True, exist_ok=True)

    summary_records = []
    subgroup_records = []
    calibration_records = []
    benchmark_records = []

    subgroup_masks = get_subgroup_masks(df_test["URL"])

    for model_name, clf in model_specs.items():
        print(f"\n{'='*60}\n[pipeline] Running Experiment: {model_name}\n{'='*60}")
        t_start = time.perf_counter()

        # 1. Out-of-Fold (OOF) cross-validation on Train
        print(f"[{model_name}] Computing 5-Fold GroupKFold OOF predictions...")
        oof_scores = np.zeros(len(df_train), dtype=float)
        fold_aps = []

        for fold, (t_idx, v_idx) in enumerate(gkf.split(X_train_proc, y_train, groups=groups_train)):
            clf_fold = getattr(clf, "clone", None)
            if clf_fold is not None:
                clf_fold = clf_fold()
            else:
                from sklearn.base import clone
                clf_fold = clone(clf)

            clf_fold.fit(X_train_proc[t_idx], y_train[t_idx])
            val_scores = clf_fold.predict_proba(X_train_proc[v_idx])[:, 1]
            oof_scores[v_idx] = val_scores
            ap_fold = average_precision_score(y_train[v_idx], val_scores)
            fold_aps.append(ap_fold)
            print(f"  Fold {fold+1}/{N_FOLDS} AP = {ap_fold:.6f}")

        cv_ap_mean = float(np.mean(fold_aps))
        cv_ap_std = float(np.std(fold_aps))
        print(f"[{model_name}] Train CV AP: {cv_ap_mean:.6f} +/- {cv_ap_std:.6f}")

        # 2. Calibration on OOF
        print(f"[{model_name}] Fitting Out-of-Fold calibrators...")
        calibrator = AdvancedCalibrator().fit(oof_scores, y_train)
        brier_train_dict = compute_brier_metrics(
            y_train,
            oof_scores,
            calibrator.predict_platt(oof_scores),
            calibrator.predict_isotonic(oof_scores),
        )
        print(f"  OOF Brier Score - Raw: {brier_train_dict['brier_raw']:.6f} | "
              f"Platt: {brier_train_dict['brier_platt']:.6f} | "
              f"Isotonic: {brier_train_dict['brier_isotonic']:.6f}")
        print(f"  Optimal OOF Decision Thresholds - Raw: {calibrator.optimal_threshold_raw_:.4f} | "
              f"Calibrated: {calibrator.optimal_threshold_calibrated_:.4f}")

        # 3. Train final model on full Train set
        print(f"[{model_name}] Training final model on entire train set ({len(X_train_proc)} rows)...")
        t_train_start = time.perf_counter()
        clf.fit(X_train_proc, y_train)
        train_duration_s = time.perf_counter() - t_train_start
        print(f"[{model_name}] Training complete in {train_duration_s:.2f} s")

        # Create full inference pipeline (ColumnTransformer + Estimator)
        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        # 4. Predict on Test set
        y_test_raw_scores = clf.predict_proba(X_test_proc)[:, 1]
        y_test_platt_scores = calibrator.predict_platt(y_test_raw_scores)
        y_test_iso_scores = calibrator.predict_isotonic(y_test_raw_scores)

        # Compute Brier scores on Test
        brier_test_dict = compute_brier_metrics(
            y_test,
            y_test_raw_scores,
            y_test_platt_scores,
            y_test_iso_scores,
        )

        calibration_records.append({
            "model": model_name,
            "train_oof_brier_raw": brier_train_dict["brier_raw"],
            "train_oof_brier_platt": brier_train_dict["brier_platt"],
            "train_oof_brier_isotonic": brier_train_dict["brier_isotonic"],
            "test_brier_raw": brier_test_dict["brier_raw"],
            "test_brier_platt": brier_test_dict["brier_platt"],
            "test_brier_isotonic": brier_test_dict["brier_isotonic"],
            "optimal_threshold_raw": calibrator.optimal_threshold_raw_,
            "optimal_threshold_calibrated": calibrator.optimal_threshold_calibrated_,
        })

        # 5. Evaluate overall Test performance
        # Evaluation at standard 0.5 threshold
        eval_std = evaluate_predictions(y_test, y_test_raw_scores, threshold=0.5)
        # Evaluation at optimal OOF-selected threshold
        eval_opt = evaluate_predictions(y_test, y_test_raw_scores, threshold=calibrator.optimal_threshold_raw_)

        summary_records.append({
            "model": model_name,
            "train_cv_ap_mean": cv_ap_mean,
            "train_cv_ap_std": cv_ap_std,
            "train_duration_s": train_duration_s,
            "test_pr_auc": eval_std["pr_auc"],
            "test_roc_auc": eval_std["roc_auc"],
            "test_f1_std_05": eval_std["f1"],
            "test_precision_std_05": eval_std["precision"],
            "test_recall_std_05": eval_std["recall"],
            "test_TP_05": eval_std["TP"],
            "test_FP_05": eval_std["FP"],
            "test_TN_05": eval_std["TN"],
            "test_FN_05": eval_std["FN"],
            "optimal_threshold_oof": calibrator.optimal_threshold_raw_,
            "test_f1_at_opt": eval_opt["f1"],
            "test_precision_at_opt": eval_opt["precision"],
            "test_recall_at_opt": eval_opt["recall"],
            "test_TP_opt": eval_opt["TP"],
            "test_FP_opt": eval_opt["FP"],
            "test_TN_opt": eval_opt["TN"],
            "test_FN_opt": eval_opt["FN"],
            "test_brier_raw": brier_test_dict["brier_raw"],
            "test_brier_platt": brier_test_dict["brier_platt"],
            "test_brier_isotonic": brier_test_dict["brier_isotonic"],
        })

        # 6. Evaluate Subgroups (G1, G2, G1_strict)
        for sg_name, sg_mask in subgroup_masks.items():
            df_sg = df_test[sg_mask].copy().reset_index(drop=True)
            y_sg = df_sg[TARGET_COL].values
            y_sg_scores = y_test_raw_scores[sg_mask.values]

            sg_eval = evaluate_predictions(y_sg, y_sg_scores, threshold=0.5)
            pr_ci, roc_ci = cluster_bootstrap_ci(df_sg, y_sg, y_sg_scores, n_boot=500, seed=RANDOM_STATE)

            subgroup_records.append({
                "model": model_name,
                "subgroup": sg_name,
                "n_samples": len(df_sg),
                "n_phishing": int(y_sg.sum()),
                "prevalence": float(y_sg.mean()),
                "pr_auc": sg_eval["pr_auc"],
                "pr_auc_ci_lower": pr_ci[0],
                "pr_auc_ci_upper": pr_ci[1],
                "roc_auc": sg_eval["roc_auc"],
                "roc_auc_ci_lower": roc_ci[0],
                "roc_auc_ci_upper": roc_ci[1],
                "f1_05": sg_eval["f1"],
                "precision_05": sg_eval["precision"],
                "recall_05": sg_eval["recall"],
                "TP_05": sg_eval["TP"],
                "FP_05": sg_eval["FP"],
                "TN_05": sg_eval["TN"],
                "FN_05": sg_eval["FN"],
            })

        # 7. Benchmarking Latency & Memory
        print(f"[{model_name}] Profiling latency on single URL, batch 10, batch 100, batch 1000...")
        for batch_sz in [1, 10, 100, 1000]:
            sample_df = X_test_df.iloc[:batch_sz].copy()
            bench_res = benchmark_model_inference(full_pipeline, sample_df, n_repetitions=30)
            bench_res["model"] = model_name
            benchmark_records.append(bench_res)

        # 8. Persist artifacts
        tag = model_name.lower()
        model_path = output_dir_models / f"{tag}_url_only.joblib"
        cal_path = output_dir_cal / f"{tag}_calibrator.joblib"
        meta_path = output_dir_models / f"{tag}_url_only_metadata.json"

        joblib.dump(full_pipeline, model_path)
        joblib.dump(calibrator, cal_path)

        metadata = {
            "model_name": model_name,
            "experiment": "C_ADVANCED",
            "features": URL_ONLY_FEATURES,
            "train_cv_ap_mean": cv_ap_mean,
            "train_cv_ap_std": cv_ap_std,
            "train_duration_seconds": train_duration_s,
            "test_pr_auc": eval_std["pr_auc"],
            "test_roc_auc": eval_std["roc_auc"],
            "test_f1": eval_std["f1"],
            "optimal_threshold": calibrator.optimal_threshold_raw_,
            "optimal_threshold_calibrated": calibrator.optimal_threshold_calibrated_,
            "brier_test_raw": brier_test_dict["brier_raw"],
            "brier_test_platt": brier_test_dict["brier_platt"],
            "brier_test_isotonic": brier_test_dict["brier_isotonic"],
            "date": pd.Timestamp.now().isoformat(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(f"[{model_name}] Artifacts successfully saved.")

    # Convert records to DataFrames and save
    reports_adv = Path("reports/advanced")
    reports_adv.mkdir(parents=True, exist_ok=True)

    df_summary = pd.DataFrame(summary_records)
    df_subgroups = pd.DataFrame(subgroup_records)
    df_calibration = pd.DataFrame(calibration_records)
    df_benchmark = pd.DataFrame(benchmark_records)

    df_summary.to_csv(reports_adv / "models_summary_comparison.csv", index=False)
    df_subgroups.to_csv(reports_adv / "subgroups_comparison.csv", index=False)
    df_calibration.to_csv(reports_adv / "calibration_comparison.csv", index=False)
    df_benchmark.to_csv(reports_adv / "benchmark_comparison.csv", index=False)

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    return df_summary, df_subgroups, df_calibration, df_benchmark


if __name__ == "__main__":
    run_experiment()
