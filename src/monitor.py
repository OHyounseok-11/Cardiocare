"""CardioCare 모니터링, 로깅, 데이터 드리프트 탐지."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import CONTINUOUS_FEATURES, prepare_dataset  # noqa: E402
from src.train import RANDOM_STATE, TEST_SIZE, build_model_pipeline, get_model_candidates  # noqa: E402

LOG_DIR = ROOT / "logs"
PLOT_DIR = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "cardiocare_pipeline.joblib"
HOLDOUT_PATH = ROOT / "models" / "holdout_test.csv"


def setup_file_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cardiocare.monitor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_or_train_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    from sklearn.model_selection import train_test_split

    X, y = prepare_dataset()
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    pipeline = build_model_pipeline(get_model_candidates()["random_forest"])
    pipeline.fit(X_train, y_train)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


def load_metadata() -> dict:
    meta_path = ROOT / "models" / "metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {"model_family": "random_forest", "random_state": RANDOM_STATE}


def shift_continuous_features(
    df: pd.DataFrame,
    feature: str = "chol",
    mean_shift: float = 30.0,
    variance_scale: float = 1.5,
) -> pd.DataFrame:
    shifted = df.copy()
    values = shifted[feature].astype(float)
    current_mean = values.mean()
    current_std = values.std(ddof=0)
    if current_std == 0:
        current_std = 1.0
    standardized = (values - current_mean) / current_std
    new_std = current_std * variance_scale
    new_mean = current_mean + mean_shift
    shifted[feature] = standardized * new_std + new_mean
    return shifted


def run_drift_analysis() -> dict:
    logger = setup_file_logger(LOG_DIR / "inference_monitor.log")
    metadata = load_metadata()
    model = load_or_train_model()

    X, y = prepare_dataset()
    train_reference = X.copy()

    if HOLDOUT_PATH.exists():
        holdout = pd.read_csv(HOLDOUT_PATH)
        X_test = holdout.drop(columns=["target"])
        y_test = holdout["target"]
    else:
        from sklearn.model_selection import train_test_split

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

    drifted = shift_continuous_features(X_test, feature="chol", mean_shift=30.0, variance_scale=1.5)

    baseline_pred = model.predict(X_test)
    drift_pred = model.predict(drifted)
    baseline_acc = balanced_accuracy_score(y_test, baseline_pred)
    drift_acc = balanced_accuracy_score(y_test, drift_pred)

    ks_results = []
    flagged: list[str] = []
    for feature in CONTINUOUS_FEATURES:
        ref = train_reference[feature].dropna().astype(float)
        cur = drifted[feature].dropna().astype(float)
        stat, p_value = ks_2samp(ref, cur)
        is_flagged = bool(p_value < 0.05)
        if is_flagged:
            flagged.append(feature)
        ks_results.append(
            {
                "feature": feature,
                "ks_statistic": float(stat),
                "p_value": float(p_value),
                "flagged": is_flagged,
            }
        )

    timestamp = datetime.now().isoformat()
    log_record = {
        "timestamp": timestamp,
        "model_version": metadata.get("model_family", "unknown"),
        "input_shape": list(X_test.shape),
        "predictions_sample": baseline_pred[:5].tolist(),
        "actual_sample": y_test.iloc[:5].tolist(),
        "baseline_balanced_accuracy": baseline_acc,
        "drift_balanced_accuracy": drift_acc,
        "flagged_features": flagged,
    }
    logger.info(json.dumps(log_record, ensure_ascii=False))

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_over_time = []
    base_time = datetime.now()
    for idx, acc in enumerate(
        np.linspace(baseline_acc, drift_acc, num=8)
    ):
        metrics_over_time.append(
            {
                "timestamp": (base_time + timedelta(days=idx)).isoformat(),
                "balanced_accuracy": float(acc),
            }
        )

    ts_df = pd.DataFrame(metrics_over_time)
    plt.figure(figsize=(8, 4))
    plt.plot(
        pd.to_datetime(ts_df["timestamp"]),
        ts_df["balanced_accuracy"],
        marker="o",
    )
    plt.axhline(baseline_acc, color="green", linestyle="--", label="baseline")
    plt.axhline(drift_acc, color="red", linestyle="--", label="drift")
    plt.title("CardioCare Balanced Accuracy (synthetic timeline)")
    plt.xlabel("Timestamp")
    plt.ylabel("Balanced Accuracy")
    plt.legend()
    plt.tight_layout()
    plot_path = PLOT_DIR / "drift_metric_timeseries.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()

    report = {
        "ks_tests": ks_results,
        "baseline_balanced_accuracy": baseline_acc,
        "drift_balanced_accuracy": drift_acc,
        "accuracy_drop": baseline_acc - drift_acc,
        "flagged_features": flagged,
        "timeseries_plot": str(plot_path),
        "log_file": str(LOG_DIR / "inference_monitor.log"),
    }

    with open(PLOT_DIR / "drift_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== KS 검정 결과 ===")
    for row in ks_results:
        flag = " [FLAGGED]" if row["flagged"] else ""
        print(
            f"{row['feature']:10s} KS={row['ks_statistic']:.4f} "
            f"p={row['p_value']:.4f}{flag}"
        )

    print("\n=== 성능 비교 ===")
    print(f"원본 테스트 balanced accuracy : {baseline_acc:.4f}")
    print(f"드리프트 테스트 balanced accuracy: {drift_acc:.4f}")
    print(f"성능 하락: {baseline_acc - drift_acc:.4f}")
    print(f"\n플래그된 특성: {flagged}")
    print(f"시계열 그래프: {plot_path}")
    print(f"로그 파일: {LOG_DIR / 'inference_monitor.log'}")

    return report


if __name__ == "__main__":
    run_drift_analysis()
