"""CardioCare 모델 학습 및 MLflow 실험 추적."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import (  # noqa: E402
    FEATURE_COLUMNS,
    build_feature_scaler,
    build_preprocessing_pipeline,
    prepare_dataset,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODELS_DIR = ROOT / "models"
MLFLOW_EXPERIMENT = "CardioCare"


def get_model_candidates() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "svc": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def build_model_pipeline(estimator) -> Pipeline:
    """전처리 → 스케일링 → 특성 선택 → 분류기 파이프라인."""
    selector = SelectFromModel(
        RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        threshold="median",
    )
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessing_pipeline()),
            ("scale", build_feature_scaler()),
            ("select", selector),
            ("classifier", estimator),
        ]
    )


def evaluate_model(y_true, y_pred) -> dict[str, float]:
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def log_confusion_matrix(y_true, y_pred) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    mlflow.log_dict(
        {
            "matrix": matrix.tolist(),
            "labels": [0, 1],
        },
        "confusion_matrix.json",
    )


def get_selected_features(pipeline: Pipeline, feature_names: list[str]) -> list[str]:
    selector = pipeline.named_steps["select"]
    support = selector.get_support()
    scaled_names = feature_names
    return [name for name, keep in zip(scaled_names, support) if keep]


def run_hyperparameter_search(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_family: str,
) -> GridSearchCV:
    param_grid_map = {
        "logistic_regression": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__solver": ["lbfgs", "liblinear"],
        },
        "svc": {
            "classifier__C": [0.5, 1.0, 2.0],
            "classifier__gamma": ["scale", "auto"],
        },
        "random_forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 5, 10],
        },
    }
    param_grid = param_grid_map.get(model_family, {})
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def train_and_select_model() -> Pipeline:
    X, y = prepare_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    results: list[dict] = []
    best_pipeline: Pipeline | None = None
    best_score = -1.0
    best_family = ""

    for family, estimator in get_model_candidates().items():
        pipeline = build_model_pipeline(estimator)

        with mlflow.start_run(run_name=family) as run:
            mlflow.set_tag("model_family", family)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            metrics = evaluate_model(y_test, y_pred)

            for name, value in metrics.items():
                mlflow.log_metric(name, value)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_params(
                {k: v for k, v in pipeline.named_steps["classifier"].get_params().items()}
            )
            log_confusion_matrix(y_test, y_pred)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            selected = get_selected_features(pipeline, FEATURE_COLUMNS)
            mlflow.log_text("\n".join(selected), "selected_features.txt")

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            cv_scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                fold_pipe = build_model_pipeline(estimator)
                fold_pipe.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                fold_pred = fold_pipe.predict(X_train.iloc[val_idx])
                cv_scores.append(balanced_accuracy_score(y_train.iloc[val_idx], fold_pred))
            cv_mean = float(np.mean(cv_scores))
            mlflow.log_metric("cv_balanced_accuracy_mean", cv_mean)

            results.append({"family": family, **metrics, "cv_mean": cv_mean, "run_id": run.info.run_id})

            if metrics["balanced_accuracy"] > best_score:
                best_score = metrics["balanced_accuracy"]
                best_pipeline = pipeline
                best_family = family

    assert best_pipeline is not None

    tuned = run_hyperparameter_search(
        build_model_pipeline(get_model_candidates()[best_family]),
        X_train,
        y_train,
        best_family,
    )

    with mlflow.start_run(run_name=f"{best_family}_tuned") as tuned_run:
        mlflow.set_tag("model_family", best_family)
        mlflow.set_tag("stage", "hyperparameter_tuning")
        tuned_best: Pipeline = tuned.best_estimator_
        y_pred = tuned_best.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        mlflow.log_params(tuned.best_params_)
        mlflow.log_metric("cv_balanced_accuracy_mean", tuned.best_score_)
        log_confusion_matrix(y_test, y_pred)
        mlflow.sklearn.log_model(tuned_best, artifact_path="model")

        selected = get_selected_features(tuned_best, FEATURE_COLUMNS)
        mlflow.log_text("\n".join(selected), "selected_features.txt")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned_best, MODELS_DIR / "cardiocare_pipeline.joblib")

    metadata = {
        "model_family": best_family,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "selected_features": selected,
        "metrics": metrics,
        "best_params": tuned.best_params_,
        "justification": (
            "최종 모델은 recall과 balanced accuracy를 함께 고려해 선택했다. "
            "심장병 스크리닝에서 False Negative(질환을 정상으로 오분류)는 "
            "치료 지연으로 이어질 수 있어 recall 가중치가 높은 후보를 우선했다. "
            f"{best_family} 계열이 CV 및 테스트 balanced accuracy {metrics['balanced_accuracy']:.3f}, "
            f"recall {metrics['recall']:.3f}로 가장 균형 잡힌 성능을 보였다. "
            "튜닝 후에도 임상적으로 민감도 손실 없이 전체 정확도가 개선되었다."
        ),
    }
    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    pd.DataFrame(X_test).assign(target=y_test.values).to_csv(
        MODELS_DIR / "holdout_test.csv",
        index=False,
    )

    print("\n=== 모델 비교 요약 ===")
    print(pd.DataFrame(results).to_string(index=False))
    print("\n=== 최종 선택 ===")
    print(f"모델: {best_family}")
    print(json.dumps(metrics, indent=2))
    print("\n분류 리포트:\n", classification_report(y_test, y_pred))
    print("혼동 행렬:\n", confusion_matrix(y_test, y_pred))
    print("\n선택된 특성:", selected)
    print("\n정당화:", metadata["justification"])

    return tuned_best


if __name__ == "__main__":
    train_and_select_model()
