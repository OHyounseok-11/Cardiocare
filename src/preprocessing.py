"""CardioCare 전처리 파이프라인 및 입력 검증."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import binarize_target, load_raw_data

CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_COLUMNS = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "target"

CLINICAL_RANGES: dict[str, tuple[float, float]] = {
    "age": (0, 120),
    "trestbps": (50, 250),
    "chol": (0, 600),
    "thalach": (40, 220),
    "oldpeak": (0, 10),
    "sex": (0, 1),
    "cp": (1, 4),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "exang": (0, 1),
    "slope": (1, 3),
    "ca": (0, 3),
    "thal": (3, 7),
}


class HeartDataPreprocessor(BaseEstimator, TransformerMixin):
    """결측 대치, 빈 컬럼 제거, IQR 이상치 클리핑을 수행한다."""

    def __init__(
        self,
        continuous_columns: Iterable[str] | None = None,
        iqr_factor: float = 1.5,
    ):
        self.continuous_columns = continuous_columns
        self.iqr_factor = iqr_factor

    def _continuous_cols(self) -> list[str]:
        return list(self.continuous_columns or CONTINUOUS_FEATURES)

    def fit(self, X, y=None):
        frame = self._to_frame(X)
        continuous_cols = self._continuous_cols()
        self.feature_columns_ = [
            col for col in frame.columns if not frame[col].isna().all()
        ]
        self.impute_values_: dict[str, float] = {}
        self.bounds_: dict[str, tuple[float, float]] = {}

        for col in self.feature_columns_:
            series = frame[col].astype(float)
            self.impute_values_[col] = float(series.median())

        for col in continuous_cols:
            if col not in self.feature_columns_:
                continue
            filled = frame[col].astype(float).fillna(self.impute_values_[col])
            q1 = filled.quantile(0.25)
            q3 = filled.quantile(0.75)
            iqr = q3 - q1
            self.bounds_[col] = (
                float(q1 - self.iqr_factor * iqr),
                float(q3 + self.iqr_factor * iqr),
            )
        return self

    def transform(self, X):
        frame = self._to_frame(X)[self.feature_columns_].copy()
        for col in self.feature_columns_:
            frame[col] = frame[col].astype(float).fillna(self.impute_values_[col])
        for col, (lower, upper) in self.bounds_.items():
            frame[col] = frame[col].clip(lower, upper)
        return frame

    @staticmethod
    def _to_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X, columns=FEATURE_COLUMNS)


def prepare_dataset(data_dir=None) -> tuple[pd.DataFrame, pd.Series]:
    """원본 데이터를 불러와 타깃 이진화 및 중복 제거까지 수행한다."""
    raw = load_raw_data(data_dir)
    labeled = binarize_target(raw)

    feature_frame = labeled[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    feature_frame.drop_duplicates(inplace=True)
    feature_frame.dropna(subset=[TARGET_COLUMN], inplace=True)

    X = feature_frame[FEATURE_COLUMNS].reset_index(drop=True)
    y = feature_frame[TARGET_COLUMN].astype(int).reset_index(drop=True)
    return X, y


def build_preprocessing_pipeline() -> Pipeline:
    """EDA 결과를 반영한 재사용 가능한 전처리 파이프라인."""
    return Pipeline([("preprocess", HeartDataPreprocessor())])


def build_feature_scaler() -> ColumnTransformer:
    """모델 학습용 스케일러 (연속형만 StandardScaler 적용)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CONTINUOUS_FEATURES),
            ("cat", "passthrough", CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def validate_clinical_ranges(
    df: pd.DataFrame,
    ranges: dict[str, tuple[float, float]] | None = None,
) -> list[str]:
    """임상 범위를 벗어난 컬럼명 목록을 반환한다."""
    bounds = ranges or CLINICAL_RANGES
    violations: list[str] = []

    for col, (lower, upper) in bounds.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().any():
            violations.append(col)
            continue
        if ((series < lower) | (series > upper)).any():
            violations.append(col)

    return violations
