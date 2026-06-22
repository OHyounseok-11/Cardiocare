"""UCI Heart Disease 데이터셋 로드 (4개 processed 파일 통합)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

PROCESSED_FILES = [
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data",
]

MISSING_MARKER = -9.0


def get_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def load_raw_data(data_dir: Path | None = None) -> pd.DataFrame:
    """4개 processed UCI 파일을 결합해 단일 DataFrame을 반환한다."""
    base = data_dir or get_data_dir()
    frames: list[pd.DataFrame] = []

    for filename in PROCESSED_FILES:
        path = base / filename
        if not path.exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")

        df = pd.read_csv(path, header=None, names=COLUMNS)
        df["source"] = filename.replace("processed.", "").replace(".data", "")
        for col in COLUMNS:
            df[col] = pd.to_numeric(df[col].replace("?", pd.NA), errors="coerce")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.replace(MISSING_MARKER, pd.NA, inplace=True)
    return combined


def binarize_target(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    """타깃을 이진 분류(0=정상, 1=심장병)로 변환한다."""
    result = df.copy()
    result[target_col] = (result[target_col] > 0).astype(int)
    return result
