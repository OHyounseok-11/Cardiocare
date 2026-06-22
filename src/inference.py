"""CardioCare 배치 추론 엔트리포인트."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import FEATURE_COLUMNS, validate_clinical_ranges  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("cardiocare.inference")

MODELS_DIR = ROOT / "models"
DEFAULT_MODEL = MODELS_DIR / "cardiocare_pipeline.joblib"


def load_model(model_path: Path | None = None):
    path = model_path or DEFAULT_MODEL
    if not path.exists():
        raise FileNotFoundError(
            f"학습된 모델이 없습니다: {path}. 먼저 `python src/train.py`를 실행하세요."
        )
    return joblib.load(path)


def load_metadata() -> dict:
    meta_path = MODELS_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {"model_family": "unknown", "random_state": 42}


def predict_batch(
    input_path: Path,
    output_path: Path | None = None,
    model_path: Path | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    violations = validate_clinical_ranges(df)
    if violations:
        raise ValueError(f"임상 범위를 벗어난 특성: {violations}")

    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 특성 누락: {missing_cols}")

    model = load_model(model_path)
    metadata = load_metadata()
    features = df[FEATURE_COLUMNS]
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    result = df.copy()
    result["prediction"] = predictions
    result["prob_no_disease"] = probabilities[:, 0]
    result["prob_disease"] = probabilities[:, 1]

    logger.info(
        "inference | model=%s | input_shape=%s | predictions=%s",
        metadata.get("model_family", "unknown"),
        features.shape,
        predictions.tolist(),
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        logger.info("결과 저장: %s", output_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="CardioCare 배치 추론")
    parser.add_argument(
        "--input",
        required=True,
        help="입력 CSV 경로 (FEATURE_COLUMNS 포함)",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs" / "predictions.csv"),
        help="출력 CSV 경로",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="학습된 파이프라인 경로",
    )
    args = parser.parse_args()

    predict_batch(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_path=Path(args.model),
    )


if __name__ == "__main__":
    main()
