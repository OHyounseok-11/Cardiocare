"""CardioCare ML 파이프라인 단위 테스트."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import load_model, predict_batch  # noqa: E402
from src.preprocessing import (  # noqa: E402
    CLINICAL_RANGES,
    FEATURE_COLUMNS,
    build_preprocessing_pipeline,
    prepare_dataset,
    validate_clinical_ranges,
)
from src.train import RANDOM_STATE, build_model_pipeline, get_model_candidates  # noqa: E402


class TestCardioCarePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = ROOT / "models" / "cardiocare_pipeline.joblib"
        if not cls.model_path.exists():
            X, y = prepare_dataset()
            pipeline = build_model_pipeline(get_model_candidates()["logistic_regression"])
            pipeline.fit(X, y)
            cls.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, cls.model_path)

        cls.sample = pd.read_csv(ROOT / "data" / "sample_batch.csv")

    def test_prediction_shape_matches_input(self):
        model = load_model(self.model_path)
        features = self.sample[FEATURE_COLUMNS]
        predictions = model.predict(features)
        self.assertEqual(predictions.shape[0], features.shape[0])

    def test_prediction_probabilities_valid(self):
        model = load_model(self.model_path)
        features = self.sample[FEATURE_COLUMNS]
        probabilities = model.predict_proba(features)
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))
        row_sums = probabilities.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5, atol=1e-5)

    def test_clinical_input_range_validation(self):
        valid = self.sample.copy()
        self.assertEqual(validate_clinical_ranges(valid), [])

        invalid = valid.copy()
        invalid.loc[0, "chol"] = 900
        violations = validate_clinical_ranges(invalid)
        self.assertIn("chol", violations)
        self.assertGreaterEqual(CLINICAL_RANGES["chol"][1], 600)

    def test_pipeline_is_deterministic_with_fixed_seed(self):
        X, y = prepare_dataset()
        pipeline_a = build_model_pipeline(get_model_candidates()["logistic_regression"])
        pipeline_b = build_model_pipeline(get_model_candidates()["logistic_regression"])

        pipeline_a.fit(X, y)
        pipeline_b.fit(X, y)

        sample = X.iloc[:10]
        pred_a = pipeline_a.predict(sample)
        pred_b = pipeline_b.predict(sample)
        np.testing.assert_array_equal(pred_a, pred_b)

    def test_inference_batch_writes_output(self):
        output_path = ROOT / "outputs" / "test_predictions.csv"
        result = predict_batch(
            input_path=ROOT / "data" / "sample_batch.csv",
            output_path=output_path,
            model_path=self.model_path,
        )
        self.assertIn("prediction", result.columns)
        self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
