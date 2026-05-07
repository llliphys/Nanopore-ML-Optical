import unittest

import numpy as np

from evaluation.metrics import per_sample_regression_metrics, regression_metrics


class MetricsTests(unittest.TestCase):
    def test_regression_metrics_perfect_prediction(self) -> None:
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])

        metrics = regression_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics["r2_global"], 1.0)
        self.assertAlmostEqual(metrics["rmse_global"], 0.0)
        self.assertAlmostEqual(metrics["mae_global"], 0.0)

    def test_per_sample_regression_metrics_returns_expected_keys(self) -> None:
        y_true = np.array([[1.0, 2.0], [2.0, 3.0]])
        y_pred = np.array([[1.0, 2.0], [2.5, 2.5]])

        metrics = per_sample_regression_metrics(y_true, y_pred)

        self.assertIn("r2_list", metrics)
        self.assertIn("rmse_list", metrics)
        self.assertIn("mae_list", metrics)
        self.assertIn("r2_mean", metrics)
        self.assertIn("rmse_mean", metrics)
        self.assertIn("mae_mean", metrics)
        self.assertEqual(len(metrics["r2_list"]), 2)
        self.assertEqual(len(metrics["rmse_list"]), 2)
        self.assertEqual(len(metrics["mae_list"]), 2)


if __name__ == "__main__":
    unittest.main()
