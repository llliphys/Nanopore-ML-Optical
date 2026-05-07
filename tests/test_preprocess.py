import unittest

import numpy as np

from features.preprocess import (
    get_pca_cumulative_variance,
    inverse_log_transform,
    invert_pca_scaled_log_predictions,
    log_transform_targets,
    preprocess_train_test,
    split_data,
)


class PreprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
                [4.0, 5.0, 6.0],
            ]
        )
        self.Y = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
                [4.0, 5.0, 6.0, 7.0],
            ]
        )

    def test_split_data_preserves_sample_count(self) -> None:
        X_train, X_test, Y_train, Y_test = split_data(
            self.X, self.Y, test_size=0.25, random_state=1234
        )

        self.assertEqual(len(X_train) + len(X_test), len(self.X))
        self.assertEqual(len(Y_train) + len(Y_test), len(self.Y))

    def test_log_transform_and_inverse_restore_values(self) -> None:
        y_log = log_transform_targets(self.Y, log_base=10, log_eps=1.0)
        y_restored = inverse_log_transform(y_log, log_base=10, log_eps=1.0)
        np.testing.assert_allclose(y_restored, self.Y)

    def test_preprocess_outputs_expected_shapes(self) -> None:
        X_train, X_test = self.X[:3], self.X[3:]
        Y_train, Y_test = self.Y[:3], self.Y[3:]

        prep = preprocess_train_test(
            X_train=X_train,
            X_test=X_test,
            Y_train=Y_train,
            Y_test=Y_test,
            n_components=2,
            log_base=10,
            log_eps=1.0,
        )

        self.assertEqual(prep["X_train_scaled"].shape, X_train.shape)
        self.assertEqual(prep["X_test_scaled"].shape, X_test.shape)
        self.assertEqual(prep["Y_train_pca"].shape, (3, 2))
        self.assertEqual(prep["Y_test_pca"].shape, (1, 2))

    def test_invert_pca_scaled_log_predictions_restores_target_shape(self) -> None:
        X_train, X_test = self.X[:3], self.X[3:]
        Y_train, Y_test = self.Y[:3], self.Y[3:]

        prep = preprocess_train_test(
            X_train=X_train,
            X_test=X_test,
            Y_train=Y_train,
            Y_test=Y_test,
            n_components=2,
            log_base=10,
            log_eps=1.0,
        )

        reconstructed = invert_pca_scaled_log_predictions(
            Y_pred_pca=prep["Y_test_pca"],
            pca=prep["pca"],
            scaler_y=prep["scaler_y"],
            log_base=10,
            log_eps=1.0,
        )

        self.assertEqual(reconstructed.shape, Y_test.shape)

    def test_get_pca_cumulative_variance_has_expected_length(self) -> None:
        prep = preprocess_train_test(
            X_train=self.X[:3],
            X_test=self.X[3:],
            Y_train=self.Y[:3],
            Y_test=self.Y[3:],
            n_components=2,
            log_base=10,
            log_eps=1.0,
        )

        pca_info = get_pca_cumulative_variance(prep["pca"])
        self.assertEqual(len(pca_info["explained_variance_ratio"]), 2)
        self.assertEqual(len(pca_info["cumulative_explained_variance"]), 2)
        self.assertEqual(len(pca_info["n_components"]), 2)


if __name__ == "__main__":
    unittest.main()
