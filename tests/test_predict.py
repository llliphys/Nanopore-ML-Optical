import unittest

import numpy as np
import torch

from inference.predict import predict_absorption, predict_pca_targets


class DummyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :2]


class PredictTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = DummyModel()
        self.X_tensor = torch.tensor(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32
        )

        self.Y_train = np.array(
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=float
        )
        self.Y_train_log = np.log10(self.Y_train + 1.0)

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import MinMaxScaler

        self.scaler_y = MinMaxScaler()
        Y_train_scaled = self.scaler_y.fit_transform(self.Y_train_log)

        self.pca = PCA(n_components=2)
        self.pca.fit(Y_train_scaled)

    def test_predict_pca_targets_returns_numpy_array(self) -> None:
        pred = predict_pca_targets(self.model, self.X_tensor, device="cpu")

        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(pred.shape, (3, 2))

    def test_predict_absorption_returns_expected_shape(self) -> None:
        pred = predict_absorption(
            model=self.model,
            X_tensor=self.X_tensor,
            pca=self.pca,
            scaler_y=self.scaler_y,
            log_base=10,
            log_eps=1.0,
            device="cpu",
        )

        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(pred.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
