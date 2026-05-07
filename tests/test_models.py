import unittest

import torch

from models.models import MLP


class ModelTests(unittest.TestCase):
    def test_mlp_forward_output_shape(self) -> None:
        model = MLP(input_dim=5, output_dim=3, hidden_dims=(8, 4), dropout=0.0)
        x = torch.randn(7, 5)

        y = model(x)

        self.assertEqual(y.shape, (7, 3))

    def test_mlp_rejects_invalid_final_activation(self) -> None:
        with self.assertRaises(ValueError):
            MLP(input_dim=5, output_dim=2, final_activation="Tanh")


if __name__ == "__main__":
    unittest.main()
