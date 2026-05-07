import unittest

import torch

from models.models import MLP
from training.train import build_optimizer, train_torch_model


class TrainModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.input_dim = 4
        self.output_dim = 2

        self.X_train = torch.randn(12, self.input_dim, dtype=torch.float32)
        self.Y_train = torch.randn(12, self.output_dim, dtype=torch.float32)
        self.X_val = torch.randn(6, self.input_dim, dtype=torch.float32)
        self.Y_val = torch.randn(6, self.output_dim, dtype=torch.float32)

    def test_build_optimizer_without_weight_decay(self) -> None:
        model = MLP(input_dim=self.input_dim, output_dim=self.output_dim)
        optimizer = build_optimizer(model, lr_rate=1e-3)

        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)

    def test_build_optimizer_with_weight_decay(self) -> None:
        model = MLP(input_dim=self.input_dim, output_dim=self.output_dim)
        optimizer = build_optimizer(model, lr_rate=1e-3, weight_decay=1e-5)

        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 1e-5)

    def test_train_torch_model_returns_expected_structure(self) -> None:
        model = MLP(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=(8, 4),
            dropout=0.0,
        )

        result = train_torch_model(
            model=model,
            X_train_tensor=self.X_train,
            Y_train_tensor=self.Y_train,
            X_val_tensor=self.X_val,
            Y_val_tensor=self.Y_val,
            num_epochs=5,
            lr_rate=1e-3,
            device="cpu",
            print_every=10,
        )

        self.assertIn("model", result)
        self.assertIn("train_loss_list", result)
        self.assertIn("val_loss_list", result)
        self.assertEqual(len(result["train_loss_list"]), 5)
        self.assertEqual(len(result["val_loss_list"]), 5)
        self.assertTrue(all(loss >= 0 for loss in result["train_loss_list"]))
        self.assertTrue(all(loss >= 0 for loss in result["val_loss_list"]))

        trained_model = result["model"]
        self.assertIsInstance(trained_model, MLP)

        with torch.no_grad():
            predictions = trained_model(self.X_val)
        self.assertEqual(predictions.shape, self.Y_val.shape)

    def test_train_torch_model_supports_l1_regularization(self) -> None:
        model = MLP(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=(8,),
            dropout=0.0,
        )

        result = train_torch_model(
            model=model,
            X_train_tensor=self.X_train,
            Y_train_tensor=self.Y_train,
            X_val_tensor=self.X_val,
            Y_val_tensor=self.Y_val,
            num_epochs=3,
            lr_rate=1e-3,
            l1_lambda=1e-4,
            device="cpu",
            print_every=10,
        )

        self.assertEqual(len(result["train_loss_list"]), 3)
        self.assertEqual(len(result["val_loss_list"]), 3)


if __name__ == "__main__":
    unittest.main()
