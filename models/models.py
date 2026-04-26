"""Neural network model definitions for nanopore biosensing."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron for regression on PCA-compressed spectra.

    Constructs a fully-connected feed-forward network with configurable
    hidden layers, ReLU activations, and dropout regularisation.

    Args:
        input_dim: Number of input features.
        output_dim: Number of output values (PCA components).
        hidden_dims: Sizes of the hidden layers (e.g. ``(128, 128, 128)``).
        dropout: Dropout probability applied after each hidden layer.
        final_activation: Optional activation applied to the output layer.
            Supported values: ``"ReLU"``, ``"Sigmoid"``, or ``None``.

    Raises:
        ValueError: If *final_activation* is not a recognised name.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        final_activation: str | None = None,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        layer_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(layer_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layer_dim = hidden_dim

        layers.append(nn.Linear(layer_dim, output_dim))

        if final_activation is not None:
            if final_activation == "ReLU":
                layers.append(nn.ReLU())
            elif final_activation == "Sigmoid":
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(
                    f"Unsupported final_activation: {final_activation!r}. "
                    f"Choose from 'ReLU', 'Sigmoid', or None."
                )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass through the network."""
        return self.model(x)
