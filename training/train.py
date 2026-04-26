"""PyTorch training loop with optional L1 regularisation."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim


def build_optimizer(
    model: nn.Module,
    lr_rate: float,
    weight_decay: float | None = None,
) -> optim.Adam:
    """Create an Adam optimiser for *model*.

    Args:
        model: PyTorch model whose parameters will be optimised.
        lr_rate: Learning rate.
        weight_decay: Optional L2 regularisation coefficient.

    Returns:
        Configured :class:`torch.optim.Adam` instance.
    """
    if weight_decay is not None:
        return optim.Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay)
    return optim.Adam(model.parameters(), lr=lr_rate)


def train_torch_model(
    model: nn.Module,
    X_train_tensor: torch.Tensor,
    Y_train_tensor: torch.Tensor,
    X_val_tensor: torch.Tensor,
    Y_val_tensor: torch.Tensor,
    num_epochs: int,
    lr_rate: float,
    l1_lambda: float | None = None,
    weight_decay: float | None = None,
    device: str = "cpu",
    print_every: int = 100,
) -> dict[str, Any]:
    """Train the MLP model and return the best checkpoint.

    The best model is selected as the one achieving the lowest training
    loss across all epochs (a deep copy is kept).

    Args:
        model: PyTorch model to train.
        X_train_tensor: Training features.
        Y_train_tensor: Training targets (PCA space).
        X_val_tensor: Validation features.
        Y_val_tensor: Validation targets (PCA space).
        num_epochs: Total number of training epochs.
        lr_rate: Learning rate for Adam.
        l1_lambda: Optional L1 regularisation strength.  ``None`` disables it.
        weight_decay: Optional L2 weight decay for Adam.
        device: Device identifier (``"cpu"`` or ``"cuda"``).
        print_every: Print progress every *N* epochs.

    Returns:
        Dictionary with ``model`` (best checkpoint), ``train_loss_list``,
        and ``val_loss_list``.
    """
    model = model.to(device)
    X_train_tensor = X_train_tensor.to(device)
    Y_train_tensor = Y_train_tensor.to(device)
    X_val_tensor = X_val_tensor.to(device)
    Y_val_tensor = Y_val_tensor.to(device)

    mse_loss = nn.MSELoss()
    optimizer = build_optimizer(model, lr_rate=lr_rate, weight_decay=weight_decay)

    train_loss_list: list[float] = []
    val_loss_list: list[float] = []

    best_loss = float("inf")
    best_model = copy.deepcopy(model)

    for epoch in range(num_epochs):
        model.train()

        outputs = model(X_train_tensor)

        if l1_lambda is not None:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            train_loss = mse_loss(outputs, Y_train_tensor) + l1_lambda * l1_norm
        else:
            train_loss = mse_loss(outputs, Y_train_tensor)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_hat = model(X_val_tensor)
            val_loss = mse_loss(y_val_hat, Y_val_tensor)

        if train_loss.item() < best_loss:
            best_loss = train_loss.item()
            best_model = copy.deepcopy(model)

        train_loss_list.append(train_loss.item())
        val_loss_list.append(val_loss.item())

        if (epoch + 1) % print_every == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] | "
                f"Train Loss: {train_loss.item():.6f} | "
                f"Val Loss: {val_loss.item():.6f} | "
                f"Best Loss : {best_loss:.6f}"
            )

    return {
        "model": best_model,
        "train_loss_list": train_loss_list,
        "val_loss_list": val_loss_list,
    }
