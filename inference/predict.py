"""Inference utilities for generating absorption spectrum predictions."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from features.preprocess import invert_pca_scaled_log_predictions


@torch.no_grad()
def predict_pca_targets(
    model: nn.Module,
    X_tensor: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """Run forward inference and return PCA-space predictions.

    The model is placed in evaluation mode and gradients are disabled
    for efficiency.

    Args:
        model: Trained PyTorch model.
        X_tensor: Input feature tensor.
        device: Device identifier (``"cpu"`` or ``"cuda"``).

    Returns:
        Predicted PCA components as a NumPy array.
    """
    model.eval()
    X_tensor = X_tensor.to(device)
    y_pred_tensor = model(X_tensor)
    return y_pred_tensor.detach().cpu().numpy()


def predict_absorption(
    model: nn.Module,
    X_tensor: torch.Tensor,
    pca: PCA,
    scaler_y: MinMaxScaler,
    log_base: float,
    log_eps: float,
    device: str = "cpu",
) -> np.ndarray:
    """Predict full absorption spectra from input features.

    Chains the model forward pass (in PCA space) with the full inverse
    preprocessing pipeline to produce absorption coefficients in the
    original (linear) scale.

    Args:
        model: Trained PyTorch model.
        X_tensor: Input feature tensor.
        pca: Fitted PCA object for inverse transform.
        scaler_y: Fitted MinMaxScaler for inverse transform.
        log_base: Log base used during preprocessing.
        log_eps: Log offset used during preprocessing.
        device: Device identifier (``"cpu"`` or ``"cuda"``).

    Returns:
        Predicted absorption spectra, shape ``(n_samples, n_energy_points)``.
    """
    Y_pred_pca = predict_pca_targets(model, X_tensor, device=device)
    Y_pred = invert_pca_scaled_log_predictions(
        Y_pred_pca=Y_pred_pca,
        pca=pca,
        scaler_y=scaler_y,
        log_base=log_base,
        log_eps=log_eps,
    )
    return Y_pred
