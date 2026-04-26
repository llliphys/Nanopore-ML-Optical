"""Regression evaluation metrics for multi-output predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute global regression metrics across all samples and outputs.

    Args:
        y_true: Ground-truth values, shape ``(n_samples, n_outputs)``.
        y_pred: Predicted values, same shape as *y_true*.

    Returns:
        Dictionary with ``r2_global``, ``rmse_global``, and ``mae_global``.
    """
    return {
        "r2_global": r2_score(y_true, y_pred),
        "rmse_global": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_global": float(mean_absolute_error(y_true, y_pred)),
    }


def per_sample_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Compute regression metrics for each sample independently.

    Evaluates R-squared, RMSE, and MAE per row (i.e. per molecular
    configuration) and returns both the per-sample lists and their means.

    Args:
        y_true: Ground-truth values, shape ``(n_samples, n_outputs)``.
        y_pred: Predicted values, same shape as *y_true*.

    Returns:
        Dictionary with ``r2_list``, ``rmse_list``, ``mae_list``
        (per-sample) and ``r2_mean``, ``rmse_mean``, ``mae_mean``
        (summary statistics).
    """
    r2_list: list[float] = []
    rmse_list: list[float] = []
    mae_list: list[float] = []

    for i in range(y_true.shape[0]):
        r2_list.append(r2_score(y_true[i, :], y_pred[i, :]))
        rmse_list.append(float(np.sqrt(mean_squared_error(y_true[i, :], y_pred[i, :]))))
        mae_list.append(float(mean_absolute_error(y_true[i, :], y_pred[i, :])))

    return {
        "r2_list": r2_list,
        "rmse_list": rmse_list,
        "mae_list": mae_list,
        "r2_mean": float(np.mean(r2_list)),
        "rmse_mean": float(np.mean(rmse_list)),
        "mae_mean": float(np.mean(mae_list)),
    }
