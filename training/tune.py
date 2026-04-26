"""Grid-search hyperparameter tuning for the MLP model."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from evaluation.metrics import per_sample_regression_metrics
from inference.predict import predict_absorption
from models.models import MLP
from training.train import train_torch_model


def build_search_space(param_grid: dict[str, list]) -> list[dict[str, Any]]:
    """Expand a parameter grid into a list of individual configurations.

    Args:
        param_grid: Mapping from parameter names to lists of candidate values.

    Returns:
        List of dictionaries, one per combination (Cartesian product).
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    return [dict(zip(keys, combo)) for combo in product(*values)]


def run_single_experiment(
    X_train_tensor: torch.Tensor,
    Y_train_tensor: torch.Tensor,
    X_val_tensor: torch.Tensor,
    Y_val_tensor: torch.Tensor,
    Y_val_true: np.ndarray,
    input_dim: int,
    output_dim: int,
    pca: PCA,
    scaler_y: MinMaxScaler,
    log_base: float,
    log_eps: float,
    device: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Train one model with given hyper-parameters and evaluate it.

    Args:
        X_train_tensor: Training features (tensor).
        Y_train_tensor: Training targets in PCA space (tensor).
        X_val_tensor: Validation features (tensor).
        Y_val_tensor: Validation targets in PCA space (tensor).
        Y_val_true: Validation targets in original space (array).
        input_dim: Number of input features.
        output_dim: Number of PCA components.
        pca: Fitted PCA object for inverse transform.
        scaler_y: Fitted MinMaxScaler for inverse transform.
        log_base: Log base used during preprocessing.
        log_eps: Log offset used during preprocessing.
        device: PyTorch device identifier.
        params: Hyper-parameter dictionary for this experiment.

    Returns:
        Dictionary with ``params``, ``model``, ``metrics``,
        ``train_loss_list``, and ``val_loss_list``.
    """
    model = MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=params["hidden_dims"],
        dropout=params["dropout"],
        final_activation=params.get("final_activation"),
    )

    train_out = train_torch_model(
        model=model,
        X_train_tensor=X_train_tensor,
        Y_train_tensor=Y_train_tensor,
        X_val_tensor=X_val_tensor,
        Y_val_tensor=Y_val_tensor,
        num_epochs=params["num_epochs"],
        lr_rate=params["lr_rate"],
        l1_lambda=params.get("l1_lambda"),
        weight_decay=params.get("weight_decay"),
        device=device,
        print_every=params.get("print_every", 1000),
    )

    trained_model = train_out["model"]

    Y_val_pred = predict_absorption(
        model=trained_model,
        X_tensor=X_val_tensor,
        pca=pca,
        scaler_y=scaler_y,
        log_base=log_base,
        log_eps=log_eps,
        device=device,
    )

    metrics = per_sample_regression_metrics(Y_val_true, Y_val_pred)

    return {
        "params": params,
        "model": trained_model,
        "metrics": metrics,
        "train_loss_list": train_out["train_loss_list"],
        "val_loss_list": train_out["val_loss_list"],
    }


def grid_search_torch(
    X_train_tensor: torch.Tensor,
    Y_train_tensor: torch.Tensor,
    X_val_tensor: torch.Tensor,
    Y_val_tensor: torch.Tensor,
    Y_val_true: np.ndarray,
    input_dim: int,
    output_dim: int,
    pca: PCA,
    scaler_y: MinMaxScaler,
    log_base: float,
    log_eps: float,
    device: str,
    param_grid: dict[str, list],
) -> dict[str, Any]:
    """Run an exhaustive grid search over hyper-parameter combinations.

    Each combination is trained and evaluated independently.  The best
    result is selected by the highest mean per-sample R-squared score.

    Args:
        X_train_tensor: Training features (tensor).
        Y_train_tensor: Training targets in PCA space (tensor).
        X_val_tensor: Validation features (tensor).
        Y_val_tensor: Validation targets in PCA space (tensor).
        Y_val_true: Validation targets in original space (array).
        input_dim: Number of input features.
        output_dim: Number of PCA components.
        pca: Fitted PCA object for inverse transforms.
        scaler_y: Fitted MinMaxScaler for inverse transforms.
        log_base: Log base used during preprocessing.
        log_eps: Log offset used during preprocessing.
        device: PyTorch device identifier.
        param_grid: Mapping from parameter names to candidate value lists.

    Returns:
        Dictionary with ``best_result`` and ``all_results``.
    """
    experiments = build_search_space(param_grid)

    best_result: dict[str, Any] | None = None
    all_results: list[dict[str, Any]] = []

    for i, params in enumerate(experiments, start=1):
        print(f"Running experiment {i}/{len(experiments)} with params: {params}")

        result = run_single_experiment(
            X_train_tensor=X_train_tensor,
            Y_train_tensor=Y_train_tensor,
            X_val_tensor=X_val_tensor,
            Y_val_tensor=Y_val_tensor,
            Y_val_true=Y_val_true,
            input_dim=input_dim,
            output_dim=output_dim,
            pca=pca,
            scaler_y=scaler_y,
            log_base=log_base,
            log_eps=log_eps,
            device=device,
            params=params,
        )

        all_results.append(result)

        if best_result is None or result["metrics"]["r2_mean"] > best_result["metrics"]["r2_mean"]:
            best_result = result

    return {
        "best_result": best_result,
        "all_results": all_results,
    }
