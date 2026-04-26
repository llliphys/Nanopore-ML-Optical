"""SHAP feature-importance analysis using a scikit-learn surrogate model.

Because SHAP with a PyTorch model can be slow, this module trains a
separate :class:`~sklearn.neural_network.MLPRegressor` on a reduced
feature subset and computes Shapley values via :mod:`shap`.
"""

from __future__ import annotations

import numpy as np
import shap
from sklearn.neural_network import MLPRegressor


def _subset_features(
    X: np.ndarray,
    num_angle_feats: int,
    num_dist_feats: int,
    num_angle_feats_shap: int,
    num_dist_feats_shap: int,
    num_elec_feats_shap: int,
) -> np.ndarray:
    """Select the reduced feature subset used for SHAP analysis.

    Args:
        X: Full feature matrix, shape ``(n_samples, n_features)``.
        num_angle_feats: Total number of angle features in *X*.
        num_dist_feats: Total number of distance features in *X*.
        num_angle_feats_shap: Number of angle features to keep.
        num_dist_feats_shap: Number of distance features to keep.
        num_elec_feats_shap: Number of electronic features to keep.

    Returns:
        Reduced feature matrix.
    """
    dist_start = num_angle_feats
    elec_start = num_angle_feats + num_dist_feats

    return np.concatenate(
        [
            X[:, :num_angle_feats_shap],
            X[:, dist_start : dist_start + num_dist_feats_shap],
            X[:, elec_start : elec_start + num_elec_feats_shap],
        ],
        axis=1,
    )


def compute_shap_values(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    num_angle_feats: int,
    num_dist_feats: int,
    num_elec_feats: int,
    num_angle_feats_shap: int,
    num_dist_feats_shap: int,
    num_elec_feats_shap: int,
    hidden_layer_sizes: tuple[int, ...] = (64, 64),
    max_iter: int = 1000,
    random_state: int = 1234,
) -> np.ndarray:
    """Train a surrogate MLP and compute SHAP feature-importance values.

    A scikit-learn :class:`MLPRegressor` is fitted on the reduced feature
    set, then SHAP Shapley values are calculated for the test samples.

    Args:
        X_train: Training features (full feature set).
        Y_train: Training targets (PCA space).
        X_test: Test features (full feature set).
        num_angle_feats: Total angle features in *X*.
        num_dist_feats: Total distance features in *X*.
        num_elec_feats: Total electronic features in *X*.
        num_angle_feats_shap: Angle features to include in SHAP analysis.
        num_dist_feats_shap: Distance features to include in SHAP analysis.
        num_elec_feats_shap: Electronic features to include in SHAP analysis.
        hidden_layer_sizes: Architecture of the surrogate MLP.
        max_iter: Maximum training iterations for the surrogate.
        random_state: Random seed for reproducibility.

    Returns:
        SHAP values array.
    """
    if num_angle_feats_shap > num_angle_feats:
        raise ValueError(f"Number of anlge features for SHAP analysis exceeds {num_angle_feats}!")

    if num_dist_feats_shap > num_dist_feats:
        raise ValueError(f"Number of distance features for SHAP analysis exceeds {num_dist_feats}!")

    if num_elec_feats_shap > num_elec_feats:
        raise ValueError(f"Number of electronic features for SHAP analysis exceeds {num_elec_feats}!")
    
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        max_iter=max_iter,
        random_state=random_state,
    )

    X_train_sub = _subset_features(
        X_train, num_angle_feats, num_dist_feats,
        num_angle_feats_shap, num_dist_feats_shap, num_elec_feats_shap,
    )
    X_test_sub = _subset_features(
        X_test, num_angle_feats, num_dist_feats,
        num_angle_feats_shap, num_dist_feats_shap, num_elec_feats_shap,
    )

    mlp.fit(X_train_sub, Y_train)

    explainer = shap.Explainer(mlp.predict, X_train_sub)
    shap_values = explainer.shap_values(X_test_sub)

    return shap_values
