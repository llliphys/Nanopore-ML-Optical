"""Preprocessing utilities for the nanopore biosensing ML pipeline.

Provides train/test splitting, log-transformation, feature/target scaling,
PCA dimensionality reduction on the target space, and the corresponding
inverse transformations for recovering original absorption spectra from
model predictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def split_data(
    X: np.ndarray,
    Y: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into random train and test subsets.

    Thin wrapper around :func:`sklearn.model_selection.train_test_split`
    for a consistent interface throughout the project.

    Args:
        X: Feature matrix.
        Y: Target matrix (or any array to be split in sync with *X*).
        test_size: Fraction of samples to reserve for testing.
        random_state: Seed for reproducible splitting.

    Returns:
        ``(X_train, X_test, Y_train, Y_test)``
    """
    return train_test_split(
        X,
        Y,
        test_size=test_size,
        random_state=random_state,
    )


def log_transform_targets(
    Y: np.ndarray,
    log_base: float,
    log_eps: float,
) -> np.ndarray:
    """Apply a log transform to stabilise the dynamic range of targets.

    Computes ``log_base(Y + log_eps)`` element-wise.

    Args:
        Y: Raw target values.
        log_base: Base of the logarithm (e.g. 10).
        log_eps: Small offset added before taking the log to avoid log(0).

    Returns:
        Log-transformed target array.
    """
    return np.emath.logn(log_base, Y + log_eps)


def inverse_log_transform(
    Y_log: np.ndarray,
    log_base: float,
    log_eps: float,
) -> np.ndarray:
    """Reverse :func:`log_transform_targets`.

    Computes ``log_base ** Y_log - log_eps`` element-wise.

    Args:
        Y_log: Log-transformed target values.
        log_base: Base of the logarithm used during the forward transform.
        log_eps: Offset that was added before the forward log transform.

    Returns:
        Reconstructed target values in the original scale.
    """
    return log_base ** Y_log - log_eps


def preprocess_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    n_components: int,
    log_base: float,
    log_eps: float,
) -> dict[str, Any]:
    """Run the full preprocessing pipeline on train/test data.

    Steps (applied in order):
      1. Log-transform targets (``Y``).
      2. Standardise features (``X``) with :class:`StandardScaler`.
      3. Min-max scale log-targets with :class:`MinMaxScaler`.
      4. Reduce target dimensionality with PCA.

    Args:
        X_train: Training features.
        X_test: Test features.
        Y_train: Training targets (raw absorption spectra).
        Y_test: Test targets (raw absorption spectra).
        n_components: Number of PCA components to retain.
        log_base: Base of the logarithm for the target transform.
        log_eps: Offset added before the log transform.

    Returns:
        Dictionary containing scaled/PCA-transformed arrays and the
        fitted scaler/PCA objects needed for inverse transforms:
        ``X_train_scaled``, ``X_test_scaled``,
        ``Y_train_log``, ``Y_test_log``,
        ``Y_train_scaled``, ``Y_test_scaled``,
        ``Y_train_pca``, ``Y_test_pca``,
        ``scaler_x``, ``scaler_y``, ``pca``.
    """
    Y_train_log = log_transform_targets(Y_train, log_base, log_eps)
    Y_test_log = log_transform_targets(Y_test, log_base, log_eps)

    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    scaler_y = MinMaxScaler()
    Y_train_scaled = scaler_y.fit_transform(Y_train_log)
    Y_test_scaled = scaler_y.transform(Y_test_log)

    pca = PCA(n_components=n_components)
    Y_train_pca = pca.fit_transform(Y_train_scaled)
    Y_test_pca = pca.transform(Y_test_scaled)

    return {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "Y_train_log": Y_train_log,
        "Y_test_log": Y_test_log,
        "Y_train_scaled": Y_train_scaled,
        "Y_test_scaled": Y_test_scaled,
        "Y_train_pca": Y_train_pca,
        "Y_test_pca": Y_test_pca,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "pca": pca,
    }


def invert_pca_scaled_log_predictions(
    Y_pred_pca: np.ndarray,
    pca: PCA,
    scaler_y: MinMaxScaler,
    log_base: float,
    log_eps: float,
) -> np.ndarray:
    """Invert the preprocessing pipeline to recover original-scale predictions.

    Applies the inverse operations in reverse order:
    PCA inverse -> MinMaxScaler inverse -> log inverse.

    Args:
        Y_pred_pca: Model predictions in PCA space.
        pca: Fitted PCA object.
        scaler_y: Fitted MinMaxScaler used on the log-transformed targets.
        log_base: Base of the logarithm used during preprocessing.
        log_eps: Offset that was added before the log transform.

    Returns:
        Predicted absorption spectra in the original (linear) scale.
    """
    Y_pred_scaled = pca.inverse_transform(Y_pred_pca)
    Y_pred_log = scaler_y.inverse_transform(Y_pred_scaled)
    return inverse_log_transform(Y_pred_log, log_base, log_eps)


def get_pca_cumulative_variance(pca: PCA) -> dict[str, np.ndarray]:
    """Extract cumulative explained variance from a fitted PCA.

    Args:
        pca: A fitted :class:`~sklearn.decomposition.PCA` instance.

    Returns:
        Dictionary with ``explained_variance_ratio``,
        ``cumulative_explained_variance``, and ``n_components`` arrays.
    """
    explained = pca.explained_variance_ratio_
    cumsum = np.cumsum(explained)
    n_components = np.arange(1, len(explained) + 1)

    return {
        "explained_variance_ratio": explained,
        "cumulative_explained_variance": cumsum,
        "n_components": n_components,
    }
