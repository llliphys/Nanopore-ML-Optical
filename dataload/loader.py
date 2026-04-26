"""Data loading and feature/target extraction for nanopore biosensing datasets.

This module reads pre-computed DFT simulation data from pickle files and
constructs the feature matrix (X) and target matrix (Y) used by the ML
pipeline.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd


# Column names for the three angle features (Euler angles of the molecule).
ANGLE_COLUMNS = ["Theta", "Phi", "Psi"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_nanopore_dataframe(
    root_dir: str,
    dataset_name: str,
) -> tuple[pd.DataFrame, str]:
    """Load a nanopore dataset from a pickle file.

    Args:
        root_dir: Project root directory containing the ``datasets/`` folder.
        dataset_name: Base name of the ``.pkl`` file (without extension).

    Returns:
        A tuple of (DataFrame, dataset_name).
    """
    dataset_dir = os.path.join(root_dir, "datasets")
    ensure_dir(dataset_dir)

    dataset_path = os.path.join(dataset_dir, f"{dataset_name}.pkl")
    data_frame = pd.read_pickle(dataset_path)
    return data_frame, dataset_name


def build_feature_target_arrays(
    df: pd.DataFrame,
    num_angle_feats: int,
    num_dist_feats: int,
    num_elec_feats: int,
    min_photo_energy: float,
    max_photo_energy: float,
) -> dict[str, Any]:
    """Extract feature and target arrays from a nanopore DataFrame.

    Features are constructed by concatenating three groups:
      - **Angle features**: molecular orientation (Theta, Phi, Psi).
      - **Distance features**: inter-atomic distances (first ``num_dist_feats``).
      - **Electronic features**: transition energies (first ``num_elec_feats``).

    Targets are the absorption coefficient spectra, filtered to the photon
    energy range [``min_photo_energy``, ``max_photo_energy``].

    Args:
        df: DataFrame with columns ``Theta``, ``Phi``, ``Psi``,
            ``Distance_Features``, ``Transition_Energies``,
            ``Photon_Energy``, and ``Absorption_Coefficient``.
        num_angle_feats: Number of angle features to use (typically 3).
        num_dist_feats: Number of distance features to keep.
        num_elec_feats: Number of electronic transition features to keep.
        min_photo_energy: Lower bound of the photon energy window (eV).
        max_photo_energy: Upper bound of the photon energy window (eV).

    Returns:
        Dictionary with keys:
          - ``X``: feature matrix, shape ``(n_samples, n_features)``.
          - ``Y``: target matrix, shape ``(n_samples, n_energy_points)``.
          - ``W``: photon energy grid, shape ``(n_samples, n_energy_points)``.
          - ``labels``: amino-acid labels per sample.
          - ``feature_columns``: list of human-readable column names.
          - ``n_samples``, ``n_features``, ``n_targets``: array dimensions.
    """
    angles = df[ANGLE_COLUMNS].values

    distance_features = np.vstack(df["Distance_Features"].values)[:, :num_dist_feats]
    transition_energies = np.vstack(df["Transition_Energies"].values)[:, :num_elec_feats]

    X = np.concatenate([angles, distance_features, transition_energies], axis=1)

    W = np.vstack(df["Photon_Energy"].values)
    Y = np.vstack(df["Absorption_Coefficient"].values)

    # Filter columns to the requested photon energy range (using the first
    # row as the reference grid — all rows share the same energy axis).
    energy_mask = (W[0] >= min_photo_energy) & (W[0] <= max_photo_energy)
    indices = np.where(energy_mask)[0]
    W = W[:, indices]
    Y = Y[:, indices]

    angle_cols = [f"A{i + 1}" for i in range(num_angle_feats)]
    dist_cols = [f"D{i + 1}" for i in range(num_dist_feats)]
    elec_cols = [f"E{i + 1}" for i in range(num_elec_feats)]

    return {
        "X": X,
        "Y": Y,
        "W": W,
        "labels": df["Amino_Acid"].values,
        "feature_columns": angle_cols + dist_cols + elec_cols,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_targets": Y.shape[1],
    }
