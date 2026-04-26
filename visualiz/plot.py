"""Plotting utilities for the nanopore biosensing ML pipeline.

All functions produce publication-quality matplotlib figures and save them
to a specified directory.  Plots include loss curves, model-score scatter
plots, absorption-spectrum comparisons, PCA variance, and SHAP summaries.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import r2_score

import shap


ABSORPTION_SCALING_FACTOR = 1e4


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_model_scores(
    Y_train_true: np.ndarray,
    Y_train_pred: np.ndarray,
    Y_test_true: np.ndarray,
    Y_test_pred: np.ndarray,
    figsave_dir: str,
    dataset_name: str,
) -> None:
    """Scatter plot of true vs. predicted absorption, coloured by per-sample R2.

    Produces a two-panel figure (training data | unseen data) where each
    sample's points are coloured by its individual R-squared score.

    Args:
        Y_train_true: True training absorption spectra.
        Y_train_pred: Predicted training absorption spectra.
        Y_test_true: True test absorption spectra.
        Y_test_pred: Predicted test absorption spectra.
        scaling_factor: Scaling factor for absorption spectra
        figsave_dir: Directory to save the figure.
        dataset_name: Name appended to the output filename.
    """
    ensure_dir(figsave_dir)

    fig, axs = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.2)

    panels = [
        ("Training Data", Y_train_true, Y_train_pred),
        ("Unseen Data", Y_test_true, Y_test_pred),
    ]

    for i, (title, Y_true, Y_pred) in enumerate(panels):
        r2_list = [
            r2_score(Y_true[idx, :], Y_pred[idx, :])
            for idx in range(Y_true.shape[0])
        ]

        ax = axs[i]
        for idx in range(Y_true.shape[0]):
            im = ax.scatter(
                Y_true[idx, :] / ABSORPTION_SCALING_FACTOR,
                Y_pred[idx, :] / ABSORPTION_SCALING_FACTOR,
                c=r2_list[idx] * np.ones(Y_true.shape[1]),
                s=50,
                alpha=0.15,
            )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        im.set_clim(min(r2_list), max(r2_list))
        cbar = fig.colorbar(im, cax=cax, cmap="RdBu")
        cbar.set_ticks([min(r2_list), max(r2_list)])
        cbar.ax.set_title(r"$R^2$", fontsize=20)

        ax.set_xlabel(r"$\alpha_{True} \times 10^4 \ \mathrm{(1/cm)}$", fontsize=25)
        if i == 0:
            ax.set_ylabel(r"$\alpha_{Pred} \times 10^4 \ \mathrm{(1/cm)}$", fontsize=25)

        ax.set_title(f"{title} ({Y_true.shape[0]} Samples)", fontsize=25)

    outpath = os.path.join(figsave_dir, f"Model_Scores_{dataset_name}.png")
    plt.savefig(outpath, bbox_inches="tight")
    plt.show()


def plot_absorption_prediction(
    X_test: np.ndarray,
    Y_test: np.ndarray,
    Y_test_pred: np.ndarray,
    W_test: np.ndarray,
    figsave_dir: str,
    dataset_name: str,
    min_photo_energy: float,
    max_photo_energy: float,
    log_base: float = 10,
    log_eps: float = 1.0,
    set_yscale_log: bool = False,
    n_groups_plot: int = 15,
    group_size_plot: int = 3,
) -> None:
    """Plot true vs. predicted absorption spectra grouped into multi-panel figures.

    Each figure shows ``group_size_plot`` samples stacked vertically, with
    ``n_groups_plot`` figures generated in total.

    Args:
        X_test: Test feature matrix (unused but kept for interface consistency).
        Y_test: True test absorption spectra.
        Y_test_pred: Predicted test absorption spectra.
        W_test: Photon energy grids for test samples.
        figsave_dir: Directory to save figures.
        dataset_name: Name appended to output filenames.
        min_photo_energy: X-axis lower limit (eV).
        max_photo_energy: X-axis upper limit (eV).
        log_base: Base for optional log-scale display.
        log_eps: Offset for optional log-scale display.
        set_yscale_log: If ``True``, display Y axis in log scale.
        n_groups_plot: Number of figure groups to generate.
        group_size_plot: Number of samples per figure.
    """
    ensure_dir(figsave_dir)

    for i in range(n_groups_plot):
        indices = np.arange(group_size_plot * i, group_size_plot * (i + 1), 1)

        nrows, ncols = len(indices), 1
        fig, axs = plt.subplots(
            nrows, ncols,
            figsize=(6 * ncols, 4 * nrows),
            sharex=True, sharey=True,
        )
        fig.subplots_adjust(wspace=0, hspace=0)

        if nrows == 1:
            axs = [axs]
        else:
            axs = axs.flat

        for count, idx in enumerate(indices):
            ax = axs[count]

            y_true = Y_test[idx, :]
            y_pred = Y_test_pred[idx, :]
            w = W_test[idx, :]

            r2 = r2_score(y_true, y_pred)

            if set_yscale_log:
                y_true_plot = np.emath.logn(log_base, y_true + log_eps)
                y_pred_plot = np.emath.logn(log_base, y_pred + log_eps)
                ax.plot(w, y_true_plot, "k--", lw=3, alpha=0.5, label="true")
                ax.plot(w, y_pred_plot, "b", lw=2, alpha=0.5, label="pred")
            else:
                ax.plot(w, y_true / ABSORPTION_SCALING_FACTOR, "k--", lw=3, alpha=0.5, label="true")
                ax.plot(w, y_pred / ABSORPTION_SCALING_FACTOR, "b", lw=2, alpha=0.5, label="pred")

            ax.text(
                0.02, 0.95, f"$R^2 = {r2:.3f}$",
                c="C1", fontsize=25,
                transform=ax.transAxes, va="top",
            )
            ax.set_xlim(min_photo_energy, max_photo_energy)
            ax.set_ylim(0, 25)
            ax.set_xticks([0, 1, 2])

            if count == 0:
                ax.legend(
                    loc="upper right", borderaxespad=0,
                    ncol=1, fontsize=23, frameon=False,
                )

        outpath = os.path.join(figsave_dir, f"Prediction_{i:02d}_{dataset_name}.png")
        plt.savefig(outpath, bbox_inches="tight")
        plt.show()


def plot_pca_cumulative_variance(
    n_components: np.ndarray,
    cumulative_explained_variance: np.ndarray,
    figsave_dir: str,
    dataset_name: str,
) -> None:
    """Plot cumulative explained variance as a function of PCA components.

    Args:
        n_components: Array of component indices (1, 2, ..., N).
        cumulative_explained_variance: Cumulative variance ratio per component.
        figsave_dir: Directory to save the figure.
        dataset_name: Name appended to the output filename.
    """
    ensure_dir(figsave_dir)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(n_components, cumulative_explained_variance, marker="o")
    ax.set_xlabel(r"$\mathrm{Number \ of \ Principal \ Components}$", fontsize=25)
    ax.set_ylabel(r"$\mathrm{Cumulative \ Explained \ Variance}$", fontsize=25)
    ax.set_ylim(cumulative_explained_variance.min() - 0.05, 1.05)

    outpath = os.path.join(figsave_dir, f"PCA_Cumsum_Variance_{dataset_name}.png")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.show()


def plot_shap_analysis_bar(
    shap_values: np.ndarray,
    X_samples: np.ndarray,
    num_angle_feats: int,
    num_dist_feats: int,
    num_elec_feats: int,
    num_angle_feats_shap: int,
    num_dist_feats_shap: int,
    num_elec_feats_shap: int,
    figsave_dir: str,
    dataset_name: str,
) -> None:
    """
    Generate a SHAP bar summary plot with custom colors for different feature groups.

    Expected shap_values shape:
        (n_samples, n_features, n_outputs)
    or
        (n_samples, n_features)

    The features are grouped as:
        A* -> angle features
        D* -> distance features
        E* -> electronic features
    """
    ensure_dir(figsave_dir)

    # Indices of each feature block in the original X_samples
    dist_start = num_angle_feats
    elec_start = num_angle_feats + num_dist_feats

    # Build the reduced feature matrix used for SHAP plotting
    X_samples_shap = np.concatenate(
        [
            X_samples[:, :num_angle_feats_shap],
            X_samples[:, dist_start: dist_start + num_dist_feats_shap],
            X_samples[:, elec_start: elec_start + num_elec_feats_shap],
        ],
        axis=1,
    )

    # Feature names
    feature_names_shap = (
        [f"A{i + 1}" for i in range(num_angle_feats_shap)]
        + [f"D{i + 1}" for i in range(num_dist_feats_shap)]
        + [f"E{i + 1}" for i in range(num_elec_feats_shap)]
    )

    # Convert multi-output SHAP to 2D: (n_samples, n_features)
    if shap_values.ndim == 3:
        # average |SHAP| over outputs
        shap_plot_values = np.mean(np.abs(shap_values), axis=2)
    elif shap_values.ndim == 2:
        shap_plot_values = shap_values
    else:
        raise ValueError(
            f"Unsupported shap_values shape: {shap_values.shape}. "
            "Expected 2D or 3D array."
        )

    # Sanity check
    if shap_plot_values.shape[1] != X_samples_shap.shape[1]:
        raise ValueError(
            f"Feature mismatch: shap_plot_values has {shap_plot_values.shape[1]} features, "
            f"but X_samples_shap has {X_samples_shap.shape[1]} features."
        )

    # Group colors
    feature_to_color = {}
    for i, name in enumerate(feature_names_shap):
        if i < num_angle_feats_shap:
            feature_to_color[name] = "#ff0052"   # red
        elif i < num_angle_feats_shap + num_dist_feats_shap:
            feature_to_color[name] = "#008bfb"   # blue
        else:
            feature_to_color[name] = "#2ecc71"   # green

    plt.figure(figsize=(8, 8))

    # Let SHAP draw the bar summary plot first
    shap.summary_plot(
        shap_plot_values,
        X_samples_shap,
        feature_names=feature_names_shap,
        plot_type="bar",
        max_display=len(feature_names_shap),
        show=False,
    )

    # Recolor bars manually based on the displayed y-axis feature labels
    ax = plt.gca()

    # Get displayed feature names directly from the axis
    displayed_feature_names = [tick.get_text() for tick in ax.get_yticklabels()]

    # SHAP bar plot uses matplotlib patches for bars
    bars = ax.patches

    # Usually patches are ordered from bottom to top, matching yticklabels
    if len(bars) != len(displayed_feature_names):
        # Fallback: only recolor the minimum matched count
        n = min(len(bars), len(displayed_feature_names))
    else:
        n = len(bars)

    for bar, feat_name in zip(bars[:n], displayed_feature_names[:n]):
        if feat_name in feature_to_color:
            bar.set_facecolor(feature_to_color[feat_name])
            bar.set_edgecolor("black")
            bar.set_linewidth(0.6)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel("Mean ($|$SHAP Value$|$)", fontsize=25)
    ax.set_ylabel("Features", fontsize=25)

    outpath = os.path.join(figsave_dir, f"SHAP_Feature_Analysis_{dataset_name}.png")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.show()


def plot_shap_analysis_beeswarm(
    shap_values: np.ndarray,
    X_samples: np.ndarray,
    num_angle_feats: int,
    num_dist_feats: int,
    num_elec_feats: int,
    num_angle_feats_shap: int,
    num_dist_feats_shap: int,
    num_elec_feats_shap: int,
    figsave_dir: str,
    dataset_name: str,
) -> None:
    """Generate a SHAP beeswarm summary plot of feature importance.

    Args:
        shap_values: SHAP values array from :func:`compute_shap_values`.
        X_samples: Full feature matrix for the explained samples.
        num_angle_feats: Total angle features in *X_samples*.
        num_dist_feats: Total distance features in *X_samples*.
        num_elec_feats: Total electronic features in *X_samples*.
        num_angle_feats_shap: Angle features used in SHAP analysis.
        num_dist_feats_shap: Distance features used in SHAP analysis.
        num_elec_feats_shap: Electronic features used in SHAP analysis.
        figsave_dir: Directory to save the figure.
        dataset_name: Name appended to the output filename.
    """
    ensure_dir(figsave_dir)

    dist_start = num_angle_feats
    elec_start = num_angle_feats + num_dist_feats

    X_samples_shap = np.concatenate(
        [
            X_samples[:, :num_angle_feats_shap],
            X_samples[:, dist_start : dist_start + num_dist_feats_shap],
            X_samples[:, elec_start : elec_start + num_elec_feats_shap],
        ],
        axis=1,
    )

    feature_names_shap = (
        [f"A{i + 1}" for i in range(num_angle_feats_shap)]
        + [f"D{i + 1}" for i in range(num_dist_feats_shap)]
        + [f"E{i + 1}" for i in range(num_elec_feats_shap)]
    )

    # Average SHAP values over the output dimensions.
    shap_mean = np.mean(np.abs(shap_values), axis=2)

    plt.figure(figsize=(8, 8))

    shap.summary_plot(shap_mean, 
                      X_samples_shap, 
                      feature_names=feature_names_shap, 
                      max_display=X_samples_shap.shape[1], 
                      show=False)

    ax = plt.gca()

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel("Mean ($|$SHAP Value$|$)", fontsize=25)
    ax.set_ylabel("Features", fontsize=25)

    outpath = os.path.join(figsave_dir, f"SHAP_Feature_Analysis_{dataset_name}.png")
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.show()


def plot_train_val_loss(
    train_loss_list: list[float],
    val_loss_list: list[float],
    figsave_dir: str,
    dataset_name: str,
) -> None:
    """Plot training and validation loss curves over epochs.

    Args:
        train_loss_list: Per-epoch training loss values.
        val_loss_list: Per-epoch validation loss values.
        figsave_dir: Directory to save the figure.
        dataset_name: Name appended to the output filename.
    """
    ensure_dir(figsave_dir)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(train_loss_list, "C0-", ms=8, lw=2, label="Train Loss")
    ax.plot(val_loss_list, "C2-", ms=8, lw=2, label="Val Loss")
    ax.legend(loc=1, fontsize=20)

    outpath = os.path.join(figsave_dir, f"Train_Val_Loss_{dataset_name}.png")
    plt.savefig(outpath, bbox_inches="tight")
    plt.show()
