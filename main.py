"""NanoporeBiosensing — main pipeline entry point.

Orchestrates the full ML workflow: data loading, preprocessing, optional
SHAP analysis, optional hyperparameter tuning, model training, prediction,
evaluation, and visualisation.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import torch

from configure.config import set_seed, get_device
from dataload.loader import load_nanopore_dataframe, build_feature_target_arrays
from features.preprocess import split_data, preprocess_train_test, get_pca_cumulative_variance
from models.models import MLP
from training.train import train_torch_model
from training.tune import grid_search_torch
from inference.predict import predict_absorption
from evaluation.metrics import per_sample_regression_metrics
from evaluation.shap_analysis import compute_shap_values
from visualiz.plot import (
    plot_absorption_prediction,
    plot_model_scores,
    plot_pca_cumulative_variance,
    plot_shap_analysis_bar,
    plot_train_val_loss,
)

warnings.filterwarnings("ignore")


def main() -> None:
    """Run the full nanopore biosensing ML pipeline."""

    root_dir = os.getcwd()

    # -------------------------
    # Parameter Settings
    # -------------------------

    nanopore_name = "GRAPHENE"
    molecule_name = "ALA"
    num_angle_feats = 3
    num_dist_feats = 100
    num_elec_feats = 200
    min_photo_energy = 0
    max_photo_energy = 2

    shap_analysis = True
    num_angle_feats_shap = 3
    num_dist_feats_shap = 20
    num_elec_feats_shap = 20

    num_epochs = 1000
    lr_rate = 1e-3
    dropout = 0.2
    l1_lambda = None
    weight_decay = None
    hidden_dims = (128, 128, 128)
    final_activation = None
    random_state = 1234
    n_components = 10
    log_base = 10
    log_eps = 1.0
    test_size = 0.1
    set_yscale_log = False
    print_every = 10

    hyper_tuning = False
    param_grid = {
        "hidden_dims": [(64, 64), (128, 128), (64, 64, 64), (128, 128, 128)],
        "dropout": [0.1, 0.2, 0.3, 0.5],
        "lr_rate": [1e-2, 1e-3, 1e-4],
        "weight_decay": [None, 1e-6, 1e-5],
        "num_epochs": [200, 400, 600, 800],
        "print_every": [1000],
    }

    # -------------------------
    # Create Figure-saving Folder
    # -------------------------

    figsave_dir = "/the_path_of_the_directory_saving_figures/"
    os.makedirs(figsave_dir, exist_ok=True)

    # -------------------------
    # Config Seed and Device
    # -------------------------

    set_seed(random_state)
    device = get_device()
    print(f"Using device: {device}")

    # -------------------------
    # Load data
    # -------------------------

    dataset_name = f"{nanopore_name}_NANOPORE_BIOMOL_{molecule_name}"
    data_frame, dataset_name = load_nanopore_dataframe(root_dir, dataset_name)

    arrays = build_feature_target_arrays(
        df=data_frame,
        num_angle_feats=num_angle_feats,
        num_dist_feats=num_dist_feats,
        num_elec_feats=num_elec_feats,
        min_photo_energy=min_photo_energy,
        max_photo_energy=max_photo_energy,
    )

    X = arrays["X"]
    Y = arrays["Y"]
    W = arrays["W"]

    # -------------------------
    # Train-Test Split
    # -------------------------

    X_train, X_test, Y_train, Y_test = split_data(
        X, Y, test_size=test_size, random_state=random_state
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")

    # Important: split W consistently with the same indices would be better;
    # For now, rebuild by splitting with the same random_state on row indices.
    idx_all = np.arange(len(X))
    idx_train, idx_test, _, _ = split_data(
        idx_all, Y, test_size=test_size, random_state=random_state
    )
    W_train = W[idx_train]
    W_test = W[idx_test]

    # -------------------------
    # Data Preprocess
    # -------------------------

    prep = preprocess_train_test(
        X_train=X_train,
        X_test=X_test,
        Y_train=Y_train,
        Y_test=Y_test,
        n_components=n_components,
        log_base=log_base,
        log_eps=log_eps,
    )

    X_train_scaled = prep["X_train_scaled"]
    X_test_scaled = prep["X_test_scaled"]
    Y_train_pca = prep["Y_train_pca"]
    Y_test_pca = prep["Y_test_pca"]
    scaler_y = prep["scaler_y"]
    pca = prep["pca"]

    print(f"PCA components: {Y_train_pca.shape[1]}")

    pca_info = get_pca_cumulative_variance(pca)

    # -------------------------
    # Convert to Tensors
    # -------------------------

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_pca, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test_pca, dtype=torch.float32)

    # -------------------------
    # Feature Importance (SHAP)
    # -------------------------

    if shap_analysis:

        shap_values = compute_shap_values(
            X_train_scaled,
            Y_train_pca,
            X_test_scaled,
            num_angle_feats,
            num_dist_feats,
            num_elec_feats,
            num_angle_feats_shap,
            num_dist_feats_shap,
            num_elec_feats_shap,
            hidden_layer_sizes=hidden_dims,
            max_iter=num_epochs,
            random_state=random_state,
        )

        plot_shap_analysis_bar(
            shap_values,
            X_test_scaled,
            num_angle_feats,
            num_dist_feats,
            num_elec_feats,
            num_angle_feats_shap,
            num_dist_feats_shap,
            num_elec_feats_shap,
            figsave_dir=figsave_dir,
            dataset_name=dataset_name,
        )

    # -------------------------
    # Model Hyper-param Tuning
    # -------------------------

    if hyper_tuning:

        tuning_out = grid_search_torch(
            X_train_tensor=X_train_tensor,
            Y_train_tensor=Y_train_tensor,
            X_val_tensor=X_test_tensor,
            Y_val_tensor=Y_test_tensor,
            Y_val_true=Y_test,
            input_dim=X_train_tensor.shape[1],
            output_dim=Y_train_tensor.shape[1],
            pca=pca,
            scaler_y=scaler_y,
            log_base=log_base,
            log_eps=log_eps,
            device=device,
            param_grid=param_grid,
        )

        best_result = tuning_out["best_result"]
        model = best_result["model"]
        train_loss_list = best_result["train_loss_list"]
        val_loss_list = best_result["val_loss_list"]

        print("Best hyperparameters:")
        print(best_result["params"])
        print("Best validation metrics:")
        print(best_result["metrics"])

    # -------------------------
    # Train / Validate a MLP Model
    # -------------------------

    if not hyper_tuning:

        model = MLP(
            input_dim=X_train_tensor.shape[1],
            output_dim=Y_train_tensor.shape[1],
            hidden_dims=hidden_dims,
            dropout=dropout,
            final_activation=final_activation,
        )

        train_out = train_torch_model(
            model=model,
            X_train_tensor=X_train_tensor,
            Y_train_tensor=Y_train_tensor,
            X_val_tensor=X_test_tensor,
            Y_val_tensor=Y_test_tensor,
            num_epochs=num_epochs,
            lr_rate=lr_rate,
            l1_lambda=l1_lambda,
            weight_decay=weight_decay,
            device=device,
            print_every=print_every,
        )

        model = train_out["model"]
        train_loss_list = train_out["train_loss_list"]
        val_loss_list = train_out["val_loss_list"]

    # -------------------------
    # Apply the Model to Predict
    # -------------------------

    Y_train_pred = predict_absorption(
        model=model,
        X_tensor=X_train_tensor,
        pca=pca,
        scaler_y=scaler_y,
        log_base=log_base,
        log_eps=log_eps,
        device=device,
    )

    Y_test_pred = predict_absorption(
        model=model,
        X_tensor=X_test_tensor,
        pca=pca,
        scaler_y=scaler_y,
        log_base=log_base,
        log_eps=log_eps,
        device=device,
    )

    # -------------------------
    # Evaluate Model Accuracy
    # -------------------------

    train_metrics = per_sample_regression_metrics(Y_train, Y_train_pred)
    test_metrics = per_sample_regression_metrics(Y_test, Y_test_pred)

    print(f"Train mean R2   = {train_metrics['r2_mean']:.3f}")
    print(f"Test mean R2    = {test_metrics['r2_mean']:.3f}")

    # -------------------------
    # Plots for Different Results
    # -------------------------

    plot_train_val_loss(
        train_loss_list,
        val_loss_list,
        figsave_dir,
        dataset_name,
    )

    plot_pca_cumulative_variance(
        n_components=pca_info["n_components"],
        cumulative_explained_variance=pca_info["cumulative_explained_variance"],
        figsave_dir=figsave_dir,
        dataset_name=dataset_name,
    )

    plot_model_scores(
        Y_train_true=Y_train,
        Y_train_pred=Y_train_pred,
        Y_test_true=Y_test,
        Y_test_pred=Y_test_pred,
        figsave_dir=figsave_dir,
        dataset_name=dataset_name,
    )

    plot_absorption_prediction(
        X_test=X_test,
        Y_test=Y_test,
        Y_test_pred=Y_test_pred,
        W_test=W_test,
        figsave_dir=figsave_dir,
        dataset_name=dataset_name,
        min_photo_energy=min_photo_energy,
        max_photo_energy=max_photo_energy,
        log_base=log_base,
        log_eps=log_eps,
        set_yscale_log=set_yscale_log,
        n_groups_plot=15,
        group_size_plot=3,
    )


if __name__ == "__main__":
    main()
