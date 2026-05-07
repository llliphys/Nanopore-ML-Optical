"""
The main entry point of the full ML pipeline integrating data loading, preprocessing, optinal PCA, 
optional SHAP analysis, optional hyperparameter tuning, model training, prediction,
evaluation, data visualisation, and CI/CD practices.
"""

from __future__ import annotations

import copy
import os
import random
import warnings
from typing import Any

import numpy as np
import torch

try:
    import yaml
except ImportError:
    yaml = None

from dataload.loader import (
    load_nanopore_dataframe, 
    build_feature_target_arrays
)
from features.preprocess import (
    split_data, 
    preprocess_train_test, 
    get_pca_cumulative_variance
)
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


default_config: dict[str, Any] = {
    "dataset": {
        "nanopore_name": "GRAPHENE",
        "molecule_name": "ALA",
        "num_angle_feats": 3,
        "num_dist_feats": 100,
        "num_elec_feats": 200,
        "min_photo_energy": 0,
        "max_photo_energy": 2,
    },
    "random_seed": 1234,
    "preprocess": {
        "n_components": 10,
        "log_base": 10,
        "log_eps": 1.0,
        "test_size": 0.1,
        "set_yscale_log": False,
    },
    "shap": {
        "shap_analysis": False,
        "num_angle_feats_shap": 3,
        "num_dist_feats_shap": 20,
        "num_elec_feats_shap": 20,
    },
    "training": {
        "device": "cpu",
        "num_epochs": 1000,
        "lr_rate": 0.001,
        "dropout": 0.2,
        "l1_lambda": None,
        "weight_decay": None,
        "hidden_dims": [128, 128, 128],
        "final_activation": None,
        "print_every": 10,
    },
    "tuning": {
        "hyper_tuning": False,
        "param_grid": {
            "hidden_dims": [
                [64, 64],
                [128, 128],
                [64, 64, 64],
                [128, 128, 128],
            ],
            "dropout": [0.1, 0.2, 0.3, 0.5],
            "lr_rate": [0.01, 0.001, 0.0001],
            "weight_decay": [None, 0.000001, 0.00001],
            "num_epochs": [200, 400, 600, 800, 1000, 2000],
            "print_every": [1000],
        },
    },
    "plotting": {
        "plot_pca_cumulative_variance": True,
        "plot_train_val_loss": False,
        "plot_shap_analysis_bar": False,
        "plot_model_scores": True,
        "plot_absorption_prediction": True,
        "plot_absorption_prediction_nrows": 4,
        "plot_absorption_prediction_ncols": 4,
    },
    "paths": {
        "dataset_dir": "datasets",
        "figure_save_dir": "results/figures",
        "model_save_dir": "results/models",
    },
}


def load_config(config_yaml: str) -> dict[str, Any]:
    """Load the YAML configuration file."""
    if yaml is None:
        raise ImportError("PyYAML is not installed. Please install it to use config files.")
    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> torch.device:
    """Get the PyTorch device to use for computations."""
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")  


def main() -> None:
    """Run the full nanopore optical biosensing ML pipeline."""
    root_dir = os.getcwd()
    config_path = os.path.join(root_dir, "config.yaml")

    try:
        config_yaml = load_config(config_path) 
        config = copy.deepcopy(config_yaml)
        print(f"Loaded configuration from {config_path}")
    except Exception as exc:
        print(f"Error loading config.yaml: {exc}")
        print("Falling back to default configuration values.")
        config = copy.deepcopy(default_config)

    dataset_config = config["dataset"]
    preprocess_config = config["preprocess"]
    shap_config = config["shap"]
    training_config = config["training"]
    tuning_config = config["tuning"]
    plot_config = config["plotting"]
    paths_config = config["paths"]

    nanopore_name = dataset_config["nanopore_name"]
    molecule_name = dataset_config["molecule_name"]
    num_angle_feats = dataset_config["num_angle_feats"]
    num_dist_feats = dataset_config["num_dist_feats"]
    num_elec_feats = dataset_config["num_elec_feats"]
    min_photo_energy = dataset_config["min_photo_energy"]
    max_photo_energy = dataset_config["max_photo_energy"]

    random_state = config.get("random_seed", 1234)

    n_components = preprocess_config["n_components"]
    log_base = preprocess_config["log_base"]
    log_eps = preprocess_config["log_eps"]
    set_yscale_log = preprocess_config["set_yscale_log"]
    test_size = preprocess_config["test_size"]

    shap_analysis = shap_config["shap_analysis"]
    num_angle_feats_shap = shap_config["num_angle_feats_shap"]
    num_dist_feats_shap = shap_config["num_dist_feats_shap"]
    num_elec_feats_shap = shap_config["num_elec_feats_shap"]

    device = training_config["device"]
    num_epochs = training_config["num_epochs"]
    hidden_dims = tuple(training_config["hidden_dims"])
    lr_rate = training_config["lr_rate"]
    dropout = training_config["dropout"]
    l1_lambda = training_config["l1_lambda"]
    weight_decay = training_config["weight_decay"]
    final_activation = training_config["final_activation"]
    print_every_training = training_config["print_every"]

    hyper_tuning = tuning_config["hyper_tuning"]
    print_every_tuning = tuning_config["print_every"]
    param_grid = copy.deepcopy(tuning_config["param_grid"])
    param_grid["hidden_dims"] = [tuple(dims) for dims in param_grid["hidden_dims"]]

    dataset_dir = paths_config["dataset_dir"]
    figure_save_dir = paths_config["figure_save_dir"]
    model_save_dir = paths_config["model_save_dir"]
    os.makedirs(os.path.join(root_dir, dataset_dir), exist_ok=True)
    os.makedirs(os.path.join(root_dir, figure_save_dir), exist_ok=True)
    os.makedirs(os.path.join(root_dir, model_save_dir), exist_ok=True)

    plot_pca = plot_config["plot_pca_cumulative_variance"]
    plot_shap = plot_config["plot_shap_analysis_bar"]
    plot_loss = plot_config["plot_train_val_loss"]
    plot_scores = plot_config["plot_model_scores"]
    plot_prediction = plot_config["plot_absorption_prediction"]
    plot_prediction_nrows = plot_config["plot_absorption_prediction_nrows"]
    plot_prediction_ncols = plot_config["plot_absorption_prediction_ncols"]

    set_seed(random_state)
    device = get_device(device)
    print(f"Using device: {device}")

    dataset_name = f"{nanopore_name}_NANOPORE_BIOMOL_{molecule_name}"

    try:
        data_frame, dataset_name = load_nanopore_dataframe(dataset_dir, dataset_name)
    except Exception as exc:
        print(f"Error loading nanopore dataframe: {exc}")
        print("No dataset found for the specified parameters. Exiting.")
        raise

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

    X_train, X_test, Y_train, Y_test = split_data(
        X, Y, test_size=test_size, random_state=random_state,
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")

    idx_all = np.arange(len(X))
    _, idx_test, _, _ = split_data(
        idx_all, Y, test_size=test_size, random_state=random_state,
    )
    W_test = W[idx_test]

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

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train_pca, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test_pca, dtype=torch.float32)

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
            print_every=print_every_tuning,
        )
        best_result = tuning_out["best_result"]
        model = best_result["model"]
        train_loss_list = best_result["train_loss_list"]
        val_loss_list = best_result["val_loss_list"]

        print("Best hyperparameters:")
        print(best_result["params"])
        print("Best validation metrics:")
        print(best_result["metrics"])

    else:

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
            print_every=print_every_training,
        )
        model = train_out["model"]
        train_loss_list = train_out["train_loss_list"]
        val_loss_list = train_out["val_loss_list"]

        torch.save(model.state_dict(), model_save_dir + f"/model_{dataset_name}.pt")

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

    train_metrics = per_sample_regression_metrics(Y_train, Y_train_pred)
    test_metrics = per_sample_regression_metrics(Y_test, Y_test_pred)
    print(f"Train mean R2   = {train_metrics['r2_mean']:.3f}")
    print(f"Test mean R2    = {test_metrics['r2_mean']:.3f}")

    if plot_loss:
        plot_train_val_loss(
            train_loss_list,
            val_loss_list,
            figure_save_dir,
            dataset_name,
        )
    if plot_pca:
        plot_pca_cumulative_variance(
            n_components=pca_info["n_components"],
            cumulative_explained_variance=pca_info["cumulative_explained_variance"],
            figsave_dir=figure_save_dir,
            dataset_name=dataset_name,
        )
    if plot_shap:
        plot_shap_analysis_bar(
            shap_values,
            X_test_scaled,
            num_angle_feats,
            num_dist_feats,
            num_elec_feats,
            num_angle_feats_shap,
            num_dist_feats_shap,
            num_elec_feats_shap,
            figsave_dir=figure_save_dir,
            dataset_name=dataset_name,
        )
    if plot_scores:
        plot_model_scores(
            Y_train_true=Y_train,
            Y_train_pred=Y_train_pred,
            Y_test_true=Y_test,
            Y_test_pred=Y_test_pred,
            figsave_dir=figure_save_dir,
            dataset_name=dataset_name,
        )
    if plot_prediction:
        plot_absorption_prediction(
            X_test=X_test,
            Y_test=Y_test,
            Y_test_pred=Y_test_pred,
            W_test=W_test,
            figsave_dir=figure_save_dir,
            dataset_name=dataset_name,
            min_photo_energy=min_photo_energy,
            max_photo_energy=max_photo_energy,
            log_base=log_base,
            log_eps=log_eps,
            set_yscale_log=set_yscale_log,
            plot_nrows=plot_prediction_nrows,
            plot_ncols=plot_prediction_ncols,

    )


if __name__ == "__main__":
    main()
