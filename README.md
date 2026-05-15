
# Machine Learning Biomolecular Absorption Spectra in Graphene Nanopores

This project implements a **machine learning (ML) pipeline** using high-throughput physically generated datasets for predicting optical absorption spectra of biomolecules (amino acids) confined within 2D material (graphene) nanopores.

## Key Features

- Physically-informed features (structural and electronic descriptors)
- Principal component Analysis (PCA) based dimensionality reduction
- PyTorch-based multilayer perceptron (MLP) regression
- Statistical evaluation and data visualization
- Feature importance analysis (Permutation + SHAP)
- Hyper-parameter tuning support using grid search method
- Data visualization and publication-ready plots
- CI/CD using GitHub Actions for automated training/inference
- Model building, containerizing, and deploying using Docker


## Project Structure

```
nanopore_ml_optical/
├── config.py
├── main.py
├── dataload/
│   └── loader.py
├── features/
│   └── preprocess.py
├── models/
│   └── models.py
├── training/
│   ├── config.py
│   ├── train.py
│   └── tune.py
├── evaluation/
│   ├── metrics.py
│   └── shap_analysis.py
├── inference/
│   └── predict.py
├── visualization/
│   └── plot.py
└── README.md
```


## Installation Guide

The Python version used: Python 3.10+

### 1. Create environment (recommended)

```
python3 -m venv .venv
source .venv/bin/activate  
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Running Pipeline
```
python main.py
```

## Simulation Datasets
Expected location: `<project_root>/datasets/`

## Future Improvements
- Integrate additional advanced models (e.g., Autoencoder)
- Enable GPU-accelerated model training and inference
- Support cloud-based deployment on platforms such as AWS

## Use Case
This pipeline is designed for:
- Computational materials science, biophysics, and biotechnology
- Nanoscale optical biosensing and spectroscopy prediction
- Machine learning-assisted scientific modelling and simulations

