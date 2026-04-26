
# Machine Learning Biomolecular Absorption Spectra in Graphene Nanopores

This project implements a **machine learning (ML) pipeline** using high-throughput density-functional theory (DFT) generated datasets for predicting optical absorption spectra of biomolecules (amino acids) confined within 2D material (graphene) nanopores.

- 📊 Data-driven features (inter-atomic distances, electronic transition energies)
- 🧮 Principal component Analysis (PCA) based dimensionality reduction
- 🤖 PyTorch-based multilayer perceptron (MLP) regression
- 📈 Statistical evaluation and data visualization
- 🔍 Feature importance analysis (Permutation + SHAP)
- ⚙️ Hyper-parameter tuning using grid search method

---

# 🚀 Key Features

- Modular, scalable ML architecture
- PCA compression of spectra (efficient learning)
- Full pipeline from raw (DFT) data → ML prediction → visualization
- Feature importance analysis (Permutation & SHAP)
- Hyperparameter tuning support
- Publication-ready plots

---

# 📁 Project Structure

```
project/
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

---

# ⚙️ Installation Guide

## 1. Create environment (recommended)

```
python -m venv venv
source venv/bin/activate  
```

## 2. Install dependencies

```
pip install -r requirements.txt
```

# ▶️ Running Pipeline
```
python main.py
```

# 📊 DFT Datasets
Expected location: `<project_root>/datasets/`

# 🚀 Future Improvements
- YAML-based configuration
- AutoEncoder & other networks
- Cross-validation (k-fold)
- Full Integration with DFT simulation
- GPU acceleration optimization

# 👨‍🔬 Use Case
This pipeline is designed for:
- Computational materials science and biophysics
- Nanopore-based optical biosensing
- Optical spectroscopy prediction
- Machine learning-assisted DFT simulations

