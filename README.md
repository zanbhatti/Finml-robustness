# FinML: Robustness of Option Pricing Models

A framework for evaluating the accuracy, robustness, and structural validity of classical and machine learning-based option pricing models under controlled input perturbations.

---

## Overview

This project studies how different models approximate option prices and how they behave under changes in input conditions.

We compare:
- Classical pricing methods
- Supervised machine learning models

The key idea is to move beyond accuracy and evaluate:
- Stability under perturbations
- Generalization behavior
- Structural consistency with financial theory

---

## Current Implementation (v0)

### Data Generation
- Synthetic dataset generated using the Black–Scholes model
- Inputs: `S, K, T, r, sigma`
- Targets: `call_price`, `put_price`

### Models
- Black–Scholes (analytical ground truth)
- MLP (neural network surrogate)

### Training Pipeline
- Supervised regression setup
- 80/20 train-test split
- PyTorch-based training loop
- Loss: Mean Squared Error (MSE)

### Evaluation
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

## Results (Initial)

Example output:
Epoch [100/100], Train Loss: ~20
MAE: ~3.4
RMSE: ~4.4

The neural network successfully learns the pricing function, with decreasing training loss and reasonable test error.

---

## Project Structure

finml-robustness/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── options_data.csv
│
├── src/
│   ├── data/
│   │   └── generate_dataset.py
│   ├── models/
│   │   ├── black_scholes.py
│   │   └── mlp.py
│   ├── training/
│   │   └── train_mlp.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── utils/
│       └── helpers.py
│
└── results/

---

## How to Run

### 1. Create environment (recommended)
conda create -n finml python=3.10
conda activate finml
pip install -r requirements.txt

### 2. Generate dataset
python -m src.data.generate_dataset

### 3. Train models
python -m src.training.train_mlp

---

## Motivation

In financial modeling, accuracy alone is not sufficient.

This project explores:
- When models perform well
- When they fail
- How robust they are to changes in inputs

This is critical for:
- Quantitative research
- Risk management
- Machine learning in finance

---

## Next Steps

- Feature normalization (improve model performance)
- Add additional models (XGBoost, Monte Carlo, Binomial)
- Implement robustness testing under input perturbations
- Evaluate structural validity (monotonicity, arbitrage constraints)
- Extend to distribution shift and stress scenarios

---

## Author

Zan Bhatti