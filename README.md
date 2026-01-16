# A-framework-for-rapid-prediction-of-volatile-chemical-dispersion

# Model 1 – PI-GNN  
**Physics-Informed Graph Neural Network for Chemical Property Learning**

---

## Overview
This repository implements **Model 1 (PI-GNN)** of a multi-modal, physics-informed AI framework designed for **rapid prediction of chemical hazard behavior**.

The primary objective of this model is to **learn latent physical and chemical representations of hazardous substances** using a graph neural network (GNN) constrained by physical priors.  
These learned representations are later used by operator-learning models (e.g., FNO) for fast concentration field prediction, enabling a practical alternative to computationally expensive CFD simulations.

This model focuses on **chemical-level learning**, not full spatio-temporal diffusion.

---

## Motivation
Conventional CFD simulators:
- Are computationally expensive
- Are difficult to deploy in time-critical or resource-limited environments
- Require expert operation and significant preprocessing

The PI-GNN model aims to:
- Learn chemical physical behavior directly from data
- Enable rapid inference once trained
- Generalize to **unseen or unreported chemical agents**
- Serve as a foundational component for downstream operator-learning models

---

## Model Concept
- Each chemical substance is represented as a **node**
- Relationships between substances are modeled as **edges**
- Message passing captures latent physical correlations between chemicals
- Physics-informed loss terms enforce physical consistency

### Outputs
- Predicted physical / thermodynamic parameters
- Latent chemical embeddings for downstream models

---

## Project Structure
```
model1_PIGNN/
├─ requirements.txt
├─ data/
│ ├─ chem_master.csv
│ ├─ chem_targets_params.csv
│ └─ splits/
│ ├─ train.txt
│ ├─ val.txt
│ └─ test.txt
└─ src/
| ├─ config.py
│ ├─ featurize.py
│ ├─ dataset.py
│ ├─ model.py
│ ├─ losses.py
│ ├─ train.py
│ └─ infer.py
```
