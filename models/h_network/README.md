# Constraint-Aware h-Network for gamma Extraction

Implements the constraint-aware extraction of gamma from B+- -> DK+- decays
using a SIREN h-network that learns cos(delta_D) from CP-tagged D data.

## Method

The h-network replaces two independent CP-tagged normalizing flows with a single
regression network h: [0,1]^2 -> [-1,+1] (output constrained by tanh). Given a
frozen flavor flow K(m', theta'), the interference observables are:

    C = sqrt(K * Kbar) * h
    |S| = sqrt(K * Kbar) * sqrt(1 - h^2)

This guarantees C^2 + S^2 = K*Kbar by construction, eliminating unphysical
constraint violations that arise in the three-flow approach.

## Architecture

- Symmetric SIREN (Sinusoidal Representation Network), 256x5, omega_0=15
- theta' -> 1-theta' symmetry enforced by averaging: h(m',th') = [g(m',th') + g(m',1-th')]/2
- ~198k trainable parameters
- SDP convention: idx=(2,3,1), pi+pi- resonant pair
- Architecture defined in: models.py

## Directory Structure

    models/h_network/
    ├── models.py                      # Architecture definitions (SIREN, flow)
    ├── __init__.py
    ├── run_ensemble_comparison.py     # Main training + gamma fit script
    ├── ensemble_comparison_config.json
    ├── README.md
    │
    ├── weights/                       # Trained h-network weights
    │   └── h_ensemble_sym_seed{0-4}.pth
    │
    ├── results/                       # All fit results
    │   ├── ensemble_comparison_symmetric_results.json  # 5-seed h-network fits
    │   ├── fit_results_symmetric.npz                   # 25-seed 3-flow baseline
    │   ├── amplitude_model_fit.npz                     # Exact isobar fit
    │   ├── ablation_results.json      # Ablation study
    │   ├── ablation_results.txt       # Human-readable ablation table
    │   ├── tuning_results.json        # Hyperparameter tuning
    │   └── scaling_results.json       # Data scaling study
    │
    ├── analysis/                      # Analysis scripts and notebooks
    │   ├── plot_comparison.py         # Generates comparison plots
    │   ├── plot_h_vs_truth.ipynb      # h-network vs true cos(delta_D)
    │   ├── run_tuning.py             # Hyperparameter tuning
    │   └── run_scaling_study.py       # Data scaling study
    │
    └── figures/                       # Publication figures
        ├── corner_comparison_pub.pdf
        ├── h_vs_truth_pub.pdf
        ├── method_comparison_symmetric.pdf
        └── triangle_clean.pdf

## Related Data

    data/
    ├── b_pseudo/BpBm_samples_rB0.1.npz     # Shared B± data (100k per charge)
    └── cp_tagged/{even,odd}_symmetric_datasets_sdp/  # 25 CP datasets (2M each)

    models/flavor_flow/
    ├── weights/trial_seed{0-24}.pth   # 25 flavor flows from collaborator
    ├── config.json
    └── summary.json

## Correspondence

Each dataset index i (0-4) uses:
- Flavor flow: models/flavor_flow/weights/trial_seed{i}.pth
- CP-even data: data/cp_tagged/even_symmetric_datasets_sdp/dataset_{i:03d}.npy
- CP-odd data:  data/cp_tagged/odd_symmetric_datasets_sdp/dataset_{i:03d}.npy
- h-network:    models/h_network/weights/h_ensemble_sym_seed{i}.pth
- B+- data:     data/b_pseudo/BpBm_samples_rB0.1.npz (SHARED across all seeds)

## Usage

    # From the repo root:
    python models/h_network/run_ensemble_comparison.py --symmetric --n-trials 5 --epochs 100 --resume

    # Generate comparison plots:
    python models/h_network/analysis/plot_comparison.py
