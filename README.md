# TAGG: Topology-aware Graph Diffusion Model with Persistent Homology

Official implementation of **"Topology-aware Graph Diffusion Model with Persistent Homology" (NeurIPS 2025)**.
Note that the code refinement is on the progress. If there are any question about the code or some discrepency in the experiment, feel free to contact me directly!

TAGG is a discrete graph diffusion model that preserves the homological structure of the reference graphs via two components:

- **Persistence Diagram Matching (PDM) loss** (Sec. 4.3): the 1-Wasserstein distance between the 0-dim persistence
  diagrams of the reference graph and the estimated probability vector, computed under a degree-based sublevel set
  filtration (`src/metrics/train_metrics.py`).
- **Topology-aware Attention Module (TAM)** (Sec. 4.2): the persistence landscape vector mu_G0 (L=4, S=16, so
  dv=64) is embedded and added as a global attention bias term in every transformer layer
  (`src/models/transformer_model.py`).

During sampling, the homological condition is the averaged landscape vector mu_G' over all training graphs (Sec. 4.4).

## Environment installation

This code was tested with PyTorch 2.0.1, cuda 11.8 and torch_geometric 2.3.1 on Python 3.9.

  - Create the conda environment (contains rdkit-free graph stack, graph-tool, cuda toolchain):

    ```
    conda env create -f tagg.yaml
    conda activate tagg
    ```

    or install manually:

    ```
    conda create -c conda-forge -n tagg python=3.9 graph-tool=2.45
    conda activate tagg
    pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    ```

  - Check that these lines do not return an error (graph_tool must be imported **before** torch,
    which `src/main.py` already does):

    ```
    python -c 'import graph_tool as gt; import torch'
    python -c 'import gudhi; from gudhi.wasserstein import wasserstein_distance; from gudhi.representations import Landscape'
    ```

  - The orbit metric requires the compiled `orca` binary. If `src/analysis/orca/orca` is missing or not executable:

    ```
    cd src/analysis/orca && g++ -O2 -std=c++11 -o orca orca.cpp
    ```

## Datasets

| dataset (`configs/dataset/`) | source | preparation |
|---|---|---|
| `comm20` (Community-small) | SPECTRE repo | auto-download |
| `planar`, `sbm` | SPECTRE repo | auto-download |
| `enzymes` (ENZYMES) | TUDataset | auto-download |
| `ego_small` | Citeseer ego subgraphs | place `ego_small.pkl` at `data/ego_small/ego_small.pkl` |

Preprocessing (train/val/test split with seed 0 and the pre-computation of the homological feature vector mu_G0)
runs automatically on first use and is cached under `data/tagg/<dataset>/`.
Delete that folder to force re-preprocessing.

## Run the code

All runs are launched from `src/`. Each experiment config selects the matching dataset and the Table 6
hyperparameters automatically:

```
cd src
python main.py +experiment=comm20               # Community-small
python main.py +experiment=ego_small            # Ego-small
python main.py +experiment=enzymes              # ENZYMES
python main.py +experiment=planar               # Planar   (Appendix A.5)
python main.py +experiment=sbm                  # SBM      (Appendix A.5)
```

Other useful commands (see [hydra](https://hydra.cc/) for the override syntax):

```
python main.py +experiment=debug.yaml                        # tiny model, fast_dev_run
python main.py general.test_only=/abs/path/to/checkpoint.ckpt dataset=comm20   # evaluate a checkpoint
python main.py +experiment=comm20 model.use_pdm_loss=0       # ablation: disable the PDM loss
```

Outputs (checkpoints, logs, generated samples) are written to `outputs/<date>/<time>-<name>/`.

## Hyperparameters (Table 6 of the paper)

The training objective is `L = L_CE^V + alpha_1 * L_CE^E + alpha_2 * L_PDM` with
`alpha_1 = model.lambda_train[0]` and `alpha_2 = model.use_pdm_loss`:

| dataset | alpha_1 | alpha_2 |
|---|---|---|
| ENZYMES | 1 | 0.0001 |
| Community-small | 0.001 | 0.001 |
| Ego-small | 0.01 | 0.0001 |

These values are already set in the corresponding `configs/experiment/*.yaml`.
Planar/SBM values are not reported in the paper; the experiment configs use the defaults (1 / 0.001).

## Notes

  - The PDM loss computes persistent homology and the Wasserstein distance on CPU (gudhi has no GPU support),
    which dominates the training time on larger graphs (see Appendix B of the paper). Inference cost is
    unaffected as the PDM loss is only used during training.
  - The homological feature vector mu_G0 is pre-computed during dataset preprocessing, so TAM adds almost no
    training-time overhead.

## Cite the paper

```
@inproceedings{park2025tagg,
title={Topology-aware Graph Diffusion Model with Persistent Homology},
author={Joonhyuk Park and Donghyun Lee and Yujee Song and Guorong Wu and Won Hwa Kim},
booktitle={Advances in Neural Information Processing Systems},
year={2025}
}
```

This codebase is built on [DiGress (Vignac et al., ICLR 2023)](https://github.com/cvignac/DiGress).
