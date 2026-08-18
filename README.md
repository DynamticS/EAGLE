# EAGLE

Training code for DEAP arousal / valence / liking classification.

## Environment

Tested with:

| Package | Version |
|---|---|
| Python | 3.8.19 |
| PyTorch | 2.2.2 (CUDA 11.8) |
| numpy | 1.24.3 |
| scikit-learn | 1.3.0 |
| scipy | 1.10.1 |
| einops | 0.8.1 |
| h5py | 3.11.0 |
| torch-geometric | 2.5.3 |
| torch-scatter | 2.1.2+pt22cu118 |
| torch-sparse | 0.6.18+pt22cu118 |

A conda environment named `GraphEEG` is assumed. Override the interpreter with `PYTHON=/path/to/python`.

A CUDA GPU is required.

## DEAP data

The DEAP dataset is **not** included. You must apply for access and download it from the official DEAP site:

https://www.eecs.qmul.ac.uk/mmv/datasets/deap/

Place the preprocessed feature files under a single directory (default `./data/features/`):

```
A_Type_DEAP_AllSub_Combined.npz
V_Type_DEAP_AllSub_Combined.npz
L_Type_DEAP_AllSub_Combined.npz
```

Each file must contain arrays `data` and `label` that reshape to `(32, 600, 32, 512)` and `(32, 600)`.

## Run

```bash
export DATA_PATH=/path/to/deap/features/
bash run.sh
```

Optional environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `DATA_PATH` | `./data/features/` | Directory of the three `.npz` files |
| `PYTHON` | `/home/wsl/.conda/envs/GraphEEG/bin/python` | Python interpreter |
| `GPU` | `0` | CUDA device id |
| `SEED` | `666` （You can also try setting other） | Random seed |

The script trains labels A, V, and L in sequence. Checkpoints are written under `./save/EAGLE-{A,V,L}-s666/`.
