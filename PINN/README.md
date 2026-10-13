# PINN for liquid chromatography

This module contains a physics-informed neural network (PINN) for the three-field
chromatography model and a frozen-network L-BFGS inverse script. The training loss
combines field labels with governing-equation, initial-condition and boundary-condition
residuals. The inverse script fits nine physical parameters to the first network output.

**Release scope:** source code and usage documentation. The reference HDF5 data, trained
checkpoints, experimental chromatograms and manuscript results are not included. Training
on your own data is needed to produce a usable checkpoint. A good curve fit alone does not
establish unique parameter recovery.

## Installation

These direct dependencies match the installed versions in the local conda `base`
environment (Python 3.9.7). Create a Python 3.9 environment and run these commands
from `PINN/`:

```bash
conda create -n piml4lc-pinn python=3.9
conda activate piml4lc-pinn
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

The pinned PyTorch build is the CUDA 11.8 build found in `base`. Multi-GPU training
requires compatible NVIDIA hardware and NCCL; the inverse script can run on CPU.
`requirements.txt` lists only packages imported directly by these PINN scripts, not
their transitive dependencies or Python's standard library.

## Train on reference data

Prepare `train_data.h5` as described in [data format](docs/data_format.md). The training
script currently expects this exact filename in the working directory:

```bash
python 100k_pinn_ddp.py
```

For multiple CUDA devices:

```bash
torchrun --standalone --nproc_per_node=2 100k_pinn_ddp.py
```

The script uses a 6000-wide network, 50 epochs, up to 10,000 training samples and a
101-by-6 spatial grid. These settings can require substantial memory and compute. It
attempts to write a checkpoint to `100k_pinn_test.pt` in the working directory. The
current validation block does not accumulate its MSE, so its reported score is zero and
does not select a scientifically meaningful best model. It also uses the first three
HDF5 rows, which overlap training. See [method notes](docs/method.md) for more limits.

## Inverse estimation

Supply a checkpoint from this training architecture and `test_data.h5` following the
[HDF5 schema](docs/data_format.md):

```bash
python reverse_infer_pinn_ddp.py \
  --checkpoint 100k_pinn_test.pt \
  --data-path /path/to/test_data.h5 \
  --sample-index 0 \
  --output reverse_infer_result.json
```

By default, the target is the selected sample's `bul` at axial node 100 for time indices
0 through 300. `--target /path/to/curve.npy` or a one-column CSV supplies an external
target in the same time/axial order. `--time-slice START STOP STRIDE`, `--axial-indices`
and `--outlet-index` select points. `--initial-json` supplies nine initial parameters;
`--use-data-initial` uses the chosen HDF5 row. Bounds can be set with `--lower` and
`--upper`, each with nine values in the documented order. The script prints optimization
history and writes best/final parameters and metrics to JSON. It loads only checkpoints
whose first layer is named `IL.weight` and has 12 inputs.

The default `--hidden-size 6000` matches training. If a checkpoint uses a different
width, pass it explicitly with `--hidden-size`. Set `--device cpu` if needed. Only load
trusted checkpoints because PyTorch checkpoint loading deserializes data.

## Validation and provenance

See [validation notes](docs/validation.md) for what has been checked locally and what
requires original data or GPU hardware. Project code follows the repository's
[BSD-3-Clause license](../LICENSE).
