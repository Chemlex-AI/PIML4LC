# FNO quick start

This module contains the three-field autoregressive forward model, training and
prediction, GA parameter inversion, optional L-BFGS inversion, and COMSOL/CADET
interfaces used in the FNO part of the manuscript. The reported FNO inverse workflow
primarily uses GA followed by COMSOL candidate selection.

This code release includes configuration and a synthetic example generator. It does not
include the original experimental data, FEM training corpus or pretrained weights. The
example uses artificial Gaussian curves to check the software workflow; it cannot
reproduce the manuscript's accuracy or establish parameter identifiability.

## Quick start

Python 3.11 is recommended. From the `FNO/` directory, create an isolated environment
and run:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python examples/run_smoke.py
```

The example generates a small dataset and exercises training, save and load, prediction,
and both inverse methods. Outputs are written under `logs/` and `results/`. The default
full model uses 101 axial nodes, 6 radial nodes, 360 time points, four FNO blocks and
width 128. The CPU example uses a smaller model and grid.

To train on reference data:

```bash
python -m src.train data.h5_file_path=/path/to/train.h5
```

When moving a trained model, keep `checkpoints/`, `.hydra/config.yaml` and
`.hydra/para_info.yaml` together. Parameter inputs must be physical values, rather than
normalized coordinates or logarithms.

A GA target file contains `outlet (time, experiments)`, `end_con (experiments,)` and
`end_time (experiments,)`. Curves in one target file must come from one analyte and share
its nine physical parameters. Experimental intensities require documented concentration
calibration and injection normalization.

```bash
python -m src.inverse --method ga --checkpoint /path/to/run/checkpoints/best.ckpt --target /path/to/target.npz
```

The COMSOL interface requires a licensed installation and the matching model. By
default, isocratic verification leaves `kd` unchanged. The manuscript's twofold `kd`
protocol requires `--isocratic-kd-scale 2` and is recorded in the output. That adjusted
protocol is not a transfer test with every physical parameter held fixed.

For details, see the [module README](../README.md), [data format](data_format.md),
[method notes](method.md), [source mapping](source_mapping.md) and
[validation record](validation.md).
