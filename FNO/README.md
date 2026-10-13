# FNO for liquid chromatography

This module provides the autoregressive Fourier neural operator (FNO) and frozen-surrogate
inverse search used in the FNO part of PIML4LC. It includes a portable training pipeline,
field prediction, genetic search (GA), optional multi-start L-BFGS, and optional COMSOL/CADET
reference interfaces. The primary manuscript FNO inverse workflow uses GA followed by
COMSOL candidate selection; L-BFGS is an alternative with a different objective.

**Release scope:** code, configuration, tests, and an artificial example generator.
The original FEM corpus, experimental chromatograms, trained checkpoints and numerical
results are not bundled. Without those inputs this release cannot independently reproduce
the manuscript's reported metrics. No trained model is downloaded by the example.

## Installation

Python 3.11 is recommended; the package metadata permits Python 3.10-3.12.
The tested CPU environment uses PyTorch 2.6, Lightning 2.5, Hydra 1.3 and TensorLy 0.9.
Create an isolated environment and run all commands below from `FNO/`:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The [quick start](docs/quick_start.md) summarizes the same basic workflow.
`constraints-tested.txt` records exact versions from the validated Python 3.11 CPU
environment; optionally add `-c constraints-tested.txt` when installing requirements.

For GPU training, install a CUDA-enabled PyTorch 2.6 build appropriate for your system
using the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/).
The primary configuration uses float32 throughout. CPU execution needs no GPU,
distributed launcher, COMSOL, CADET, external account or logging-service credential.

## Quick software check

```bash
python examples/run_smoke.py
```

This creates 20 artificial samples on a 12-time-step, 17-axial-node, 2-radial-node grid,
runs two training batches and one validation/test batch, reloads the checkpoint, predicts
fields, then exercises both inverse methods. Generated files live under `data/`, `logs/`
and `results/`; they are ignored by Git. Re-running the example overwrites its named
example inputs and result files.

The example fields are Gaussian profiles, **not GRM/FEM solutions**. The miniature model
is deliberately untrained for scientific use. Inverse fits to these curves demonstrate
execution only and do not establish physical parameter recovery.

## Train and evaluate on reference data

Prepare an HDF5 corpus following [the data format](docs/data_format.md), then run:

```bash
python -m src.train data.h5_file_path=/path/to/train.h5
```

The default configuration corresponds to the primary three-field FNO: 101 axial nodes,
6 radial nodes, 360 one-second steps, 13 state channels, four Fourier blocks, configured
mode count 64, and width/conditioning dimension 128. Bulk initial conditions are learned
from the static parameters; particle and adsorbed fields start at zero. See
[the method notes](docs/method.md) for the real-FFT mode convention and conditioning.

```bash
# Single GPU
python -m src.train data.h5_file_path=/path/to/train.h5 trainer.accelerator=gpu

# Optional bulk-only ablation
python -m src.train data.h5_file_path=/path/to/train.h5 model.bul_only=true

# Evaluate with the same architecture, normalization and grid used for training
python -m src.eval ckpt_path=/path/to/run/checkpoints/best_epoch000.ckpt data.h5_file_path=/path/to/train.h5
```

For a non-default model, pass the same overrides (including `experiment=smoke` for a
smoke checkpoint) to `src.eval`. Alternatively train with `test=true` (the default) to
evaluate the selected best checkpoint immediately. `src.eval` reports held-out loss and
peak-index error; it does not produce the manuscript's complete statistical result tables.

The data loader shuffles row indices with seed 42 and splits approximately 80/10/10;
validation and test counts are each `min(10000, floor(0.1*N))`. At least 10 rows are required.
This matches the 15929/1990/1990 split for 19909 rows. For a corpus containing repeated
physical parameter sets, arrange a parameter-disjoint split before claiming out-of-sample
generalization: this implementation splits rows, not groups.

Runs are saved under `logs/fno_rollout/runs/<timestamp>/` with checkpoints, CSV metrics,
`.hydra/config.yaml` and `.hydra/para_info.yaml`. **Keep these files together when moving
a trained model.** Checkpoints are not self-describing without their saved configuration.
Only load checkpoints from trusted sources; Lightning checkpoint loading uses Python
deserialization.

## Predict fields and outlet chromatograms

Provide an NPZ file with physical parameter vectors, one vector per key:

```bash
python -m src.predict --checkpoint /path/to/run/checkpoints/best_epoch000.ckpt --parameters /path/to/parameters.npz --output results/prediction.npz
```

Outputs contain `bul`, `par`, `sol` (for a three-field model), nonnegative `outlet` and
`time_s`. Field normalizations are reversed on export. Spatial/temporal dimensions come
from the checkpoint configuration. Optional `--config` and `--parameter-config` arguments
allow explicitly supplying metadata for older checkpoints. A sidecar normalization file
is preferred over machine-specific paths stored in historical configurations.

## Inverse estimation

Supply dimensionless outlet curves on the model's one-second full grid, together with
the fixed gradient settings for each experiment. See [NPZ schemas](docs/data_format.md).
All curves in one target file must belong to one analyte and share its nine physical
parameters. Detector intensities require a documented experimental calibration and
injection normalization before use; arbitrary rescaling can change the inferred parameters.

```bash
# Primary FNO search: default population 1024, 15 generations
python -m src.inverse --method ga --checkpoint /path/to/run/checkpoints/best.ckpt --target /path/to/target.npz --output-dir results/ga

# Optional differentiable search; gradients pass through the frozen FNO
python -m src.inverse --method lbfgs --checkpoint /path/to/run/checkpoints/best.ckpt --target /path/to/target.npz --n-starts 8 --max-iter 30 --output-dir results/lbfgs
```

The CLI writes ranked physical parameters, curves, arguments, optimization history and
an overlay plot. GA retains globally ranked candidates, removes neighbors within radius
0.1 in the normalized nine-dimensional physical space, and keeps up to 128 candidates.
`--chunk-size` controls memory use. The default value is conservative for CPU and small GPUs;
increase it for faster batched GA inference if memory permits.

The default artifact suppression is ten initial samples: GA zeroes predicted values before
scoring; L-BFGS excludes those time points from its objective. Set `--zero-first-n 0` for
the twelve-point artificial example. GA and L-BFGS loss values use different objectives
and should not be directly compared. Good surrogate fits do not prove parameter uniqueness.

## Optional reference simulation and verification

COMSOL requires a compatible licensed installation, the `mph` Python bridge, and the
GRM model already stored in the parent repository:

```bash
python -m pip install -r requirements-simulation.txt
python -m src.simulate --backend comsol --parameters /path/to/parameters.npz --model ../COMSOL_models/GRM.mph --output data/train.h5
```

This sequential command exclusively creates its HDF5 output, records completed samples,
and fails on solver errors. Match the returned field shapes to the training grid. COMSOL
model parameter names, the study name and named export nodes must match the adapter.
Core FNO execution does not import this bridge.

To rank the top eight FNO candidates by COMSOL retention-time RMSE on the gradient data:

```bash
python -m src.verify --candidates results/ga/parameters.csv --gradient-target /path/to/gradient_target.npz --model ../COMSOL_models/GRM.mph --output-dir results/verification
```

Optionally add `--isocratic-target /path/to/isocratic_target.npz` for independent isocratic
evaluation. Candidate selection uses gradient data only. The default isocratic desorption
scale is one; to explicitly follow the manuscript's adjusted protocol, add
`--isocratic-kd-scale 2`. This adjustment means the reported isocratic protocol is not
a strict transfer test with all physical parameters unchanged. The scale is saved in the
verification report. Traces without an observed peak are excluded from score summaries.

Historical CADET adapters are also retained. Install the matching `cadet-python` interface
and CADET executable, then set `CADET_CLI_PATH` or put `cadet-cli` on `PATH`; select
`--backend cadet` in `src.simulate`. CADET and COMSOL execution were not tested in the
release preparation environment. The primary FNO results use COMSOL, and compatibility
of the historical CADET adapter with current CADET versions is not established.

## Tests and provenance

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover physical normalization and conditioning, full/bulk-only rollout layouts,
gradient propagation, chunked multi-start optimization against a known analytic optimum,
reference peak metrics and a real train/save/relocate/load/predict/inverse software check.
The GitHub workflow runs these CPU checks on Linux and Windows; hosted CI execution
is pending until the changes are pushed. See [source mapping](docs/source_mapping.md)
for what was retained, changed and excluded from the research workspace.

Project code follows the repository's BSD-3-Clause license. NeuralOperator-derived
layers and Lightning-Hydra-Template-derived utilities retain their MIT notices in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
