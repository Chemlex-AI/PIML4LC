# PIML4LC

Code and model resources for differentiable mass-transfer modeling and inverse
analysis of liquid chromatography.

## Resources

| Directory | Contents |
| --- | --- |
| [FNO](FNO/README.md) | Parameter-conditioned autoregressive Fourier neural operator, training, prediction, GA and L-BFGS inversion, and optional reference-solver interfaces |
| [PINN](PINN/README.md) | Physics-informed neural network training and frozen-model parameter inversion |
| [COMSOL_models](COMSOL_models/README.md) | COMSOL model files and resumable GRM batch simulation; require a licensed COMSOL installation |
| [wetlab_params](wetlab_params) | Existing extracted parameter table |

The FNO module accompanies the FNO sections of the manuscript *Differentiable mass
transfer modeling enables wet lab validated inverse analysis of liquid chromatography*.
The PINN module contains a separate physics-informed training script and parameter
inversion script.

## Run COMSOL batch simulations

The [GRM batch script](COMSOL_models/batch_simulation.py) requires an installed, licensed
COMSOL. From the repository root, create the Python environment and parameter plan, then
run the simulations:

```bash
conda env create -f COMSOL_models/environment.yml
conda activate piml4lc-comsol
python COMSOL_models/batch_simulation.py --plan-only
python COMSOL_models/batch_simulation.py --resume --workers 4
```

By default, the script samples 20,000 parameter sets. It writes the parameter table and each completed sample's
`cl`, `cp` and `q` CSV exports under
`COMSOL_models/comsol_sim/`. Run the last command again to retry unfinished samples.
See the [COMSOL module README](COMSOL_models/README.md) for more options.

## Run the FNO example

Use Python 3.11. From a checkout of this repository:

```bash
cd FNO
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python examples/run_smoke.py
```

The example generates artificial Gaussian fields locally. It checks that the software
works; it does not reproduce the manuscript's accuracy or scientific conclusions.
Raw chromatograms, the FEM training corpus, and pretrained FNO weights are not
included in this FNO code release. See [the module documentation](FNO/README.md)
for training with your own reference data and for input formats.

## Run the PINN example

The PINN scripts require your own HDF5 data; this repository does not include a PINN
training set or checkpoint. From the repository root, prepare `PINN/train_data.h5` as
described in the [PINN data format](PINN/docs/data_format.md), then run:

```bash
cd PINN
conda create -n piml4lc-pinn python=3.9
conda activate piml4lc-pinn
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
python 100k_pinn_ddp.py
```

## License

Project code follows the existing [BSD-3-Clause license](LICENSE). Adapted third-party
components retain their original MIT licenses; see
[third-party notices](FNO/THIRD_PARTY_NOTICES.md).
