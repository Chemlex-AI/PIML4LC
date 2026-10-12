# PIML4LC

Code and model resources for differentiable mass-transfer modeling and inverse
analysis of liquid chromatography.

## Resources

| Directory | Contents |
| --- | --- |
| [FNO](FNO/README.md) | Parameter-conditioned autoregressive Fourier neural operator, training, prediction, GA and L-BFGS inversion, and optional reference-solver interfaces |
| [COMSOL_models](COMSOL_models) | Existing COMSOL transport, GRM and equilibrium-dispersive model files; require a licensed COMSOL installation |
| [wetlab_params](wetlab_params) | Existing extracted parameter table |

The FNO module accompanies the FNO sections of the manuscript *Differentiable mass
transfer modeling enables wet lab validated inverse analysis of liquid chromatography*.
It does not implement the separate PINN or Fisher-information observability workflows.

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

## License

Project code follows the existing [BSD-3-Clause license](LICENSE). Adapted third-party
components retain their original MIT licenses; see
[third-party notices](FNO/THIRD_PARTY_NOTICES.md).
