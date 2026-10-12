# Numerical data formats

All NPZ inputs are loaded with `allow_pickle=False`. No Python object arrays are needed.
All parameter values passed to the public CLIs are **physical values**, not logarithms,
normalized coordinates or embedding values.

## Parameter order and ranges

The canonical recorded order is:

```text
end_con, end_time, sa, sd, sq, ka, kd, qmax, col_disp, fil_diff, par_diff
```

| Key | Physical training range | Meaning |
| --- | --- | --- |
| end_con | 0.50-0.95 | Terminal organic fraction; initial fraction is 0.05 |
| end_time | 120-300 | Ramp duration in seconds |
| sa, sd, sq | 0-10 | Solvent-strength coefficients |
| ka, kd | 1-1000 | Reference adsorption/desorption constants in the GRM convention |
| qmax | 10-10000 | Reference capacity in the GRM convention |
| col_disp | 1e-9-1e-7 | Axial dispersion, m^2/s |
| fil_diff | 1e-3-1e-1 | Film transfer, m/s |
| par_diff | 1e-9-1e-7 | Particle diffusion, m^2/s |

The last six logarithmic parameters have log10 bounds in `data/para_info.yaml`.
Their public numerical values are positive physical values. The configuration contains
eight sampled programs from `end_con={0.50,0.95}` and `end_time={120,210,270,300}`.
The time-varying operating input is a linear ramp followed by a hold, not an arbitrary
user-defined gradient waveform.

## Training HDF5

| Dataset | Shape | Notes |
| --- | --- | --- |
| Each of the eleven parameter keys above | `(N,)` | One physical value per simulated trajectory |
| bul | `(N, total_time, n_z)` | Dimensionless bulk concentration |
| par | `(N, total_time, n_z*n_r)` | Dimensionless particle concentration, flattened axial-major/radial-minor |
| sol | `(N, total_time, n_z*n_r)` | Dimensionless adsorbed state, same flattening |

All three field datasets are expected, including for a bulk-only training run. The loader
checks their shapes against `data.var_dims`. The primary grid is `(360,101,6)`; an override
must match the actual data. Concentrations use the unit injection reference of the FEM
corpus; `c0`/`bul0` is not one of the eleven FNO inputs.

Training takes the absolute value of targets by default (`data.apply_abs=true`). Particle
and adsorbed fields are scaled by `0.1` and `2e-5` respectively before the weighted loss;
bulk scale is fixed to one in this release. Equal loss weights apply after scaling.
Prediction exports reverse the field scaling. Configure normalization intentionally and
keep the exact YAML beside checkpoints.

The code treats row index zero as the first modeled sample. Operating-rate features
start at `t=1s`; full-resolution output and inverse targets are indexed at `1..360s`.
Align any reference export to that convention before training. With time downsampling,
the historical loader selects rows `0,stride,2*stride,...`, while rate features use
`stride,2*stride,...`; inspect time alignment for a new corpus rather than assuming a
different timestamp origin. The primary configuration has stride one.

## Prediction NPZ

One nonempty one-dimensional array `(N,)` for each of the eleven parameter keys,
all arrays of equal length. Example:

```python
import numpy as np
np.savez('parameters.npz', end_con=np.array([.95]), end_time=np.array([120.]),
         sa=np.array([5.]), sd=np.array([5.]), sq=np.array([5.]),
         ka=np.array([10.]), kd=np.array([10.]), qmax=np.array([100.]),
         col_disp=np.array([1e-8]), fil_diff=np.array([1e-2]), par_diff=np.array([1e-8]))
```

Exported field arrays have leading dimensions `(N, model_time, ...)`. `outlet` is
`(N, model_time)` and `time_s` is `(model_time,)`. The outlet is clipped at zero; raw
field predictions may contain negative values because the model's final activation
is LeakyReLU. Downsampled prediction exports stay on the coarse grid; the inverse
wrapper interpolates to the full grid as in the research implementation.

## Gradient inverse / verification target NPZ

| Key | Shape | Meaning |
| --- | --- | --- |
| outlet | `(full_time, E)` | Dimensionless measured/reference outlet curves |
| end_con | `(E,)` | Fixed terminal organic fractions |
| end_time | `(E,)` | Fixed ramp durations in seconds |

All curves refer to one analyte with shared physical parameters. The FNO inverse CLI
requires its full temporal length and trained operating range. For example, three
gradient experiments have `end_con=[.95,.95,.95]`, `end_time=[120,210,270]` and an outlet
array of shape `(360,3)`. Experimental calibration and concentration normalization
are supplied by the caller; the CLI does not infer injection concentration.

## Isocratic verification target NPZ

Use `outlet (time,E)` and `mobile_fraction (E,)`. COMSOL supports this independently
of the FNO's gradient-only conditioning. Curves use one-second spacing, and the
explicit `--isocratic-kd-scale` is applied only to isocratic COMSOL simulation.
