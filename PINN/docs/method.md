# PINN method and implementation notes

## Model and training loss

The network maps three coordinates and nine physical parameters to bulk, particle and
stationary concentrations. It has a 12-to-6000 input layer in the training script, two
hidden layers, sigmoid activations and a three-value output. The training loss weights
field MSE, three governing-equation residuals, initial conditions and four boundary
conditions by `[1.0, 1e-10, 1e-5, 1e-5]`. Spatial and temporal derivatives are computed by
PyTorch autograd. `bul0`, gradient phase and injection state are used by the loss but are
not direct network inputs.

The training script reads the whole HDF5 file into memory, remaps the flattened particle
fields, and builds a full sample's point set before batching. It trains at most the first
10,000 samples for 50 epochs. There is no configurable data split: the validation loop
uses rows 0, 1 and 2, which also fall inside training for a normal dataset.

The validation block constructs predictions and targets but does not add an MSE to
`local_mse_sum`. Its reported validation score is consequently zero after the first
nonempty pass. Checkpoints written by this script should be treated as training snapshots
until a real held-out validation metric is implemented. The script has not been revised
here; these notes document its current behavior.

## Frozen-model inverse

The inverse script loads a matching `IL.weight` checkpoint and freezes all model weights.
It optimizes nine input parameters against the first output using L-BFGS and MSE. A
sigmoid transform keeps parameters inside user-supplied bounds; `ka`, `kd`, `qmax`,
`col_disp`, `fil_diff` and `par_diff` are interpolated in log space when their lower
bounds are positive. The other three are interpolated linearly.

The fixed coordinates use `time_index / 10.6`, `axial_index / 100` and `--radius` (default
1). The inverse script does not use `bul0`, `end_con` or `end_time` as network inputs,
which matches the 12-input architecture. It reports an in-sample fit to the supplied
curve. Parameter identifiability and agreement with physical experiments require
independent validation.
