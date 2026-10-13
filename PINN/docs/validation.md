# PINN validation status

## Checked from the released source

- The training script reads `train_data.h5` and expects a fixed 101-by-6 spatial mapping.
- The inverse script exposes a command-line interface and accepts checkpoints with a
  12-input `IL.weight` layer.
- The inverse input and output schema, parameter order, and current validation behavior
  are documented in [data format](data_format.md) and [method notes](method.md).

## Outside this documentation update

No original HDF5 training corpus or trained PINN weights are bundled, so full training,
reported manuscript metrics and physical inverse recovery were not checked. Multi-GPU
execution was not checked. The current training validation loop does not calculate its
MSE and reuses training rows; its checkpoint score is not a held-out quality estimate.
