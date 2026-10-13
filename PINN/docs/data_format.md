# PINN data formats

The scripts read HDF5 files from local paths. Dataset names are case-sensitive. All
arrays use the same sample order; neither script checks that physical parameter rows
match their field rows.

## Training HDF5

`100k_pinn_ddp.py` reads `./train_data.h5` and expects these datasets:

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `ka`, `kd`, `qmax`, `sa`, `sd`, `sq`, `col_disp`, `fil_diff`, `par_diff`, `bul0` | `(N,)` or `(N, 1)` each | Nine physical inputs, then initial bulk concentration |
| `end_con`, `end_time` | `(N,)` or `(N, 1)` each | Gradient endpoint concentration and time |
| `bul` | `(N, T, 101)` | Bulk field |
| `par`, `sol` | `(N, T, 606)` | Particle and stationary fields in the script's flattened spatial order |

The script remaps each 606-column `par` and `sol` row to `(101, 6)` using the fixed
axial/radial coordinate lists at the top of the source file. The flattened column order
must match those lists. An HDF5 file already stored as `(N, T, 101, 6)` does not match
this loader. Training loops over every time, axial and radial location, so memory use
scales with `N × T × 101 × 6`. At least three rows are needed by the current validation
loop. `end_time` must be nonzero because it divides the gradient slope.

The 12 network inputs are dimensionless time (`time_index / 10.6`), dimensionless axial
position (`axial_index / 100`), one of six radial values (`0.7037`, `0.76296`, `0.82222`,
`0.88148`, `0.94074`, `1`), then `ka`, `kd`, `qmax`, `sa`, `sd`, `sq`, `col_disp`,
`fil_diff`, `par_diff`. `bul0`, current gradient and injection terms enter the physics
loss rather than the network. The three outputs are `bul`, `par` and `sol`.

## Inverse HDF5 and target curve

`reverse_infer_pinn_ddp.py` reads the nine physical parameter datasets above, `bul0`
and `bul` from `--data-path` (default `./test_data.h5`). The nine parameters and `bul0`
are concatenated to obtain the row count and optional initial guess. `bul` needs shape
`(N, T, L)`, with enough time and axial entries for the selected indices. The default
selection uses time indices `0:301:1` and axial index `100`.

An optional `--target` file may be a one-dimensional `.npy` array or a headerless numeric
CSV. Its length must equal `len(time_indices) × len(axial_indices)`. Values are ordered
with time as the outer loop and axial index as the inner loop. The model output being fit
is the first field (`bul`); use the same scale and units as the training labels.

`--initial-json` accepts either a list of nine numbers or an object with the keys below.
The same order applies to `--lower` and `--upper`:

| Position | Parameter |
| --- | --- |
| 1–3 | `ka`, `kd`, `qmax` |
| 4–6 | `sa`, `sd`, `sq` |
| 7–9 | `col_disp`, `fil_diff`, `par_diff` |

The inverse JSON result contains the initial guess, selected coordinates, optimization
history, and best/final loss, RMSE, R² and parameter values. It does not contain the full
predicted curve. If the target is external, the HDF5 still supplies parameter arrays and
`bul` shape information.
