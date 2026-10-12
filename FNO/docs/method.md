# Method and manuscript correspondence

The primary FNO implements manuscript sections 2.2.2 and 4.5. It is a surrogate of
GRM trajectories trained on FEM labels, not a PINN trained with PDE residuals.

## State and conditioning

The axial state has `1+2*n_r=13` channels: bulk concentration, six particle channels
and six adsorbed-state channels on 101 axial nodes. The physical vectors contain
eleven recorded scalars: two fixed operating settings and nine searchable physical
parameters. Min-max/log10 encoding of all eleven values is concatenated with linear-domain
encoding of the six logarithmic values, producing seventeen static inputs in `[-1,1]`.

Three-layer MLPs embed the seventeen static inputs and the two per-time effective-rate
features to 128 dimensions. For the prescribed ramp/hold organic fraction `psi(t)`, the
effective-rate features are `0.1*ka*exp(-sa*psi)` and `0.1*kd*exp(-sd*psi)`. The factor 0.1
is part of the implementation's input scaling. Static and temporal embeddings are added,
broadcast along the grid and concatenated with the current state. The initial bulk
profile is a learned linear projection of the static inputs; other initial channels
are zero. Fourier time-index embeddings are disabled in the primary configuration.

The FNO has four Fourier blocks, configured mode count 64, hidden width 128, lifting
and projection widths 256, GELU activations and a normalized axial grid embedding.
No domain padding or residual state prediction is used in the default configuration.
It preserves the NeuralOperator-derived implementation, including pointwise skip and
channel-MLP paths; the simplified Fourier formula in the manuscript is not the full
implementation of a block.

**Mode convention:** `n_modes_height=64` is passed to the vendored real-FFT spectral
layer, which uses `64//2+1=33` nonredundant coefficients. A real FFT on 101 axial nodes
has 51 coefficients total. Thus the configuration value 64 does not mean 64 independent
positive-frequency coefficients. The release preserves this research-code behavior
and existing weight shapes.

## Training

The three-field loss sums separately averaged MSEs with equal weights after particle
and adsorbed-field normalization. AdamW uses learning rate `5e-4`, betas `(0.9,0.95)` and
zero weight decay. ReduceLROnPlateau uses factor 0.6 and patience five; gradient clipping
is unit norm. Primary training uses float32, a minimum 50/maximum 1000 epochs, validation
every epoch, early stopping with patience 20, and retention of the top three plus last
checkpoints. Peak error and outlet MSE are diagnostics, not extra training loss terms.

The default split algorithm reproduces the manuscript's 19909-row corpus counts if
the original corpus and row order are provided. The release does not supply that
corpus, original weights or raw measurements, so paper-level accuracy is unverified.

## Inverse and reference selection

The primary FNO inverse method is GA: population 1024, fifteen generations, mutation
probability 0.2, elite size two, tournament size three and maximum five copies per
parent. Mutations sample normalized coordinates, and repeated physical candidates
reuse memoized FNO predictions. Fitness combines normalized waveform MSE (weight .05),
peak-index and peak-height errors, peak-collapse and excessive-total-variation penalties.
Operating settings are fixed across each curve, and nine physical parameters are shared
across curves. Candidates are sorted and filtered in normalized nine-dimensional space.

The optional L-BFGS method optimizes bounded normalized coordinates through a sigmoid
and differentiates through the frozen FNO. Its weighted waveform and peak-summary objective
differs from GA and is not the PINN observability objective. No Fisher-information matrix,
profile-likelihood analysis or diffusion initializer is claimed in this release.

The optional `src.verify` workflow simulates the top eight candidates with COMSOL,
selects minimum gradient retention-time RMSE, resolves ties by FNO rank, then evaluates
the selected physical vector on separately supplied isocratic curves. Applying the
manuscript's `kd` factor of two requires the explicit `--isocratic-kd-scale 2` option and
is recorded. Isocratic measurements are not used for candidate selection. Agreement
with the high-fidelity model and wet-lab observations still does not establish uniqueness.
