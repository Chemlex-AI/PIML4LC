# Research source mapping and release changes

This module was curated from the supplied `fno_forward/` and `fno_inverse/` research
workspace. Original research files were left unchanged. The release layout avoids
placing another `src/` package at the repository root: commands run from `FNO/`.

| Research source | Release location / treatment |
| --- | --- |
| `fno_forward/src/models/net/rollout_net.py` | Retained under `src/models/net/`; primary autoregressive model |
| `fno_forward/src/models/net/FNO/` | Required dependency closure retained; unrelated operator layers and unused account helpers excluded |
| `fno_forward/src/data/` | Solver dataset, HDF5 loader and conditioning retained; initializer datamodule excluded |
| `fno_forward/src/models/module/` | Rollout training module retained; diffusion/peak/parameter initializers excluded |
| `fno_forward/src/models/inverse/ga.py`, `gd.py`, `inverse_alg.py` | Algorithms retained; generic `src.inverse` replaces internal experiment-specific entry scripts |
| `fno_forward/src/train.py`, `eval.py` and Hydra utilities | Retained with portable defaults, local CSV logging and normalization metadata |
| `fno_forward/src/models/simulator/` | Optional COMSOL/CADET adapters retained with configurable paths |
| `fno_inverse/verify_comsol.py` | Replaced by portable `src.verify` with supplied curves/candidates; internal scheduler and job-lock infrastructure excluded |
| Internal shell launchers, handoff notes and plotting notebooks | Excluded; internal credentials, usernames, absolute machine paths and historical jobs are not published |
| Training corpora, wet-lab chromatograms, checkpoints, results and manuscript PDF | Not included in the FNO overlay |

Release fixes and deliberate changes:

- Default training is the manuscript's three-field model, single-process float32 execution,
  complete dependencies and local CSV logging; machine-specific DDP/GPU selection is removed.
- Root detection prefers the module's `pyproject.toml`, even inside the parent Git repository.
- Checkpoint loading uses explicit metadata and the saved normalization; it no longer
  initializes a distributed process group for single-process inference.
- L-BFGS recomputes the sigmoid graph per chunk, preventing a second backward through a freed
  graph; the post-step candidate is scored using the same coordinates that are saved.
- Test metrics are logged through Lightning; `seed=0` now also seeds the trainer.
- The released generic GA CLI clusters using **all nine physical coordinates**. The original
  high-level `Optimize.normed_opt_paras` removed all three transport coefficients and clustered
  in only six coordinates. This is a correction to match the manuscript and may change the
  retained candidate list. The release does not claim bit-for-bit reproduction of old inverse outputs.
- Inputs require explicit physical values and valid shapes, and fewer than ten training rows
  produce an actionable error rather than empty validation/test splits.
- COMSOL errors include their cause; its model path and the CADET executable are configurable.
- Default isocratic `kd` scaling is one; the manuscript's factor two is an explicit recorded
  protocol option. Existing high-fidelity adapter behavior otherwise remains research code.

Historical checkpoints remain usable when their model structure is supported and their exact
configuration and normalization are provided. Checkpoints relying on excluded latent, diffusion,
UNet or peak/parameter networks are outside this release's scope.
