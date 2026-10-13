# COMSOL batch simulation

This directory contains all the three COMSOL models and the `batch_simulation.py` script for running the
model in batches. By default, the script samples 20,000 parameter sets.

## Environment

Install and license a COMSOL version that can run `GRM.mph` and the packages needed
by this script:

```bash
conda env create -f COMSOL_models/environment.yml
conda activate piml4lc-comsol
```

COMSOL itself and its license must be
configured separately.

## Run simulations

Generate the parameter plan, then start the simulations:

```bash
python COMSOL_models/batch_simulation.py --plan-only
python COMSOL_models/batch_simulation.py --resume --workers 4
```

By default, output goes to `COMSOL_models/comsol_sim/`. `parameters.csv` records every
sampled parameter set; each completed `sample_00000/` directory contains `cl.csv`,
`cp.csv` and `q.csv`. After an interruption, repeat the command with `--resume` to skip
completed samples. Use `--count`, `--seed`, `--workers`, `--cores-per-job`, `--timeout`,
`--model` and `--output-dir` to adjust the run.

If an existing parameter plan uses older gradient combinations, resuming rejects it.
Choose a new `--output-dir` to generate a new plan. Full simulation requires sufficient
COMSOL licenses, memory and storage.
