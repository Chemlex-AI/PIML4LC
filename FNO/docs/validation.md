# Validation performed during release preparation

Preparation date: 2026-10-12. The code was prepared from the user-supplied repository
ZIP snapshot; it has not been pushed or reconciled with a newer authenticated remote.

## Confirmed locally

- Windows, Python 3.11.15, PyTorch 2.6.0+cpu, Lightning 2.5.6, Hydra 1.3.2,
  NumPy 1.26.4 and TensorLy 0.9.0.
- Six pytest cases passed, including an integration test that trains a real small FNO,
  saves its checkpoint, moves the run directory, invalidates the original normalization
  path, reloads the model from its sidecar metadata and runs prediction/GA/L-BFGS.
- Physical-parameter normalization round trips and the exact seventeen-input order
  were checked, together with the first time-varying effective-rate feature.
- Bulk-only and three-field rollout shapes and finite nonzero input gradients passed.
- Chunked L-BFGS with three starts and chunk size one converged to the known optimum
  of an independent analytic curve model.
- Reference retention-time metric and exclusion of missing observed peaks passed.
- `examples/run_smoke.py` completed the documented CPU workflow; a separate `src.eval`
  invocation completed against the resulting checkpoint.
- Python compilation and dependency compatibility checks passed.

## Outside the local validation

The original scientific data and model weights were intentionally excluded. No manuscript
metrics, physical parameter recovery on FEM/wet-lab data, GPU run or COMSOL/CADET solve
was validated. The GitHub Actions workflow is prepared for Linux/Windows but has not yet
run on GitHub. Synthetic-example losses and fitted parameters are not scientific results.

The historical NeuralOperator copy is retained without an exact original upstream revision;
its provenance and license are documented separately. The primary model architecture and
state-dictionary key layout are preserved, but excluded research variants are unsupported.
