# Third-party notices

Original project contributions follow the parent repository's BSD-3-Clause license.
The following adapted components retain their MIT license terms and notices.

## NeuralOperator

- Upstream: https://github.com/neuraloperator/neuraloperator
- Copyright (c) 2023 NeuralOperator developers.
- License text: [licenses/NeuralOperator-MIT.txt](licenses/NeuralOperator-MIT.txt).
- Scope: `src/models/net/FNO/base_model.py`, `fno.py`, `utils.py`, and the retained
  `layers/` modules. The chromatography wrapper and release cleanup modify this code.
- The supplied research workspace did not record the exact upstream revision. This
  is a vendored research copy, not a claim of synchronization with current NeuralOperator.
  The license was checked against the upstream license on 2026-10-12 (upstream main
  observed at `00b7d86f8d74ff0af55da53eb585fe26df9c71f0`).

## Lightning-Hydra-Template

- Upstream: https://github.com/ashleve/lightning-hydra-template
- Copyright (c) 2021 ashleve.
- License text: [licenses/lightning-hydra-template-MIT.txt](licenses/lightning-hydra-template-MIT.txt).
- Scope: adapted training/evaluation entry points, configuration conventions and
  utilities in `src/utils/{instantiators,logging_utils,pylogger,rich_utils,utils}.py`.
- The upstream README includes the MIT grant; the exact inherited revision is not
  recorded in the supplied workspace.

Other libraries are installed as dependencies and retain their respective licenses.
COMSOL is separate proprietary software requiring an appropriate license; it is not
redistributed in the FNO code overlay. The existing parent repository's `.mph` model
files are separate from the COMSOL application.
