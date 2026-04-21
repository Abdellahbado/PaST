# Problem

Phase M is the protocol-repair branch that replaces benchmark-derived development data as train/validation input with a synthetic-only protocol.

The branch requirement is to establish a paper-clean split contract:

- train and validation are generated synthetic VLS only,
- benchmark `61-90` is primary external test only,
- benchmark `1-60` is secondary robustness/OOD test only.

This branch must produce deterministic generator, manifests, and protocol artifacts so the next branch can execute synthetic-only labeling/training without benchmark leakage.
