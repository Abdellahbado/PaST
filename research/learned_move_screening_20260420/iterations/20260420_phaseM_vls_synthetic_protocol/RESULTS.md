# Results

Phase M protocol branch is implemented and artifacts are generated under `temp/phaseM_vls_synthetic_protocol/`.

Implemented:

- synthetic VLS generator `scripts/phaseM_vls_synthetic_protocol.py`,
- Data-format instance triplets (`Data_p`, `Data_e`, `Data_c`) for generated instances,
- train/val/test manifests with strict role separation,
- generated-vs-benchmark comparison tables and family summaries,
- deterministic config export.

Pilot corpus:

- 180 synthetic instances total,
- full support coverage for `M={25,30,40}`, `N={250,300,350,400,500}`, `K={350,500}`,
- split counts: train 150, val 30,
- benchmark manifests: primary 30 (`61-90`), secondary 60 (`1-60`).

Alignment checks (synthetic vs benchmark `61-90`):

- exact match for `M/N/K` support moments,
- low TV distances for sampled value distributions (`p=0.0171`, `e=0.0363`, `c=0.0113`).

Outcome:

- Phase M passes protocol setup and enables the next branch (Phase N) to execute synthetic-only labeling and offline learning preparation.
