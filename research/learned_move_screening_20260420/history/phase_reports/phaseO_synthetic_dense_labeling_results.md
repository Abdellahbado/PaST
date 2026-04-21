# Phase O Synthetic Dense Labeling Results

Date: 2026-04-20

## 1) Branch objective and strict pass condition

Phase O objective was to replace the Phase N one-sided synthetic labeling policy and recover mixed-sign exact labels under the same synthetic-only manifest protocol.

Strict pass rule:

- fail if all-positive,
- fail if all-negative,
- pass only if mixed-sign exact labels are present and manifest gating is preserved.

## 2) Policy replacement implemented

Implemented policy change:

- previous path: `stageL1_dataset_logging` (screened Stage L1 path)
- new path: `stageO_synthetic_dense_logging` wrapper using `vnd_exact_dp_insert_rank_dense_labeling`

Code updates:

- `solvers/cpp/parallel_heuristic_compare.cpp`
  - added new variant `stageO_synthetic_dense_logging`
  - writes per-instance broad/exact outputs to `temp/phaseO_synthetic_dense_labeling/`
- `scripts/phaseO_synthetic_dense_labeling.py`
  - manifest-driven bounded execution over Phase M train/val only
  - per-instance epsilon derivation and run aggregation
  - split/merged dataset emission and schema summary

## 3) Bounded run configuration

- train sample: 12
- val sample: 4
- total selected instances: 16
- solver variant: `stageO_synthetic_dense_logging`
- epsilon policy: `epsilon=min(K, max(ceil(sum(p)/M), max(p)) + 8)`
- dense run controls: `ls_max_rounds=2`, `ls_time_cap_sec=2.0`, `ls_max_moves_per_round=8000`

## 4) Required metrics

From `temp/phaseO_synthetic_dense_labeling/labeling_run_summary.json`:

- train instances labeled: 12
- val instances labeled: 4
- total exact-labeled rows: 28669
- positives (`label_improving=1`): 20292
- negatives (`label_improving=0`): 8377
- positive rate: 0.7078028532561303
- negative rate: 0.2921971467438697
- total wall runtime: 41.605333 sec
- max RSS: 84639744 bytes
- manifest gating preserved: true

Split aggregate (`labeling_subset_aggregate.csv`):

- train: 21504 rows, 13774 positive, 7730 negative
- val: 7165 rows, 6518 positive, 647 negative

## 5) Decision

Phase O bounded run passes the strict rule:

- output is non-empty,
- labels are mixed-sign,
- not all-positive and not all-negative,
- manifest gating is preserved,
- no schema or loader mismatch observed.

## 6) Interpretation

- Phase N blocker is resolved for bounded synthetic extraction: policy replacement produces usable negative coverage.
- Class balance is still split-dependent (val is more positive-skewed than train), so scale-up should monitor per-split and per-combo drift before freezing a full training table.
