# Phase Q HPC Setup

This note is specific to the synthetic-only offline move-ranking branch.

## Files this setup assumes already exist

- `temp/phaseP_full_synthetic_freeze/synthetic_moves_exact_labeled_train_frozen.csv.gz`
- `temp/phaseP_full_synthetic_freeze/synthetic_moves_exact_labeled_val_frozen.csv`
- `research/learned_move_screening_20260420/requirements_phaseQ_synthetic_offline_training.txt`

## Recommended environment setup

Create a dedicated virtual environment for this branch:

```bash
python -m venv .venv_phaseq
source .venv_phaseq/bin/activate
python -m pip install --upgrade pip
python -m pip install -r research/learned_move_screening_20260420/requirements_phaseQ_synthetic_offline_training.txt
```

If your HPC image already provides a suitable Python environment, you can skip the `venv` creation and install directly into that environment instead.

## Prepare the frozen train file

The train file is stored compressed on GitHub to stay under the GitHub file-size limit.

Decompress it once before training:

```bash
gzip -dk temp/phaseP_full_synthetic_freeze/synthetic_moves_exact_labeled_train_frozen.csv.gz
```

This produces:

- `temp/phaseP_full_synthetic_freeze/synthetic_moves_exact_labeled_train_frozen.csv`

## Quick dependency sanity check

Run this before starting Phase Q implementation:

```bash
python - <<'PY'
import numpy
import pandas
import scipy
import sklearn
import tqdm
import xgboost
import lightgbm
import catboost
print("phaseQ python deps: OK")
PY
```

## Expected working files for Phase Q

The coder should read these first:

- `research/learned_move_screening_20260420/START_HERE.md`
- `research/learned_move_screening_20260420/ACTIVE.md`
- `research/learned_move_screening_20260420/iterations/20260421_phaseP_full_synthetic_freeze/SUMMARY.md`
- `research/learned_move_screening_20260420/reference/PLAN_supervised_move_ranking.md`
- `research/learned_move_screening_20260420/reference/phaseM_benchmark_role_note.md`
- `research/learned_move_screening_20260420/phaseP_full_synthetic_freeze_results.md`
- `research/learned_move_screening_20260420/phaseP_full_synthetic_freeze_readiness.md`
- `temp/phaseP_full_synthetic_freeze/freeze_manifest.json`
- `temp/phaseP_full_synthetic_freeze/dataset_summary_global.json`
- `temp/phaseP_full_synthetic_freeze/dataset_summary_by_split.csv`
- `temp/phaseP_full_synthetic_freeze/dataset_summary_by_bucket.csv`
- `temp/phaseP_full_synthetic_freeze/feature_schema_frozen.json`

## Suggested Phase Q run pattern

After the coder implements the Phase Q training script, the intended usage pattern is:

```bash
source .venv_phaseq/bin/activate
python <phaseQ_script>.py
```

The exact script path depends on the implementation created in Phase Q.
