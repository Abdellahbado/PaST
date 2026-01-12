#!/usr/bin/env sh
set -eu

# Default entrypoint for PaST containers.
# - If args are provided, run them.
# - Otherwise, run the experiment suite.

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

exec python -m PaST.run_experiments \
  --variants all \
  --seeds 0 \
  --config PaST/configs/a100_full.yaml \
  --output_dir runs \
  --device cuda \
  --eval_seed 1337 \
  --eval_seed_mode per_update
