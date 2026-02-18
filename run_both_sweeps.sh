#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/new-ml-env/bin/python}"
export PYTHON_BIN

echo "=== SWEEP 1: all models (linear/poly/mlp/lgbm) ==="
NOHUP=0 IN_NOHUP=1 bash /Users/mac/Documents/Study/PFE/PaST/run_all_models_deploy_epsilon_profiles_all_sizes.sh

echo ""
echo "=== SWEEP 2: MLP variants (mlp_hist/mlp_price/mlp_meta/mlp_all) ==="
NOHUP=0 IN_NOHUP=1 bash /Users/mac/Documents/Study/PFE/PaST/run_all_new_mlp_variants_generalization.sh

echo ""
echo "=== BOTH SWEEPS COMPLETE ==="
