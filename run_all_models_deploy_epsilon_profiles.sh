#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Run from this script's directory (PaST/)
cd "$(dirname "$0")"

# -----------------------------
# Config (deployment-like)
# -----------------------------
TRAIN_SEEDS="0-49"          # pooled training seeds
EVAL_SEEDS="300-319"        # held-out seeds for single-machine eval (optional)

SAMPLES=4000
PMAX=12
TARGET_UTIL=0.80

# Vary sizes within the small benchmark interval
N_RANGE="20-60"
D_RANGE="2-4"
M_RANGE="3-7"

# Epsilon-sim settings
REPLICATES=10
EPS_SEED=123
BEAM=50
PRUNE_FACTOR=2.0
ASSIGN_ALPHA=1.2
ASSIGN_UNIFORM_MIX=0.15

LOG_DIR="ADP/logs/deploy_epsilon_profiles"
MODEL_DIR="ADP/models/deploy_epsilon_profiles"
mkdir -p "$LOG_DIR" "$MODEL_DIR"

# Two profiles (train+eval within profile; no cross-profile testing here)
PROFILES=("daily_tou" "generate_data")
MODELS=("linear" "poly" "mlp" "lgbm")

for PROFILE in "${PROFILES[@]}"; do
	echo "========================================================================"
	echo "PROFILE=$PROFILE (20-hour repeating)"
	echo "========================================================================"

	PROFILE_ARGS=()
	if [[ "$PROFILE" == "generate_data" ]]; then
		# Match the flags in both scripts.
		PROFILE_ARGS+=(--daily-price-profile generate_data --gd-seed 20260109 --gd-ck-low 1 --gd-ck-high 8)
	else
		PROFILE_ARGS+=(--daily-price-profile daily_tou)
	fi

	for MODEL in "${MODELS[@]}"; do
		echo ""
		echo ">>> Training pooled Vhat: model=$MODEL profile=$PROFILE"

		if [[ "$MODEL" == "lgbm" ]]; then
			if ! conda run -n new-ml-env python -c "import lightgbm" >/dev/null 2>&1; then
				echo "    [skip] lightgbm is not installed in conda env 'new-ml-env'"
				continue
			fi
		fi

		CKPT="$MODEL_DIR/vhat_${PROFILE}_${MODEL}.npz"

		# 1) Train pooled model on variable N/D within interval
		conda run -n new-ml-env python sandbox/eval_pooled_vhat.py \
			--D 3 --N 40 --pmax "$PMAX" \
			--train-seeds "$TRAIN_SEEDS" --samples-per-instance "$SAMPLES" \
			--train-N-range "$N_RANGE" --train-D-range "$D_RANGE" \
			--eval-seeds "$EVAL_SEEDS" \
			--eval-N-range "$N_RANGE" --eval-D-range "$D_RANGE" --eval-pmax "$PMAX" \
			--transferable-features --normalize --normalize-labels \
			--target-util "$TARGET_UTIL" \
			"${PROFILE_ARGS[@]}" \
			--model-type "$MODEL" --beams 2,5 \
			--save-model "$CKPT" \
			--out-csv "$LOG_DIR/pooled_${PROFILE}_${MODEL}.csv" \
			2>&1 | tee "$LOG_DIR/pooled_${PROFILE}_${MODEL}.log"

		echo ">>> Epsilon-constraint deployment sim: model=$MODEL profile=$PROFILE"

		# 2) Deployment-like multi-machine epsilon simulation
		conda run -n new-ml-env python sandbox/eval_epsilon_constraint_sim.py \
			--category small \
			--N 40 --N-range "$N_RANGE" \
			--M 5 --M-range "$M_RANGE" \
			--D 3 --D-range "$D_RANGE" \
			--replicates "$REPLICATES" --seed "$EPS_SEED" \
			--pmax "$PMAX" --target-util "$TARGET_UTIL" \
			--price-mode daily_tou \
			"${PROFILE_ARGS[@]}" \
			--assign-alpha "$ASSIGN_ALPHA" --assign-uniform-mix "$ASSIGN_UNIFORM_MIX" \
			--guided --beam "$BEAM" --prune-factor "$PRUNE_FACTOR" \
			--load-model "$CKPT" --transferable-features --normalize \
			--out-csv "$LOG_DIR/epsilon_${PROFILE}_${MODEL}.csv" \
			2>&1 | tee "$LOG_DIR/epsilon_${PROFILE}_${MODEL}.log"

		echo ">>> Done: model=$MODEL profile=$PROFILE"
	done
done

echo "Done. Logs: $LOG_DIR  Models: $MODEL_DIR"
