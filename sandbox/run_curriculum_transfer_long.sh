#!/usr/bin/env bash
set -u

# Robust curriculum runner:
# - trains small -> medium -> large
# - loops multiple seeds
# - saves checkpoints to a stable folder for reuse
# - does NOT abort the whole run if one stage fails

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-python}

RUN_TAG=${RUN_TAG:-"curriculum_transfer_$(date +%Y%m%d_%H%M%S)"}
OUT_DIR=${OUT_DIR:-"$ROOT_DIR/checkpoints/vhat_tou_ridge/$RUN_TAG"}
LOG_DIR="$OUT_DIR/logs"

SEEDS=${SEEDS:-"0 1 2 3 4"}

# Instance sizes (defaults chosen to keep exact DP tractable by keeping K modest).
SMALL_D=${SMALL_D:-2}
SMALL_N=${SMALL_N:-40}
SMALL_PMAX=${SMALL_PMAX:-6}

MEDIUM_D=${MEDIUM_D:-6}
MEDIUM_N=${MEDIUM_N:-60}
MEDIUM_PMAX=${MEDIUM_PMAX:-6}

LARGE_D=${LARGE_D:-10}
LARGE_N=${LARGE_N:-80}
LARGE_PMAX=${LARGE_PMAX:-6}

# Training budgets (increase if you want “more than good enough”)
SMALL_SAMPLES=${SMALL_SAMPLES:-3000}
MEDIUM_SAMPLES=${MEDIUM_SAMPLES:-4000}

# Large stage is typically run as frozen-transfer (no labeling) by default.
# If you want to fine-tune on large, set LARGE_FINE_TUNE_SAMPLES>0.
LARGE_FINE_TUNE_SAMPLES=${LARGE_FINE_TUNE_SAMPLES:-0}

# Beams (tune for speed/quality Pareto)
SMALL_BEAM=${SMALL_BEAM:-6000}
MEDIUM_BEAM=${MEDIUM_BEAM:-30000}
LARGE_BEAM=${LARGE_BEAM:-60000}

PRUNE_FACTOR=${PRUNE_FACTOR:-2.0}
L2=${L2:-1e-3}
PRIOR_STRENGTH=${PRIOR_STRENGTH:-10.0}

mkdir -p "$OUT_DIR" "$LOG_DIR"

common_flags=(
	"--transferable-features"
	"--l2" "$L2"
	"--prune-factor" "$PRUNE_FACTOR"
)

run_stage () {
	local stage_name="$1"; shift
	local seed="$1"; shift
	local log_file="$LOG_DIR/${stage_name}_seed${seed}.log"

	echo "[$(date +%F\ %T)] stage=$stage_name seed=$seed" | tee -a "$log_file"
	( set -x; "$PYTHON_BIN" PaST/sandbox/train_eval_vhat_beam_dp.py "$@" ) >>"$log_file" 2>&1
	local rc=$?
	echo "[$(date +%F\ %T)] rc=$rc" | tee -a "$log_file"
	return $rc
}

echo "Run tag: $RUN_TAG" | tee "$OUT_DIR/README.txt"
echo "Out dir: $OUT_DIR" | tee -a "$OUT_DIR/README.txt"
echo "Seeds:   $SEEDS" | tee -a "$OUT_DIR/README.txt"

for seed in $SEEDS; do
	small_ckpt="$OUT_DIR/small_seed${seed}.npz"
	medium_ckpt="$OUT_DIR/medium_seed${seed}.npz"
	large_ckpt="$OUT_DIR/large_seed${seed}.npz"

	# SMALL (train from scratch)
	run_stage "small" "$seed" \
		--seed "$seed" --D "$SMALL_D" --N "$SMALL_N" --pmax "$SMALL_PMAX" \
		--samples "$SMALL_SAMPLES" --beam "$SMALL_BEAM" \
		--save-model "$small_ckpt" \
		"${common_flags[@]}" \
	|| true

	# MEDIUM (load small, fine-tune with quadratic prior)
	run_stage "medium" "$seed" \
		--seed "$seed" --D "$MEDIUM_D" --N "$MEDIUM_N" --pmax "$MEDIUM_PMAX" \
		--samples "$MEDIUM_SAMPLES" --beam "$MEDIUM_BEAM" \
		--load-model "$small_ckpt" --prior-strength "$PRIOR_STRENGTH" \
		--save-model "$medium_ckpt" \
		"${common_flags[@]}" \
	|| true

	# LARGE
	if [[ "$LARGE_FINE_TUNE_SAMPLES" -gt 0 ]]; then
		# Fine-tune on large (expensive; still skips full exact solve)
		run_stage "large_ft" "$seed" \
			--seed "$seed" --D "$LARGE_D" --N "$LARGE_N" --pmax "$LARGE_PMAX" \
			--samples "$LARGE_FINE_TUNE_SAMPLES" --beam "$LARGE_BEAM" \
			--skip-exact \
			--load-model "$medium_ckpt" --prior-strength "$PRIOR_STRENGTH" \
			--save-model "$large_ckpt" \
			"${common_flags[@]}" \
		|| true
	else
		# Frozen transfer on large (fast; no labeling; skips full exact solve)
		run_stage "large" "$seed" \
			--seed "$seed" --D "$LARGE_D" --N "$LARGE_N" --pmax "$LARGE_PMAX" \
			--beam "$LARGE_BEAM" \
			--skip-exact \
			--load-model "$medium_ckpt" --freeze-loaded \
			--save-model "$large_ckpt" \
			"${common_flags[@]}" \
		|| true
	fi

done

echo "Done. Logs in: $LOG_DIR" | tee -a "$OUT_DIR/README.txt"
echo "Models in: $OUT_DIR/*.npz" | tee -a "$OUT_DIR/README.txt"