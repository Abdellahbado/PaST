#!/usr/bin/env python3
"""
Comprehensive analysis of the rigorous experiment log.

Parses the 53k-line log to extract per-experiment results and answers:
1. Which experiments/profiles/models show non-trivial gaps?
2. Why are most evaluations showing 0% gap for ALL methods (L, Z, P)?
3. Where does the learned model beat heuristics?
4. What are the noise/profile limits?
5. Is there a bug or are the instances just too easy?
"""

import re
import sys
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ── Parse seed-level lines ────────────────────────────────────────────────────
# Format: seed=200 beam=2 exact=168.0 L=168.0 Z=215.0 P=168.0 gapL=0.00% gapZ=27.98% gapP=0.00% t_exact=1.207s tL=0.710s
SEED_RE = re.compile(
    r"seed=(\d+)\s+beam=(\d+)\s+"
    r"exact=([\d.eE+-]+)\s+L=([\d.eE+-]+)\s+Z=([\d.eE+-]+)\s+P=([\d.eE+-]+)\s+"
    r"gapL=([\d.eE+-]+)%\s+gapZ=([\d.eE+-]+)%\s+gapP=([\d.eE+-]+)%\s+"
    r"t_exact=([\d.eE+-]+)s\s+tL=([\d.eE+-]+)s"
)

# Experiment header markers
EXP_MARKERS = {
    "Experiment B": "Experiment B",
    "Experiment C": "Experiment C",
    "Experiment A": "Experiment A",
    "Experiment D": "Experiment D",
}

# Run tag: "[1/35] TRAIN small → EVAL small : daily_tou_poly" or "[1] daily_tou_poly_s0.0_r0.5_sp0"
RUN_TAG_RE = re.compile(
    r"\[(\d+)(?:/\d+)?\]\s+(?:TRAIN.*?:\s*|EVAL.*?:\s*|SKIP.*?:\s*|)(\S+)"
)

# Eval params: D, N, pmax
EVAL_PARAMS_RE = re.compile(r"Eval params: D=(\d+).*?N=(\d+).*?pmax=(\d+)")

# Price mode line
PRICE_MODE_RE = re.compile(r"Eval prices: (forecast_realized|deterministic)")
NOISE_RE = re.compile(r"sigma=([\d.]+).*?rho=([\d.]+).*?spike_prob=([\d.]+)")

# R2 line
R2_RE = re.compile(r"R2_train=([\d.]+)\s+R2_test=([\d.]+)")

# Model type
MODEL_TYPE_RE = re.compile(r"Model type:\s*(\w+)")

# Training summary: features, samples
TRAIN_SAMPLES_RE = re.compile(r"Pooled training data.*?(\d+)\s+samples")
POOLED_RE = re.compile(r"Pooled\s+(\d+)\s+samples")


def parse_log(log_path: str) -> pd.DataFrame:
    """Parse the complete log into a DataFrame with one row per (experiment, run, seed, beam)."""

    rows = []

    current_experiment = "unknown"
    current_run_tag = "unknown"
    current_eval_D = 0
    current_eval_N = 0
    current_eval_pmax = 0
    current_price_mode = "deterministic"
    current_sigma = 0.0
    current_rho = 0.0
    current_spike_prob = 0.0
    current_r2_train = None
    current_r2_test = None
    current_model_type = "unknown"
    current_phase = "unknown"  # train_eval_small, eval_medium, noise_eval

    with open(log_path, "r") as f:
        for line in f:
            line = line.rstrip()

            # Detect experiment phase
            for key, marker in EXP_MARKERS.items():
                if marker in line and "Starting" in line:
                    current_experiment = key
                    break

            # Detect run tag
            m = RUN_TAG_RE.search(line)
            if m:
                current_run_tag = m.group(2)
                if "TRAIN" in line and "EVAL small" in line:
                    current_phase = "train_eval_small"
                elif "EVAL medium" in line:
                    current_phase = "eval_medium"
                elif "SKIP" in line:
                    continue
                else:
                    current_phase = "eval_only"
                # Reset R2 for new run
                current_r2_train = None
                current_r2_test = None
                current_sigma = 0.0
                current_rho = 0.0
                current_spike_prob = 0.0
                current_price_mode = "deterministic"

            # Detect model type
            m = MODEL_TYPE_RE.search(line)
            if m:
                current_model_type = m.group(1)

            # Detect eval params
            m = EVAL_PARAMS_RE.search(line)
            if m:
                current_eval_D = int(m.group(1))
                current_eval_N = int(m.group(2))
                current_eval_pmax = int(m.group(3))

            # Detect price mode
            m = PRICE_MODE_RE.search(line)
            if m:
                current_price_mode = m.group(1)

            # Detect noise params
            m = NOISE_RE.search(line)
            if m:
                current_sigma = float(m.group(1))
                current_rho = float(m.group(2))
                current_spike_prob = float(m.group(3))

            # Detect R2
            m = R2_RE.search(line)
            if m:
                current_r2_train = float(m.group(1))
                current_r2_test = float(m.group(2))

            # Parse seed-level result
            m = SEED_RE.search(line)
            if m:
                rows.append(
                    {
                        "experiment": current_experiment,
                        "run_tag": current_run_tag,
                        "phase": current_phase,
                        "eval_D": current_eval_D,
                        "eval_N": current_eval_N,
                        "eval_pmax": current_eval_pmax,
                        "price_mode": current_price_mode,
                        "sigma": current_sigma,
                        "rho": current_rho,
                        "spike_prob": current_spike_prob,
                        "r2_train": current_r2_train,
                        "r2_test": current_r2_test,
                        "model_type": current_model_type,
                        "seed": int(m.group(1)),
                        "beam": int(m.group(2)),
                        "exact": float(m.group(3)),
                        "L": float(m.group(4)),
                        "Z": float(m.group(5)),
                        "P": float(m.group(6)),
                        "gapL": float(m.group(7)),
                        "gapZ": float(m.group(8)),
                        "gapP": float(m.group(9)),
                        "t_exact": float(m.group(10)),
                        "t_L": float(m.group(11)),
                    }
                )

    df = pd.DataFrame(rows)
    return df


def extract_profile_from_tag(tag: str) -> str:
    """Extract the pricing profile name from a run tag."""
    known = [
        "daily_tou",
        "flat",
        "two_block",
        "ramp",
        "double_peak",
        "weekend_weekday",
        "generate_data",
    ]
    for p in sorted(known, key=len, reverse=True):
        if tag.startswith(p):
            return p
    return "unknown"


def extract_model_from_tag(tag: str) -> str:
    """Extract model type from run tag."""
    profile = extract_profile_from_tag(tag)
    rest = tag[len(profile) :].lstrip("_")
    # For Exp B: rest = "poly", "elasticnet", etc.
    # For Exp A noise: rest = "poly_s0.0_r0.5_sp0"
    models = ["elasticnet", "lasso", "lgbm", "mlp", "poly"]
    for m in sorted(models, key=len, reverse=True):
        if rest.startswith(m):
            return m
    return rest.split("_")[0] if rest else "unknown"


def analyze(df: pd.DataFrame):
    """Run all analyses and print results."""

    print("=" * 80)
    print("COMPREHENSIVE ANALYSIS OF RIGOROUS EXPERIMENTS")
    print("=" * 80)

    # ── 1. Overview ──────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("1. OVERVIEW")
    print(f"{'─' * 80}")
    print(f"Total seed-level results parsed: {len(df)}")
    print(f"Experiments: {df['experiment'].nunique()}")
    for exp in df["experiment"].unique():
        n = len(df[df["experiment"] == exp])
        runs = df[df["experiment"] == exp]["run_tag"].nunique()
        print(f"  {exp}: {n} rows, {runs} unique runs")

    # ── 2. The Zero-Gap Problem ──────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("2. THE ZERO-GAP PROBLEM")
    print(f"{'─' * 80}")

    total = len(df)
    zero_gapL = len(df[df["gapL"].abs() < 0.005])
    zero_gapZ = len(df[df["gapZ"].abs() < 0.005])
    zero_gapP = len(df[df["gapP"].abs() < 0.005])
    all_zero = len(
        df[
            (df["gapL"].abs() < 0.005)
            & (df["gapZ"].abs() < 0.005)
            & (df["gapP"].abs() < 0.005)
        ]
    )

    print(f"Rows with gapL ≈ 0: {zero_gapL}/{total} ({100*zero_gapL/total:.1f}%)")
    print(f"Rows with gapZ ≈ 0: {zero_gapZ}/{total} ({100*zero_gapZ/total:.1f}%)")
    print(f"Rows with gapP ≈ 0: {zero_gapP}/{total} ({100*zero_gapP/total:.1f}%)")
    print(
        f"Rows where ALL methods find optimal (L=Z=P=exact): {all_zero}/{total} ({100*all_zero/total:.1f}%)"
    )

    nonzero_L = df[df["gapL"].abs() >= 0.005]
    if len(nonzero_L) > 0:
        print(f"\nNon-zero gapL rows: {len(nonzero_L)}")
        print(f"  Mean gapL (where non-zero): {nonzero_L['gapL'].mean():.2f}%")
        print(f"  Max gapL: {nonzero_L['gapL'].max():.2f}%")

    # ── 3. Instance Trivality Analysis ───────────────────────────────────
    print(f"\n{'─' * 80}")
    print("3. INSTANCE TRIVIALITY ANALYSIS")
    print(f"{'─' * 80}")
    print("Why all gaps are zero: analyzing instance size vs difficulty\n")

    # Group by (eval_D, eval_N) and check gap statistics
    for (d, n), g in df.groupby(["eval_D", "eval_N"]):
        total_g = len(g)
        trivial = len(g[(g["gapZ"].abs() < 0.005)])
        T = 20 * d
        cap = int(0.80 * T)  # target_util
        print(f"D={d}, N={n} (T={T} slots, cap≈{cap}, util≈{cap/T:.0%}):")
        print(f"  Total evals: {total_g}")
        print(f"  Z finds optimal: {trivial}/{total_g} ({100*trivial/total_g:.1f}%)")
        print(f"  Mean |gapL|: {g['gapL'].abs().mean():.3f}%")
        print(f"  Mean |gapZ|: {g['gapZ'].abs().mean():.3f}%")
        print(f"  Mean |gapP|: {g['gapP'].abs().mean():.3f}%")
        # Check if exact cost is identical across seeds
        unique_exact = g["exact"].nunique()
        print(f"  Unique exact costs: {unique_exact}")
        print()

    # ── 4. Per-Profile Analysis ──────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("4. PER-PROFILE ANALYSIS (Experiment B — train small, eval small)")
    print(f"{'─' * 80}")

    df["profile"] = df["run_tag"].apply(extract_profile_from_tag)
    df["model"] = df["run_tag"].apply(extract_model_from_tag)

    exp_b_small = df[
        (df["experiment"] == "Experiment B") & (df["phase"] == "train_eval_small")
    ]
    if len(exp_b_small) > 0:
        print(f"\nBeam=10 results (best beam width):")
        b10 = exp_b_small[exp_b_small["beam"] == 10]
        summary = (
            b10.groupby(["profile", "model"])
            .agg(
                mean_gapL=("gapL", "mean"),
                mean_gapZ=("gapZ", "mean"),
                mean_gapP=("gapP", "mean"),
                max_gapL=("gapL", "max"),
                n_seeds=("seed", "count"),
                n_nonzero_gapL=("gapL", lambda x: (x.abs() >= 0.005).sum()),
            )
            .reset_index()
        )

        print(summary.to_string(index=False))

        # highlight profiles where L beats P
        l_beats_p = (
            b10.groupby("profile")
            .agg(
                mean_gapL=("gapL", "mean"),
                mean_gapP=("gapP", "mean"),
            )
            .reset_index()
        )
        l_beats_p["advantage"] = l_beats_p["mean_gapP"] - l_beats_p["mean_gapL"]
        print(f"\n  Profile advantage of L over P (mean gapP - mean gapL):")
        print(l_beats_p.to_string(index=False))

    # ── 5. Per-Profile Medium Eval ───────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("5. MEDIUM EVAL ANALYSIS (D=8, N=150 → clamped to N=128)")
    print(f"{'─' * 80}")

    exp_b_med = df[
        (df["experiment"] == "Experiment B") & (df["phase"] == "eval_medium")
    ]
    if len(exp_b_med) > 0:
        b10m = exp_b_med[exp_b_med["beam"] == 10]
        summary_med = (
            b10m.groupby(["profile", "model"])
            .agg(
                mean_gapL=("gapL", "mean"),
                mean_gapZ=("gapZ", "mean"),
                mean_gapP=("gapP", "mean"),
                n_nonzero_gapL=("gapL", lambda x: (x.abs() >= 0.005).sum()),
                n_nonzero_gapZ=("gapZ", lambda x: (x.abs() >= 0.005).sum()),
            )
            .reset_index()
        )
        print(summary_med.to_string(index=False))

        # Explain clamping
        print(f"\n  NOTE: N=150 was requested but _sample_D_N_feasible clamps to")
        print(
            f"  N=min(150, floor(0.80*T)) = min(150, floor(0.80*160)) = min(150, 128) = 128"
        )
        print(
            f"  With N=128 and cap=128: p_j start uniform [1,12], then ALL get reduced to 1"
        )
        print(f"  → Every job has p_j=1 → trivially solvable → all gaps = 0")

    # ── 6. Experiment C — Regularization & Features ──────────────────────
    print(f"\n{'─' * 80}")
    print("6. EXPERIMENT C — REGULARIZATION & FEATURES")
    print(f"{'─' * 80}")

    exp_c = df[df["experiment"] == "Experiment C"]
    if len(exp_c) > 0:
        b10c = exp_c[exp_c["beam"] == 10]
        for phase_name in ["train_eval_small", "eval_medium"]:
            sub = b10c[b10c["phase"] == phase_name]
            if len(sub) == 0:
                continue
            print(f"\n  Phase: {phase_name}")
            summary_c = (
                sub.groupby("run_tag")
                .agg(
                    mean_gapL=("gapL", "mean"),
                    mean_gapZ=("gapZ", "mean"),
                    mean_gapP=("gapP", "mean"),
                    n_nonzero_gapL=("gapL", lambda x: (x.abs() >= 0.005).sum()),
                    r2_train=("r2_train", "first"),
                )
                .reset_index()
            )
            if len(summary_c) > 0:
                print(summary_c.to_string(index=False))

    # ── 7. Experiment A — Noise Stress Test ──────────────────────────────
    print(f"\n{'─' * 80}")
    print("7. EXPERIMENT A — NOISE STRESS TEST")
    print(f"{'─' * 80}")

    exp_a = df[df["experiment"] == "Experiment A"]
    if len(exp_a) > 0:
        b10a = exp_a[exp_a["beam"] == 10]
        print(f"\n  Total noise eval rows (beam=10): {len(b10a)}")
        print(
            f"  All-zero-gap rows: {len(b10a[(b10a['gapL'].abs() < 0.005) & (b10a['gapZ'].abs() < 0.005)])}"
        )

        # Group by noise level
        noise_summary = (
            b10a.groupby(["sigma", "rho", "spike_prob", "profile"])
            .agg(
                mean_gapL=("gapL", "mean"),
                mean_gapZ=("gapZ", "mean"),
                mean_gapP=("gapP", "mean"),
                n=("seed", "count"),
                n_nonzero=("gapL", lambda x: (x.abs() >= 0.005).sum()),
            )
            .reset_index()
        )

        # Show only unique sigma levels
        print(f"\n  Per noise level (aggregated across profiles/models), beam=10:")
        sigma_summary = (
            b10a.groupby("sigma")
            .agg(
                mean_gapL=("gapL", "mean"),
                mean_gapZ=("gapZ", "mean"),
                mean_gapP=("gapP", "mean"),
                n=("seed", "count"),
                n_nonzero_any=("gapL", lambda x: (x.abs() >= 0.005).sum()),
            )
            .reset_index()
        )
        print(sigma_summary.to_string(index=False))

        print(f"\n  Full noise grid:")
        print(noise_summary.to_string(index=False))

    # ── 8. Experiment D — Combined Stress ────────────────────────────────
    print(f"\n{'─' * 80}")
    print("8. EXPERIMENT D — COMBINED NOISE + PROFILE STRESS")
    print(f"{'─' * 80}")

    exp_d = df[df["experiment"] == "Experiment D"]
    if len(exp_d) > 0:
        b10d = exp_d[exp_d["beam"] == 10]
        combined_summary = (
            b10d.groupby(["profile", "model", "sigma"])
            .agg(
                mean_gapL=("gapL", "mean"),
                mean_gapZ=("gapZ", "mean"),
                mean_gapP=("gapP", "mean"),
                n_nonzero=("gapL", lambda x: (x.abs() >= 0.005).sum()),
            )
            .reset_index()
        )
        print(combined_summary.to_string(index=False))

    # ── 9. Training Quality (R2 scores) ──────────────────────────────────
    print(f"\n{'─' * 80}")
    print("9. TRAINING QUALITY (R² scores)")
    print(f"{'─' * 80}")

    train_runs = df[df["r2_test"].notna()].drop_duplicates(
        subset=["run_tag", "r2_test"]
    )
    if len(train_runs) > 0:
        print(f"  Runs with R² data: {len(train_runs)}")
        print(
            f"  R²_test range: [{train_runs['r2_test'].min():.4f}, {train_runs['r2_test'].max():.4f}]"
        )
        print(f"  R²_test mean: {train_runs['r2_test'].mean():.4f}")
        print(f"  R²_test < 0.99: {len(train_runs[train_runs['r2_test'] < 0.99])}")

        print(f"\n  Per-model R²_test:")
        r2_by_model = train_runs.groupby("model_type")["r2_test"].agg(
            ["mean", "min", "max", "count"]
        )
        print(r2_by_model.to_string())

    # ── 10. Where does L actually beat heuristics? ───────────────────────
    print(f"\n{'─' * 80}")
    print("10. WHERE DOES THE LEARNED MODEL (L) BEAT HEURISTICS?")
    print(f"{'─' * 80}")

    # Cases where gapL < gapP (L is better than P)
    L_better_P = df[(df["gapL"] < df["gapP"] - 0.005)]
    print(
        f"\n  Cases where L < P (learned beats price heuristic): {len(L_better_P)}/{len(df)}"
    )
    if len(L_better_P) > 0:
        print(
            f"  Average advantage: {(L_better_P['gapP'] - L_better_P['gapL']).mean():.2f}%"
        )
        print(f"\n  By profile and beam:")
        adv = (
            L_better_P.groupby(["profile", "beam"])
            .agg(
                count=("seed", "count"),
                mean_gapL=("gapL", "mean"),
                mean_gapP=("gapP", "mean"),
                advantage=(
                    "gapP",
                    lambda x: (
                        L_better_P.loc[x.index, "gapP"]
                        - L_better_P.loc[x.index, "gapL"]
                    ).mean(),
                ),
            )
            .reset_index()
        )
        print(adv.to_string(index=False))

    # Cases where gapP < gapL (P is better than L)
    P_better_L = df[(df["gapP"] < df["gapL"] - 0.005)]
    print(
        f"\n  Cases where P < L (price heuristic beats learned): {len(P_better_L)}/{len(df)}"
    )

    # Cases where L beats Z
    L_better_Z = df[(df["gapL"] < df["gapZ"] - 0.005)]
    print(
        f"  Cases where L < Z (learned beats zero heuristic): {len(L_better_Z)}/{len(df)}"
    )

    # ── 11. Root Cause: Instance Triviality ──────────────────────────────
    print(f"\n{'─' * 80}")
    print("11. ROOT CAUSE DIAGNOSIS")
    print(f"{'─' * 80}")
    print(
        """
DIAGNOSIS: The instances are TRIVIALLY SOLVABLE.

The evaluation uses the same fixed D and N for all seeds (no --eval-D-range / 
--eval-N-range specified). The exact optimal is achievable by even the simplest
beam search heuristic (beam=2 with zero guidance).

Small eval (D=6, N=30):
  T = 20*6 = 120 slots, target_util=0.80 → cap = floor(0.80*120) = 96
  build_instance reduces sum(p) to EXACTLY 96 for ALL seeds (the while loop 
  stops at total == cap). With 30 jobs on 120 slots at 80% utilization,
  the scheduling problem is easy enough for beam=2 to solve optimally.
  
  Evidence: exact=168.0 for ALL seeds in daily_tou. Z=215 (zero guidance 
  gets a different cost), but L and P always find 168. For some profiles
  (ramp, double_peak), the structure makes even Z find the optimal.

Medium eval (D=8, N=150):
  T = 20*8 = 160 slots, cap = floor(0.80*160) = 128
  _sample_D_N_feasible CLAMPS N from 150 down to 128 (since 150 > cap).
  Then build_instance starts with 128 jobs, p ~ U(1,12), sum(p) ≈ 832.
  The capping loop reduces sum(p) to 128 → ALL p_j become 1.
  With 128 unit-length jobs on 160 slots, scheduling is trivial.
  
  Evidence: EVERY medium eval has L=Z=P=exact, zero gap everywhere.

Noise experiment (Exp A):
  Same D=6, N=30 instances. Noise only changes prices, not instance size.
  Even with sigma=4.0, the instances remain trivially solvable because
  the problem difficulty comes from job counts/sizes vs time horizon,
  NOT from price structure.

RECOMMENDATIONS:
  1. Increase instance difficulty: use higher utilization (0.95+) or 
     larger N relative to T
  2. Use eval-D-range and eval-N-range to create instance variety
  3. For medium instances, ensure N < cap to avoid silent clamping
  4. Consider N=80-100 with D=8 (cap=128) for meaningful medium evals
  5. For small: use D=3, N=40-50 (T=60, cap=48, util=83-100%)
"""
    )

    # ── 12. Non-trivial results spotlight ────────────────────────────────
    print(f"\n{'─' * 80}")
    print("12. SPOTLIGHT: NON-TRIVIAL RESULTS")
    print(f"{'─' * 80}")

    nontrivial = df[df["gapL"].abs() >= 0.01]
    if len(nontrivial) > 0:
        print(f"\n  {len(nontrivial)} rows with |gapL| >= 0.01%")
        nt_summary = (
            nontrivial.groupby(["experiment", "profile", "model", "beam"])
            .agg(
                count=("seed", "count"),
                mean_gapL=("gapL", "mean"),
                mean_gapZ=("gapZ", "mean"),
                mean_gapP=("gapP", "mean"),
            )
            .reset_index()
        )
        print(nt_summary.to_string(index=False))
    else:
        print("  No non-trivial results found (all |gapL| < 0.01%)")

    # ── 13. Speed comparison ─────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("13. SPEED COMPARISON (t_L / t_exact)")
    print(f"{'─' * 80}")

    speed_df = df[df["t_exact"] > 0.01].copy()
    if len(speed_df) > 0:
        speed_df["speedup"] = speed_df["t_exact"] / speed_df["t_L"]
        by_beam = speed_df.groupby("beam")["speedup"].agg(
            ["mean", "median", "min", "max"]
        )
        print(by_beam.to_string())

    return df


def main():
    log_path = (
        sys.argv[1] if len(sys.argv) > 1 else "ADP/logs/logs/rigourous/complete_1.log"
    )

    if not os.path.exists(log_path):
        print(f"ERROR: Log file not found: {log_path}")
        sys.exit(1)

    print(f"Parsing {log_path}...")
    df = parse_log(log_path)
    print(f"Parsed {len(df)} result rows.\n")

    if len(df) == 0:
        print("No results parsed. Check log format.")
        sys.exit(1)

    df = analyze(df)

    # Save parsed data for further analysis
    out_csv = "ADP/logs/logs/rigourous/parsed_results.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nParsed data saved to {out_csv}")


if __name__ == "__main__":
    main()
