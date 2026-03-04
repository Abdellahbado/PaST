#!/usr/bin/env python3
"""
summarize_hard_results.py — Read CSVs from exp_hard_profile_sweep and print
a clean text summary grouped by profile / variant / model / beam.

Usage:
    python scripts/summarize_hard_results.py
    python scripts/summarize_hard_results.py --log-dir ADP/logs/exp_hard_profile_sweep
    python scripts/summarize_hard_results.py --out ADP/logs/summary_hard.txt
    python scripts/summarize_hard_results.py --profile generate_data
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

# ── CLI ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument(
    "--log-dir",
    default="ADP/logs/exp_hard_profile_sweep",
    help="Directory containing the CSV files.",
)
ap.add_argument(
    "--out", default=None, help="Also write output to this file (default: stdout only)."
)
ap.add_argument(
    "--profile", default=None, help="Filter to one profile (e.g. generate_data)."
)
ap.add_argument(
    "--beams", default="2,5,10", help="Beam widths to show (comma-separated)."
)
args = ap.parse_args()

BEAMS = [int(b) for b in args.beams.split(",")]
LOG_DIR = args.log_dir

# ── Variant labels ───────────────────────────────────────────────────────────
# Suffix in filename → human label
VARIANTS = [
    ("hard_small", "small  (in-domain)"),
    ("hard_medium", "medium (cross-size)"),
    ("hard_large", "large  (cross-size)"),
    ("hard_noise_ar1", "noise  AR(1)"),
    ("hard_noise_drift", "noise  drift"),
    ("hard_noise_bias", "noise  bias"),
]

MODELS = ["mlp", "poly", "poly_mlp", "factored_mlp"]


# ── Collect CSVs ─────────────────────────────────────────────────────────────
def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def median(vals):
    vals = sorted(v for v in vals if v is not None)
    n = len(vals)
    if n == 0:
        return float("nan")
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def fmt(v, pct=False, decimals=2):
    if v != v:  # nan
        return " " * (8 if pct else 6)
    if pct:
        return f"{v:6.2f}%"
    return f"{v:6.2f}x"


# ── Auto-detect profiles ─────────────────────────────────────────────────────
all_csvs = glob.glob(os.path.join(LOG_DIR, "*.csv"))
profiles_found = set()
for p in all_csvs:
    base = os.path.basename(p)
    for suffix, _ in VARIANTS:
        for model in MODELS:
            tag = f"_{model}_{suffix}.csv"
            if base.endswith(tag):
                profile = base[: base.index(f"_{model}_{suffix}")]
                profiles_found.add(profile)

if args.profile:
    profiles_found = {args.profile}

profiles_found = sorted(profiles_found)

# ── Build output ─────────────────────────────────────────────────────────────
lines = []


def emit(*args_):
    line = " ".join(str(a) for a in args_)
    lines.append(line)
    print(line)


emit("=" * 80)
emit(f"  HARD PROFILE SWEEP — SUMMARY")
emit(f"  Log dir : {LOG_DIR}")
emit(f"  Profiles: {', '.join(profiles_found) if profiles_found else '(none found)'}")
emit("=" * 80)

if not profiles_found:
    emit(f"\n  No CSVs found in {LOG_DIR}  —  nothing to summarize.")
    sys.exit(0)

for profile in profiles_found:
    emit("")
    emit("─" * 80)
    emit(f"  PROFILE: {profile}")
    emit("─" * 80)

    for suffix, variant_label in VARIANTS:
        # gather per-model data for this variant
        model_data = {}
        for model in MODELS:
            fname = os.path.join(LOG_DIR, f"{profile}_{model}_{suffix}.csv")
            if not os.path.exists(fname):
                continue
            rows = load_csv(fname)
            if not rows:
                continue

            beam_stats = {}
            for beam in BEAMS:
                brows = [r for r in rows if int(float(r.get("beam", 0))) == beam]
                if not brows:
                    continue
                gL = [float(r["gap_learned_pct"]) for r in brows]
                gZ = [float(r["gap_zero_pct"]) for r in brows]
                gP = [float(r["gap_price_pct"]) for r in brows]
                spd = [float(r["speedup_learned"]) for r in brows]
                beam_stats[beam] = dict(
                    n=len(brows),
                    mean_gL=mean(gL),
                    med_gL=median(gL),
                    mean_gZ=mean(gZ),
                    mean_gP=mean(gP),
                    mean_spd=mean(spd),
                )
            if beam_stats:
                model_data[model] = beam_stats

        if not model_data:
            continue

        emit("")
        emit(f"  ── {variant_label}")
        emit(
            f"  {'Model':<14}  {'beam':>4}  {'n':>4}  {'gapL(mean)':>10}  {'gapL(med)':>10}  {'gapZ':>8}  {'gapP':>8}  {'speedL':>8}"
        )
        emit(
            f"  {'-'*14}  {'-'*4}  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}"
        )

        for model in MODELS:
            if model not in model_data:
                continue
            first = True
            for beam in BEAMS:
                s = model_data[model].get(beam)
                if s is None:
                    continue
                label = f"{model:<14}" if first else f"{'':14}"
                first = False
                emit(
                    f"  {label}  {beam:>4}  {s['n']:>4}  "
                    f"{fmt(s['mean_gL'], pct=True):>10}  "
                    f"{fmt(s['med_gL'],  pct=True):>10}  "
                    f"{fmt(s['mean_gZ'], pct=True):>8}  "
                    f"{fmt(s['mean_gP'], pct=True):>8}  "
                    f"{fmt(s['mean_spd']):>8}"
                )

# ── Overall across all variants per profile ───────────────────────────────────
emit("")
emit("=" * 80)
emit("  OVERALL SUMMARY (mean gapL @ beam=10, across all found variants)")
emit("=" * 80)
emit(
    f"  {'Profile':<20}  {'Variant':<22}  {'mlp':>8}  {'poly':>8}  {'poly_mlp':>10}  {'factored_mlp':>12}"
)
emit(f"  {'-'*20}  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*12}")

for profile in profiles_found:
    first_profile = True
    for suffix, variant_label in VARIANTS:
        row_vals = {}
        for model in MODELS:
            fname = os.path.join(LOG_DIR, f"{profile}_{model}_{suffix}.csv")
            if not os.path.exists(fname):
                continue
            rows = load_csv(fname)
            brows = [r for r in rows if int(float(r.get("beam", 0))) == 10]
            if not brows:
                continue
            vals = [float(r["gap_learned_pct"]) for r in brows]
            row_vals[model] = mean(vals)
        if not row_vals:
            continue
        pfx = f"{profile:<20}" if first_profile else f"{'':20}"
        first_profile = False
        cells = "  ".join(
            f"{fmt(row_vals.get(m), pct=True):>10}" if m in row_vals else f"{'—':>10}"
            for m in ["mlp", "poly", "poly_mlp", "factored_mlp"]
        )
        emit(f"  {pfx}  {variant_label:<22}  {cells}")

emit("")
emit("=" * 80)

# ── Write to file if requested ────────────────────────────────────────────────
if args.out:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[summary] Written to {args.out}")
