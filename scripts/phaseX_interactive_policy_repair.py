#!/usr/bin/env python3
"""Phase X — Interactive LLM Policy Repair orchestration script.

Subcommands:
  --generate-random-policy    Write a valid random DSL policy JSON.
  --eval-policy               Run phaseX_policy_json and return parsed CSV.
  --eval-baselines            Run trimmed, LLM exc, random exc, score escape.
  --smoke                     Full X2 smoke on 3 dev cells × 6 arms.
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BINARY = PROJECT_ROOT / "solvers/cpp/build/parallel_heuristic_compare"
DATA_DIR = "temp/paper_exact_repo/instances"

PHASEX_DIR = (
    PROJECT_ROOT
    / "research/learned_move_screening_20260420"
    / "iterations/20260508_phaseX_interactive_llm_policy_repair"
)
POLICIES_DIR = PHASEX_DIR / "policies"
EVAL_DIR = PHASEX_DIR / "eval"
NOTES_DIR = PHASEX_DIR / "notes"

SCHEMA_PATH = POLICIES_DIR / "schema.json"
EXAMPLE_POLICY_PATH = POLICIES_DIR / "example_policy.json"

DEFAULT_DP_LIMIT = "30.0"
DEFAULT_LS_TIME = "10.0"
DEFAULT_LS_ROUNDS = "5"
DEFAULT_LS_MOVES = "20000"

SMOKE_CELLS = [
    (61, 347, "guard"),
    (62, 290, "secondary"),
    (65, 195, "primary"),
]

BASELINE_VARIANTS = [
    "vnd_exact_dp_insert_rank_diverse_trimmed",
    "phaseS_llm_exception_lane",
    "phaseS_random_exception_lane",
    "phaseV_score_escape_sampler",
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_schema():
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def _ensure_dirs():
    POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)


def _run_variant(variant, inst, eps, *, extra_env=None, timeout=1800):
    """Run one variant on one (inst, eps). Returns (returncode, stdout, stderr)."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    cmd = [
        str(BINARY),
        "paper-instance",
        str(inst),
        str(eps),
        variant,
        DATA_DIR,
        DEFAULT_DP_LIMIT,
        DEFAULT_LS_TIME,
        DEFAULT_LS_ROUNDS,
        DEFAULT_LS_MOVES,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return (-1, "", f"timed out after {timeout}s")
    return (result.returncode, result.stdout or "", result.stderr or "")


def _parse_csv(stdout):
    """Parse CSV output from the C++ binary. Returns dict or None."""
    lines = [l.strip() for l in stdout.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return None
    header = [h.strip() for h in lines[-2].split(",")]
    data = lines[-1].split(",")
    if len(data) < len(header):
        return None
    return dict(zip(header, data))


def _is_feasible(row):
    if row is None:
        return False
    tec = row.get("tec_total", "")
    if tec == "" or tec is None:
        return False
    try:
        val = float(tec)
    except (ValueError, TypeError):
        return False
    return val > 0.0  # positive energy cost = feasible


# ── random policy generator ──────────────────────────────────────────────────

def generate_random_policy(output_path=None, seed=None):
    """Generate a valid random DSL policy JSON and write it.

    Returns the policy dict.
    """
    schema = _load_schema()
    props = schema["properties"]

    rng = random.Random(seed)

    normal_modes = props["normal_mode"]["enum"]
    escape_modes = props["escape_mode"]["enum"]

    initial_budget = rng.randint(1, 8)
    max_budget = rng.randint(max(4, initial_budget), 16)
    grow_on_hit = rng.randint(0, min(4, max_budget - initial_budget + 2))
    shrink_on_miss = rng.randint(0, 4)
    max_per_source = rng.randint(1, 4)
    max_per_target = rng.randint(1, 4)
    switch_after_no_hit = rng.randint(0, 4)
    switch_back_on_hit = rng.choice([True, False])
    require_positive_cheap_lb = rng.choice([True, False])
    guard_max_budget = rng.randint(0, 4)
    coverage_bonus = round(rng.uniform(0.0, 3.0), 2)
    random_mix = round(rng.uniform(0.0, 1.0), 2)

    # Scoring weights for hybrid mode: sum doesn't need to be 1, but keep them reasonable
    cheap_lb_weight = round(rng.uniform(0.0, 1.0), 2)
    s2_weight = round(rng.uniform(0.0, 1.0), 2)
    slack_weight = round(rng.uniform(0.0, 1.0), 2)

    policy = {
        "policy_name": f"random_{rng.randint(0, 9999):04d}",
        "normal_mode": rng.choice(normal_modes),
        "escape_mode": rng.choice(escape_modes),
        "switch_after_no_hit": switch_after_no_hit,
        "switch_back_on_hit": switch_back_on_hit,
        "initial_budget": initial_budget,
        "max_budget": max_budget,
        "grow_on_hit": grow_on_hit,
        "shrink_on_miss": shrink_on_miss,
        "max_per_source": max_per_source,
        "max_per_target": max_per_target,
        "require_positive_cheap_lb": require_positive_cheap_lb,
        "coverage_bonus": coverage_bonus,
        "random_mix": random_mix,
        "cheap_lb_weight": cheap_lb_weight,
        "s2_weight": s2_weight,
        "slack_weight": slack_weight,
        "guard_max_budget": guard_max_budget,
    }

    if output_path is None:
        idx = 0
        while True:
            output_path = POLICIES_DIR / f"random_policy_{idx:03d}.json"
            if not output_path.exists():
                break
            idx += 1

    with open(output_path, "w") as f:
        json.dump(policy, f, indent=2)
        f.write("\n")

    print(f"Wrote random policy → {output_path}")
    return policy, output_path


# ── eval helpers ─────────────────────────────────────────────────────────────

def eval_policy(policy_path, inst, eps):
    """Evaluate a Phase X policy on one cell. Returns dict row or None."""
    if not Path(policy_path).exists():
        print(f"ERROR: policy file not found: {policy_path}", file=sys.stderr)
        sys.exit(1)

    extra_env = {"PHASEX_POLICY_PATH": str(Path(policy_path).resolve())}
    rc, stdout, stderr = _run_variant(
        "phaseX_policy_json", inst, eps, extra_env=extra_env
    )

    if rc != 0:
        print(
            f"ERROR: binary exited {rc} for phaseX_policy_json on {inst}/{eps}\n{stderr}",
            file=sys.stderr,
        )
        sys.exit(1)

    row = _parse_csv(stdout)
    if row is None:
        print(f"ERROR: could not parse CSV on {inst}/{eps}", file=sys.stderr)
        sys.exit(1)

    if not _is_feasible(row):
        print(
            f"ERROR: infeasible result for phaseX_policy_json on {inst}/{eps}  TEC={row.get('tec_total','?')}",
            file=sys.stderr,
        )
        sys.exit(1)

    row["_inst"] = str(inst)
    row["_eps"] = str(eps)
    row["_policy_name"] = row.get("phaseX_policy_name", "")
    return row


def eval_baselines(inst, eps):
    """Run all baseline variants on one cell. Returns list of dict rows."""
    rows = []
    for variant in BASELINE_VARIANTS:
        extra_env = {}
        if variant == "phaseS_random_exception_lane":
            extra_env["PHASES_RANDOM_SEED"] = "0"

        rc, stdout, stderr = _run_variant(variant, inst, eps, extra_env=extra_env)
        if rc != 0:
            print(
                f"ERROR: binary exited {rc} for {variant} on {inst}/{eps}\n{stderr}",
                file=sys.stderr,
            )
            sys.exit(1)

        row = _parse_csv(stdout)
        if row is None:
            print(f"ERROR: parse failed for {variant} on {inst}/{eps}", file=sys.stderr)
            sys.exit(1)

        if not _is_feasible(row):
            print(
                f"ERROR: infeasible {variant} on {inst}/{eps} TEC={row.get('tec_total','?')}",
                file=sys.stderr,
            )
            sys.exit(1)

        row["_inst"] = str(inst)
        row["_eps"] = str(eps)
        row["_variant_short"] = variant
        rows.append(row)

    return rows


# ── smoke ────────────────────────────────────────────────────────────────────

def run_smoke():
    """Run full X2 smoke on 3 dev cells × 6 arms."""
    _ensure_dirs()

    arms = []
    # Baselines
    for v in BASELINE_VARIANTS:
        arms.append(
            {
                "type": "baseline",
                "variant": v,
                "label": v,
                "extra_env": {"PHASES_RANDOM_SEED": "0"}
                if v == "phaseS_random_exception_lane"
                else {},
            }
        )
    # Example policy
    arms.append(
        {
            "type": "phaseX",
            "policy_path": str(EXAMPLE_POLICY_PATH.resolve()),
            "label": "phaseX_example_policy",
        }
    )
    # Random policy
    rpol, rpath = generate_random_policy(seed=42)
    arms.append(
        {
            "type": "phaseX",
            "policy_path": str(rpath.resolve()),
            "label": "phaseX_random_policy",
        }
    )

    total = len(SMOKE_CELLS) * len(arms)
    all_rows = []
    n = 0

    for inst, eps, role in SMOKE_CELLS:
        for arm in arms:
            n += 1
            label = f"{inst}/{eps} {arm['label']}"
            print(f"[{n}/{total}] {label}...", end=" ", flush=True)
            t0 = time.time()

            if arm["type"] == "baseline":
                rc, stdout, stderr = _run_variant(
                    arm["variant"], inst, eps, extra_env=arm["extra_env"]
                )
            else:
                extra_env = {
                    "PHASEX_POLICY_PATH": arm["policy_path"],
                }
                rc, stdout, stderr = _run_variant(
                    "phaseX_policy_json", inst, eps, extra_env=extra_env
                )

            if rc != 0:
                print(f"FAILED rc={rc}")
                print(f"STDERR: {stderr[:200]}", file=sys.stderr)
                n_err = n
                continue

            row = _parse_csv(stdout)
            if row is None:
                print("PARSE FAILED")
                continue

            if not _is_feasible(row):
                print(f"INFEASIBLE tec={row.get('tec_total','?')}")
                continue

            elapsed = time.time() - t0
            row["_inst"] = str(inst)
            row["_eps"] = str(eps)
            row["_role"] = role
            row["_arm"] = arm["label"]
            row["_wall_sec"] = f"{elapsed:.1f}"
            all_rows.append(row)

            tec = row.get("tec_total", "?")
            print(f"TEC={tec} {elapsed:.1f}s")

    if not all_rows:
        print("ERROR: no feasible rows produced", file=sys.stderr)
        sys.exit(1)

    # ── raw CSV ───────────────────────────────────────────────────────────────
    raw_path = EVAL_DIR / "x2_smoke_raw.csv"
    raw_fields = sorted(set(k for r in all_rows for k in r))
    priority = [
        "_inst",
        "_eps",
        "_role",
        "_arm",
        "instance_id",
        "epsilon",
        "variant",
        "tec_total",
        "runtime_sec",
        "_wall_sec",
        "stop_reason",
    ]
    ordered = priority + [k for k in raw_fields if k not in priority]
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nRaw CSV → {raw_path}")

    # ── summary CSV ───────────────────────────────────────────────────────────
    summary_path = EVAL_DIR / "x2_smoke_summary.csv"
    sf = [
        "inst",
        "eps",
        "role",
        "arm",
        "tec_total",
        "runtime_sec",
        "wall_sec",
        "stop_reason",
        "accepted_insert_inter_moves",
        "evaluated_insert_inter_moves",
        "exception_candidates_considered",
        "exception_candidates_evaluated",
        "exception_budget_used",
        "exception_improvement_count",
        "exception_best_delta",
        "outside_pool_distinct_src",
        "outside_pool_distinct_tgt",
        "selected_distinct_src",
        "selected_distinct_tgt",
        "exception_hit_rate",
        "phaseV_score_escape_candidates_considered",
        "phaseV_score_escape_candidates_evaluated",
        "phaseV_score_escape_improvement_count",
        "phaseV_score_escape_best_delta",
        "phaseV_score_escape_escape_rounds",
        "phaseV_score_escape_normal_rounds",
        "phaseX_candidates_considered",
        "phaseX_candidates_evaluated",
        "phaseX_improvement_count",
        "phaseX_best_delta",
        "phaseX_normal_rounds",
        "phaseX_escape_rounds",
        "phaseX_policy_name",
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sf, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            sr = {k: r.get(k, "") for k in sf}
            sr["inst"] = r.get("_inst", "")
            sr["eps"] = r.get("_eps", "")
            sr["role"] = r.get("_role", "")
            sr["arm"] = r.get("_arm", "")
            sr["wall_sec"] = r.get("_wall_sec", "")
            w.writerow(sr)
    print(f"Summary CSV → {summary_path}")

    # ── print table ───────────────────────────────────────────────────────────
    print("\n=== X2 SMOKE RESULTS ===\n")
    print(
        f"{'Inst/Eps':>10s}  {'Arm':<35s}  {'TEC':>12s}  {'Ins':>4s}  {'ExcImp':>6s}  {'Stop':<18s}"
    )
    print("-" * 100)
    for r in all_rows:
        inst_eps = f"{r['_inst']}/{r['_eps']}"
        arm = r["_arm"][:35]
        tec = r.get("tec_total", "?")[:12]
        ins = r.get("accepted_insert_inter_moves", "?")
        exc_imp = r.get("exception_improvement_count", "?")
        stop = r.get("stop_reason", "?")[:18]
        print(f"{inst_eps:>10s}  {arm:<35s}  {tec:>12s}  {ins:>4s}  {exc_imp:>6s}  {stop:<18s}")

    return all_rows


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase X — Interactive LLM Policy Repair orchestration"
    )
    parser.add_argument(
        "--generate-random-policy",
        action="store_true",
        help="Write a valid random DSL policy JSON.",
    )
    parser.add_argument(
        "--eval-policy",
        nargs=3,
        metavar=("INST", "EPS", "POLICY_PATH"),
        help="Evaluate a Phase X policy on one cell.",
    )
    parser.add_argument(
        "--eval-baselines",
        nargs=2,
        metavar=("INST", "EPS"),
        help="Run all baselines on one cell.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run full X2 smoke on 3 dev cells.",
    )
    parser.add_argument(
        "--policy-output",
        type=str,
        default=None,
        help="Path for generated random policy (with --generate-random-policy).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for --generate-random-policy.",
    )

    args = parser.parse_args()

    if args.generate_random_policy:
        _ensure_dirs()
        generate_random_policy(output_path=args.policy_output, seed=args.seed)

    elif args.eval_policy:
        inst, eps, policy_path = args.eval_policy
        row = eval_policy(policy_path, int(inst), int(eps))
        print(json.dumps(row, indent=2))

    elif args.eval_baselines:
        inst, eps = args.eval_baselines
        rows = eval_baselines(int(inst), int(eps))
        for r in rows:
            print(
                f"{r['_variant_short']}: TEC={r.get('tec_total','?')}  "
                f"insert={r.get('accepted_insert_inter_moves','?')}  "
                f"stop={r.get('stop_reason','?')}"
            )

    elif args.smoke:
        run_smoke()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
