#!/usr/bin/env python3
"""Phase X — Interactive LLM Policy Repair orchestration script.

Subcommands:
  --generate-random-policy    Write a valid random DSL policy JSON.
  --eval-policy               Run phaseX_policy_json and return parsed CSV.
  --eval-baselines            Run trimmed, LLM exc, random exc, score escape.
  --smoke                     Full X2 smoke on 3 dev cells × 6 arms.
  --x3-random-campaign        X3: 20 random policies × 3 cells + baselines.
  --x4-interactive            X4: 5-round interactive DeepSeek policy repair.
"""

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
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
PROMPTS_DIR = PHASEX_DIR / "prompts"
RESPONSES_DIR = PHASEX_DIR / "responses"
LLM_INTERACTIVE_DIR = POLICIES_DIR / "llm_interactive"

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
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    LLM_INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)


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


# ── DeepSeek client ───────────────────────────────────────────────────────────

def _load_env_deepseek():
    """Source .env.deepseek.sh into os.environ if not already set."""
    if os.environ.get("DEEPSEEK_API_KEY", "").startswith("sk-"):
        return
    env_file = PROJECT_ROOT / ".env.deepseek.sh"
    if not env_file.exists():
        raise RuntimeError(".env.deepseek.sh not found")
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

DS_MODEL = "deepseek-v4-pro"
DS_MAX_TOKENS = 16000
DS_TEMPERATURE = 0.5


def _call_deepseek(messages):
    _load_env_deepseek()
    url = f"{os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')}/chat/completions"
    api_key = os.environ["DEEPSEEK_API_KEY"]
    body = json.dumps({
        "model": DS_MODEL,
        "messages": messages,
        "temperature": DS_TEMPERATURE,
        "max_tokens": DS_MAX_TOKENS,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    })
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"\nDeepSeek API error: {e}", file=sys.stderr)
        raise
    elapsed = time.time() - t0
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    meta = {
        "model": data.get("model", DS_MODEL),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "elapsed_sec": round(elapsed, 1),
    }
    return content, meta


def _extract_json_from_response(content):
    """Extract JSON object from LLM response. Handles code fences and inline JSON."""
    json_str = None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if m:
        json_str = m.group(1)
    else:
        m = re.search(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", content, re.DOTALL)
        if m:
            json_str = m.group(1)
    if json_str is None:
        return None
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def _validate_policy(policy):
    """Validate policy dict against schema. Returns (ok, errors_list)."""
    schema = _load_schema()
    errors = []
    required = schema.get("required", [])
    props = schema.get("properties", {})
    for key in required:
        if key not in policy:
            errors.append(f"missing required field: {key}")
    for key, val in policy.items():
        if key not in props:
            continue
        spec = props[key]
        if "enum" in spec and val not in spec["enum"]:
            errors.append(f"{key}: {val} not in enum {spec['enum']}")
        if "type" in spec:
            t = spec["type"]
            if t == "integer" and not isinstance(val, int):
                errors.append(f"{key}: expected int, got {type(val).__name__}")
            elif t == "number" and not isinstance(val, (int, float)):
                errors.append(f"{key}: expected number, got {type(val).__name__}")
            elif t == "string" and not isinstance(val, str):
                errors.append(f"{key}: expected string, got {type(val).__name__}")
            elif t == "boolean" and not isinstance(val, bool):
                errors.append(f"{key}: expected bool, got {type(val).__name__}")
        if "minimum" in spec and isinstance(val, (int, float)):
            if val < spec["minimum"]:
                errors.append(f"{key}: {val} < minimum {spec['minimum']}")
        if "maximum" in spec and isinstance(val, (int, float)):
            if val > spec["maximum"]:
                errors.append(f"{key}: {val} > maximum {spec['maximum']}")
    return len(errors) == 0, errors


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


# ── X3 random campaign ────────────────────────────────────────────────────────

def run_x3_campaign():
    """X3: 30 random policies × 3 cells + baselines + example_policy."""
    _ensure_dirs()

    campaign_dir = POLICIES_DIR / "random_campaign"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    # ── Baselines and example policy ──────────────────────────────────────────
    print("=== X3: Collecting baselines and example policy ===\n")
    baseline_rows = []
    for inst, eps, role in SMOKE_CELLS:
        print(f"  Baselines for {inst}/{eps}...", end=" ", flush=True)
        t0 = time.time()
        rows = eval_baselines(inst, eps)
        for r in rows:
            r["_role"] = role
            r["_arm"] = r.get("_variant_short", "?")
            r["_policy_id"] = ""
        baseline_rows.extend(rows)

        # Example policy
        r = eval_policy(str(EXAMPLE_POLICY_PATH.resolve()), inst, eps)
        r["_role"] = role
        r["_arm"] = "phaseX_example_policy"
        r["_policy_id"] = "example"
        baseline_rows.append(r)
        print(f"{time.time()-t0:.1f}s")

    # ── Baseline reference metrics ────────────────────────────────────────────
    ref = {}  # (inst,eps) -> {'trimmed': tec, 'example': tec, 'score_esc': tec}
    for r in baseline_rows:
        inst = int(r["_inst"])
        eps = int(r["_eps"])
        arm = r["_arm"]
        tec = float(r.get("tec_total", 0))
        key = (inst, eps)
        if key not in ref:
            ref[key] = {}
        ref[key][arm] = tec

    def _ref_tec(inst, eps, arm):
        return ref.get((inst, eps), {}).get(arm, None)

    raw_path = EVAL_DIR / "x3_random_campaign_raw.csv"
    agg_path = EVAL_DIR / "x3_random_campaign_aggregate.csv"
    N_POLICIES = 20

    # ── Generate and evaluate N random policies ──────────────────────────────
    print(f"\n=== X3: Evaluating {N_POLICIES} random policies on 3 cells ===\n")
    campaign_rows = []
    policy_index = []
    n_failed = 0
    n_infeasible = 0

    for i in range(N_POLICIES):
        seed = 100 + i
        local_rng = random.Random(seed)

        # Generate policy with a unique name including its seed
        _, policy_path = generate_random_policy(
            output_path=campaign_dir / f"x3_campaign_{i:03d}.json",
            seed=seed,
        )
        # overwrite auto-generated name with a seed-tracked one
        with open(policy_path) as f:
            pol = json.load(f)
        pol["policy_name"] = f"x3_campaign_{i:03d}_s{seed}"
        with open(policy_path, "w") as f:
            json.dump(pol, f, indent=2)
            f.write("\n")

        policy_id = f"c{i:03d}"

        pol_tecs = {}
        pol_feasible = True
        for inst, eps, role in SMOKE_CELLS:
            print(f"  [{i+1:2d}/{N_POLICIES}] {policy_id} on {inst}/{eps}...",
                  end=" ", flush=True)
            t0 = time.time()

            rc, stdout, stderr = _run_variant(
                "phaseX_policy_json",
                inst, eps,
                extra_env={"PHASEX_POLICY_PATH": str(policy_path.resolve())},
            )

            if rc != 0:
                print(f"FAILED rc={rc}")
                n_failed += 1
                pol_feasible = False
                break

            row = _parse_csv(stdout)
            if row is None:
                print("PARSE FAILED")
                n_failed += 1
                pol_feasible = False
                break

            if not _is_feasible(row):
                print(f"INFEASIBLE tec={row.get('tec_total','?')}")
                n_infeasible += 1
                pol_feasible = False
                break

            elapsed = time.time() - t0
            tec = float(row.get("tec_total", 0))
            pol_tecs[(inst, eps)] = tec
            row["_inst"] = str(inst)
            row["_eps"] = str(eps)
            row["_role"] = role
            row["_arm"] = "phaseX_random_campaign"
            row["_policy_id"] = policy_id
            row["_policy_seed"] = str(seed)
            row["_policy_name"] = pol["policy_name"]
            row["_wall_sec"] = f"{elapsed:.1f}"
            campaign_rows.append(row)
            print(f"TEC={tec:.0f} {elapsed:.1f}s")

        if not pol_feasible:
            continue

        # Per-policy metrics
        tecs_list = [pol_tecs.get((inst, eps), 0) for inst, eps, _ in SMOKE_CELLS]
        mean_tec = sum(tecs_list) / len(tecs_list)

        beats_example = 0
        beats_score_esc = 0
        regresses_trimmed = 0
        deltas_vs_trimmed = []
        deltas_vs_example = []
        for inst, eps, role in SMOKE_CELLS:
            t = pol_tecs.get((inst, eps), 0)
            t_trimmed = _ref_tec(inst, eps, "vnd_exact_dp_insert_rank_diverse_trimmed")
            t_example = _ref_tec(inst, eps, "phaseX_example_policy")
            t_score = _ref_tec(inst, eps, "phaseV_score_escape_sampler")
            if t_trimmed is not None:
                deltas_vs_trimmed.append(t - t_trimmed)
                if t > t_trimmed:
                    regresses_trimmed += 1
            if t_example is not None and t < t_example:
                beats_example += 1
            if t_score is not None and t < t_score:
                beats_score_esc += 1
            if t_example is not None:
                deltas_vs_example.append(t - t_example)

        mean_delta_trimmed = sum(deltas_vs_trimmed) / len(deltas_vs_trimmed)
        mean_delta_example = sum(deltas_vs_example) / len(deltas_vs_example) if deltas_vs_example else 0.0

        policy_index.append({
            "policy_id": policy_id,
            "policy_seed": seed,
            "policy_name": pol["policy_name"],
            "tecs": pol_tecs,
            "mean_tec": mean_tec,
            "mean_delta_vs_trimmed": mean_delta_trimmed,
            "mean_delta_vs_example": mean_delta_example,
            "beats_example": beats_example,
            "beats_score_escape": beats_score_esc,
            "regresses_trimmed": regresses_trimmed,
            "n_failed": 0,
        })

    # ── Save raw CSV ──────────────────────────────────────────────────────────
    raw_path = EVAL_DIR / "x3_random_campaign_raw.csv"
    all_raw_rows = baseline_rows + campaign_rows
    raw_fields = sorted(set(k for r in all_raw_rows for k in r))
    priority = [
        "_inst", "_eps", "_role", "_arm", "_policy_id",
        "instance_id", "epsilon", "variant", "tec_total",
        "runtime_sec", "_wall_sec", "stop_reason",
    ]
    ordered = priority + [k for k in raw_fields if k not in priority]
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_raw_rows)
    print(f"\nRaw CSV → {raw_path}")

    # ── Summary CSV (per-policy, per-cell) ────────────────────────────────────
    summary_path = EVAL_DIR / "x3_random_campaign_summary.csv"
    with open(summary_path, "w", newline="") as f:
        sf = [
            "policy_id", "policy_seed", "policy_name",
            "inst", "eps", "role",
            "tec_total", "wall_sec",
            "beats_example_on_cell", "beats_score_esc_on_cell",
        ]
        w = csv.DictWriter(f, fieldnames=sf, extrasaction="ignore")
        w.writeheader()
        for r in campaign_rows:
            inst = int(r["_inst"])
            eps = int(r["_eps"])
            pid = r.get("_policy_id", "")
            pname = r.get("_policy_name", "")
            pseed = r.get("_policy_seed", "")
            tec = r.get("tec_total", "")
            ws = r.get("_wall_sec", "")
            t_example = _ref_tec(inst, eps, "phaseX_example_policy")
            t_score = _ref_tec(inst, eps, "phaseV_score_escape_sampler")
            t = float(tec)
            beats_ex = (t_example is not None and t < t_example)
            beats_se = (t_score is not None and t < t_score)
            w.writerow({
                "policy_id": pid, "policy_seed": pseed, "policy_name": pname,
                "inst": inst, "eps": eps, "role": r.get("_role", ""),
                "tec_total": tec, "wall_sec": ws,
                "beats_example_on_cell": str(beats_ex),
                "beats_score_esc_on_cell": str(beats_se),
            })
    print(f"Summary CSV → {summary_path}")

    # ── Aggregate policy metrics CSV ──────────────────────────────────────────
    agg_path = EVAL_DIR / "x3_random_campaign_aggregate.csv"
    agg_sf = [
        "policy_id", "policy_seed", "policy_name",
        "tec_61_347", "tec_62_290", "tec_65_195",
        "mean_tec", "mean_delta_vs_trimmed", "mean_delta_vs_example",
        "beats_example_cells", "beats_score_esc_cells",
        "regresses_trimmed_cells",
    ]
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_sf, extrasaction="ignore")
        w.writeheader()
        for pi in policy_index:
            w.writerow({
                "policy_id": pi["policy_id"],
                "policy_seed": pi["policy_seed"],
                "policy_name": pi["policy_name"],
                "tec_61_347": pi["tecs"].get((61, 347), ""),
                "tec_62_290": pi["tecs"].get((62, 290), ""),
                "tec_65_195": pi["tecs"].get((65, 195), ""),
                "mean_tec": f"{pi['mean_tec']:.1f}",
                "mean_delta_vs_trimmed": f"{pi['mean_delta_vs_trimmed']:.1f}",
                "mean_delta_vs_example": f"{pi['mean_delta_vs_example']:.1f}",
                "beats_example_cells": pi["beats_example"],
                "beats_score_esc_cells": pi["beats_score_escape"],
                "regresses_trimmed_cells": pi["regresses_trimmed"],
            })
    print(f"Aggregate CSV → {agg_path}")

    # ── Analysis ──────────────────────────────────────────────────────────────
    print("\n=== X3 ANALYSIS ===\n")

    # Baseline reference table
    print("Baseline TECs:")
    print(f"  {'Cell':<10s} {'Trimmed':>8s} {'LLM Exc':>8s} {'Rand Exc':>8s} {'ScoreEsc':>8s} {'Example':>8s}")
    for inst, eps, role in SMOKE_CELLS:
        t_tr = _ref_tec(inst, eps, "vnd_exact_dp_insert_rank_diverse_trimmed")
        t_ll = _ref_tec(inst, eps, "phaseS_llm_exception_lane")
        t_re = _ref_tec(inst, eps, "phaseS_random_exception_lane")
        t_se = _ref_tec(inst, eps, "phaseV_score_escape_sampler")
        t_ex = _ref_tec(inst, eps, "phaseX_example_policy")
        print(f"  {inst}/{str(eps):<5s} {int(t_tr or 0):>8d} {int(t_ll or 0):>8d} {int(t_re or 0):>8d} {int(t_se or 0):>8d} {int(t_ex or 0):>8d}")

    # Random policy stats
    mean_tecs = [pi["mean_tec"] for pi in policy_index]
    example_mean_tecs = [sum([
        _ref_tec(61,347,"phaseX_example_policy"),
        _ref_tec(62,290,"phaseX_example_policy"),
        _ref_tec(65,195,"phaseX_example_policy"),
    ]) / 3]
    trimmed_mean_tecs = [sum([
        _ref_tec(61,347,"vnd_exact_dp_insert_rank_diverse_trimmed"),
        _ref_tec(62,290,"vnd_exact_dp_insert_rank_diverse_trimmed"),
        _ref_tec(65,195,"vnd_exact_dp_insert_rank_diverse_trimmed"),
    ]) / 3]

    if mean_tecs:
        mean_tecs_sorted = sorted(mean_tecs)
        median_rnd = mean_tecs_sorted[len(mean_tecs_sorted)//2]
        best_rnd = min(mean_tecs_sorted)
        worst_rnd = max(mean_tecs_sorted)

        beats_example = sum(1 for pi in policy_index if pi["mean_tec"] < example_mean_tecs[0])
        beats_trimmed = sum(1 for pi in policy_index if pi["mean_tec"] < trimmed_mean_tecs[0])

        print(f"\nSummary over {len(policy_index)} viable random policies:")
        print(f"  Failed / infeasible: {n_failed} / {n_infeasible}")
        print(f"  Example mean TEC: {example_mean_tecs[0]:.1f}")
        print(f"  Trimmed mean TEC: {trimmed_mean_tecs[0]:.1f}")
        print(f"  Random best mean: {best_rnd:.1f}")
        print(f"  Random median mean: {median_rnd:.1f}")
        print(f"  Random worst mean: {worst_rnd:.1f}")
        print(f"  Beat example mean TEC: {beats_example}/{len(policy_index)}")
        print(f"  Beat trimmed mean TEC: {beats_trimmed}/{len(policy_index)}")

        # Top 3 random policies
        sorted_idx = sorted(policy_index, key=lambda pi: pi["mean_tec"])
        print(f"\nTop policies by mean TEC:")
        print(f"  {'Rank':<5s} {'ID':<8s} {'Mean':>10s} {'ΔTrimmed':>10s} {'ΔExample':>10s} {'BeatEx':>6s} {'BeatSE':>6s} {'RegrTr':>6s}")
        for rank, pi in enumerate(sorted_idx[:5]):
            print(f"  {rank+1:<5d} {pi['policy_id']:<8s} "
                  f"{pi['mean_tec']:>10.1f} "
                  f"{pi['mean_delta_vs_trimmed']:>10.1f} "
                  f"{pi['mean_delta_vs_example']:>10.1f} "
                  f"{pi['beats_example']:>6d} "
                  f"{pi['beats_score_escape']:>6d} "
                  f"{pi['regresses_trimmed']:>6d}")

        # ── Case classification ────────────────────────────────────────────────
        example_wins_vs_median = example_mean_tecs[0] < median_rnd
        best_beats_example = best_rnd < example_mean_tecs[0]

        print(f"\nCase classification:")
        if not example_wins_vs_median and not best_beats_example:
            # Random beats example even at median → random-searchable
            print(f"  Case A: DSL is easy/random-searchable.")
            print(f"    Rationale: random median ({median_rnd:.1f}) beats example ({example_mean_tecs[0]:.1f})")
            print(f"    Implication: X4 must compare LLM vs best-of-random, not median.")
            case = "A"
        elif example_wins_vs_median and best_beats_example:
            # Median bad but best beats example → searchable with effort
            print(f"  Case B: DSL contains useful policies but search is noisy.")
            print(f"    Rationale: random median ({median_rnd:.1f}) worse than example ({example_mean_tecs[0]:.1f}),")
            print(f"    but best random ({best_rnd:.1f}) beats example.")
            print(f"    Implication: X4 tests whether interactive LLM finds good policies faster than random.")
            case = "B"
        else:
            # Neither median nor best beats example → example is strong
            print(f"  Case C: example_policy is a strong baseline.")
            print(f"    Rationale: neither median ({median_rnd:.1f}) nor best ({best_rnd:.1f}) beats example ({example_mean_tecs[0]:.1f})")
            print(f"    Implication: X4 tests whether LLM can repair beyond example.")
            case = "C"

        return {
            "case": case,
            "n_policies": len(policy_index),
            "n_failed": n_failed,
            "n_infeasible": n_infeasible,
            "example_mean_tec": example_mean_tecs[0],
            "trimmed_mean_tec": trimmed_mean_tecs[0],
            "random_median_mean": median_rnd,
            "random_best_mean": best_rnd,
            "policy_index": sorted_idx,
        }


# ── X4 interactive LLM loop ────────────────────────────────────────────────────

# Fixed reference values from X3 campaign (2026-05-09)
X3_REF = {
    "example_mean_tec": 14292.0,
    "trimmed_mean_tec": 14534.0,
    "random_median_mean": 14362.0,
    "random_best_mean": 14254.3,
    "example_per_cell": {61: 6884, 62: 9484, 65: 26508},
    "score_esc_per_cell": {61: 6884, 62: 9484, 65: 26508},
    "trimmed_per_cell": {61: 6884, 62: 9687, 65: 27031},
}

X4_CELLS = [
    (61, 347, "guard"),
    (62, 290, "secondary"),
    (65, 195, "primary"),
]


def _build_round0_prompt():
    schema_text = json.dumps(_load_schema(), indent=2)
    example_text = json.dumps(json.load(open(EXAMPLE_POLICY_PATH)), indent=2)
    best_random = json.dumps(json.load(
        open(POLICIES_DIR / "random_campaign/x3_campaign_000.json")
    ), indent=2)

    prompt = f"""# Phase X — Interactive LLM Policy Repair — Round 0

You are a scheduling optimization expert designing exception-lane policies for a
parallel machine scheduling solver with exact DP per-machine cost evaluation.

## Problem

We have a VND local search solver for parallel machine scheduling with:
- DiverseTrimmed core shortlist (per-source top-K with per-target quota=1)
- Exception lane: evaluates candidates rejected by the shortlist
- Exact DP verification per proposed move

Your job: design a JSON policy that controls the exception lane to minimize total
energy cost (TEC). Lower TEC is better.

## Policy DSL

The policy is a JSON object with 17 fields controlling the exception lane.
You generate exactly ONE policy JSON. The C++ solver reads it and applies it.

```json
{schema_text}
```

### Field Summary

| Field | Range | Meaning |
|-------|-------|---------|
| normal_mode | llm_score, s2, random, cheap_lb, hybrid | Scoring in normal rounds |
| escape_mode | none, cheap_lb_pair, random_pair, coverage, anti_s2 | Scoring after consecutive misses |
| switch_after_no_hit | 0-4 | Rounds before escape (0=never) |
| switch_back_on_hit | true/false | Return to normal after escape hit |
| initial_budget | 1-8 | Starting exception evals per round |
| max_budget | 4-16 | Upper bound on budget |
| grow_on_hit | 0-4 | Add candidates on improvement |
| shrink_on_miss | 0-4 | Remove after 2+ misses |
| max_per_source | 1-4 | Diversity quota per source |
| max_per_target | 1-4 | Diversity quota per target |
| require_positive_cheap_lb | true/false | Drop candidates with cheap_lb_delta ≤ 0 |
| coverage_bonus | 0.0-3.0 | Bonus for novel machines (coverage mode) |
| random_mix | 0.0-1.0 | Random fraction in hybrid mode |
| cheap_lb_weight | 0.0-1.0 | cheap_lb_delta weight in hybrid |
| s2_weight | 0.0-1.0 | s2 weight in hybrid |
| slack_weight | 0.0-1.0 | slack_bonus weight in hybrid |
| guard_max_budget | 0-4 | Budget cap on tight-epsilon rounds (eps_per_job≤3.0). 0 = skip on guard |

### Scoring Mode Details

- llm_score: s2 + slack_bonus + tightness_bonus (current example behavior)
- s2: Raw s2 score only
- random: Uniform random via seeded RNG
- cheap_lb: cheap_lb_delta (lower-bound improvement estimate)
- hybrid: Weighted mix of cheap_lb_delta + s2 + slack_bonus + random

### Escape Mode Details

- none: No escape — stay in normal mode
- cheap_lb_pair: Best cheap_lb_delta per (source, target) pair
- random_pair: Random pairs
- coverage: Reward uncovered machines (needs coverage_bonus)
- anti_s2: score = max(0, cheap_lb_delta) - s2 (inverts s2 for when s2 mis-ranks)

### Budget Adaptation

1. Start: budget = initial_budget
2. On improvement: budget = min(max_budget, budget + grow_on_hit)
3. On 2+ consecutive misses: budget = max(1, budget - shrink_on_miss)
4. Guard rounds: capped at guard_max_budget (0 = skip exception lane entirely)

## Constraints

- Output EXACTLY ONE valid JSON object matching the schema above.
- NO C++ code, NO Python, NO pseudocode.
- NO instance IDs (61/347, 62/290, 65/195) in policy values.
- NO arbitrary thresholds outside the DSL.
- Policy fields are in the JSON; NO external if/then logic.
- Include a short rationale BEFORE the JSON, but evaluation uses ONLY the JSON.

## Baseline Reference (3 development cells)

### Example Policy (current baseline)
```json
{example_text}
```

### X3 Random Campaign — 20 random DSL policies

| Metric | Value |
|--------|------|
| Example mean TEC | 14292.0 |
| Random median mean TEC | 14362.0 (worse than example by +70) |
| Random best mean TEC | 14254.3 (better than example by -37.7) |
| Random worst mean TEC | 14471.0 |
| Policies beating example on mean | 2/20 (10%) |
| Policies beating trimmed on mean | 20/20 (100%) |

### Best Random Policy (c000, mean TEC = 14254.3)
```json
{best_random}
```

This random policy achieved:
- 61/347: 6877 (vs example 6884, Δ = -7)
- 62/290: 9561 (vs example 9484, Δ = +77)
- 65/195: 26325 (vs example 26508, Δ = -183)

### Per-Cell Context

The three cells have different characteristics:
- 61/347: guard cell, tight epsilon. Exception lane finds no improvements (TEC same as trimmed).
- 62/290: secondary cell, medium epsilon. Exception lane can find ~200 improvement.
- 65/195: primary cell, loose epsilon. Exception lane can find ~500 improvement.

## Your Task

Design ONE policy JSON that should beat the example_policy (mean TEC < 14292.0)
and ideally approach or beat the random best (mean TEC < 14254.3).

Key insights from X3:
1. Most random policies beat trimmed (all 20/20) — exception lane always helps.
2. Only 2/20 beat the example policy — the DSL is NOT trivially random-searchable.
3. The best random policy uses random normal mode + cheap_lb_pair escape
   with require_positive_cheap_lb=true and diverse quotas (4,3). It keeps
   guard_max_budget=0 (skip on tight guard cell).
4. The guard cell (61/347) is hard to improve — most policies tie the baseline there.

Think strategically:
- Scoring mode matters for the primary cell (65/195) where most improvement comes from.
- The hybrid mode lets you blend multiple signals — use it if a pure mode underperforms.
- Budget adaptation (grow/shrink) controls exploration depth.
- Escape mode matters when normal mode gets stuck.
- guard_max_budget=0 protects the guard cell from bad exception moves.

Output format: short rationale first, then:
```json
{{...}}
```"""

    return prompt


def _build_round_n_prompt(round_num, rounds_history):
    """Build feedback prompt for rounds 1-4."""
    prev = rounds_history[-1]
    policy_json = json.dumps(prev["policy"], indent=2)
    tecs = prev["tecs"]
    example_tecs = X3_REF["example_per_cell"]
    score_tecs = X3_REF["score_esc_per_cell"]

    prev_rows = []
    for r in rounds_history:
        prev_rows.append(f"| Round {r['round']} | {r['tecs'].get(61, '?')} | {r['tecs'].get(62, '?')} | {r['tecs'].get(65, '?')} | {r['mean_tec']:.1f} |")

    prev_table = "\n".join(prev_rows)

    prompt = f"""# Phase X — Interactive LLM Policy Repair — Round {round_num}

## Previous Policy (Round {round_num - 1})
```json
{policy_json}
```

### Per-Cell TEC Results

| Cell | Your TEC | Example TEC | Δ vs Example | ScoreEsc TEC | Δ vs ScoreEsc |
|------|---------|------------|-------------|-------------|--------------|
"""

    eps_map = {61: 347, 62: 290, 65: 195}
    for inst in [61, 62, 65]:
        t = tecs.get(inst, 0)
        te = example_tecs[inst]
        ts = score_tecs[inst]
        eps = eps_map[inst]
        prompt += f"| {inst}/{eps} | {t:.0f} | {te} | {t-te:+.0f} | {ts} | {t-ts:+.0f} |\n"

    prompt += f"""
| **Mean** | **{prev['mean_tec']:.1f}** | **{X3_REF['example_mean_tec']:.1f}** | **{prev['mean_tec']-X3_REF['example_mean_tec']:+.1f}** | — | — |

### Comparison to Baselines

| Baseline | Mean TEC | Δ vs Your Policy |
|----------|---------:|-----------------:|
| Example policy | {X3_REF['example_mean_tec']:.1f} | {X3_REF['example_mean_tec'] - prev['mean_tec']:+.1f} |
| Random median | {X3_REF['random_median_mean']:.1f} | {X3_REF['random_median_mean'] - prev['mean_tec']:+.1f} |
| Random best c000 | {X3_REF['random_best_mean']:.1f} | {X3_REF['random_best_mean'] - prev['mean_tec']:+.1f} |
| Trimmed baseline | {X3_REF['trimmed_mean_tec']:.1f} | {X3_REF['trimmed_mean_tec'] - prev['mean_tec']:+.1f} |

### All Rounds History

| Round | 61/347 | 62/290 | 65/195 | Mean TEC |
|-------|--------|--------|--------|----------|
{prev_table}

## Your Task

Analyze the results above and propose ONE REVISED policy JSON.

1. Which cells improved vs regressed? Why?
2. What specific field change should fix the regression while preserving gains?
3. State explicitly what you changed in this round and WHY.

Output format: analysis first, then:
```json
{{...}}
```"""

    return prompt


def run_x4_interactive():
    """X4: 5-round interactive DeepSeek policy repair."""
    _ensure_dirs()

    print("=" * 60)
    print("Phase X4 — 5-Round Interactive DeepSeek Policy Repair")
    print("=" * 60)

    rounds_history = []

    for round_num in range(5):
        print(f"\n{'─'*60}")
        print(f"ROUND {round_num}/4")
        print(f"{'─'*60}")

        # ── Build prompt ───────────────────────────────────────────────────
        if round_num == 0:
            prompt = _build_round0_prompt()
        else:
            prompt = _build_round_n_prompt(round_num, rounds_history)

        prompt_path = PROMPTS_DIR / f"x4_round_{round_num}.md"
        with open(prompt_path, "w") as f:
            f.write(prompt)
        print(f"  Prompt → {prompt_path} ({len(prompt)} chars)")

        # ── Call DeepSeek ──────────────────────────────────────────────────
        print(f"  Calling DeepSeek...", end=" ", flush=True)
        messages = []
        if round_num >= 2:
            prev_prompt = open(PROMPTS_DIR / f"x4_round_{round_num - 1}.md").read()
            prev_resp = open(RESPONSES_DIR / f"x4_round_{round_num - 1}_raw.md").read()
            messages.append({"role": "user", "content": prev_prompt[:4000]})
            messages.append({"role": "assistant", "content": prev_resp[:4000]})
        messages.append({"role": "user", "content": prompt})
        content, meta = _call_deepseek(messages)

        resp_path = RESPONSES_DIR / f"x4_round_{round_num}_raw.md"
        with open(resp_path, "w") as f:
            f.write(content)
        meta_path = RESPONSES_DIR / f"x4_round_{round_num}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"{len(content)} chars, {meta['elapsed_sec']:.0f}s")
        print(f"  Response → {resp_path}")

        # ── Extract & validate JSON ────────────────────────────────────────
        policy = _extract_json_from_response(content)
        if policy is None:
            print("  ERROR: No JSON found in response. Saving full response for manual extraction.")
            continue

        ok, errors = _validate_policy(policy)
        if not ok:
            print(f"  JSON validation errors: {errors}")
            if round_num > 0:
                print("  Attempting JSON repair via DeepSeek...")
                repair_prompt = (
                    f"The JSON you generated failed validation:\n"
                    + "\n".join(f"  - {e}" for e in errors)
                    + f"\n\nOriginal JSON:\n```json\n{json.dumps(policy, indent=2)}\n```\n\n"
                    + "Please output the FIXED JSON (same policy, just fix validation errors):\n```json\n{...}\n```"
                )
                repair_content, _ = _call_deepseek([{"role": "user", "content": repair_prompt}])
                repair_path = RESPONSES_DIR / f"x4_round_{round_num}_repair_raw.md"
                with open(repair_path, "w") as f:
                    f.write(repair_content)
                fixed = _extract_json_from_response(repair_content)
                if fixed is not None:
                    ok2, err2 = _validate_policy(fixed)
                    if ok2:
                        policy = fixed
                        print(f"  JSON repaired successfully.")
                    else:
                        print(f"  Repair also failed: {err2}")
                        continue
                else:
                    print("  Could not extract JSON from repair response.")
                    continue
            else:
                continue

        print(f"  Policy: {policy.get('policy_name', '?')}")

        # ── Save policy JSON ───────────────────────────────────────────────
        policy_path = LLM_INTERACTIVE_DIR / f"x4_round_{round_num}.json"
        with open(policy_path, "w") as f:
            json.dump(policy, f, indent=2)
            f.write("\n")
        print(f"  Policy JSON → {policy_path}")

        # ── Evaluate on 3 cells ────────────────────────────────────────────
        tecs = {}
        print(f"  Evaluating on 3 cells...")
        for inst, eps, role in X4_CELLS:
            print(f"    {inst}/{eps}...", end=" ", flush=True)
            rc, stdout, stderr = _run_variant(
                "phaseX_policy_json", inst, eps,
                extra_env={"PHASEX_POLICY_PATH": str(policy_path.resolve())},
            )
            if rc != 0:
                print(f"FAILED rc={rc}")
                tecs[inst] = None
                continue
            row = _parse_csv(stdout)
            if row is None or not _is_feasible(row):
                print("INFEASIBLE")
                tecs[inst] = None
                continue
            tec = float(row.get("tec_total", 0))
            tecs[inst] = tec
            example_tec = X3_REF["example_per_cell"].get(inst, 0)
            delta = tec - example_tec
            best_rand_tec = 0
            if inst == 61:
                best_rand_tec = 6877
            elif inst == 62:
                best_rand_tec = 9561
            elif inst == 65:
                best_rand_tec = 26325
            print(f"TEC={tec:.0f} (Δex={delta:+.0f}, Δbest={tec-best_rand_tec:+.0f})")

        valid_tecs = [v for v in tecs.values() if v is not None]
        mean_tec = sum(valid_tecs) / len(valid_tecs) if valid_tecs else float("inf")

        print(f"  Mean TEC = {mean_tec:.1f}" + (
            f" (Δ example = {mean_tec - X3_REF['example_mean_tec']:+.1f})"
            if valid_tecs else ""
        ))

        rounds_history.append({
            "round": round_num,
            "policy": policy,
            "tecs": tecs,
            "mean_tec": mean_tec,
            "n_valid": len(valid_tecs),
        })

    # ── Final Report ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("X4 FINAL REPORT")
    print(f"{'='*60}")

    all_valid = [r for r in rounds_history if r["n_valid"] == 3]
    if not all_valid:
        print("ERROR: No round had all 3 cells feasible.")
        return

    best_round = min(all_valid, key=lambda r: r["mean_tec"])

    print(f"\nPer-Round Summary:")
    print(f"  {'Round':<8s} {'61/347':>8s} {'62/290':>8s} {'65/195':>8s} {'Mean':>10s} {'ΔEx':>10s} {'ΔMed':>10s} {'ΔBest':>10s}")
    for r in rounds_history:
        if r["n_valid"] < 3:
            continue
        dm_ex = r["mean_tec"] - X3_REF["example_mean_tec"]
        dm_med = r["mean_tec"] - X3_REF["random_median_mean"]
        dm_best = r["mean_tec"] - X3_REF["random_best_mean"]
        marker = "← BEST" if r is best_round else ""
        print(f"  Round {r['round']:<3d} "
              f"{r['tecs'].get(61, 0):>8.0f} "
              f"{r['tecs'].get(62, 0):>8.0f} "
              f"{r['tecs'].get(65, 0):>8.0f} "
              f"{r['mean_tec']:>10.1f} "
              f"{dm_ex:>+10.1f} "
              f"{dm_med:>+10.1f} "
              f"{dm_best:>+10.1f} "
              f"{marker}")

    beats_example = sum(1 for r in all_valid if r["mean_tec"] < X3_REF["example_mean_tec"])
    beats_median = sum(1 for r in all_valid if r["mean_tec"] < X3_REF["random_median_mean"])
    beats_best = sum(1 for r in all_valid if r["mean_tec"] < X3_REF["random_best_mean"])
    improved = sum(1 for i in range(1, len(rounds_history))
                   if rounds_history[i]["n_valid"] == 3 and rounds_history[i-1]["n_valid"] == 3
                   and rounds_history[i]["mean_tec"] < rounds_history[i-1]["mean_tec"])

    print(f"\nAggregate:")
    print(f"  Best round: Round {best_round['round']} (mean TEC = {best_round['mean_tec']:.1f})")
    print(f"  Δ vs example_policy: {best_round['mean_tec'] - X3_REF['example_mean_tec']:+.1f}")
    print(f"  Δ vs random median: {best_round['mean_tec'] - X3_REF['random_median_mean']:+.1f}")
    print(f"  Δ vs random best c000: {best_round['mean_tec'] - X3_REF['random_best_mean']:+.1f}")
    print(f"  Rounds beating example: {beats_example}/5")
    print(f"  Rounds beating random median: {beats_median}/5")
    print(f"  Rounds beating random best: {beats_best}/5")
    print(f"  Rounds improved over previous round: {improved}/4")

    success_level = "FAILURE"
    if beats_example > 0:
        success_level = "MINIMUM SUCCESS"
    if beats_best > 0:
        success_level = "STRONG SUCCESS"

    print(f"\n  Verdict: {success_level}")

    # ── Save eval CSVs ─────────────────────────────────────────────────────
    # Per-round CSV
    rounds_csv_path = EVAL_DIR / "x4_interactive_rounds.csv"
    with open(rounds_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "round", "policy_name",
            "tec_61_347", "tec_62_290", "tec_65_195",
            "mean_tec", "delta_vs_example", "delta_vs_random_median",
            "delta_vs_random_best", "n_valid",
        ])
        w.writeheader()
        for r in rounds_history:
            w.writerow({
                "round": r["round"],
                "policy_name": r["policy"].get("policy_name", ""),
                "tec_61_347": r["tecs"].get(61, ""),
                "tec_62_290": r["tecs"].get(62, ""),
                "tec_65_195": r["tecs"].get(65, ""),
                "mean_tec": f"{r['mean_tec']:.1f}" if r["n_valid"] == 3 else "",
                "delta_vs_example": f"{r['mean_tec'] - X3_REF['example_mean_tec']:+.1f}" if r["n_valid"] == 3 else "",
                "delta_vs_random_median": f"{r['mean_tec'] - X3_REF['random_median_mean']:+.1f}" if r["n_valid"] == 3 else "",
                "delta_vs_random_best": f"{r['mean_tec'] - X3_REF['random_best_mean']:+.1f}" if r["n_valid"] == 3 else "",
                "n_valid": r["n_valid"],
            })
    print(f"\nRounds CSV → {rounds_csv_path}")

    return {
        "best_round": best_round,
        "rounds_history": rounds_history,
        "beats_example": beats_example,
        "beats_best": beats_best,
        "success_level": success_level,
    }


# ── X5 equal-budget random comparison ─────────────────────────────────────────

X5_N_BATCHES = 20
X5_N_PER_BATCH = 5
X5_LLM_BEST = 14285.7  # LLM best-of-5 mean TEC from X4 Round 2
X5_SEED_BASE = 5000
X5_CELLS = [(61, 347, "guard"), (62, 290, "secondary"), (65, 195, "primary")]


def run_x5_equal_budget():
    """X5: 50 batches × 5 random policies on 3 dev cells.

    Checkpointed to eval/x5_batch_checkpoint.csv. Safe to resume.
    """
    _ensure_dirs()
    checkpoint_path = EVAL_DIR / "x5_batch_checkpoint.csv"
    campaign_dir = POLICIES_DIR / "random_bestof5_batches"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase X5 — Random Best-of-5 Distribution Estimator")
    print(f"  {X5_N_BATCHES} independent batches × {X5_N_PER_BATCH} policies = "
          f"{X5_N_BATCHES * X5_N_PER_BATCH} random policies total")
    print(f"  Unit of comparison: best-of-5 (LLM) vs best-of-5 (random batch)")
    print(f"  LLM best-of-5 target: mean TEC = {X5_LLM_BEST:.1f}")
    print("=" * 60)

    # Load or init checkpoint
    completed = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for row in csv.DictReader(f):
                completed.add(row["batch_id"])
        print(f"  Resuming from checkpoint: {len(completed)} batches done")

    # Main loop
    for batch_idx in range(X5_N_BATCHES):
        batch_id = f"b{batch_idx:03d}"
        if batch_id in completed:
            continue

        print(f"\n  Batch {batch_idx+1}/{X5_N_BATCHES} ({batch_id})...", flush=True)
        t0_batch = time.time()

        batch_tecs = {}
        batch_feasible = True

        for p in range(X5_N_PER_BATCH):
            policy_seed = X5_SEED_BASE + batch_idx * X5_N_PER_BATCH + p
            policy_name = f"x5_b{batch_idx:03d}_p{p}_{policy_seed}"

            # Generate
            _, pol_path = generate_random_policy(
                output_path=campaign_dir / f"{policy_name}.json",
                seed=policy_seed,
            )
            with open(pol_path) as f:
                pol = json.load(f)
            pol["policy_name"] = policy_name
            with open(pol_path, "w") as f:
                json.dump(pol, f, indent=2)
                f.write("\n")

            # Evaluate
            pol_tecs = {}
            for inst, eps, _ in X5_CELLS:
                rc, stdout, stderr = _run_variant(
                    "phaseX_policy_json", inst, eps,
                    extra_env={"PHASEX_POLICY_PATH": str(pol_path.resolve())},
                )
                if rc != 0:
                    print(f"    p{p}: {policy_name} on {inst}/{eps} FAILED rc={rc}", file=sys.stderr)
                    pol_tecs[inst] = None
                    continue
                row = _parse_csv(stdout)
                if row is None or not _is_feasible(row):
                    print(f"    p{p}: {policy_name} on {inst}/{eps} INFEASIBLE", file=sys.stderr)
                    pol_tecs[inst] = None
                    continue
                tec = float(row.get("tec_total", 0))
                pol_tecs[inst] = tec

            valid = [v for v in pol_tecs.values() if v is not None]
            if len(valid) == 3:
                mean_tec = sum(valid) / 3
                batch_tecs[policy_name] = {
                    "tecs": pol_tecs,
                    "mean_tec": mean_tec,
                }
            else:
                print(f"    p{p}: {policy_name} not fully feasible ({len(valid)}/3 cells)")

        elapsed = time.time() - t0_batch

        # Find best in batch
        if batch_tecs:
            best_name = min(batch_tecs, key=lambda k: batch_tecs[k]["mean_tec"])
            best_mean = batch_tecs[best_name]["mean_tec"]
            best_tecs = batch_tecs[best_name]["tecs"]
            beats_llm = best_mean < X5_LLM_BEST
            delta_llm = best_mean - X5_LLM_BEST

            print(f"    Best: {best_name} mean={best_mean:.1f} (ΔLLM={delta_llm:+.1f}) "
                  f"{'BEATS LLM' if beats_llm else ''}  ({elapsed:.0f}s)")

            # Save checkpoint line
            ck_row = {
                "batch_id": batch_id,
                "batch_idx": batch_idx,
                "n_feasible": len(batch_tecs),
                "best_policy": best_name,
                "tec_61_347": f"{best_tecs.get(61, 0):.0f}",
                "tec_62_290": f"{best_tecs.get(62, 0):.0f}",
                "tec_65_195": f"{best_tecs.get(65, 0):.0f}",
                "mean_tec": f"{best_mean:.1f}",
                "delta_vs_llm": f"{delta_llm:+.1f}",
                "beats_llm": str(beats_llm),
                "wall_sec": f"{elapsed:.0f}",
            }
        else:
            print(f"    ALL FAILED ({elapsed:.0f}s)")
            ck_row = {
                "batch_id": batch_id, "batch_idx": batch_idx,
                "n_feasible": 0, "best_policy": "", "tec_61_347": "", "tec_62_290": "",
                "tec_65_195": "", "mean_tec": "", "delta_vs_llm": "", "beats_llm": "",
                "wall_sec": f"{elapsed:.0f}",
            }

        # Append checkpoint
        fieldnames = list(ck_row.keys())
        write_header = not checkpoint_path.exists()
        with open(checkpoint_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow(ck_row)

    # ── Analysis ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("X5 ANALYSIS — Random Best-of-5 Distribution")
    print(f"{'='*60}")

    # Read complete checkpoint
    all_batches = []
    with open(checkpoint_path) as f:
        for row in csv.DictReader(f):
            all_batches.append(row)

    ml = [float(r["mean_tec"]) for r in all_batches if r["mean_tec"]]
    if not ml:
        print("ERROR: No valid batches")
        return

    ml_sorted = sorted(ml)
    n = len(ml_sorted)
    median_ml = ml_sorted[n // 2]
    best_ml = ml_sorted[0]
    worst_ml = ml_sorted[-1]

    # IQR (p25, p75)
    q1_idx = max(0, n // 4)
    q3_idx = min(n - 1, 3 * n // 4)
    q1 = ml_sorted[q1_idx]
    q3 = ml_sorted[q3_idx]
    iqr = q3 - q1

    n_beats_llm = sum(1 for m in ml if m < X5_LLM_BEST)  # random batches BETTER than LLM
    n_llm_beats_random = n - n_beats_llm  # random batches WORSE than LLM
    rank = n_beats_llm + 1
    pct_rank = (n - rank) / n * 100  # fraction of random batches WORSE than LLM

    print(f"\n  LLM best-of-5 mean TEC: {X5_LLM_BEST:.1f}")
    print(f"\n  Random best-of-5 distribution (N={n} batches):")
    print(f"    Best (global best-of-{X5_N_BATCHES * X5_N_PER_BATCH} oracle):  {best_ml:.1f}")
    print(f"    Q1:  {q1:.1f}")
    print(f"    Median: {median_ml:.1f}")
    print(f"    Q3:  {q3:.1f}")
    print(f"    IQR: {iqr:.1f}")
    print(f"    Worst: {worst_ml:.1f}")
    print(f"\n  LLM vs random best-of-5 distribution:")
    print(f"    Random batches beating LLM: {n_beats_llm}/{n} ({n_beats_llm/n*100:.0f}%)")
    print(f"    LLM beats {n_llm_beats_random}/{n} ({n_llm_beats_random/n*100:.0f}%) random batches")
    print(f"    LLM percentile rank: {pct_rank:.0f}% (rank {rank} of {n})")
    print(f"    LLM beats median best-of-5: {'YES' if X5_LLM_BEST < median_ml else 'NO'}")
    print(f"    LLM in top quartile (≥p75): {'YES' if pct_rank >= 75 else 'NO'}")

    # Oracle reference (not equal budget)
    print(f"\n  Oracle reference (global best-of-{X5_N_BATCHES * X5_N_PER_BATCH}, NOT equal budget):")
    print(f"    Global best random: {best_ml:.1f}")
    print(f"    Δ LLM vs global best: {X5_LLM_BEST - best_ml:+.1f}")

    if pct_rank >= 75:
        strength = "STRONG"
    elif pct_rank >= 50:
        strength = "MODERATE"
    else:
        strength = "WEAK"

    print(f"\n  Signal strength: {strength}")
    if pct_rank < 50:
        print(f"  WARNING: LLM is worse than random median best-of-5.")

    # ── Save summary CSVs ────────────────────────────────────────────────────
    batches_path = EVAL_DIR / "x5_random_bestof5_batches.csv"
    with open(checkpoint_path) as f:
        content = f.read()
    with open(batches_path, "w") as f:
        f.write(content)
    print(f"\nBatches CSV → {batches_path}")

    # Summary CSV
    summary_path = EVAL_DIR / "x5_random_bestof5_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        w.writerow({"metric": "llm_best_of_5_mean_tec", "value": f"{X5_LLM_BEST:.1f}"})
        w.writerow({"metric": "n_batches", "value": str(n)})
        w.writerow({"metric": "random_best_bestof5_oracle", "value": f"{best_ml:.1f}"})
        w.writerow({"metric": "random_q1_bestof5", "value": f"{q1:.1f}"})
        w.writerow({"metric": "random_median_bestof5", "value": f"{median_ml:.1f}"})
        w.writerow({"metric": "random_q3_bestof5", "value": f"{q3:.1f}"})
        w.writerow({"metric": "random_iqr_bestof5", "value": f"{iqr:.1f}"})
        w.writerow({"metric": "random_worst_bestof5", "value": f"{worst_ml:.1f}"})
        w.writerow({"metric": "n_random_beats_llm", "value": str(n_beats_llm)})
        w.writerow({"metric": "n_llm_beats_random", "value": str(n_llm_beats_random)})
        w.writerow({"metric": "llm_percentile_rank", "value": f"{pct_rank:.1f}"})
        w.writerow({"metric": "signal_strength", "value": strength})
    print(f"Summary CSV → {summary_path}")

    return {
        "llm_best": X5_LLM_BEST,
        "random_best": best_ml,
        "random_median": median_ml,
        "n_beats_llm": n_beats_llm,
        "pct_rank": pct_rank,
        "strength": strength,
    }


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
        "--x3-random-campaign",
        action="store_true",
        help="X3: 20 random policies × 3 cells + baselines.",
    )
    parser.add_argument(
        "--x4-interactive",
        action="store_true",
        help="X4: 5-round interactive DeepSeek policy repair.",
    )
    parser.add_argument(
        "--x5-equal-budget",
        action="store_true",
        help="X5: 20 batches × 5 random policies — best-of-5 distribution estimator.",
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

    elif args.x3_random_campaign:
        run_x3_campaign()

    elif args.x4_interactive:
        run_x4_interactive()

    elif args.x5_equal_budget:
        run_x5_equal_budget()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
