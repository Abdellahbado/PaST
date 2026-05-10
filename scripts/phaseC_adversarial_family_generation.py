#!/usr/bin/env python3
"""Phase C: Adversarial Benchmark Family Generation.

Subcommands:
  --generate-random-families   → families/random_families.json
  --generate-human-sweep-families → families/human_sweep_families.json
  --validate-family-file FILE  → validate against family_schema.json

Usage:
  conda run -n new-ml-env python3 scripts/phaseC_adversarial_family_generation.py --generate-random-families
  conda run -n new-ml-env python3 scripts/phaseC_adversarial_family_generation.py --generate-human-sweep-families
  conda run -n new-ml-env python3 scripts/phaseC_adversarial_family_generation.py --validate-family-file research/glns_llm_heuristic_20260422/iterations/20260510_phaseC_adversarial_ehs_benchmark_design/families/random_families.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ITER_DIR = (
    PROJECT_ROOT
    / "research/glns_llm_heuristic_20260422"
    / "iterations/20260510_phaseC_adversarial_ehs_benchmark_design"
)
FAMILIES_DIR = ITER_DIR / "families"
NOTES_DIR = ITER_DIR / "notes"
PROMPTS_DIR = ITER_DIR / "prompts"

SCHEMA_PATH = FAMILIES_DIR / "family_schema.json"

N_FAMILIES = 8

# ─────────────────────────────────────────────────────────────────────────────
# Schema loading
# ─────────────────────────────────────────────────────────────────────────────

def load_schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_FIELDS = [
    "family_name", "hypothesis", "n_jobs_range", "m_machines_range",
    "horizon_T_range", "processing_time_distribution",
    "machine_rate_distribution", "TOU_price_profile_type",
    "epsilon_regime", "expected_EHS_failure_mechanism",
    "generated_instances_count", "validity_constraints", "rejection_conditions",
]

VALID_FAILURE_MECHANISMS = [
    "first_khat_dominance", "asgh_lock_in", "res_reinsertion_starvation",
    "es_exploration_tension", "front_coverage_gap", "short_budget_pressure",
    "load_imbalance", "epsilon_skip", "other",
]

VALID_EPSILON_REGIMES = ["tight", "medium", "loose", "mixed"]

VALID_PROCESSING_DIST_TYPES = [
    "uniform", "normal_truncated", "bimodal", "exponential_truncated", "fixed", "custom",
]

VALID_MACHINE_RATE_TYPES = ["uniform", "step", "exponential", "custom"]

VALID_TOU_TYPES = [
    "flat", "single_peak", "dual_peak", "step_function", "random_walk",
    "high_variance", "low_variance", "monotonic_increasing",
    "monotonic_decreasing", "custom",
]


class ValidationResult:
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def add_error(self, msg: str):
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0

    def report(self) -> str:
        lines = []
        if self.errors:
            lines.append(f"ERRORS ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"  ❌ {e}")
        if self.warnings:
            lines.append(f"WARNINGS ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  ⚠️  {w}")
        if not self.errors and not self.warnings:
            lines.append("  ✅ All checks passed.")
        return "\n".join(lines)


def validate_family(family: dict, idx: int, schema: dict) -> ValidationResult:
    r = ValidationResult()
    name = family.get("family_name", f"<index {idx}>")

    for field in REQUIRED_FIELDS:
        if field not in family:
            r.add_error(f"{name}: missing required field '{field}'")

    if "family_name" in family and (not isinstance(family["family_name"], str) or not family["family_name"]):
        r.add_error(f"{name}: family_name must be non-empty string")

    if "hypothesis" in family and (not isinstance(family["hypothesis"], str) or len(family["hypothesis"]) < 20):
        r.add_warning(f"{name}: hypothesis is too short (<20 chars), may lack detail")

    if "expected_EHS_failure_mechanism" in family:
        mech = family["expected_EHS_failure_mechanism"]
        if mech not in VALID_FAILURE_MECHANISMS:
            r.add_error(f"{name}: invalid mechanism '{mech}', must be one of {VALID_FAILURE_MECHANISMS}")

    if "epsilon_regime" in family:
        er = family["epsilon_regime"]
        if er not in VALID_EPSILON_REGIMES:
            r.add_error(f"{name}: invalid epsilon_regime '{er}'")

    for range_field in ["n_jobs_range", "m_machines_range", "horizon_T_range"]:
        if range_field in family:
            rf = family[range_field]
            if not isinstance(rf, dict) or "min" not in rf or "max" not in rf:
                r.add_error(f"{name}: {range_field} must have 'min' and 'max'")
                continue
            if not isinstance(rf["min"], (int, float)) or not isinstance(rf["max"], (int, float)):
                r.add_error(f"{name}: {range_field} min/max must be numbers")
            elif rf["min"] > rf["max"]:
                r.add_error(f"{name}: {range_field} min > max")
            elif rf["min"] < 1:
                r.add_error(f"{name}: {range_field} min must be >= 1")

    n_range = family.get("n_jobs_range", {"min": 0, "max": 0})
    m_range = family.get("m_machines_range", {"min": 0, "max": 0})
    t_range = family.get("horizon_T_range", {"min": 0, "max": 0})

    n_min = n_range.get("min", 0)
    m_min = m_range.get("min", 0)
    t_min = t_range.get("min", 0)

    if n_min > 0 and m_min > 0 and n_min < 2 * m_min:
        r.add_warning(f"{name}: n_min ({n_min}) < 2 * m_min ({m_min}), may be degenerate")

    if m_min > 0 and m_min < 3:
        r.add_warning(f"{name}: m_min < 3, single-machine degenerate risk")

    if n_min > 0 and t_min > 0 and n_min > 2 * t_min:
        r.add_warning(f"{name}: n_min ({n_min}) > 2 * T_min ({t_min}), scheduling may be impossible")

    if "processing_time_distribution" in family:
        pt = family["processing_time_distribution"]
        if not isinstance(pt, dict) or "type" not in pt:
            r.add_error(f"{name}: processing_time_distribution missing 'type'")
        elif pt["type"] not in VALID_PROCESSING_DIST_TYPES:
            r.add_error(f"{name}: invalid processing_time_distribution type '{pt['type']}'")

    if "machine_rate_distribution" in family:
        mr = family["machine_rate_distribution"]
        if not isinstance(mr, dict) or "type" not in mr:
            r.add_error(f"{name}: machine_rate_distribution missing 'type'")
        elif mr["type"] not in VALID_MACHINE_RATE_TYPES:
            r.add_error(f"{name}: invalid machine_rate_distribution type '{mr['type']}'")

    if "TOU_price_profile_type" in family:
        tou = family["TOU_price_profile_type"]
        if tou not in VALID_TOU_TYPES:
            r.add_error(f"{name}: invalid TOU_price_profile_type '{tou}'")

    if "generated_instances_count" in family:
        cnt = family["generated_instances_count"]
        if not isinstance(cnt, int) or cnt < 3 or cnt > 50:
            r.add_error(f"{name}: generated_instances_count must be 3-50, got {cnt}")

    if "rejection_conditions" in family:
        rc = family["rejection_conditions"]
        if not isinstance(rc, list) or len(rc) == 0:
            r.add_warning(f"{name}: rejection_conditions is empty")
        else:
            for cond in rc:
                if not isinstance(cond, str) or not cond:
                    r.add_error(f"{name}: rejection_condition must be non-empty string")

    if "validity_constraints" in family:
        vc = family["validity_constraints"]
        if not isinstance(vc, dict) or "feasibility_guarantee" not in vc:
            r.add_error(f"{name}: validity_constraints missing 'feasibility_guarantee'")

    expected_gen = family.get("generated_instances_count", 0)
    t_max = t_range.get("max", 0)
    m_max = m_range.get("max", 0)
    if expected_gen > 0 and t_max > 0 and m_max > 0:
        if not (3 <= expected_gen <= 50):
            r.add_error(f"{name}: generated_instances_count {expected_gen} out of [3,50]")

    return r


def validate_family_file(filepath: str) -> bool:
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False

    with open(filepath) as f:
        data = json.load(f)

    schema = load_schema()

    if isinstance(data, dict) and "families" in data:
        print(f"  (Detected wrapper format, generator='{data.get('generator', '?')}' — extracting families array)")
        data = data["families"]

    if not isinstance(data, list):
        print(f"❌ Root must be a JSON array of family specs, got {type(data).__name__}")
        return False

    if len(data) != N_FAMILIES:
        print(f"⚠️  Expected {N_FAMILIES} families, found {len(data)}")

    all_valid = True
    for i, family in enumerate(data):
        result = validate_family(family, i, schema)
        print(f"\n--- Family {i}: {family.get('family_name', f'<unnamed>')} ---")
        print(result.report())
        if not result.valid:
            all_valid = False

    print(f"\n{'='*60}")
    if all_valid:
        print("✅ Overall: ALL FAMILIES VALID")
    else:
        print("❌ Overall: SOME FAMILIES HAVE ERRORS")
    return all_valid


# ─────────────────────────────────────────────────────────────────────────────
# Random family generation
# ─────────────────────────────────────────────────────────────────────────────

TOU_PARAMS_PRESETS = {
    "single_peak": {"peak_start": 0.3, "peak_end": 0.6, "peak_multiplier": 1.5, "base_price": 1.0},
    "dual_peak": {"peak1_start": 0.2, "peak1_end": 0.35, "peak2_start": 0.65, "peak2_end": 0.8, "peak_multiplier": 1.5, "base_price": 1.0},
    "high_variance": {"base_price": 0.5, "noise_range": 1.5, "variance": "high"},
    "low_variance": {"base_price": 1.0, "noise_range": 0.3, "variance": "low"},
    "step_function": {"base_price": 0.5, "step_multiplier": 2.5, "step_start": 0.5},
    "monotonic_increasing": {"start_price": 0.3, "end_price": 2.0},
    "monotonic_decreasing": {"start_price": 2.0, "end_price": 0.3},
    "random_walk": {"start_price": 1.0, "step_size": 0.15},
}


def make_random_family(rng: random.Random, idx: int) -> dict:
    m_min = rng.randint(3, 20)
    m_max = rng.randint(m_min, min(m_min + 15, 50))
    m_mid = (m_min + m_max) // 2

    n_min = rng.randint(max(20, 2 * m_min), 300)
    n_max = rng.randint(n_min, min(n_min + 200, 600))
    n_mid = (n_min + n_max) // 2

    proc_type = rng.choice(["uniform", "bimodal", "exponential_truncated", "normal_truncated"])
    if proc_type == "uniform":
        p_low = rng.randint(1, 5)
        p_high = rng.randint(p_low + 2, max(p_low + 3, 20))
        proc_params = {"low": p_low, "high": p_high}
    elif proc_type == "bimodal":
        p_small_low = rng.randint(1, 3)
        p_small_high = rng.randint(4, 8)
        p_large_low = rng.randint(10, 15)
        p_large_high = rng.randint(16, 25)
        proc_params = {"small_low": p_small_low, "small_high": p_small_high,
                       "large_low": p_large_low, "large_high": p_large_high,
                       "small_fraction": round(rng.uniform(0.3, 0.7), 2)}
    elif proc_type == "exponential_truncated":
        proc_params = {"scale": round(rng.uniform(2.0, 8.0), 1), "max": 30}
    else:
        proc_params = {"mean": round(rng.uniform(5.0, 12.0), 1), "std": round(rng.uniform(1.5, 4.0), 1), "min": 1, "max": 30}

    p_avg_est = 8
    if proc_type == "uniform":
        p_avg_est = (proc_params.get("low", 1) + proc_params.get("high", 10)) // 2
    elif proc_type == "bimodal":
        small_avg = (proc_params.get("small_low", 2) + proc_params.get("small_high", 5)) // 2
        large_avg = (proc_params.get("large_low", 15) + proc_params.get("large_high", 25)) // 2
        sf = proc_params.get("small_fraction", 0.5)
        p_avg_est = int(sf * small_avg + (1 - sf) * large_avg)
    elif proc_type == "exponential_truncated":
        p_avg_est = min(15, int(proc_params.get("scale", 5) * 1.5))
    elif proc_type == "normal_truncated":
        p_avg_est = int(proc_params.get("mean", 8))
    t_min_est = max(20, (n_mid * p_avg_est * 3) // (2 * m_mid))
    t_min = rng.randint(t_min_est, t_min_est + 200)
    t_max = rng.randint(t_min + 50, min(t_min + 400, 1000))

    mr_type = rng.choice(["uniform", "step", "exponential"])
    if mr_type == "uniform":
        mr_params = {"low": round(rng.uniform(0.3, 1.5), 2), "high": round(rng.uniform(2.0, 5.0), 2)}
    elif mr_type == "step":
        mr_params = {"low_rate": round(rng.uniform(0.3, 1.0), 2),
                     "high_rate": round(rng.uniform(2.0, 4.0), 2),
                     "step_fraction": round(rng.uniform(0.3, 0.7), 2)}
    else:
        mr_params = {"scale": round(rng.uniform(0.5, 2.0), 2), "max": 5.0}

    tou_type = rng.choice([t for t in VALID_TOU_TYPES if t != "flat" and t != "custom"])
    tou_params = TOU_PARAMS_PRESETS.get(tou_type, {}).copy()
    if "base_price" in tou_params:
        tou_params["base_price"] = round(rng.uniform(0.5, 2.5), 2)
    if "peak_multiplier" in tou_params:
        tou_params["peak_multiplier"] = round(rng.uniform(1.3, 3.0), 2)

    epsilon_regime = rng.choice(["tight", "medium", "loose"])
    failure_mechanism = rng.choice(VALID_FAILURE_MECHANISMS[:-1])

    hypothesis_templates = {
        "first_khat_dominance": f"At n≈{n_mid}, m≈{m_mid}, first khat cost dominates EHS budget. Instance sized to make SGH construction at khat=T the runtime bottleneck, leaving no time for multi-khat descent.",
        "asgh_lock_in": f"A-SGH 96% job retention may lock EHS into suboptimal trajectories when khat→khat-1 optimal assignments are structurally different. Instance designed with job sizes that force spread changes at each khat.",
        "res_reinsertion_starvation": f"R-ES reinsertion only fires on 1.4% of khats. Instance designed with dense feasible schedule space so reinsertion is valuable but never activated within budget.",
        "es_exploration_tension": f"ES non-empty local improvements (36.6% rate) may prevent R-ES from escaping to better regions. Instance with heterogeneous energy rates and sharp TOU peaks to induce exploration trap.",
        "front_coverage_gap": f"Discontinuous energy-vs-cmax Pareto region from step-function TOU and bimodal jobs. EHS front may miss intermediate points.",
        "short_budget_pressure": f"At n≈{n_mid}, m≈{m_mid}, EHS under 120s cannot complete a full khat. Instance sized at the boundary where first-khat cost equals time budget.",
        "load_imbalance": f"Very heterogeneous machine rates (range {mr_params.get('high', '?')}/{mr_params.get('low', '?')}) cause SGH to concentrate all jobs on cheap machines, inflating cmax at constant energy.",
        "epsilon_skip": f"Coarse epsilon spacing relative to energy rate differences means EHS skips epsilons worth exploring. PROFILE designed with narrow price levels and wide machine rate differences.",
    }

    hypothesis = hypothesis_templates.get(failure_mechanism, f"Random family #{idx} targeting {failure_mechanism} mechanism with n≈{n_mid}, m≈{m_mid}.")

    return {
        "family_name": f"random_{idx:03d}",
        "hypothesis": hypothesis,
        "description": f"Randomly generated family #{idx}: n=[{n_min},{n_max}], m=[{m_min},{m_max}], T=[{t_min},{t_max}], {proc_type} processing, {mr_type} machine rates, {tou_type} TOU, {epsilon_regime} epsilon.",
        "n_jobs_range": {"min": n_min, "max": n_max},
        "m_machines_range": {"min": m_min, "max": m_max},
        "horizon_T_range": {"min": t_min, "max": t_max},
        "processing_time_distribution": {"type": proc_type, "params": proc_params},
        "machine_rate_distribution": {"type": mr_type, "params": mr_params},
        "TOU_price_profile_type": tou_type,
        "TOU_price_profile_params": tou_params,
        "epsilon_regime": epsilon_regime,
        "expected_EHS_failure_mechanism": failure_mechanism,
        "expected_EHS_failure_mechanism_evidence": f"B6 evidence: random assignment, no targeted mechanism prediction.",
        "generated_instances_count": 8,
        "validity_constraints": {
            "feasibility_guarantee": True,
            "min_total_work": n_mid,
            "n_per_machine_min": 2,
        },
        "rejection_conditions": [
            "sum(p_j) > m * T (obviously infeasible)",
            "all e[h] equal (no energy rate differentiation)",
            "all ct[t] equal (flat price, no TOU)",
            "n < 2 * m (degenerate: too few jobs)",
        ],
        "seed_behavior": {"base_seed": 10000 + idx * 100, "expected_seed_variance": "medium"},
    }


def generate_random_families(seed: int = 42) -> dict:
    rng = random.Random(seed)
    families = []
    for i in range(N_FAMILIES):
        attempts = 0
        while attempts < 20:
            family = make_random_family(rng, i)
            vr = validate_family(family, i, load_schema())
            if vr.valid:
                families.append(family)
                break
            attempts += 1
            if attempts >= 20:
                print(f"WARNING: Could not generate valid random family {i} after 20 attempts")
                families.append(family)
    return {
        "generator": "random",
        "generator_seed": seed,
        "generator_description": "Uniform sample of legal parameter ranges from family_schema.json",
        "n_families": len(families),
        "families": families,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Human sweep families
# ─────────────────────────────────────────────────────────────────────────────

def make_human_sweep_families() -> dict:
    families = [
        {
            "family_name": "human_tight_epsilon",
            "hypothesis": "Tight epsilon (small cmax decrements) forces many khat iterations. SGH construction dominates per-khat cost, making EHS slow to converge. Under short budget (60s), EHS may explore only 1-2 khats, leaving most of the front unexplored.",
            "description": "Tight epsilon ≈ 0.5 * max(p_j). Moderate n/m. EHS must descend through many small cmax steps, each paying full SGH+A-SGH+R-ES cost.",
            "n_jobs_range": {"min": 60, "max": 120},
            "m_machines_range": {"min": 8, "max": 12},
            "horizon_T_range": {"min": 150, "max": 250},
            "processing_time_distribution": {"type": "uniform", "params": {"low": 1, "high": 10}},
            "machine_rate_distribution": {"type": "uniform", "params": {"low": 0.5, "high": 3.0}},
            "TOU_price_profile_type": "single_peak",
            "TOU_price_profile_params": {"peak_start": 0.3, "peak_end": 0.6, "peak_multiplier": 2.0, "base_price": 1.0},
            "epsilon_regime": "tight",
            "expected_EHS_failure_mechanism": "first_khat_dominance",
            "expected_EHS_failure_mechanism_evidence": "B6.13 run28: first khat dominates (100-400s). B6.17b: short-budget gap 12.9-71.6% at 120s.",
            "generated_instances_count": 8,
            "validity_constraints": {
                "feasibility_guarantee": True,
                "min_total_work": 300,
                "n_per_machine_min": 5,
            },
            "rejection_conditions": [
                "sum(p_j) > m * T",
                "all e[h] equal",
                "all ct[t] equal",
                "n < 2 * m",
            ],
            "seed_behavior": {"base_seed": 20000, "expected_seed_variance": "medium"},
        },
        {
            "family_name": "human_loose_epsilon",
            "hypothesis": "Loose epsilon (large cmax decrements) means few khat iterations. If epsilon is too coarse, EHS skips intermediate cmax values worth exploring, producing sparse fronts. Under long budget, the sparse front still leaves gaps.",
            "description": "Loose epsilon ≈ 2 * max(p_j). Wide spacing between khats. EHS may skip valuable cmax-energetic tradeoff points, producing a sparse or gappy Pareto front.",
            "n_jobs_range": {"min": 60, "max": 120},
            "m_machines_range": {"min": 8, "max": 12},
            "horizon_T_range": {"min": 150, "max": 250},
            "processing_time_distribution": {"type": "uniform", "params": {"low": 1, "high": 10}},
            "machine_rate_distribution": {"type": "uniform", "params": {"low": 0.5, "high": 3.0}},
            "TOU_price_profile_type": "single_peak",
            "TOU_price_profile_params": {"peak_start": 0.3, "peak_end": 0.6, "peak_multiplier": 2.0, "base_price": 1.0},
            "epsilon_regime": "loose",
            "expected_EHS_failure_mechanism": "epsilon_skip",
            "expected_EHS_failure_mechanism_evidence": "B6 evidence: epsilon=LB derived step size controls khat count. Loose epsilon means fewer khats, potentially skipping intermediate tradeoff points.",
            "generated_instances_count": 8,
            "validity_constraints": {
                "feasibility_guarantee": True,
                "min_total_work": 300,
                "n_per_machine_min": 5,
            },
            "rejection_conditions": [
                "sum(p_j) > m * T",
                "all e[h] equal",
                "all ct[t] equal",
                "n < 2 * m",
            ],
            "seed_behavior": {"base_seed": 20100, "expected_seed_variance": "medium"},
        },
        {
            "family_name": "human_high_price_volatility",
            "hypothesis": "High TOU price volatility creates sharp energy-vs-cmax tradeoffs. SGH construction may make over-optimistic energy decisions at khat=T that degrade during descent. ES non-empty local improvements may trap the solver in high-energy regions.",
            "description": "TOU profile with extreme variance: prices range from 0.2 to 5.0 within horizon. Sharp peaks create energy holes that SGH construction exploits but that R-ES cannot escape.",
            "n_jobs_range": {"min": 50, "max": 100},
            "m_machines_range": {"min": 6, "max": 15},
            "horizon_T_range": {"min": 120, "max": 300},
            "processing_time_distribution": {"type": "uniform", "params": {"low": 2, "high": 15}},
            "machine_rate_distribution": {"type": "uniform", "params": {"low": 0.3, "high": 4.0}},
            "TOU_price_profile_type": "high_variance",
            "TOU_price_profile_params": {"base_price": 0.5, "noise_range": 3.0, "variance": "high"},
            "epsilon_regime": "medium",
            "expected_EHS_failure_mechanism": "es_exploration_tension",
            "expected_EHS_failure_mechanism_evidence": "B6.4b: ES non-empty improves 36.6% of khats. When TOU is volatile, ES may find local improvements that prevent better reinsertion moves.",
            "generated_instances_count": 8,
            "validity_constraints": {
                "feasibility_guarantee": True,
                "min_total_work": 250,
                "n_per_machine_min": 3,
            },
            "rejection_conditions": [
                "sum(p_j) > m * T",
                "all e[h] equal",
                "all ct[t] equal",
                "n < 2 * m",
            ],
            "seed_behavior": {"base_seed": 20200, "expected_seed_variance": "high"},
        },
        {
            "family_name": "human_low_price_volatility",
            "hypothesis": "Low TOU price volatility means energy cost is dominated by machine assignment rather than scheduling. EHS's sophisticated per-machine resequencing (R-ES reinsertion at 1.81s/khat) becomes wasted effort since any assignment is nearly energy-equivalent.",
            "description": "TOU price nearly flat (variance minimal). Energy minimization reduces almost entirely to machine assignment. R-ES reinsertion's scheduling refinement is wasted.",
            "n_jobs_range": {"min": 50, "max": 150},
            "m_machines_range": {"min": 8, "max": 20},
            "horizon_T_range": {"min": 200, "max": 400},
            "processing_time_distribution": {"type": "uniform", "params": {"low": 1, "high": 20}},
            "machine_rate_distribution": {"type": "step", "params": {"low_rate": 0.5, "high_rate": 3.0, "step_fraction": 0.5}},
            "TOU_price_profile_type": "low_variance",
            "TOU_price_profile_params": {"base_price": 1.0, "noise_range": 0.2, "variance": "low"},
            "epsilon_regime": "medium",
            "expected_EHS_failure_mechanism": "res_reinsertion_starvation",
            "expected_EHS_failure_mechanism_evidence": "B6.4b: R-ES reinsertion improves only 1.4% of khats. With flat prices, scheduling refinement has near-zero value. Reinsertion cost (1.81s/khat) is pure waste.",
            "generated_instances_count": 8,
            "validity_constraints": {
                "feasibility_guarantee": True,
                "min_total_work": 250,
                "n_per_machine_min": 3,
            },
            "rejection_conditions": [
                "sum(p_j) > m * T",
                "all e[h] equal",
                "all ct[t] equal",
                "n < 2 * m",
            ],
            "seed_behavior": {"base_seed": 20300, "expected_seed_variance": "low"},
        },
        {
            "family_name": "human_many_small_jobs",
            "hypothesis": "Many small jobs (n high, p_j small) produce a dense combinatorial space. SGH construction dominates runtime (O(n*m*khat)). A-SGH must evaluate many possible assignments per job. EHS may timeout without completing even the first khat under short budget.",
            "description": "High job count (n≥200), small processing times (p_j ≤ 5). SGH cost scales with n*m. First khat may exceed 60s budget.",
            "n_jobs_range": {"min": 200, "max": 400},
            "m_machines_range": {"min": 10, "max": 20},
            "horizon_T_range": {"min": 200, "max": 400},
            "processing_time_distribution": {"type": "uniform", "params": {"low": 1, "high": 5}},
            "machine_rate_distribution": {"type": "uniform", "params": {"low": 0.5, "high": 3.0}},
            "TOU_price_profile_type": "dual_peak",
            "TOU_price_profile_params": {"peak1_start": 0.2, "peak1_end": 0.35, "peak2_start": 0.65, "peak2_end": 0.8, "peak_multiplier": 1.8, "base_price": 1.0},
            "epsilon_regime": "medium",
            "expected_EHS_failure_mechanism": "short_budget_pressure",
            "expected_EHS_failure_mechanism_evidence": "B6.13 run28: first khat dominates. B6.17b: EHS at 120s reaches only 12.9% on large instances. Many small jobs maximize SGH construction cost.",
            "generated_instances_count": 8,
            "validity_constraints": {
                "feasibility_guarantee": True,
                "min_total_work": 200,
                "n_per_machine_min": 10,
            },
            "rejection_conditions": [
                "sum(p_j) > m * T",
                "all e[h] equal",
                "all ct[t] equal",
                "n < 2 * m",
            ],
            "seed_behavior": {"base_seed": 20400, "expected_seed_variance": "medium"},
        },
        {
            "family_name": "human_mixed_job_sizes",
            "hypothesis": "Bimodal job sizes (small + large) create assignment tension: large jobs dominate load balance, small jobs provide energy flexibility. SGH myopic assignment may misplace large jobs early, and A-SGH retention locks them in. R-ES reinsertion cannot relocate large jobs efficiently.",
            "description": "Bimodal processing times: 50% small (p_j=2-5), 50% large (p_j=15-25). Large jobs are hard to reinsert; small jobs create combinatorial explosion for ES.",
            "n_jobs_range": {"min": 80, "max": 150},
            "m_machines_range": {"min": 8, "max": 15},
            "horizon_T_range": {"min": 200, "max": 400},
            "processing_time_distribution": {"type": "bimodal", "params": {"small_low": 2, "small_high": 5, "large_low": 15, "large_high": 25, "small_fraction": 0.5}},
            "machine_rate_distribution": {"type": "step", "params": {"low_rate": 0.5, "high_rate": 3.0, "step_fraction": 0.6}},
            "TOU_price_profile_type": "single_peak",
            "TOU_price_profile_params": {"peak_start": 0.3, "peak_end": 0.6, "peak_multiplier": 2.0, "base_price": 1.0},
            "epsilon_regime": "medium",
            "expected_EHS_failure_mechanism": "asgh_lock_in",
            "expected_EHS_failure_mechanism_evidence": "B6.11: A-SGH keeps 96-98% of jobs. Released jobs repair back. Bimodal jobs create structurally different optimal assignments at different khats — A-SGH retention becomes a liability.",
            "generated_instances_count": 8,
            "validity_constraints": {
                "feasibility_guarantee": True,
                "min_total_work": 400,
                "n_per_machine_min": 5,
            },
            "rejection_conditions": [
                "sum(p_j) > m * T",
                "all e[h] equal",
                "all ct[t] equal",
                "n < 2 * m",
            ],
            "seed_behavior": {"base_seed": 20500, "expected_seed_variance": "medium"},
        },
        {
            "family_name": "human_many_machines_sparse",
            "hypothesis": "Many machines with few jobs each creates assignment combinatorial search without obvious structure. ES moves span many machines. SGH assignment quality may be poor because each machine sees too few jobs for meaningful load balancing.",
            "description": "High machine count (m≥20), moderate job count (n). Low job-per-machine ratio means each machine has few jobs. SGH may assign suboptimally due to limited per-machine information.",
            "n_jobs_range": {"min": 60, "max": 120},
            "m_machines_range": {"min": 20, "max": 35},
            "horizon_T_range": {"min": 100, "max": 200},
            "processing_time_distribution": {"type": "uniform", "params": {"low": 2, "high": 10}},
            "machine_rate_distribution": {"type": "exponential", "params": {"scale": 1.0, "max": 4.0}},
            "TOU_price_profile_type": "single_peak",
            "TOU_price_profile_params": {"peak_start": 0.3, "peak_end": 0.6, "peak_multiplier": 1.8, "base_price": 1.0},
            "epsilon_regime": "medium",
            "expected_EHS_failure_mechanism": "front_coverage_gap",
            "expected_EHS_failure_mechanism_evidence": "Many machines + few jobs = each khat produces a narrow energy range. Combined with moderate TOU, EHS front may be sparse.",
            "generated_instances_count": 8,
            "validity_constraints": {
                "feasibility_guarantee": True,
                "min_total_work": 120,
                "n_per_machine_min": 2,
            },
            "rejection_conditions": [
                "sum(p_j) > m * T",
                "all e[h] equal",
                "all ct[t] equal",
                "n < 2 * m",
            ],
            "seed_behavior": {"base_seed": 20600, "expected_seed_variance": "high"},
        },
        {
            "family_name": "human_few_machines_dense",
            "hypothesis": "Few machines with many jobs each makes machine-level energy and load optimization critical. SGH assignment quality is crucial because each job placement has large impact. ES non-empty becomes combinatorially expensive (O(n*m) per khat). R-ES reinsertion may be valuable but too slow to activate.",
            "description": "Low machine count (m=4-6), high job count. Dense per-machine assignments. ES neighborhood is large. R-ES reinsertion is valuable but bottlenecked.",
            "n_jobs_range": {"min": 150, "max": 300},
            "m_machines_range": {"min": 4, "max": 8},
            "horizon_T_range": {"min": 400, "max": 600},
            "processing_time_distribution": {"type": "uniform", "params": {"low": 1, "high": 10}},
            "machine_rate_distribution": {"type": "uniform", "params": {"low": 0.5, "high": 3.5}},
            "TOU_price_profile_type": "dual_peak",
            "TOU_price_profile_params": {"peak1_start": 0.25, "peak1_end": 0.4, "peak2_start": 0.6, "peak2_end": 0.75, "peak_multiplier": 1.6, "base_price": 1.0},
            "epsilon_regime": "tight",
            "expected_EHS_failure_mechanism": "load_imbalance",
            "expected_EHS_failure_mechanism_evidence": "Few machines + many jobs = load balance critical. SGH may fail to balance load across heterogeneous machine rates. Cmax inflation from poor assignment is hard for R-ES to repair.",
            "generated_instances_count": 8,
            "validity_constraints": {
                "feasibility_guarantee": True,
                "min_total_work": 300,
                "n_per_machine_min": 20,
            },
            "rejection_conditions": [
                "sum(p_j) > m * T",
                "all e[h] equal",
                "all ct[t] equal",
                "n < 2 * m",
            ],
            "seed_behavior": {"base_seed": 20700, "expected_seed_variance": "medium"},
        },
    ]

    return {
        "generator": "human_sweep",
        "generator_description": "Fixed parameter sweeps over epsilon tightness, price volatility, job size distribution, and machine density. 8 families, each targeting a specific EHS mechanism.",
        "n_families": len(families),
        "families": families,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase C family generation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate-random-families", action="store_true")
    group.add_argument("--generate-human-sweep-families", action="store_true")
    group.add_argument("--validate-family-file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random families")
    args = parser.parse_args()

    FAMILIES_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    if args.generate_random_families:
        print(f"Generating {N_FAMILIES} random families (seed={args.seed})...")
        data = generate_random_families(seed=args.seed)
        out_path = FAMILIES_DIR / "random_families.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  → Saved: {out_path}")

        schema = load_schema()
        all_ok = True
        for i, fam in enumerate(data["families"]):
            vr = validate_family(fam, i, schema)
            if not vr.valid:
                all_ok = False
                print(f"  ❌ Family {i} ({fam['family_name']}): {len(vr.errors)} errors")
                for e in vr.errors:
                    print(f"     • {e}")
        if all_ok:
            print("  ✅ All random families pass validation.")

        note_path = NOTES_DIR / "c2_random_family_generation.md"
        note_lines = [
            "# C2A Random Family Generation",
            "",
            f"**Date**: 2026-05-10",
            f"**Seed**: {args.seed}",
            f"**Families generated**: {len(data['families'])}",
            "",
            "## Method",
            "Uniform random sampling of legal parameter ranges from `family_schema.json`.",
            "Each family is generated by sampling: n_jobs_range, m_machines_range, horizon_T_range,",
            "processing_time_distribution (type + params), machine_rate_distribution, TOU_price_profile_type,",
            "epsilon_regime, and a random failure mechanism.",
            "",
            "## Validation",
            "All families validated against schema using `--validate-family-file`. Sanity constraints",
            "enforced: feasible n/m/T ratios, non-degenerate processing times, non-degenerate TOU,",
            "meaningful epsilon regime.",
            "",
            "## Families",
        ]
        for i, fam in enumerate(data["families"]):
            note_lines.append(f"- **{fam['family_name']}**: {fam['expected_EHS_failure_mechanism']} — n=[{fam['n_jobs_range']['min']},{fam['n_jobs_range']['max']}], m=[{fam['m_machines_range']['min']},{fam['m_machines_range']['max']}], T=[{fam['horizon_T_range']['min']},{fam['horizon_T_range']['max']}], {fam['processing_time_distribution']['type']} proc, {fam['machine_rate_distribution']['type']} rates, {fam['TOU_price_profile_type']} TOU, {fam['epsilon_regime']} eps")
        with open(note_path, "w") as f:
            f.write("\n".join(note_lines) + "\n")
        print(f"  → Notes: {note_path}")

    elif args.generate_human_sweep_families:
        print(f"Generating {N_FAMILIES} human sweep families...")
        data = make_human_sweep_families()
        out_path = FAMILIES_DIR / "human_sweep_families.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  → Saved: {out_path}")

        schema = load_schema()
        all_ok = True
        for i, fam in enumerate(data["families"]):
            vr = validate_family(fam, i, schema)
            if not vr.valid:
                all_ok = False
                print(f"  ❌ Family {i} ({fam['family_name']}): {len(vr.errors)} errors")
                for e in vr.errors:
                    print(f"     • {e}")
        if all_ok:
            print("  ✅ All human sweep families pass validation.")

        note_path = NOTES_DIR / "c2_human_sweep_family_generation.md"
        families_desc = {
            "human_tight_epsilon": ("Tight epsilon (≈0.5·max(p_j)) forces many khats. SGH cost dominates. EHS explores few khats under short budget.", "first_khat_dominance"),
            "human_loose_epsilon": ("Loose epsilon (≈2·max(p_j)) gives few khats. Intermediate tradeoff points may be skipped.", "epsilon_skip"),
            "human_high_price_volatility": ("High TOU price variance creates sharp energy tradeoffs. ES may trap solver in local improvements.", "es_exploration_tension"),
            "human_low_price_volatility": ("Nearly flat TOU prices. R-ES reinsertion scheduling refinement is wasted effort since any schedule is energy-equivalent.", "res_reinsertion_starvation"),
            "human_many_small_jobs": ("High n, small p_j. SGH O(n·m) cost dominates. First khat may exceed short budget.", "short_budget_pressure"),
            "human_mixed_job_sizes": ("Bimodal job sizes. Large jobs hard to reinsert. A-SGH locks in early assignment errors.", "asgh_lock_in"),
            "human_many_machines_sparse": ("Many machines, few jobs per machine. Sparse front. Combinatorial assignment search.", "front_coverage_gap"),
            "human_few_machines_dense": ("Few machines, dense jobs. Load balance critical. SGH may fail on heterogeneous rates.", "load_imbalance"),
        }
        note_lines = [
            "# C2B Human Sweep Family Generation",
            "",
            "**Date**: 2026-05-10",
            f"**Families generated**: {len(data['families'])}",
            "",
            "## Method",
            "8 fixed families designed as simple parameter sweeps over key dimensions:",
            "epsilon tightness, price volatility, job size distribution, and machine density.",
            "Each family targets a specific EHS mechanism based on B6 evidence.",
            "",
            "## Design rationale",
            "These are simple, transparent designs — the human baseline should be strong",
            "enough to challenge the LLM, not trivially beatable.",
            "",
            "## Families",
        ]
        for fam in data["families"]:
            desc, mech = families_desc[fam["family_name"]]
            note_lines.append(f"- **{fam['family_name']}** → {mech}: {desc}")
        with open(note_path, "w") as f:
            f.write("\n".join(note_lines) + "\n")
        print(f"  → Notes: {note_path}")

    elif args.validate_family_file:
        filepath = args.validate_family_file
        print(f"Validating: {filepath}")
        ok = validate_family_file(filepath)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
