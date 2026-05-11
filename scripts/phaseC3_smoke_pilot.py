#!/usr/bin/env python3
"""Phase C3: DeepSeek family generation + smoke pilot.

Usage:
  # C3A: Call DeepSeek for LLM families
  conda run -n new-ml-env python3 scripts/phaseC3_smoke_pilot.py --c3a-deepseek-call

  # C3A validate only (no API call)
  conda run -n new-ml-env python3 scripts/phaseC3_smoke_pilot.py --c3a-validate-llm

  # C3B: Select smoke families (from existing family files)
  conda run -n new-ml-env python3 scripts/phaseC3_smoke_pilot.py --c3b-select-families

  # C3C: Generate smoke instances
  conda run -n new-ml-env python3 scripts/phaseC3_smoke_pilot.py --c3c-generate-instances

  # C3D: Run EHS evaluation
  conda run -n new-ml-env python3 scripts/phaseC3_smoke_pilot.py --c3d-ehs-eval

  # C3E: Compare and summarize
  conda run -n new-ml-env python3 scripts/phaseC3_smoke_pilot.py --c3e-compare

  # All phases (no DeepSeek call)
  conda run -n new-ml-env python3 scripts/phaseC3_smoke_pilot.py --c3-all-no-llm
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from glns.paper_heuristics import run_ehs, PaperSchedule
from glns.sequencing import _CYTHON_AVAILABLE

ITER_DIR = (
    PROJECT_ROOT
    / "research/glns_llm_heuristic_20260422"
    / "iterations/20260510_phaseC_adversarial_ehs_benchmark_design"
)
FAMILIES_DIR = ITER_DIR / "families"
PROMPTS_DIR = ITER_DIR / "prompts"
RESPONSES_DIR = ITER_DIR / "responses"
EVAL_DIR = ITER_DIR / "eval"
NOTES_DIR = ITER_DIR / "notes"
GEN_INST_DIR = ITER_DIR / "generated_instances" / "c3_smoke"

SCHEMA_PATH = FAMILIES_DIR / "family_schema.json"
LLM_PROMPT_PATH = PROMPTS_DIR / "call1_llm_family_designer.md"
RANDOM_FAMILIES_PATH = FAMILIES_DIR / "random_families.json"
HUMAN_FAMILIES_PATH = FAMILIES_DIR / "human_sweep_families.json"
LLM_FAMILIES_RAW_PATH = FAMILIES_DIR / "llm_families_raw.json"
LLM_FAMILIES_PATH = FAMILIES_DIR / "llm_families.json"
SELECTED_FAMILIES_PATH = NOTES_DIR / "c3_selected_families.json"

# EHS evaluation settings
EHS_SHORT_BUDGET = 30  # reduced: LLM families have n≈800, m≈40 — very slow
EHS_LONG_BUDGET = 60   # reduced from 300: even 60s may not complete one khat on large instances
SMOKE_INSTANCES_PER_FAMILY = 3
SMOKE_SEED_BASE = 50000

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_env_deepseek():
    if "DEEPSEEK_API_KEY" in os.environ:
        return
    env_file = PROJECT_ROOT / ".env.deepseek.sh"
    if not env_file.exists():
        raise RuntimeError(f"{env_file} not found. Source it first or set DEEPSEEK_API_KEY.")
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and not os.environ.get(k):
                    os.environ[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# C3A: DeepSeek Call
# ─────────────────────────────────────────────────────────────────────────────

def call_deepseek(prompt: str, max_tokens: int = 16000) -> Tuple[str, dict]:
    load_env_deepseek()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")

    import subprocess
    import tempfile
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        tf.write(payload)
        tmp_path = tf.name

    try:
        result = subprocess.run(
            ["curl", "-s", "--connect-timeout", "30", "--max-time", "300",
             "-X", "POST", f"{base_url}/chat/completions",
             "-H", f"Authorization: Bearer {api_key}",
             "-H", "Content-Type: application/json",
             "-d", f"@{tmp_path}"],
            capture_output=True, text=True, timeout=310,
        )
        if result.returncode != 0:
            raise RuntimeError(f"curl failed: {result.stderr}")

        data = json.loads(result.stdout)
        if "error" in data:
            raise RuntimeError(f"DeepSeek API error: {data['error']}")
        content = data["choices"][0]["message"]["content"]
        meta = {
            "model": data.get("model", model),
            "usage": data.get("usage", {}),
            "finish_reason": data["choices"][0].get("finish_reason", "unknown"),
        }
        return content, meta
    finally:
        os.unlink(tmp_path)


def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON block from markdown/text response."""
    # Try ```json ... ``` first
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    # Try ``` ... ```
    m = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        inner = m.group(1)
        try:
            json.loads(inner)
            return inner
        except json.JSONDecodeError:
            pass
    # Try raw JSON
    text_stripped = text.strip()
    if text_stripped.startswith("{") or text_stripped.startswith("["):
        try:
            json.loads(text_stripped)
            return text_stripped
        except json.JSONDecodeError:
            pass
    return None


def run_c3a_deepseek_call():
    print("=" * 60)
    print("C3A: DeepSeek Call 1 — LLM Family Designer")
    print("=" * 60)

    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    with open(LLM_PROMPT_PATH) as f:
        prompt = f.read()

    print(f"  Prompt: {len(prompt)} chars from {LLM_PROMPT_PATH}")
    print("  Calling DeepSeek V4 Pro...")

    content, meta = call_deepseek(prompt, max_tokens=16000)
    print(f"  Response: {len(content)} chars")
    print(f"  Usage: {meta['usage']}")

    # Save raw response
    raw_path = RESPONSES_DIR / "call1_family_designer_raw.md"
    with open(raw_path, "w") as f:
        f.write(content)
    print(f"  → Saved raw response: {raw_path}")

    # Save metadata
    meta_path = RESPONSES_DIR / "call1_family_designer_metadata.json"
    save_json(meta, meta_path)
    print(f"  → Saved metadata: {meta_path}")

    # Extract JSON
    json_str = extract_json_from_text(content)
    if json_str is None:
        print("  ⚠️  No JSON found in response. Attempting repair call...")
        repair_prompt = f"""Your previous response did not contain valid JSON. Please output ONLY the JSON object (no markdown, no explanation) containing 8 family specs in the exact format specified.

The output must start with {{ and end with }} and contain a "families" array.

Previous response:
{content[:2000]}
"""
        content2, meta2 = call_deepseek(repair_prompt, max_tokens=16000)
        json_str = extract_json_from_text(content2)
        if json_str is None:
            json_str2 = content2.strip()
            if json_str2.startswith("{"):
                json_str = json_str2
        if json_str is None:
            raise RuntimeError("Cannot extract valid JSON even after repair call.")

    # Validate and save raw
    try:
        families_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parse error: {e}")
        raise

    save_json(families_data, LLM_FAMILIES_RAW_PATH)
    print(f"  → Saved raw LLM families: {LLM_FAMILIES_RAW_PATH}")

    # Validate against schema
    families_list = families_data if isinstance(families_data, list) else families_data.get("families", [])
    if isinstance(families_data, dict) and "families" in families_data:
        pass  # Wrapper format
    elif isinstance(families_data, list):
        families_data = {"generator": "deepseek_v4_pro", "generator_call": "call1_family_designer",
                         "n_families": len(families_list), "families": families_list}
    else:
        families_data = {"generator": "deepseek_v4_pro", "n_families": 0, "families": []}
        raise RuntimeError("LLM output is not a valid family wrapper or array")

    n_fams = len(families_data.get("families", []))
    print(f"  LLM produced {n_fams} families")

    # Import validation from generation script
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from phaseC_adversarial_family_generation import validate_family, load_schema

    schema = load_schema()
    valid_families = []
    for i, fam in enumerate(families_data.get("families", [])):
        vr = validate_family(fam, i, schema)
        if vr.valid:
            valid_families.append(fam)
            print(f"  ✅ Family {i} ({fam.get('family_name', '?')}): valid")
        else:
            print(f"  ❌ Family {i} ({fam.get('family_name', '?')}): {len(vr.errors)} errors")
            for e in vr.errors:
                print(f"     • {e}")

    if not valid_families:
        print("  ❌ NO valid families after filtering!")
        # Save what we have
        save_json({"generator": "deepseek_v4_pro", "n_families": 0, "families": [],
                    "note": "All families had validation errors"}, LLM_FAMILIES_PATH)
    else:
        output = {"generator": "deepseek_v4_pro",
                   "generator_call": "call1_family_designer",
                   "generator_description": "LLM-designed adversarial instance families targeting EHS failure mechanisms M1-M8 based on B6 closure evidence.",
                   "n_families": len(valid_families),
                   "families": valid_families}
        save_json(output, LLM_FAMILIES_PATH)
        print(f"  → Saved {len(valid_families)} valid families: {LLM_FAMILIES_PATH}")

    print("  ✅ C3A complete.")


def run_c3a_validate_llm():
    """Validate existing LLM families without API call."""
    print("=" * 60)
    print("C3A Validation: Checking LLM families")
    print("=" * 60)

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from phaseC_adversarial_family_generation import validate_family, load_schema

    for path, label in [(LLM_FAMILIES_PATH, "validated"), (LLM_FAMILIES_RAW_PATH, "raw")]:
        if not path.exists():
            print(f"  {label}: file not found")
            continue
        data = load_json(path)
        families = data if isinstance(data, list) else data.get("families", [])
        schema = load_schema()
        ok = 0
        for i, fam in enumerate(families):
            vr = validate_family(fam, i, schema)
            if vr.valid:
                ok += 1
            else:
                print(f"  ❌ {label} family {i}: {len(vr.errors)} errors")
        print(f"  {label}: {ok}/{len(families)} valid")


# ─────────────────────────────────────────────────────────────────────────────
# C3B: Family Selection
# ─────────────────────────────────────────────────────────────────────────────

def run_c3b_select_families():
    print("=" * 60)
    print("C3B: Smoke Family Selection")
    print("=" * 60)

    llm_data = load_json(LLM_FAMILIES_PATH) if LLM_FAMILIES_PATH.exists() else None
    random_data = load_json(RANDOM_FAMILIES_PATH)
    human_data = load_json(HUMAN_FAMILIES_PATH)

    llm_fams = llm_data.get("families", []) if llm_data else []
    random_fams = random_data.get("families", [])
    human_fams = human_data.get("families", [])

    print(f"  LLM: {len(llm_fams)} families available")
    print(f"  Random: {len(random_fams)} families available")
    print(f"  Human: {len(human_fams)} families available")

    selected_llm = llm_fams[:2] if len(llm_fams) >= 2 else llm_fams
    selected_random = random_fams[:2]
    selected_human = human_fams[:2]

    selection = {
        "description": "First 2 valid families from each arm. No cherry-picking.",
        "selected_families": {
            "llm": [{"family_name": f["family_name"],
                      "mechanism": f.get("expected_EHS_failure_mechanism", "?")}
                     for f in selected_llm],
            "random": [{"family_name": f["family_name"],
                         "mechanism": f.get("expected_EHS_failure_mechanism", "?")}
                        for f in selected_random],
            "human": [{"family_name": f["family_name"],
                        "mechanism": f.get("expected_EHS_failure_mechanism", "?")}
                       for f in selected_human],
        },
        "total_families": len(selected_llm) + len(selected_random) + len(selected_human),
    }
    save_json(selection, SELECTED_FAMILIES_PATH)
    print(f"  Selected: {len(selected_llm)} LLM + {len(selected_random)} random + {len(selected_human)} human")
    for arm, fams in [("LLM", selected_llm), ("Random", selected_random), ("Human", selected_human)]:
        for f in fams:
            print(f"    {arm}: {f['family_name']} → {f.get('expected_EHS_failure_mechanism', '?')}")
    print(f"  → Saved: {SELECTED_FAMILIES_PATH}")
    return selection


# ─────────────────────────────────────────────────────────────────────────────
# C3C: Instance Generation
# ─────────────────────────────────────────────────────────────────────────────

def _sample_int(rng: random.Random, lo: int, hi: int) -> int:
    if lo == hi:
        return lo
    return rng.randint(lo, hi)


def _generate_processing_times(rng: random.Random, dist: dict, n: int) -> List[int]:
    ptype = dist.get("type", "uniform")
    params = dist.get("params", {})
    times = []
    if ptype == "uniform":
        lo = params.get("low", 1)
        hi = params.get("high", 10)
        times = [rng.randint(lo, hi) for _ in range(n)]
    elif ptype == "bimodal":
        s_lo, s_hi = params.get("small_low", 1), params.get("small_high", 5)
        l_lo, l_hi = params.get("large_low", 10), params.get("large_high", 20)
        sf = params.get("small_fraction", 0.5)
        for _ in range(n):
            if rng.random() < sf:
                times.append(rng.randint(s_lo, s_hi))
            else:
                times.append(rng.randint(l_lo, l_hi))
    elif ptype == "exponential_truncated":
        scale = params.get("scale", 5.0)
        pmax = params.get("max", 30)
        for _ in range(n):
            v = int(rng.expovariate(1.0 / scale)) + 1
            times.append(min(v, pmax))
    elif ptype == "normal_truncated":
        mean = params.get("mean", 8.0)
        std = params.get("std", 3.0)
        pmin = params.get("min", 1)
        pmax = params.get("max", 30)
        for _ in range(n):
            v = int(rng.gauss(mean, std) + 0.5)
            times.append(max(pmin, min(v, pmax)))
    else:
        times = [rng.randint(1, 10) for _ in range(n)]
    return times


def _generate_machine_rates(rng: random.Random, dist: dict, m: int) -> List[float]:
    mtype = dist.get("type", "uniform")
    params = dist.get("params", {})
    if mtype == "uniform":
        lo = params.get("low", 0.5)
        hi = params.get("high", 3.0)
        return [round(rng.uniform(lo, hi), 4) for _ in range(m)]
    elif mtype == "step":
        lo = params.get("low_rate", 0.5)
        hi = params.get("high_rate", 3.0)
        sf = params.get("step_fraction", 0.5)
        n_hi = max(1, int(m * (1 - sf)))
        n_lo = m - n_hi
        return [round(hi, 4) for _ in range(n_hi)] + [round(lo, 4) for _ in range(n_lo)]
    elif mtype == "exponential":
        scale = params.get("scale", 1.0)
        pmax = params.get("max", 5.0)
        rates = []
        for _ in range(m):
            v = rng.expovariate(1.0 / scale) * scale
            rates.append(round(min(v, pmax) + 0.1, 4))
        return rates
    else:
        return [round(rng.uniform(0.5, 3.0), 4) for _ in range(m)]


def _generate_tou_profile(rng: random.Random, tou_type: str, tou_params: dict, T: int) -> List[float]:
    ct = [0.0] * T
    if tou_type == "flat":
        base = tou_params.get("base_price", 1.0)
        ct = [round(base + rng.uniform(-0.05, 0.05), 4) for _ in range(T)]
    elif tou_type == "single_peak":
        base = tou_params.get("base_price", 1.0)
        peak_mult = tou_params.get("peak_multiplier", 2.0)
        ps = int(tou_params.get("peak_start", 0.3) * T)
        pe = int(tou_params.get("peak_end", 0.6) * T)
        for t in range(T):
            if ps <= t < pe:
                ct[t] = round(base * peak_mult + rng.uniform(-0.1, 0.1), 4)
            else:
                ct[t] = round(base + rng.uniform(-0.1, 0.1), 4)
    elif tou_type == "dual_peak":
        base = tou_params.get("base_price", 1.0)
        peak_mult = tou_params.get("peak_multiplier", 2.0)
        p1s = int(tou_params.get("peak1_start", 0.2) * T)
        p1e = int(tou_params.get("peak1_end", 0.35) * T)
        p2s = int(tou_params.get("peak2_start", 0.65) * T)
        p2e = int(tou_params.get("peak2_end", 0.8) * T)
        for t in range(T):
            if p1s <= t < p1e or p2s <= t < p2e:
                ct[t] = round(base * peak_mult + rng.uniform(-0.1, 0.1), 4)
            else:
                ct[t] = round(base + rng.uniform(-0.1, 0.1), 4)
    elif tou_type == "high_variance":
        base = tou_params.get("base_price", 1.0)
        noise = tou_params.get("noise_range", 2.0)
        for t in range(T):
            ct[t] = round(max(0.1, base + rng.uniform(-noise, noise)), 4)
    elif tou_type == "low_variance":
        base = tou_params.get("base_price", 1.0)
        noise = tou_params.get("noise_range", 0.2)
        for t in range(T):
            ct[t] = round(max(0.1, base + rng.uniform(-noise, noise)), 4)
    elif tou_type == "step_function":
        base = tou_params.get("base_price", 1.0)
        mult = tou_params.get("step_multiplier", 3.0)
        ss = int(tou_params.get("step_start", 0.5) * T)
        for t in range(T):
            ct[t] = round(base * mult if t >= ss else base, 4)
    elif tou_type == "monotonic_increasing":
        start = tou_params.get("start_price", 0.5)
        end = tou_params.get("end_price", 3.0)
        for t in range(T):
            frac = t / max(T - 1, 1)
            ct[t] = round(start + (end - start) * frac, 4)
    elif tou_type == "monotonic_decreasing":
        start = tou_params.get("start_price", 3.0)
        end = tou_params.get("end_price", 0.5)
        for t in range(T):
            frac = t / max(T - 1, 1)
            ct[t] = round(start + (end - start) * frac, 4)
    elif tou_type == "random_walk":
        cur = tou_params.get("start_price", 1.0)
        step = tou_params.get("step_size", 0.2)
        for t in range(T):
            ct[t] = round(max(0.1, cur), 4)
            cur += rng.uniform(-step, step)
            cur = max(0.1, cur)
    elif tou_type == "custom":
        ct = [tou_params.get("base_price", 1.0)] * T
    else:
        ct = [1.0] * T
    return ct


def _generate_one_instance(rng: random.Random, family: dict, instance_idx: int, base_seed: int) -> dict:
    n_r = family["n_jobs_range"]
    m_r = family["m_machines_range"]
    t_r = family["horizon_T_range"]

    n = _sample_int(rng, n_r["min"], n_r["max"])
    m = _sample_int(rng, m_r["min"], m_r["max"])
    T = _sample_int(rng, t_r["min"], t_r["max"])

    p_j = _generate_processing_times(rng, family["processing_time_distribution"], n)
    e_h = _generate_machine_rates(rng, family["machine_rate_distribution"], m)
    ct = _generate_tou_profile(rng, family.get("TOU_price_profile_type", "single_peak"),
                                family.get("TOU_price_profile_params", {}), T)

    inst = {
        "n": n,
        "m": m,
        "T": T,
        "p": p_j,
        "e": e_h,
        "ct": ct,
        "instance_id": f"c3_{family['family_name']}_{base_seed}",
        "metadata": {
            "family_name": family["family_name"],
            "arm": family.get("_arm", "unknown"),
            "seed": base_seed,
            "instance_idx": instance_idx,
            "expected_mechanism": family.get("expected_EHS_failure_mechanism", "?"),
            "feasible": sum(p_j) <= m * T,
        },
    }
    return inst


def run_c3c_generate_instances():
    print("=" * 60)
    print("C3C: Instance Generation")
    print("=" * 60)

    if not SELECTED_FAMILIES_PATH.exists():
        print("  Running C3B first...")
        run_c3b_select_families()

    selection = load_json(SELECTED_FAMILIES_PATH)

    llm_data = load_json(LLM_FAMILIES_PATH) if LLM_FAMILIES_PATH.exists() else {"families": []}
    random_data = load_json(RANDOM_FAMILIES_PATH)
    human_data = load_json(HUMAN_FAMILIES_PATH)

    llm_fams = {f["family_name"]: f for f in llm_data.get("families", [])}
    random_fams = {f["family_name"]: f for f in random_data.get("families", [])}
    human_fams = {f["family_name"]: f for f in human_data.get("families", [])}

    GEN_INST_DIR.mkdir(parents=True, exist_ok=True)

    all_instances = []
    rng = random.Random(42)

    for arm, fam_dict, fam_key in [
        ("llm", llm_fams, "llm"),
        ("random", random_fams, "random"),
        ("human", human_fams, "human"),
    ]:
        fam_names = [f["family_name"] for f in selection["selected_families"][fam_key]]
        for fname in fam_names:
            fam = fam_dict.get(fname)
            if fam is None:
                print(f"  ⚠️  Family {fname} not found in {fam_key} data")
                continue
            fam["_arm"] = arm
            base_seed = fam.get("seed_behavior", {}).get("base_seed", SMOKE_SEED_BASE)
            for i in range(SMOKE_INSTANCES_PER_FAMILY):
                seed = base_seed + i * 17
                inst = _generate_one_instance(rng, fam, i, seed)
                inst["metadata"]["_arm"] = arm

                # Rejection check
                p_sum = sum(inst["p"])
                if p_sum > inst["m"] * inst["T"]:
                    print(f"  ⚠️  {inst['instance_id']}: infeasible (sum(p)={p_sum} > m*T={inst['m']*inst['T']}), re-generating...")
                    for retry in range(10):
                        new_rng = random.Random(seed + retry * 100)
                        new_n = max(10, inst["n"] - retry * 10)
                        new_T = min(1000, inst["T"] + retry * 50)
                        p_j2 = _generate_processing_times(new_rng, fam["processing_time_distribution"], new_n)
                        if sum(p_j2) <= inst["m"] * new_T:
                            inst["n"] = new_n
                            inst["T"] = new_T
                            inst["p"] = p_j2
                            inst["metadata"]["feasible"] = True
                            break
                    else:
                        print(f"  ❌ Could not fix infeasibility for {inst['instance_id']}")
                        inst["metadata"]["feasible"] = False

                fpath = GEN_INST_DIR / f"{inst['instance_id']}.json"
                save_json(inst, fpath)
                all_instances.append(inst)
                print(f"  ✅ {inst['instance_id']}: n={inst['n']}, m={inst['m']}, T={inst['T']}, sum(p)={sum(inst['p'])}, feasible={inst['metadata']['feasible']}")

    # Save manifest
    manifest = {
        "n_instances": len(all_instances),
        "n_feasible": sum(1 for i in all_instances if i["metadata"]["feasible"]),
        "families": selection["selected_families"],
        "instances": [{"id": i["instance_id"], "n": i["n"], "m": i["m"], "T": i["T"],
                        "sum_p": sum(i["p"]), "feasible": i["metadata"]["feasible"]}
                       for i in all_instances],
    }
    save_json(manifest, GEN_INST_DIR / "manifest.json")
    print(f"  → Generated {len(all_instances)} instances ({manifest['n_feasible']} feasible)")
    print(f"  → Manifest: {GEN_INST_DIR / 'manifest.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# C3D: EHS Smoke Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_hv(front_points: List[Tuple[float, float]], ref_point: Tuple[float, float]) -> float:
    """Hypervolume for 2-objective minimization. Front must be sorted by first objective ascending."""
    if not front_points:
        return 0.0
    hv = 0.0
    prev_x = ref_point[0]
    for x, y in sorted(front_points, key=lambda p: p[0]):
        if y < ref_point[1]:
            hv += (ref_point[0] - prev_x) * (ref_point[1] - y)
            prev_x = x
    return hv


def _compute_toy_hv(ehs_front: List[Tuple[float, float]], inst: dict) -> float:
    """Simple HV using (max_cmax, max_energy) as reference."""
    if not ehs_front:
        return 0.0
    max_cmax = max(p[0] for p in ehs_front) + 10
    max_energy = max(p[1] for p in ehs_front) * 1.2
    ref = (max_cmax, max_energy)
    return _compute_hv(ehs_front, ref)


def run_ehs_on_instance(inst: dict, time_limit: float, eps_ordering: str = "default") -> dict:
    """Run EHS on a generated instance. Returns metrics dict."""
    try:
        t0 = time.time()
        front = run_ehs(
            inst, rng=random.Random(42), time_limit_seconds=time_limit, eps_ordering=eps_ordering,
        )
        elapsed = time.time() - t0

        schedule = front[0] if front else None
        if schedule is None:
            return {"error": "EHS returned empty front", "feasible": False}

        metrics = {
            "feasible": schedule.feasible if hasattr(schedule, 'feasible') else True,
            "front_size": len(front),
            "runtime_s": round(elapsed, 2),
            "timeout": elapsed >= time_limit * 0.95,
            "deepest_cmax": min(s.cmax for s in front) if front else -1,
            "best_energy": min(s.energy for s in front) if front else float("inf"),
            "shallowest_cmax": max(s.cmax for s in front) if front else -1,
            "shallowest_energy": max(s.energy for s in front) if front else float("inf"),
        }

        # HV
        points = [(s.cmax, s.energy) for s in front if s.feasible]
        metrics["toy_hv"] = round(_compute_toy_hv(points, inst), 2)

        return metrics
    except Exception as e:
        return {"error": str(e), "feasible": False, "runtime_s": 0}


def run_c3d_ehs_eval():
    print("=" * 60)
    print("C3D: EHS Smoke Evaluation")
    print("=" * 60)

    manifest_path = GEN_INST_DIR / "manifest.json"
    if not manifest_path.exists():
        print("  No manifest found. Run --c3c-generate-instances first.")
        return

    manifest = load_json(manifest_path)
    instances = manifest.get("instances", [])

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = EVAL_DIR / "c3_smoke_raw.csv"
    fieldnames = [
        "instance_id", "arm", "family_name", "expected_mechanism",
        "n", "m", "T", "sum_p", "feasible_instance",
        "budget", "feasible_ehs", "front_size", "runtime_s", "timeout",
        "deepest_cmax", "best_energy", "shallowest_cmax", "toy_hv", "error",
    ]

    rows = []
    for inst_info in instances:
        inst_path = GEN_INST_DIR / f"{inst_info['id']}.json"
        if not inst_path.exists():
            print(f"  ⚠️  Instance file not found: {inst_path}")
            continue
        inst = load_json(inst_path)
        arm = inst["metadata"].get("_arm", inst["metadata"].get("arm", "?"))
        fam_name = inst["metadata"]["family_name"]
        mech = inst["metadata"].get("expected_mechanism", "?")

        for budget, blabel in [(EHS_SHORT_BUDGET, "short"), (EHS_LONG_BUDGET, "long")]:
            print(f"  Running EHS on {inst_info['id']} @ {blabel} ({budget}s)...", end=" ", flush=True)
            metrics = run_ehs_on_instance(inst, budget, eps_ordering="expensive_source_first")
            row = {
                "instance_id": inst_info["id"],
                "arm": arm,
                "family_name": fam_name,
                "expected_mechanism": mech,
                "n": inst["n"],
                "m": inst["m"],
                "T": inst["T"],
                "sum_p": sum(inst["p"]),
                "feasible_instance": inst_info.get("feasible", True),
                "budget": blabel,
                "feasible_ehs": metrics.get("feasible", False),
                "front_size": metrics.get("front_size", 0),
                "runtime_s": metrics.get("runtime_s", 0),
                "timeout": metrics.get("timeout", False),
                "deepest_cmax": metrics.get("deepest_cmax", -1),
                "best_energy": metrics.get("best_energy", float("inf")),
                "shallowest_cmax": metrics.get("shallowest_cmax", -1),
                "toy_hv": metrics.get("toy_hv", 0),
                "error": metrics.get("error", ""),
            }
            rows.append(row)
            status = "✅" if metrics.get("feasible") else "❌"
            print(f"{status} front={metrics.get('front_size', 0)}, hv={metrics.get('toy_hv', 0)}, {metrics.get('runtime_s', 0):.1f}s", flush=True)

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → Saved raw results: {csv_path} ({len(rows)} rows)")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# C3E: Compare Arms
# ─────────────────────────────────────────────────────────────────────────────

def run_c3e_compare():
    print("=" * 60)
    print("C3E: Arm Comparison and Diagnostic Yield")
    print("=" * 60)

    raw_path = EVAL_DIR / "c3_smoke_raw.csv"
    if not raw_path.exists():
        print("  No raw CSV found. Run --c3d-ehs-eval first.")
        return

    rows = []
    with open(raw_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["front_size"] = int(row["front_size"])
            row["feasible_ehs"] = row["feasible_ehs"] == "True"
            row["timeout"] = row["timeout"] == "True"
            row["runtime_s"] = float(row["runtime_s"])
            row["toy_hv"] = float(row["toy_hv"])
            row["feasible_instance"] = row["feasible_instance"] == "True"
            rows.append(row)

    # Group by instance and budget
    from collections import defaultdict
    inst_data = defaultdict(lambda: {"short": None, "long": None})
    for r in rows:
        inst_data[r["instance_id"]][r["budget"]] = r

    # Per-instance analysis
    summary_rows = []
    for inst_id, budgets in sorted(inst_data.items()):
        short = budgets.get("short")
        long = budgets.get("long")
        if short is None or long is None:
            continue

        arm = short["arm"]
        fam = short["family_name"]
        mech = short["expected_mechanism"]

        # Yield criteria
        hv_short = short["toy_hv"]
        hv_long = long["toy_hv"]
        hv_gap_pct = ((hv_long - hv_short) / max(hv_short, 0.01)) * 100 if hv_short > 0 else 100.0
        fs_short = short["front_size"]
        fs_long = long["front_size"]
        fs_gap = fs_long - fs_short

        zero_at_short = fs_short <= 1
        meaningful_long = fs_long >= 2
        first_khat_miss = short["timeout"] and fs_short <= 1

        is_high_yield = (
            hv_gap_pct >= 5.0 or
            fs_gap >= 2 or
            (zero_at_short and meaningful_long) or
            first_khat_miss or
            not short["feasible_ehs"]
        )

        summary_rows.append({
            "instance_id": inst_id,
            "arm": arm,
            "family_name": fam,
            "expected_mechanism": mech,
            "n": short["n"],
            "m": short["m"],
            "T": short["T"],
            "sum_p": short["sum_p"],
            "short_feasible": short["feasible_ehs"],
            "short_front_size": fs_short,
            "short_hv": hv_short,
            "short_runtime_s": short["runtime_s"],
            "short_timeout": short["timeout"],
            "long_feasible": long["feasible_ehs"],
            "long_front_size": fs_long,
            "long_hv": hv_long,
            "long_runtime_s": long["runtime_s"],
            "long_timeout": long["timeout"],
            "hv_gap_pct": round(hv_gap_pct, 1),
            "fs_gap": fs_gap,
            "high_yield": is_high_yield,
        })

    # Arm-level summary
    arm_summary = defaultdict(lambda: {"total": 0, "high_yield": 0, "families": set()})
    for r in summary_rows:
        arm_summary[r["arm"]]["total"] += 1
        if r["high_yield"]:
            arm_summary[r["arm"]]["high_yield"] += 1
        arm_summary[r["arm"]]["families"].add(r["family_name"])

    # Write summary CSV
    summary_path = EVAL_DIR / "c3_smoke_summary.csv"
    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"  → Saved summary: {summary_path}")

    # Print arm comparison
    print()
    print("  Arm Comparison:")
    print(f"  {'Arm':<10} {'Instances':>10} {'High-Yield':>12} {'Rate':>8} {'Families':>10}")
    print(f"  {'-'*50}")
    for arm in ["llm", "random", "human"]:
        s = arm_summary[arm]
        rate = f"{s['high_yield']/s['total']*100:.0f}%" if s["total"] > 0 else "N/A"
        print(f"  {arm:<10} {s['total']:>10} {s['high_yield']:>12} {rate:>8} {len(s['families']):>10}")

    # Per-instance display
    print()
    for r in summary_rows:
        tag = "🔴 HIGH" if r["high_yield"] else "🟢 low"
        print(f"  {tag} {r['instance_id']:<35} short={r['short_front_size']:>3}pts hv={r['short_hv']:>10.1f} → long={r['long_front_size']:>3}pts hv={r['long_hv']:>10.1f} | gap={r['hv_gap_pct']:>5.1f}% fsΔ={r['fs_gap']:>3}")

    # Decision
    print()
    print("  Gate Assessment:")
    llm_rate = arm_summary["llm"]["high_yield"] / max(arm_summary["llm"]["total"], 1)
    random_rate = arm_summary["random"]["high_yield"] / max(arm_summary["random"]["total"], 1)
    human_rate = arm_summary["human"]["high_yield"] / max(arm_summary["human"]["total"], 1)

    if llm_rate > random_rate and llm_rate > human_rate and llm_rate >= 0.33:
        print("  → STRONG: LLM yield clearly above both baselines.")
    elif llm_rate >= max(random_rate, human_rate) and llm_rate >= 0.20:
        print("  → MODERATE: LLM ties best baseline but mechanisms may be more interpretable.")
    elif llm_rate < max(random_rate, human_rate) * 0.8:
        print("  → FAIL: LLM yield lower than baselines.")
    else:
        print("  → WEAK: LLM similar to random/human, no clear advantage.")

    # Write decision note
    decision_path = NOTES_DIR / "c3_smoke_decision.md"
    lines = [
        "# C3 Smoke Decision",
        "",
        f"**Date**: 2026-05-10",
        f"**Instances evaluated**: {len(summary_rows)}",
        "",
        "## Yield Rates",
        f"| Arm | Instances | High-Yield | Rate | Families |",
        f"|-----|----------|------------|------|----------|",
    ]
    for arm in ["llm", "random", "human"]:
        s = arm_summary[arm]
        rate = f"{s['high_yield']/s['total']*100:.0f}%" if s["total"] > 0 else "N/A"
        lines.append(f"| {arm} | {s['total']} | {s['high_yield']} | {rate} | {len(s['families'])} |")

    lines += [
        "",
        "## Gate",
        f"- LLM yield: {llm_rate:.0%}",
        f"- Random yield: {random_rate:.0%}",
        f"- Human yield: {human_rate:.0%}",
        "",
        "## Decision",
    ]

    if llm_rate > random_rate and llm_rate > human_rate and llm_rate >= 0.33:
        lines.append("**STRONG**: Proceed to full campaign.")
    elif llm_rate >= max(random_rate, human_rate):
        lines.append("**MODERATE**: LLM ties baseline. Further smoke or selective campaign.")
    else:
        lines.append("**FAIL/WEAK**: LLM does not beat baselines. Do not proceed to full campaign.")

    lines += [
        "",
        "## Per-Instance Details",
    ]
    for r in summary_rows:
        lines.append(f"- `{r['instance_id']}` ({r['arm']}): {r['expected_mechanism']} — yield={'HIGH' if r['high_yield'] else 'low'}, ΔHV={r['hv_gap_pct']:.1f}%, ΔFS={r['fs_gap']}")

    with open(decision_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  → Decision note: {decision_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase C3 Smoke Pilot")
    parser.add_argument("--c3a-deepseek-call", action="store_true", help="C3A: Call DeepSeek")
    parser.add_argument("--c3a-validate-llm", action="store_true", help="C3A: Validate existing LLM families")
    parser.add_argument("--c3b-select-families", action="store_true", help="C3B: Select smoke families")
    parser.add_argument("--c3c-generate-instances", action="store_true", help="C3C: Generate instances")
    parser.add_argument("--c3d-ehs-eval", action="store_true", help="C3D: Run EHS evaluation")
    parser.add_argument("--c3e-compare", action="store_true", help="C3E: Compare arms")
    parser.add_argument("--c3-all-no-llm", action="store_true", help="Run C3B-C3E (skip DeepSeek call)")
    args = parser.parse_args()

    if args.c3a_deepseek_call:
        run_c3a_deepseek_call()
    elif args.c3a_validate_llm:
        run_c3a_validate_llm()
    elif args.c3b_select_families:
        run_c3b_select_families()
    elif args.c3c_generate_instances:
        run_c3c_generate_instances()
    elif args.c3d_ehs_eval:
        run_c3d_ehs_eval()
    elif args.c3e_compare:
        run_c3e_compare()
    elif args.c3_all_no_llm:
        run_c3b_select_families()
        run_c3c_generate_instances()
        run_c3d_ehs_eval()
        run_c3e_compare()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
