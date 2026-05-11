"""Helper: process C3A DeepSeek curl response → save all artifacts."""
import json, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from pathlib import Path
from phaseC3_smoke_pilot import load_json, save_json, extract_json_from_text
from phaseC_adversarial_family_generation import validate_family, load_schema

ITER_DIR = Path(PROJECT_ROOT) / "research/glns_llm_heuristic_20260422/iterations/20260510_phaseC_adversarial_ehs_benchmark_design"
RESPONSES_DIR = ITER_DIR / "responses"
FAMILIES_DIR = ITER_DIR / "families"
RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

with open(Path(PROJECT_ROOT) / "temp/c3a_deepseek_response.json") as f:
    api_data = json.load(f)

content = api_data["choices"][0]["message"]["content"]
meta = {
    "model": api_data.get("model"),
    "usage": api_data.get("usage", {}),
    "finish_reason": api_data["choices"][0].get("finish_reason"),
    "completion_tokens": api_data["usage"]["completion_tokens"],
    "prompt_tokens": api_data["usage"]["prompt_tokens"],
    "reasoning_tokens": api_data["usage"]["completion_tokens_details"]["reasoning_tokens"],
    "cached_tokens": api_data["usage"]["prompt_tokens_details"]["cached_tokens"],
}

with open(RESPONSES_DIR / "call1_family_designer_raw.md", "w") as f:
    f.write(content)
save_json(meta, RESPONSES_DIR / "call1_family_designer_metadata.json")
print(f"Raw response: {len(content)} chars saved")

json_str = extract_json_from_text(content)
if json_str is None:
    json_str = content.strip()
families_data = json.loads(json_str)
families_list = families_data if isinstance(families_data, list) else families_data.get("families", [])

if isinstance(families_data, list):
    families_data = {"generator": "deepseek_v4_pro", "generator_call": "call1_family_designer",
                     "n_families": len(families_list), "families": families_list}

save_json(families_data, FAMILIES_DIR / "llm_families_raw.json")
print(f"Raw JSON: {len(families_list)} families saved")

schema = load_schema()
valid = []
for i, fam in enumerate(families_list):
    vr = validate_family(fam, i, schema)
    if vr.valid:
        valid.append(fam)
        print(f"  ✅ {i}: {fam.get('family_name', '?')} → {fam.get('expected_EHS_failure_mechanism', '?')}")
    else:
        print(f"  ❌ {i}: {fam.get('family_name', '?')}: {len(vr.errors)} errors")
        for e in vr.errors:
            print(f"     • {e}")
        for w in vr.warnings:
            print(f"     ⚠️  {w}")

est_cost = (api_data["usage"]["prompt_tokens"] * 0.14 + api_data["usage"]["completion_tokens"] * 1.10) / 1_000_000

output = {
    "generator": "deepseek_v4_pro",
    "generator_call": "call1_family_designer",
    "generator_description": "LLM-designed adversarial instance families targeting EHS failure mechanisms based on B6 closure evidence.",
    "n_families": len(valid),
    "n_total": len(families_list),
    "cost_usd": round(est_cost, 4),
    "families": valid,
}
save_json(output, FAMILIES_DIR / "llm_families.json")
print(f"\nValid: {len(valid)}/{len(families_list)} → {FAMILIES_DIR / 'llm_families.json'}")
print(f"Estimated cost: ${est_cost:.4f}")
