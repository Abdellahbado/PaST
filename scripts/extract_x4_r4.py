#!/usr/bin/env python3
import json, re, sys, os
os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None); os.environ.pop("HTTP_PROXY", None); os.environ.pop("HTTPS_PROXY", None)

BASE = "research/learned_move_screening_20260420/iterations/20260508_phaseX_interactive_llm_policy_repair"
with open(f"{BASE}/responses/x4_round_4_raw.md") as f:
    content = f.read()

m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
if m:
    policy = json.loads(m.group(1))
else:
    m = re.search(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", content, re.DOTALL)
    if m:
        policy = json.loads(m.group(1))
    else:
        print("ERROR: No JSON found")
        sys.exit(1)

print(f"Policy: {policy.get('policy_name', '?')}")

with open(f"{BASE}/policies/schema.json") as f:
    schema = json.load(f)
props = schema.get("properties", {})
req = schema.get("required", [])
errors = []

for key in req:
    if key not in policy:
        errors.append(f"missing {key}")
for key, val in policy.items():
    if key not in props:
        continue
    spec = props[key]
    if "enum" in spec and val not in spec["enum"]:
        errors.append(f"{key}: {val} not in {spec['enum']}")
    if "minimum" in spec and isinstance(val, (int, float)) and val < spec["minimum"]:
        errors.append(f"{key}: {val} < min {spec['minimum']}")
    if "maximum" in spec and isinstance(val, (int, float)) and val > spec["maximum"]:
        errors.append(f"{key}: {val} > max {spec['maximum']}")

if errors:
    print(f"Validation errors: {errors}")
    if len(errors) == 1 and "max_per_target" in errors[0]:
        policy["max_per_target"] = 4
        print("Auto-fixed max_per_target to 4")
    else:
        print("Cannot auto-fix")
        sys.exit(1)
else:
    print("Validation OK")

pol_path = f"{BASE}/policies/llm_interactive/x4_round_4.json"
with open(pol_path, "w") as f:
    json.dump(policy, f, indent=2)
    f.write("\n")
print(f"Saved to {pol_path}")
print(json.dumps(policy, indent=2))
