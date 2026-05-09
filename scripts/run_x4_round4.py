#!/usr/bin/env python3
"""Run X4 Round 4: custom prompt → DeepSeek → extract → save → validate."""
import os, sys, urllib.request, json, time, re

BASE = "research/learned_move_screening_20260420/iterations/20260508_phaseX_interactive_llm_policy_repair"

# Read Round 4 prompt
with open(f"{BASE}/prompts/x4_round_4.md") as f:
    prompt = f.read()

print(f"Prompt: {len(prompt)} chars")

# Read previous context
with open(f"{BASE}/responses/x4_round_2_raw.md") as f:
    prev_resp = f.read()
with open(f"{BASE}/prompts/x4_round_2.md") as f:
    prev_prompt = f.read()

messages = [
    {"role": "user", "content": prev_prompt[:4000]},
    {"role": "assistant", "content": prev_resp[:4000]},
    {"role": "user", "content": prompt},
]

url = "https://api.deepseek.com/chat/completions"
body = json.dumps({
    "model": "deepseek-v4-pro",
    "messages": messages,
    "temperature": 0.5,
    "max_tokens": 16000,
}).encode()
req = urllib.request.Request(url, data=body, headers={
    "Content-Type": "application/json",
    "Authorization": f'Bearer {os.environ["DEEPSEEK_API_KEY"]}',
})
t0 = time.time()
with urllib.request.urlopen(req, timeout=600) as resp:
    data = json.loads(resp.read().decode())
elapsed = time.time() - t0
content = data["choices"][0]["message"]["content"]
print(f"Response: {len(content)} chars, {elapsed:.0f}s")

# Save raw
resp_path = f"{BASE}/responses/x4_round_4_raw.md"
with open(resp_path, "w") as f:
    f.write(content)
print(f"Saved to {resp_path}")

meta_path = f"{BASE}/responses/x4_round_4_meta.json"
usage = data.get("usage", {})
meta = {
    "model": data.get("model", "deepseek-v4-pro"),
    "prompt_tokens": usage.get("prompt_tokens", 0),
    "completion_tokens": usage.get("completion_tokens", 0),
    "elapsed_sec": round(elapsed, 1),
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

# Extract JSON
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

print(f"Policy name: {policy.get('policy_name', '?')}")

# Validate
with open(f"{BASE}/policies/schema.json") as f:
    schema = json.load(f)
props = schema.get("properties", {})
errors = []
for key in schema.get("required", []):
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
    # Auto-fix max_per_target if that's the only error
    if len(errors) == 1 and "max_per_target" in errors[0]:
        policy["max_per_target"] = 4
        print("Auto-fixed max_per_target to 4")
    else:
        print("Cannot auto-fix")
        sys.exit(1)
else:
    print("Validation OK")

# Save
pol_path = f"{BASE}/policies/llm_interactive/x4_round_4.json"
with open(pol_path, "w") as f:
    json.dump(policy, f, indent=2)
    f.write("\n")
print(f"Saved policy to {pol_path}")
print(json.dumps(policy, indent=2))
