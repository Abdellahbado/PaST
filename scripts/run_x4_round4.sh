#!/bin/bash
set -e
source .env.deepseek.sh

BASE="research/learned_move_screening_20260420/iterations/20260508_phaseX_interactive_llm_policy_repair"
PROMPT_FILE="$BASE/prompts/x4_round_4.md"
RESP_FILE="$BASE/responses/x4_round_4_raw.md"
META_FILE="$BASE/responses/x4_round_4_meta.json"
POL_FILE="$BASE/policies/llm_interactive/x4_round_4.json"

echo "Reading prompt from $PROMPT_FILE"
PROMPT=$(cat "$PROMPT_FILE")
PREV_PROMPT=$(cat "$BASE/prompts/x4_round_2.md" | head -c 4000)
PREV_RESP=$(cat "$BASE/responses/x4_round_2_raw.md" | head -c 4000)

# Build messages JSON
MESSAGES=$(python3 -c "
import json
msgs = [
    {'role': 'user', 'content': '''$PREV_PROMPT'''},
    {'role': 'assistant', 'content': '''$PREV_RESP'''},
    {'role': 'user', 'content': '''$PROMPT'''},
]
print(json.dumps(msgs))
")

echo "Calling DeepSeek API..."
START=$(date +%s)
RESP=$(curl -s -w "\n%{http_code}" --connect-timeout 30 --max-time 600 \
    https://api.deepseek.com/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
    -d "{\"model\":\"deepseek-v4-pro\",\"messages\":$MESSAGES,\"temperature\":0.5,\"max_tokens\":16000}")
END=$(date +%s)
ELAPSED=$((END - START))

HTTP_CODE=$(echo "$RESP" | tail -1)
BODY=$(echo "$RESP" | sed '$d')

echo "HTTP $HTTP_CODE, ${ELAPSED}s"

if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: HTTP $HTTP_CODE"
    echo "$BODY"
    exit 1
fi

CONTENT=$(echo "$BODY" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
USAGE=$(echo "$BODY" | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin).get('usage',{})))")

echo "$CONTENT" > "$RESP_FILE"
echo "Saved response to $RESP_FILE"

python3 -c "
import json
meta = {'model': 'deepseek-v4-pro', 'prompt_tokens': 0, 'completion_tokens': 0, 'elapsed_sec': $ELAPSED}
with open('$META_FILE', 'w') as f: json.dump(meta, f, indent=2)
"
echo "Saved metadata to $META_FILE"

# Extract and validate JSON
python3 << 'PYEOF'
import json, re, sys

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
PYEOF
