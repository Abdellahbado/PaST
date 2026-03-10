#!/bin/bash
set -e

export PATH="/opt/homebrew/opt/dotnet@8/bin:$PATH"
export DOTNET_ROOT="/opt/homebrew/opt/dotnet@8"

REPO="/Users/mac/Documents/Study/PFE/PaST/data/green-scheduling-bab/Iirc.EnergyStatesAndCostsScheduling"
PROJ="$REPO/Iirc.EnergyStatesAndCostsScheduling.DatasetGenerators/Iirc.EnergyStatesAndCostsScheduling.DatasetGenerators.csproj"
DATA_ROOT="$REPO/data"

PRESCRIPTIONS=(
    "aghelinejad2017a_1.json"
    "benedikt2025a_large_nosby.json"
    "benedikt2025a_large_twosby.json"
    "benedikt2025a_medium_nosby.json"
    "benedikt2025a_medium_twosby.json"
    "benedikt2025a_prelim.json"
    "benedikt2025_groups.json"
    "benedikt2025b_groups.json"
    "benedikt2025b_test.json"
    "benedikt2025b_gcd.json"
    "benedikt2025b_drops.json"
)

cd "$REPO"
for p in "${PRESCRIPTIONS[@]}"; do
    echo "=== Generating: $p ==="
    dotnet run -c Release --project "$PROJ" -- "$DATA_ROOT" "$p" 2>&1 | grep -E "generated|error|Exception" || true
done

echo ""
echo "=== Generated datasets ==="
for d in "$DATA_ROOT/datasets"/*/; do
    count=$(ls "$d"/*.json 2>/dev/null | wc -l)
    echo "  $(basename "$d"): $count instances"
done
