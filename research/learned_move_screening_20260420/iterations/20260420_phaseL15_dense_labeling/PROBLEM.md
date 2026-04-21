# Problem

Stage L1 dataset was clean but too small/narrow for serious model evaluation (`112` exact-labeled, `27` positives, single anchor context).

Stage L1.5 must generate a substantially denser exact-labeled `insert_inter` dataset while expanding context diversity in a controlled way.

Key requirement:

- data generation only (no model training/inference)
- exact-DP oracle labels remain authoritative
