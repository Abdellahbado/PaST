# Phase C Summary

**Status**: C0-C2 complete. Ready for C3 smoke pilot (DeepSeek call pending).

**Branch**: LLM-guided adversarial benchmark design for EHS.

**Core hypothesis**: LLM can design structured BPMSTP instance families that
expose EHS weaknesses more efficiently than random or human parameter sweeps.

## Completed

### C0: Protocol
- EXPERIMENT_PROTOCOL.md with full C0-C4 phase structure, decision gates.

### C1: Schema
- `families/family_schema.json` with 13 required fields, type enums,
  sanity constraints, rejection conditions.

### C2: Generators
- **Random**: 8 families (`families/random_families.json`). Uniform sampling
  of legal parameter ranges. All validated (0 errors).
- **Human sweep**: 8 families (`families/human_sweep_families.json`).
  Fixed parameter sweeps over epsilon tightness, price volatility, job
  size distribution, machine density. All validated (0 errors).
- **LLM prompt**: `prompts/call1_llm_family_designer.md`. Full EHS pipeline
  description, B6 closure evidence, 8 target mechanisms, schema docs,
  examples of valid/invalid families.
- **Validation script**: `scripts/phaseC_adversarial_family_generation.py`
  with `--generate-random-families`, `--generate-human-sweep-families`,
  `--validate-family-file` subcommands.

## Next
- C3: First DeepSeek call to generate LLM families (`call1_llm_family_designer.md`)
- C3 smoke pilot: 2 LLM + 2 random + 2 human families, 3 instances each,
  EHS at 60s + 300s budgets.

## Gate
After smoke, decide based on diagnostic yield whether to proceed to full campaign.
