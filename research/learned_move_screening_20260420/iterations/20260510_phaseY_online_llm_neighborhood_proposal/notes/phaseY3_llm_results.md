# Phase Y3 — First DeepSeek Online Neighborhood Proposal Results

## Summary

Y3 tests whether DeepSeek can read a concrete solver state trace and propose a better bounded neighborhood than random under the same DP budget (K=20). One DeepSeek call per dev cell. **Result: FAIL — LLM does not outperform random or manual on any cell.**

## Per-Cell Results

### Cell_A (61/347, ε=347 medium, TEC=6946)

| Variant | TEC | Gen | Ev | Imp | Δ vs baseline |
|---------|-----|-----|----|----|--------------|
| Baseline (trace) | 6946 | — | — | — | 0 |
| **LLM (y3)** | **6946** | **645** | **20** | **0** | **0** |
| Manual | 6946 | 325 | 20 | 0 | 0 |
| Random best | 6893 | 313 | 20 | 11 | -53 |
| Random median | 6924 | — | — | — | -22 |

**LLM result**: No improvements found. Tied with manual proposal, worse than both random median and random best. Cell_A appears near-optimal (prior best 6852 from Phase S is only -94 away). The LLM's strategy (attack high-gap rate-2 sources with small/medium to rate-3/4 targets) generated many candidates (645) but none improved TEC.

**LLM proposal**: `y3_attack_highgap_rate2_sources_small_medium`
- Sources: M24, M19, M22, M6, M17 (top-5 gap machines)
- Targets: M12, M0, M20, M15, M21 (rate-3/4 near-empty)
- Sizes: small, medium
- Ranking: cost_gap, Diversity: per_source

### Cell_B (62/290, ε=290 medium, TEC=9435)

| Variant | TEC | Gen | Ev | Imp | Δ |
|---------|-----|-----|----|----|---|
| Baseline (trace) | 9435 | — | — | — | 0 |
| **LLM (y3)** | **CRASH** | **—** | **—** | **—** | **—** |
| Manual | CRASH | — | — | — | — |
| Random best | 9366 | 194 | 19 | 6 | -69 |
| Random median | ~9395 | — | — | — | ~-40 |

**LLM result**: SIGBUS (exit 138) on macOS release build, heap-buffer-overflow on ASAN debug build. Same failure mode as Y2.1 manual proposal — not specific to the LLM proposal. Instance 62 appears to trigger a memory safety bug in the Phase Y execution path on Apple Silicon. Validated on 4/5 random seeds (100, 200, 300, 400 pass; 500 crashes).

**LLM proposal**: `y3_attack_underexplored_highgap_small_medium` (saved but not executed)

### Cell_C (65/195, ε=195 tight, TEC=27031)

| Variant | TEC | Gen | Ev | Imp | Δ vs baseline |
|---------|-----|-----|----|----|--------------|
| Baseline (trace) | 27031 | — | — | — | 0 |
| **LLM (y3)** | **27013** | **590** | **20** | **1** | **-18** |
| Manual | 26715 | 550 | 20 | 12 | -316 |
| Random best | 26789 | 205 | 15 | 10 | -242 |
| Random median | 26814 | — | — | — | -217 |

**LLM result**: 1 improvement found (Δ=-18). LLM proposal beats baseline but significantly underperforms both manual (12 improvements, Δ=-316) and random best (Δ=-242). The LLM's strategy (continue the proven large-job pattern from rate-3→high-gap targets) was directionally correct but failed to exploit the range of opportunities the manual and random proposals captured. Only 1 source-target pair produced improvements (vs manual's broader success).

**LLM proposal**: `y3_large_jobs_rate3_to_highgap_targets`
- Sources: M7, M19, M0, M2, M13
- Targets: M1, M6, M12, M8, M5
- Sizes: small, medium, large
- Ranking: cost_gap, Diversity: per_source

## Gate Assessment

| Gate | Condition | Result |
|------|-----------|--------|
| Strong | LLM beats random best on ≥2/3 cells | **FAIL** (0/2 evaluable, 1 crash) |
| Moderate | LLM beats random median + manual on ≥2/3 | **FAIL** (0/2) |
| Weak | LLM beats baseline but not random median | **FAIL** (Cell_A tied, Cell_C barely beats) |
| Fail | LLM loses to random/manual on most cells | **CONFIRMED** |

## Analysis

### Why did the LLM fail?

1. **Proposal format too constrained**: With only 5 sources × 5 targets × job sizes, the LLM has limited expressiveness. The manual proposal designer could tune these choices based on trial-and-error over many runs; the LLM had one shot per cell.

2. **State trace ≠ full information**: The LLM sees per-machine aggregates but not the actual job-level schedule (which jobs are where). Without job-level visibility, the LLM can only make broad suggestions about which machines to attack.

3. **Single-call protocol**: The LLM had one call per cell. Interactive feedback (like Phase X's interactive protocol) might allow the LLM to refine after seeing results.

4. **Cell_A is near-optimal**: Prior best TEC=6852 (Δ=-94). Current=6946. Room for improvement is only ~1.4%. Random best found Δ=-53, leaving only ~1.1% remaining to prior best. This is genuinely hard.

5. **Cell_C: LLM was too conservative**: The LLM's proposal matched the proven pattern (large jobs from rate-3→high-gap targets) but didn't explore enough. The manual proposal succeeded by targeting a broader set of machines. The LLM also used all 3 job sizes, diluting the search — focusing on just large might have been better.

6. **Cell_B: Infrastructure bug**: Cannot evaluate LLM quality due to SIGBUS crash. The LLM proposal exists but the execution path crashes regardless of proposal content.

## Next Steps

- **Y3 is a FAIL**. The LLM does not beat random/median under the single-call, bounded-neighborhood protocol.
- Cell_B's SIGBUS bug should be fixed before any further Phase Y evaluation on instance 62.
- If continuing Phase Y: consider interactive multi-call protocol (LLM sees execution results and refines), or relaxing the proposal constraints (more sources/targets).
- Alternative: stop Phase Y and conclude that LLM neighborhood proposals from state traces are not competitive with random search under these constraints.

## Artifacts

- `prompts/y3_Cell_A_prompt.md`, `y3_Cell_B_prompt.md`, `y3_Cell_C_prompt.md`
- `responses/y3_Cell_A_raw.md`, `y3_Cell_B_raw.md`, `y3_Cell_C_raw.md`
- `proposals/llm/y3_Cell_A.json`, `y3_Cell_B.json`, `y3_Cell_C.json`
- `eval/y3_llm_raw.csv`, `eval/y3_llm_summary.csv`
- `notes/phaseY3_llm_results.md` (this file)

## Binary Configuration

- Cell_A, Cell_C, and random baselines: **Release build** (`-O3 -DNDEBUG`)
- Cell_B: crashes on both Release (SIGBUS) and Debug+ASAN (heap-buffer-overflow)
- Cell_C LLM confirmed on both Debug (TEC=27013) and Release (TEC=27013) — consistent
- Random seeds chosen to avoid known SIGBUS seeds: Cell_A 1-400 all pass, Cell_B 100-400 pass, Cell_C seeds 1,7,13,17 pass
