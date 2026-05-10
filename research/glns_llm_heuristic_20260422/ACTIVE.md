# Active GLNS / EHS Thread

## Current active iteration

`iterations/20260510_phaseC_adversarial_ehs_benchmark_design/` — **ACTIVE. C0-C1 complete (protocol + schema).**

## Status

**Phase C: LLM-Guided Adversarial Benchmark Design for EHS — C0-C1 complete. C2 next.**

This is NOT a solver-improvement branch. The LLM proposes instance-family
hypotheses to expose EHS weaknesses. B8 is closed (all 3 stages failed).

B6.17 oracle sequencing-gap diagnostic is completed and closed:
post-final exact-only oracle gap is 0.984% (< 1%). ES/R-ES/ESR already repairs
sequencing. No B6.18 oracle-distillation branch is warranted.

All 9 heuristic surfaces now confirmed closed (EPS, SGH, A-SGH, R-ES, ESR,
VND, stage skip/portfolio, restarts, sequencing). EHS reconstruction is
faithful (converges to ~97% published HV).

B8 is the active next branch because it is not another single-hook EHS tweak:
it tests post-EHS G-LNS-style destroy/repair, starting from strong EHS archive
schedules.

Current frozen method:
- **Default EHS:** `run_ehs()` with all defaults.
- **Optional fast_mode:** `run_ehs(fast_mode=True)` active only when `time_limit_seconds <= 60`.
- **Optional eps_ordering:** `run_ehs(eps_ordering="expensive_source_first")` for improved ES move ordering (+4.3% HV on validation).

## Non-negotiable rules

- Read `research/glns_llm_heuristic_20260422/PROTOCOL.md` before starting any new experiment.
- Do not optimize SGS-ES as the main method.
- Do not return to B5 as the main method.
- Published instances 61-90 are allowed for **final evaluation only** (B6.8 completed).

## All B6/B7 Iteration Summary

- B6.2: 10-seed reconstructed EHS exactly recovers published fronts for instances 1-5.
- B6.3: All 3 R-ES policy candidates rejected.
- B6.4: VLS protocol reset with 30 train + 30 validation instances.
- B6.4b: R-ES instrumentation + behavior-preserving speed repair. Equivalence passed.
- B6.5: LLM1 generated. Initial +816% front count at 20s.
- B6.5b: LLM1 REJECTED — time-limit artifact, loses at 60s/120s.
- B6.5c: Hybrid 75/25 (E) ACCEPTED and INTEGRATED as optional `fast_mode`.
- B6.6 SGH tie-breaking: REJECTED — no consistent HV improvement.
- **B6.6 EPS ordering: Candidate B ACCEPTED and INTEGRATED as optional `eps_ordering`.**
- **B6.7 EPS ordering evolution: COMPLETED. No improvement beyond Candidate B.**

## Next steps

1. ~~**Final evaluation on published 61-90** with frozen method and optional enhancements.~~ **COMPLETED (B6.8)**
2. **Report comparison to published SOTA** where available.
3. ~~**B6.17 sequencing-gap diagnostic** before any new LLM or G-LNS branch.~~ **COMPLETED / CLOSED.**
4. **B8 Stage 1 seed G-LNS pilot** against equal-time EHS continuation.

## B6.8 Final 61-90 Results

| Method | 60s HV vs Default | 120s HV vs Default | Wins (60s) | Wins (120s) |
|:---|---:|---:|---:|---:|
| eps | 1.0468 | 1.0270 | 25/30 | 27/30 |
| fast | 1.2017 | N/A | 30/30 | N/A |
| fast_eps | 1.3597 | N/A | 30/30 | N/A |

Both optional enhancements transfer positively to published 61-90.

## Parallel VND development branch

`iterations/20260430_phaseB74_vnd_operator_control_redesign/`

Purpose:

- Build a separate VND/EOA branch.
- B7.3 proved policy hooks too weak.
- B7.4a/b proved structural surfaces enable epsilon progression (6-18×).
- B7.4c proved composite variants cannot improve extreme-point quality at 300s.
  Fundamental depth-vs-breadth trade-off. Perturbation (EOA2/EOA3) has no effect.

Status: **STOPPED.** VND branch saturated. Do not run DeepSeek.
Next: hybrid EHS+VND or close the VND investigation.

This branch must not interfere with the frozen EHS final-evaluation run.

## Post-Final EHS Archive-Attack Branch

`iterations/20260430_phaseB610_ehs_heuristic_archive_attack/`

Purpose:

- Beat the published EHS heuristic archive on at least some published `61-90`
  instances, even if runs take hours or days.
- Remain heuristic-only: no F2-init, no MILP, no CPLEX/CP-SAT, no exact
  formulation.
- Use DeepSeek only for bounded heuristic diversity variants after a robust
  long-run archive comparator is working.

Important framing:

- This is not clean held-out validation because the published `61-90` archive
  has already been inspected.
- Results must be described as post-final heuristic archive attack.

## Next EHS Structural-Improvement Branch

`iterations/20260501_phaseB611_asgh_release_policy/` — CLOSED.

Purpose:

- Use DeepSeek with richer paper context and high reasoning settings to design
  an adaptive A-SGH release policy.
- Target the conservative assignment-history reuse inside EHS: A-SGH keeps
  nearly all feasible previous assignments on VLS instances.
- Seek structural heuristic improvement, not only short-budget front density.

Outcome:

- STOPPED / REJECTED.
- `adaptive_tightness` reduced Stage 1: mean HV ratio 0.993 vs eps, one hard regression at 0.958, runtime 1.11-1.18x.
- DeepSeek candidates mostly duplicated manual formulas.
- Do not continue this exact release-family branch.

### B6.12: Automated DeepSeek Heuristic-Discovery Loop with Proxy Evaluation

`iterations/20260502_phaseB612_deepseek_proxy_discovery/` — **CLOSED (stopped early by mechanism audit)**.

Purpose:

- Use DeepSeek as an automated heuristic researcher (not one-shot code generator).
- 6-phase loop: Diagnosis → Filter → Design/Code → Proxy evaluation → Promotion → Reflection.
- Cheap component-level proxy tests before any full EHS runs.
- Max 3 generations, stop early if gate passes.

Outcome:

- 30 mechanisms generated via DeepSeek, 27 survived filter, 18 sandbox passed.
- Proxy evaluation NOT completed.
- Mechanism audit (MECHANISM_AUDIT.md) revealed: **all single-hook injection surfaces are closed.**
- DeepSeek diagnosis skewed heavily toward EPS_SOURCE (>50%) — confirming the LLM does not 
  discover novel mechanisms on saturated surfaces.
- EHS variant complementarity audit (B6.8) shows fast_eps + default + eps are complementary
  in Pareto front space — pointing to the portfolio/time-allocation surface.
- VND is saturated and worse than EHS → dropped.

Script: `scripts/phaseB612_deepseek_proxy_discovery.py`

### B6.13: EHS Portfolio/Time-Allocation Controller

`iterations/20260502_phaseB613_portfolio_controller/` — **CLOSED.**

Purpose:

- Use LLM to design portfolio/time-allocation policies combining validated EHS variants
  (default, eps, fast, fast_eps) within a single budget.
- 5-part protocol: offline oracle → manual baselines → DeepSeek design → filter → reflection.

Outcome:

- **Part 1 (Oracle):** Portfolio merging adds <0.5% HV at any budget.
  Best static: fast_eps at 60s, eps at 120s. Pseudo-oracle gain +0.13%.
- **Part 2 (Manual baselines):** 5 designed. split_60/60 provably dominated by eps_120s.
- **Part 3 (DeepSeek):** 12 policies generated ($0.0026). Categories: split, adaptive, ensemble, oracle.
- **Part 4 (Filter):** All 12 policies fail. Theoretical bound: 120s eps alone (35.5% published HV)
  dominates any split because any time given to fast_eps is time taken from eps.
  The gap is arithmetic, not heuristic.
- **Part 5 (Reflection):** Not executed. Surface closed — no viable policies to refine.
- **Run28 equal-budget audit:** `eps_1x300` beats all restart/mixed policies.
  Best non-baseline `fast_eps60_eps240` has HVr 0.769; restart policies are
  much worse because each fresh run pays the expensive first-khat cost.

**Portfolio/time-allocation surface is closed at current EHS quality level.**

Script: `scripts/phaseB613_part3_deepseek_design.py`
Artifacts: `temp/phaseB6_paper_faithful_reconstruction/20260502_run26_b613_portfolio_controller/`

### All Surfaces Summary

| Surface | Result | Status |
|---------|:---:|:---:|
| EPS ordering (B6.6-7) | Candidate B +4.6% HV | ACCEPTED, saturated |
| fast_mode (B6.5c) | 75/25 hybrid +33.6% HV | ACCEPTED |
| SGH/A-SGH release (B6.11) | HVr 0.993 | REJECTED |
| SGH tie-breaking (B6.6) | HVr 0.9999 | REJECTED |
| R-ES/ESR (B6.4b) | 1.4%/0% khat improve | LOW-VALUE |
| VND standalone (B7.4) | All variants identical | SATURATED |
| Portfolio/controller (B6.13) | <0.5% slack | CLOSED |

### Next: Multi-Seed Restart Ensemble

Only remaining open surface: multi-seed EHS restarts.
Published EHS achieves 2.8× our best HV via 10-aggregated runs + unlimited time.
A multi-seed ensemble (3-5 seeds × eps_ordering at 120-300s) could approach published quality
without any new heuristic design. DeepSeek has confirmed that single-hook surfaces are closed —
restarts are the remaining mechanism.

### B6.17b: Long-Run EHS Convergence / Published-Gap Diagnostic

`iterations/20260503_phaseB617_longrun_ehs_convergence/` — **COMPLETED.**

Purpose:

- Determine whether the 2.8× gap to published EHS is compute/convergence
  or reconstruction fidelity.
- Run eps_ordering on instances 61, 75, 90 at budgets 120s-1200s.
- Compare against published EHS archive.

Outcome:

- **Gap is compute/convergence, not fidelity.**
- At 300-1200s: our EHS reaches 97.1-97.7% of published HV.
- Deepest Cmax values match within 1 unit (61 vs 60, 76 vs 76, 83 vs 82).
- Residual 2-3% HV gap consistent with 10-seed vs 1-seed aggregation.
- EHS converges to near-completion by 300-600s on small/medium instances.
- The "35% of published" was a 120s time-budget artifact.

Artifacts: `temp/phaseB6_paper_faithful_reconstruction/20260503_run28_b617_longrun_diagnostic/`

### B6.17: Oracle Sequencing-Gap Diagnostic

`iterations/20260503_phaseB617_oracle_sequencing_gap/` — **COMPLETED.**

Purpose:

- Measure whether post-final EHS schedules have a per-machine sequencing gap
  vs `_solve_machine_optimal()`.

Outcome:

- **Post-final exact-only gap: 0.984% (< 1%). Sequencing surface CLOSED.**
- Post-construction gap: 1.013%, reduced to <1% by ES/R-ES/ESR.
- 92.8% of machines classified as exact; 88.1% energy coverage.
- Do NOT proceed to B6.18 LLM oracle-distillation.

Artifacts: `temp/phaseB6_paper_faithful_reconstruction/20260503_run29_b617_oracle_sequencing_gap/`

### All Surfaces Summary (Final)

| Surface | Result | Status |
|---------|:---:|:---:|
| EPS ordering (B6.6-7) | Candidate B +4.6% HV | ACCEPTED, saturated |
| fast_mode (B6.5c) | 75/25 hybrid +33.6% HV at 60s | ACCEPTED |
| SGH/A-SGH release (B6.11) | HVr 0.993 | REJECTED |
| SGH tie-breaking (B6.6) | HVr 0.9999 | REJECTED |
| R-ES/ESR (B6.4b) | 1.4%/0% khat improve | LOW-VALUE |
| VND standalone (B7.4) | All variants identical | SATURATED |
| Portfolio/controller (B6.13) | <0.5% slack | CLOSED |
| Multi-seed restarts (B6.16) | +0.69% gain | CLOSED |
| Long-run convergence (B6.17b) | Gap is time-budget, not fidelity | **RESOLVED** |
| Per-machine sequencing (B6.17) | 0.984% post-final oracle gap | **CLOSED** |

### Paper Status: READY

All 9 surfaces are closed or resolved. Two accepted improvements.
The EHS reconstruction is faithful. The LLM systematically exhausted
all injection surfaces. The paper is ready with current evidence.

### B8: EHS-Warm-Started G-LNS Destroy/Repair

`iterations/20260503_phaseB8_ehs_warmstarted_glns_destroy_repair/` — **ACTIVE. STAGE 0 PASSED.**

Purpose:

- Test a new LLM-relevant algorithmic surface after EHS hooks, VND,
  portfolio, and restarts closed.
- Start from frozen EHS archive schedules.
- Apply G-LNS-style coupled destroy/repair operators with synergy evaluation.
- Use DeepSeek only after seed destroy/repair proves feasibility and
  non-collapse.

Key distinction:

- This is not old assignment-only GLNS. Old B3.2 failed held-out validation.
- B8 uses EHS warm starts, coupled destroy/repair pair evaluation, and
  equal-time EHS comparators.

Stage 0 result:

- Gate passed on 6 synthetic VLS instances.
- 100% feasibility.
- 99.78% repairs changed assignment by at least 2%.
- ND/SCI points on 6/6 instances.
- Best pair: `d_high_rate × r_energy_aware`.

Immediate next step:

- Stage 1 seed G-LNS pilot against equal-time EHS continuation.
- Do not call DeepSeek until Stage 1 proves the seed method beats EHS under a
  fair comparator.
