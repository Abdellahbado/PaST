# C5 Decision: Multi-Budget Robust Validation

**Date**: 2026-05-13
**Gate**: **STRONG**

## Decision Questions

### 1. Does Phase C still pass under robust multi-budget validation?

**Yes.** LLM persistent_hard rate = 67% (4/6) vs Literature 0% (0/6) vs Random 50% (3/6).
The LLM advantage is NOT only a 30s artifact — all persistent-hard instances show meaningful growth from 30s to 120s, and LLM M2 instances continue growing to 300s.

### 2. Which family is strongest?

**hybrid_M2_tight_steprates_asghlock** (LLM Call2): 2/3 persistent, 2/3 MC, 2/3 NTM.
d30-300: [+42, +62, +76]. d120-300: [-3, +37, +50].
Two of three instances show continued growth from 120s to 300s (+37, +50), confirming persistent A-SGH lock-in behavior.

**human_mixed_job_sizes** (agent_manual_sweep, internal): 3/3 persistent, 3/3 MC, 3/3 NTM.
d30-300: [+41, +67, +28]. d120-300: [+29, +54, +17].
Even stronger, but is an internal control, not an external baseline.

### 3. Does the LLM beat random?

**Yes, narrowly.** LLM 67% vs Random 50% persistent-hard.
Caveat: random_005 (3/3 persistent) happened to land on tight epsilon + step rates through random generation. This shows the configuration space contains high-leverage ingredients, but the LLM finds them more consistently (both M1 and M2 produce persistent instances, across diverse seeding).

### 4. Does the LLM beat literature-derived baselines?

**Yes, clearly.** Literature 0% persistent-hard (0/6). Wang and Anghinolfi instances produce tiny fronts that don't grow meaningfully from 30s to 120s. The well-behaved benchmark generators are poor adversarial test instances.

### 5. Does the LLM beat or lose to the internal structured sweep?

**Loses.** Agent manual sweep: 83% PH (5/6) vs LLM 67% (4/6).
Agent mixed_job_sizes: 3/3 PH, 3/3 MC, 3/3 NTM.
This is honest and expected — the agent/coder designed families with deep knowledge of the metric. The agent remains an internal control, not an external baseline.

### 6. Is the weakness genuinely EHS-mechanism-specific or only a short-budget artifact?

**Mechanism-specific for M2 instances.** 
- 30s budget truncates front (fs=1 to 19 at 30s)
- 120s unlocks significant growth (fs=26 to 64)
- 300s continues growth in 2/3 instances (+37, +50 additional points)
- The growth pattern matches A-SGH lock-in: early cheap-machine assignments break as epsilon tightens, creating a burst of new front points

For M1 instances, the growth is large (+75 to +78) but mechanism not confirmed — first_khat_dominance rules are too strict or the mechanism is more subtle.

### 7. What exact paper-safe claim can we make?

> "Interactive LLM prompting can design realistic, feasible, mechanism-aware
> adversarial benchmark families for EHS that expose persistent stress
> mechanisms. Under multi-budget evaluation (30s/120s/300s), LLM-designed
> families achieve 67% persistent-hard rate versus 0% for literature-derived
> generators (Wang 2018, Anghinolfi 2021) and 50% for random generation.
> The strongest LLM family (hybrid_M2) combines tight epsilon, step machine
> rates, and single-peak TOU to induce A-SGH lock-in that persists beyond
> the trivial 30s short-budget regime, with continued front growth from
> 120s to 300s in 2/3 instances. This confirms that LLM-generated families
> are not merely exploiting a too-short time limit."

### Unacceptable Claims (avoid)

- "LLM found the hardest instances" — agent_manual_sweep is stronger
- "LLM proves EHS is weak" — this tests EHS stress, not solver quality
- "LLM dominates all baselines" — random_005 matches LLM per-family rate; agent beats LLM
