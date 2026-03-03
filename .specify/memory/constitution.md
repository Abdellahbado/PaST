<!--
SYNC IMPACT REPORT
==================
Version change : 1.0.0 => 1.1.0 (MINOR)
Reason         : Three amendments following first review session.

Modified principles :
  - I. Problem Canonicalization: objective corrected from single (energy only)
    to bi-objective (makespan Cmax + total energy). Tone made higher-level.
  - III. Benchmark Integrity: softened from NON-NEGOTIABLE single suite to
    two-tier policy (primary + secondary). Detailed table removed (belongs in specs).
  - VI. Scope Lock: removed erroneous prohibition on bi-objective optimization
    (now the actual problem). Method generalization to adjacent problems allowed.
  - Overall: reduced specificity throughout; numerical thresholds moved out of
    the constitution and into individual experiment plans/specs.

Added sections : none
Removed sections : none

Templates reviewed & status :
  OK .specify/templates/plan-template.md -- no structural change needed.
  OK .specify/templates/spec-template.md -- no structural change needed.
  OK .specify/templates/tasks-template.md -- no structural change needed.

Deferred TODOs :
  - TODO(LLM_MODEL): Specific LLM model/API for Track B not yet finalized.
-->

# PaST -- Parallel Machine Scheduling with Energy Considerations -- Constitution

## Core Principles

### I. Problem Identity

The project addresses **Parallel Machine Scheduling** under Time-of-Use (TOU)
electricity pricing with a bi-objective goal.

**Setting:**
- A set of parallel machines, each with a machine-specific energy consumption rate.
- A set of non-preemptive jobs, each with a given processing time.
  Jobs with identical processing times are interchangeable.
- A time-varying electricity price profile covering the scheduling horizon T,
  which also serves as a common deadline for all jobs.

**Objectives (bi-objective):**
1. Minimize **makespan** (Cmax): the completion time of the last job.
2. Minimize **total energy cost**: the sum of energy consumed across all machines
   weighted by the time-varying price.

These two objectives are in tension; work in this project explores their trade-off
or aggregated formulations, depending on the experiment context.

**Core constraints:** non-preemptive execution, one job per machine at a time,
all jobs complete by deadline T.

**Scope of applicability:** While the project is centered on this problem, learning
components and algorithmic ideas developed here may be studied on related scheduling
variants. Such generalizations are welcome as long as the primary problem remains
the reference point.

---

### II. Two-Track Solution Strategy

The project pursues two complementary approaches. New wholesale directions require
discussion and a constitution amendment; extensions or experiments within these
tracks do not.

#### Track A -- Learning-Accelerated Dynamic Programming

A novel exact DP algorithm solves the single-machine subproblem optimally. The role
of learning is to **accelerate** this DP by approximating the value function and
guiding beam-pruned search, reducing computation while preserving solution quality.

Key commitments:
- The exact DP remains the correctness and optimality reference.
- The learned approximator is trained from solved optimal examples (supervised).
- Feature engineering and model selection are open areas of investigation within this track.

#### Track B -- LLM-Guided Assignment

An LLM proposes the **assignment** of jobs to machines (first stage). The per-machine
sequencing and timing are then handled by Track A. The LLM is used as a heuristic
reasoner, not as an end-to-end optimizer.

Key commitments:
- The LLM handles assignment only; it does not determine sequencing or timing.
- The full pipeline (LLM assignment + Track A scheduling) is always evaluated together.
- TODO(LLM_MODEL): The specific model/API to be confirmed before Track B experiments begin.

**Combined pipeline:**

    Instance -> [Track B: LLM proposes job-to-machine assignment]
             -> [Track A: per-machine DP finds optimal schedule]
             -> Evaluate Cmax and total energy cost

---

### III. Benchmark Policy

Evaluation is organized in two tiers.

**Primary benchmark:** The established 90-instance set derived from Wang et al. (2018)
and Anghinolfi et al. (2021), covering small, medium-large, and very-large scales.
Results claimed as general contributions SHOULD be validated on this set.

**Secondary benchmarks:** Custom instance sets (e.g., with specific price profile
structures, learning-friendly properties, or controlled difficulty) are permitted and
encouraged, particularly for studying learning components in isolation. Results on
secondary benchmarks are clearly scoped to the conditions they test.

**Integrity rules (both tiers):**
- The instance generation procedure and random seed MUST be recorded with every result.
- Selectively reporting only favorable instances from a set is FORBIDDEN.
- Scale-stratified reporting (when multiple scales are tested) is preferred alongside
  aggregate summaries.

---

### IV. Evaluation Standards

**Primary metric:** Optimality gap relative to the best known solution (or exact
optimum when available), reported for both objectives where applicable.

**Expectations:**
- Results MUST be reported with sufficient statistical context (mean, variance,
  number of instances) to be interpretable.
- Timing results MUST specify the hardware used and MUST compare structurally
  equivalent implementations.
- Claims of improvement over a baseline MUST be supported by evidence; specific
  significance thresholds and test choices are defined in individual experiment plans.

**Rationale:** Quantitative rigor is non-negotiable; exact thresholds are
experiment-specific and belong in the plan or spec, not here.

---

### V. Reproducibility

Every experiment MUST be reproducible from its logged artifacts.

- Fixed random seeds MUST be set and recorded.
- All hyperparameters, software versions, and the git state MUST be logged per run.
- Results MUST be persisted to structured files. Terminal-only results are not official.
- Datasets MUST be versioned; changing a dataset requires a new identifier, not
  an overwrite.

---

### VI. Scope Discipline

The project maintains focus on the primary problem (Principle I). The following
require explicit discussion and a constitution amendment before proceeding:

- Abandoning or replacing Track A or Track B entirely.
- Introducing the LLM as a full end-to-end scheduler (sequencing + timing).
- Pursuing problem variants so far from the primary problem that the benchmark
  and solver infrastructure cannot be reused.

Exploratory work in adjacent scheduling problems, ablation studies, or alternative
learning paradigms is acceptable and encouraged -- but must not silently redirect
the main development line. Sandbox directories exist for this purpose.

---

## Technical Foundations

The project is Python-based. Core dependencies include standard scientific computing
libraries (NumPy, SciPy), a machine learning toolkit (scikit-learn and/or PyTorch),
and the custom exact DP solver maintained in this repository. Docker is available for
reproducible execution environments.

Specific library choices, model architectures, and infrastructure decisions are made
at the plan/spec level and do not require a constitution amendment unless they
represent a fundamental change to the solution strategy.

---

## Governance

- This constitution is the highest-level guiding document. In conflicts with READMEs,
  experiment plans, or other docs, the constitution takes precedence.
- Before starting a new experiment or significant coding task, verify alignment with
  Principles I (problem identity), II (track strategy), and VI (scope discipline).
- **Amendment procedure:**
  1. Update this file with a version bump (see rules below).
  2. Update the Sync Impact Report (HTML comment at top).
  3. Note any downstream template or doc changes required.
  4. Record the rationale in the commit message.
- **Version bump rules:**
  - MAJOR (X.0.0): Fundamental redefinition of the problem (Principle I) or
    removal of an active solution track (Principle II).
  - MINOR (x.Y.0): Correcting or materially expanding a principle, adding a new
    principle, or changing benchmark policy.
  - PATCH (x.y.Z): Wording, clarifications, or editorial fixes.

**Version**: 1.1.0 | **Ratified**: 2026-03-03 | **Last Amended**: 2026-03-03
