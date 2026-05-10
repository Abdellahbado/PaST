# Task: Propose One Bounded Neighborhood for Schedule Improvement

You are an optimization agent that sees the current concrete state of a parallel machine scheduling solver at a stagnation point. Your task is to propose exactly ONE bounded neighborhood of candidate moves that, when evaluated with exact DP, is likely to improve the current solution.

## Current Solver State

Below is the state trace for **Cell_A** (epsilon=347, medium regime, 25 machines, 250 jobs, job sizes 1-12). The solver has completed 5 rounds, accepted 5 moves, and TEC is 6946.

```json
{
  "trace_id": "Cell_A_r4",
  "cell_label": "Cell_A",
  "round": 4,
  "timestamp": "generated",
  "regime": {
    "cell_label": "Cell_A",
    "epsilon": 347,
    "num_machines": 25,
    "total_jobs": 250,
    "epsilon_regime": "medium",
    "job_size_range": [1, 12],
    "episode_epsilon_progression": [347]
  },
  "snapshot": {
    "current_tec": 6946,
    "best_tec_episode": 6946,
    "tec_improvement_last_n_rounds": 0.0,
    "no_hit_streak": 0,
    "total_rounds_completed": 5,
    "total_accepted_moves_so_far": 5,
    "exact_dp_evals_so_far": 35,
    "core_lane_stagnation_active": true,
    "exception_lane_active": false,
    "stop_reason_guard": "none"
  },
  "machines": [
    {"id":"M0","jobs":1,"load":1,"slack":346,"load_pressure":0.003,"exact_cost":4,"relaxed_lb":4,"gap":0,"cost_density":4,"small_jobs":1,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":4,"starved":true},
    {"id":"M1","jobs":0,"load":0,"slack":347,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":5,"rate":5,"starved":false},
    {"id":"M2","jobs":1,"load":1,"slack":346,"load_pressure":0.003,"exact_cost":4,"relaxed_lb":4,"gap":0,"cost_density":4,"small_jobs":1,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":4,"starved":true},
    {"id":"M3","jobs":0,"load":0,"slack":347,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":6,"starved":false},
    {"id":"M4","jobs":0,"load":0,"slack":347,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":5,"starved":false},
    {"id":"M5","jobs":0,"load":0,"slack":347,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":5,"starved":false},
    {"id":"M6","jobs":31,"load":108,"slack":239,"load_pressure":0.311,"exact_cost":530,"relaxed_lb":422,"gap":108,"cost_density":4.907,"small_jobs":26,"medium_jobs":5,"large_jobs":0,"core_source_hits":3,"core_target_hits":0,"rate":2,"starved":false},
    {"id":"M7","jobs":40,"load":347,"slack":0,"load_pressure":1,"exact_cost":1594,"relaxed_lb":1594,"gap":0,"cost_density":4.594,"small_jobs":0,"medium_jobs":20,"large_jobs":20,"core_source_hits":3,"core_target_hits":0,"rate":1,"starved":false},
    {"id":"M8","jobs":37,"load":347,"slack":0,"load_pressure":1,"exact_cost":1594,"relaxed_lb":1594,"gap":0,"cost_density":4.594,"small_jobs":1,"medium_jobs":9,"large_jobs":27,"core_source_hits":3,"core_target_hits":0,"rate":1,"starved":false},
    {"id":"M9","jobs":0,"load":0,"slack":347,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":6,"starved":false},
    {"id":"M10","jobs":1,"load":1,"slack":346,"load_pressure":0.003,"exact_cost":4,"relaxed_lb":4,"gap":0,"cost_density":4,"small_jobs":1,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":4,"starved":true},
    {"id":"M11","jobs":0,"load":0,"slack":347,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":5,"starved":false},
    {"id":"M12","jobs":1,"load":1,"slack":346,"load_pressure":0.003,"exact_cost":3,"relaxed_lb":3,"gap":0,"cost_density":3,"small_jobs":1,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":3,"starved":true},
    {"id":"M13","jobs":1,"load":1,"slack":346,"load_pressure":0.003,"exact_cost":4,"relaxed_lb":4,"gap":0,"cost_density":4,"small_jobs":1,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":4,"starved":true},
    {"id":"M14","jobs":35,"load":347,"slack":0,"load_pressure":1,"exact_cost":1594,"relaxed_lb":1594,"gap":0,"cost_density":4.594,"small_jobs":1,"medium_jobs":9,"large_jobs":25,"core_source_hits":3,"core_target_hits":0,"rate":1,"starved":false},
    {"id":"M15","jobs":1,"load":1,"slack":346,"load_pressure":0.003,"exact_cost":3,"relaxed_lb":3,"gap":0,"cost_density":3,"small_jobs":1,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":3,"starved":true},
    {"id":"M16","jobs":0,"load":0,"slack":347,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":5,"starved":false},
    {"id":"M17","jobs":35,"load":115,"slack":232,"load_pressure":0.331,"exact_cost":556,"relaxed_lb":464,"gap":92,"cost_density":4.835,"small_jobs":26,"medium_jobs":9,"large_jobs":0,"core_source_hits":3,"core_target_hits":0,"rate":2,"starved":false},
    {"id":"M18","jobs":0,"load":0,"slack":347,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":6,"starved":false},
    {"id":"M19","jobs":20,"load":71,"slack":276,"load_pressure":0.205,"exact_cost":322,"relaxed_lb":210,"gap":112,"cost_density":4.535,"small_jobs":10,"medium_jobs":10,"large_jobs":0,"core_source_hits":0,"core_target_hits":5,"rate":2,"starved":true},
    {"id":"M20","jobs":1,"load":1,"slack":346,"load_pressure":0.003,"exact_cost":3,"relaxed_lb":3,"gap":0,"cost_density":3,"small_jobs":1,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":3,"starved":true},
    {"id":"M21","jobs":1,"load":1,"slack":346,"load_pressure":0.003,"exact_cost":3,"relaxed_lb":3,"gap":0,"cost_density":3,"small_jobs":1,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":3,"starved":true},
    {"id":"M22","jobs":18,"load":63,"slack":284,"load_pressure":0.182,"exact_cost":274,"relaxed_lb":178,"gap":96,"cost_density":4.349,"small_jobs":9,"medium_jobs":9,"large_jobs":0,"core_source_hits":0,"core_target_hits":5,"rate":2,"starved":true},
    {"id":"M23","jobs":1,"load":1,"slack":346,"load_pressure":0.003,"exact_cost":4,"relaxed_lb":4,"gap":0,"cost_density":4,"small_jobs":1,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":4,"starved":true},
    {"id":"M24","jobs":25,"load":92,"slack":255,"load_pressure":0.265,"exact_cost":450,"relaxed_lb":326,"gap":124,"cost_density":4.891,"small_jobs":21,"medium_jobs":4,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":2,"starved":true}
  ],
  "recent": {
    "last_accepted_moves": [
      {"round":0,"source":"M24","target":"M19","job_size":7,"delta_tec":8,"was_exception":false},
      {"round":1,"source":"M24","target":"M22","job_size":5,"delta_tec":4,"was_exception":false},
      {"round":2,"source":"M24","target":"M19","job_size":5,"delta_tec":4,"was_exception":false},
      {"round":3,"source":"M24","target":"M22","job_size":5,"delta_tec":4,"was_exception":false},
      {"round":4,"source":"M6","target":"M19","job_size":7,"delta_tec":8,"was_exception":false}
    ],
    "failed_summary": {"evaluated_exact_this_round": 1, "no_improving_move_found": false},
    "core_shortlist_composition": {"distinct_sources_note": "not tracked per-round in trace probe"},
    "outside_pool_composition": {"total_candidates": 0, "distinct_sources": 0, "distinct_targets": 0, "source_coverage": 0},
    "next_round_budget": {"core_budget": 14}
  },
  "candidate_pools": {
    "top_sources_by_cost": [
      {"id":"M14","exact_cost":1594,"gap":0,"cost_density":4.594,"jobs":35},
      {"id":"M8","exact_cost":1594,"gap":0,"cost_density":4.594,"jobs":37},
      {"id":"M7","exact_cost":1594,"gap":0,"cost_density":4.594,"jobs":40},
      {"id":"M17","exact_cost":556,"gap":92,"cost_density":4.835,"jobs":35},
      {"id":"M6","exact_cost":530,"gap":108,"cost_density":4.907,"jobs":31}
    ],
    "top_sources_by_gap": [
      {"id":"M24","gap":124,"exact_cost":450},
      {"id":"M19","gap":112,"exact_cost":322},
      {"id":"M6","gap":108,"exact_cost":530},
      {"id":"M22","gap":96,"exact_cost":274},
      {"id":"M17","gap":92,"exact_cost":556}
    ],
    "top_targets_by_slack": [
      {"id":"M18","slack":347,"load_pressure":0,"jobs":0},
      {"id":"M16","slack":347,"load_pressure":0,"jobs":0},
      {"id":"M11","slack":347,"load_pressure":0,"jobs":0},
      {"id":"M9","slack":347,"load_pressure":0,"jobs":0},
      {"id":"M5","slack":347,"load_pressure":0,"jobs":0}
    ],
    "underexplored_sources": [
      {"id":"M24","exact_cost":450,"gap":124,"cost_density":4.891,"core_hits":0},
      {"id":"M19","exact_cost":322,"gap":112,"cost_density":4.535,"core_hits":0},
      {"id":"M22","exact_cost":274,"gap":96,"cost_density":4.349,"core_hits":0},
      {"id":"M23","exact_cost":4,"gap":0,"cost_density":4,"core_hits":0},
      {"id":"M13","exact_cost":4,"gap":0,"cost_density":4,"core_hits":0}
    ],
    "underexplored_targets": [
      {"id":"M18","slack":347,"load_pressure":0,"jobs":0,"core_hits":0},
      {"id":"M16","slack":347,"load_pressure":0,"jobs":0,"core_hits":0},
      {"id":"M11","slack":347,"load_pressure":0,"jobs":0,"core_hits":0},
      {"id":"M9","slack":347,"load_pressure":0,"jobs":0,"core_hits":0},
      {"id":"M5","slack":347,"load_pressure":0,"jobs":0,"core_hits":0}
    ],
    "job_size_by_cost_quartile": {
      "q4_highest": {"small_pct":38,"medium_pct":30,"large_pct":32},
      "q3": {"small_pct":61,"medium_pct":39,"large_pct":0},
      "q2": {"small_pct":100,"medium_pct":0,"large_pct":0},
      "q1_lowest": {"small_pct":0,"medium_pct":0,"large_pct":0}
    }
  },
  "prior_arms": {
    "trimmed": 6884,
    "llm_exception": 6869,
    "random_best": 6852,
    "score_escape": 6884,
    "phaseX_random_best": 6884,
    "phaseX_llm_best": 6884
  }
}
```

## Reference Results

- Trace probe TEC: 6946 (same as manual proposal)
- Manual proposal: TEC=6946 (0 improvements from 325 candidates, 20 evaluated)
- Random baseline (5 seeds): range 6893-6946, median ~6924, best 6893 (Δ=-53)
- Prior best from Phase S: 6852

## Proposal Schema

You must output exactly one valid JSON proposal conforming to this schema:

```json
{
  "proposal_name": "<string, max 64 chars>",
  "move_family": "insert_inter",
  "source_machines": ["M<n>", ...],  // 1-5 machine IDs
  "target_machines": ["M<n>", ...],  // 1-5 machine IDs
  "job_size_classes": ["small"|"medium"|"large", ...],  // 1-3 classes
  "max_candidates": <int 1-30>,
  "ranking_hint": "cheap_lb"|"s2"|"random"|"cost_gap"|"slack"|"hybrid",
  "diversity_rule": "per_source"|"per_target"|"source_target_pair"|"none",
  "fallback_if_empty": "random_same_budget"|"top_s2_same_budget",
  "rationale": "<string, max 500 chars>"
}
```

Constraints:
- source_machines: 1-5 distinct machine IDs from the state trace
- target_machines: 1-5 distinct machine IDs from the state trace
- job_size_classes: 1-3 from ["small", "medium", "large"]
- max_candidates: 1-30 (recommended: 20)
- diversity_rule: how to spread candidates across machines
- ranking_hint: how to rank candidates before selection
- fallback_if_empty: what to do if no candidates generated

## Task

Analyze the solver state trace above. Propose exactly ONE bounded neighborhood of (source, job, target) moves that is likely to find TEC improvements when evaluated with exact DP. Output your proposal as a JSON object.

Think step by step about:
1. Which machines are the biggest problems (high EC, high gap, high cost density)?
2. Which machines are underexplored (0 core hits) but have high potential?
3. What job sizes offer the best re-packing opportunities?
4. Which target machines have available slack at reasonable energy rates?
5. What ranking strategy is most appropriate given the gaps?

Your response must be valid JSON only (no markdown, no explanation outside the JSON).
