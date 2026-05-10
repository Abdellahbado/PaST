# Task: Propose One Bounded Neighborhood for Schedule Improvement

You are an optimization agent that sees the current concrete state of a parallel machine scheduling solver at a stagnation point. Your task is to propose exactly ONE bounded neighborhood of candidate moves that, when evaluated with exact DP, is likely to improve the current solution.

## Current Solver State

Below is the state trace for **Cell_B** (epsilon=290, medium regime, 25 machines, 250 jobs, job sizes 1-12). The solver has completed 5 rounds, accepted 5 moves, and TEC is 9435.

```json
{
  "trace_id": "Cell_B_r4",
  "cell_label": "Cell_B",
  "round": 4,
  "timestamp": "generated",
  "regime": {
    "cell_label": "Cell_B",
    "epsilon": 290,
    "num_machines": 25,
    "total_jobs": 250,
    "epsilon_regime": "medium",
    "job_size_range": [1, 12],
    "episode_epsilon_progression": [290]
  },
  "snapshot": {
    "current_tec": 9435,
    "best_tec_episode": 9435,
    "tec_improvement_last_n_rounds": 0.0,
    "no_hit_streak": 0,
    "total_rounds_completed": 5,
    "total_accepted_moves_so_far": 5,
    "exact_dp_evals_so_far": 43,
    "core_lane_stagnation_active": true,
    "exception_lane_active": false,
    "stop_reason_guard": "none"
  },
  "machines": [
    {"id":"M0","jobs":0,"load":0,"slack":290,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":5,"rate":4,"starved":false},
    {"id":"M1","jobs":5,"load":45,"slack":245,"load_pressure":0.155,"exact_cost":405,"relaxed_lb":165,"gap":240,"cost_density":9,"small_jobs":0,"medium_jobs":0,"large_jobs":5,"core_source_hits":0,"core_target_hits":5,"rate":3,"starved":true},
    {"id":"M2","jobs":0,"load":0,"slack":290,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":5,"rate":6,"starved":false},
    {"id":"M3","jobs":0,"load":0,"slack":290,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":6,"starved":false},
    {"id":"M4","jobs":25,"load":168,"slack":122,"load_pressure":0.579,"exact_cost":1176,"relaxed_lb":994,"gap":182,"cost_density":7,"small_jobs":1,"medium_jobs":22,"large_jobs":2,"core_source_hits":3,"core_target_hits":0,"rate":2,"starved":false},
    {"id":"M5","jobs":30,"load":159,"slack":131,"load_pressure":0.548,"exact_cost":1036,"relaxed_lb":904,"gap":132,"cost_density":6.516,"small_jobs":6,"medium_jobs":23,"large_jobs":1,"core_source_hits":3,"core_target_hits":0,"rate":2,"starved":false},
    {"id":"M6","jobs":0,"load":0,"slack":290,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":4,"starved":false},
    {"id":"M7","jobs":26,"load":290,"slack":0,"load_pressure":1,"exact_cost":1319,"relaxed_lb":1319,"gap":0,"cost_density":4.548,"small_jobs":0,"medium_jobs":1,"large_jobs":25,"core_source_hits":3,"core_target_hits":0,"rate":1,"starved":false},
    {"id":"M8","jobs":0,"load":0,"slack":290,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":3,"starved":false},
    {"id":"M9","jobs":0,"load":0,"slack":290,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":6,"starved":false},
    {"id":"M10","jobs":0,"load":0,"slack":290,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":5,"starved":false},
    {"id":"M11","jobs":40,"load":186,"slack":104,"load_pressure":0.641,"exact_cost":1280,"relaxed_lb":1184,"gap":96,"cost_density":6.882,"small_jobs":25,"medium_jobs":14,"large_jobs":1,"core_source_hits":3,"core_target_hits":0,"rate":2,"starved":false},
    {"id":"M12","jobs":0,"load":0,"slack":290,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":5,"starved":false},
    {"id":"M13","jobs":0,"load":0,"slack":290,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":5,"starved":false},
    {"id":"M14","jobs":36,"load":140,"slack":150,"load_pressure":0.483,"exact_cost":842,"relaxed_lb":718,"gap":124,"cost_density":6.014,"small_jobs":21,"medium_jobs":14,"large_jobs":1,"core_source_hits":0,"core_target_hits":0,"rate":2,"starved":true},
    {"id":"M15","jobs":22,"load":126,"slack":164,"load_pressure":0.434,"exact_cost":784,"relaxed_lb":606,"gap":178,"cost_density":6.222,"small_jobs":7,"medium_jobs":14,"large_jobs":1,"core_source_hits":0,"core_target_hits":0,"rate":2,"starved":true},
    {"id":"M16","jobs":0,"load":0,"slack":290,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":4,"starved":false},
    {"id":"M17","jobs":0,"load":0,"slack":290,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":4,"starved":false},
    {"id":"M18","jobs":0,"load":0,"slack":290,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":5,"starved":false},
    {"id":"M19","jobs":0,"load":0,"slack":290,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":4,"starved":false},
    {"id":"M20","jobs":0,"load":0,"slack":290,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":5,"starved":false},
    {"id":"M21","jobs":28,"load":290,"slack":0,"load_pressure":1,"exact_cost":1319,"relaxed_lb":1319,"gap":0,"cost_density":4.548,"small_jobs":1,"medium_jobs":0,"large_jobs":27,"core_source_hits":0,"core_target_hits":0,"rate":1,"starved":true},
    {"id":"M22","jobs":38,"load":186,"slack":104,"load_pressure":0.641,"exact_cost":1274,"relaxed_lb":1184,"gap":90,"cost_density":6.849,"small_jobs":22,"medium_jobs":15,"large_jobs":1,"core_source_hits":3,"core_target_hits":0,"rate":2,"starved":false},
    {"id":"M23","jobs":0,"load":0,"slack":290,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":3,"starved":false},
    {"id":"M24","jobs":0,"load":0,"slack":290,"load_pressure":0,"exact_cost":0,"relaxed_lb":0,"gap":0,"cost_density":0,"small_jobs":0,"medium_jobs":0,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":5,"starved":false}
  ],
  "recent": {
    "last_accepted_moves": [
      {"round":0,"source":"M4","target":"M1","job_size":9,"delta_tec":13,"was_exception":false},
      {"round":1,"source":"M4","target":"M1","job_size":9,"delta_tec":8,"was_exception":false},
      {"round":2,"source":"M5","target":"M1","job_size":9,"delta_tec":4,"was_exception":false},
      {"round":3,"source":"M5","target":"M1","job_size":9,"delta_tec":4,"was_exception":false},
      {"round":4,"source":"M5","target":"M1","job_size":9,"delta_tec":2,"was_exception":false}
    ],
    "failed_summary": {"evaluated_exact_this_round": 2, "no_improving_move_found": false},
    "core_shortlist_composition": {"distinct_sources_note": "not tracked per-round in trace probe"},
    "outside_pool_composition": {"total_candidates": 0, "distinct_sources": 0, "distinct_targets": 0, "source_coverage": 0},
    "next_round_budget": {"core_budget": 14}
  },
  "candidate_pools": {
    "top_sources_by_cost": [
      {"id":"M21","exact_cost":1319,"gap":0,"cost_density":4.548,"jobs":28},
      {"id":"M7","exact_cost":1319,"gap":0,"cost_density":4.548,"jobs":26},
      {"id":"M11","exact_cost":1280,"gap":96,"cost_density":6.882,"jobs":40},
      {"id":"M22","exact_cost":1274,"gap":90,"cost_density":6.849,"jobs":38},
      {"id":"M4","exact_cost":1176,"gap":182,"cost_density":7,"jobs":25}
    ],
    "top_sources_by_gap": [
      {"id":"M1","gap":240,"exact_cost":405},
      {"id":"M4","gap":182,"exact_cost":1176},
      {"id":"M15","gap":178,"exact_cost":784},
      {"id":"M5","gap":132,"exact_cost":1036},
      {"id":"M14","gap":124,"exact_cost":842}
    ],
    "top_targets_by_slack": [
      {"id":"M24","slack":290,"load_pressure":0,"jobs":0},
      {"id":"M23","slack":290,"load_pressure":0,"jobs":0},
      {"id":"M20","slack":290,"load_pressure":0,"jobs":0},
      {"id":"M19","slack":290,"load_pressure":0,"jobs":0},
      {"id":"M18","slack":290,"load_pressure":0,"jobs":0}
    ],
    "underexplored_sources": [
      {"id":"M21","exact_cost":1319,"gap":0,"cost_density":4.548,"core_hits":0},
      {"id":"M14","exact_cost":842,"gap":124,"cost_density":6.014,"core_hits":0},
      {"id":"M15","exact_cost":784,"gap":178,"cost_density":6.222,"core_hits":0},
      {"id":"M1","exact_cost":405,"gap":240,"cost_density":9,"core_hits":0}
    ],
    "underexplored_targets": [
      {"id":"M24","slack":290,"load_pressure":0,"jobs":0,"core_hits":0},
      {"id":"M23","slack":290,"load_pressure":0,"jobs":0,"core_hits":0},
      {"id":"M20","slack":290,"load_pressure":0,"jobs":0,"core_hits":0},
      {"id":"M19","slack":290,"load_pressure":0,"jobs":0,"core_hits":0},
      {"id":"M18","slack":290,"load_pressure":0,"jobs":0,"core_hits":0}
    ],
    "job_size_by_cost_quartile": {
      "q4_highest": {"small_pct":34,"medium_pct":40,"large_pct":26},
      "q3": {"small_pct":26,"medium_pct":52,"large_pct":22},
      "q2": {"small_pct":0,"medium_pct":0,"large_pct":0},
      "q1_lowest": {"small_pct":0,"medium_pct":0,"large_pct":0}
    }
  },
  "prior_arms": {
    "trimmed": 9687,
    "llm_exception": 9455,
    "random_best": 9583,
    "score_escape": 9484,
    "phaseX_random_best": 9495,
    "phaseX_llm_best": 9495
  }
}
```

## Reference Results

- Trace probe TEC: 9435 (already beats prior best from Phase S: trimmed=9687, llm_exception=9455)
- Manual proposal: FAILED (SIGBUS on macOS, results unavailable)
- Random baseline (5 seeds, only seed 1 failed): range 9366-9406, best 9366 (Δ=-69)

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
