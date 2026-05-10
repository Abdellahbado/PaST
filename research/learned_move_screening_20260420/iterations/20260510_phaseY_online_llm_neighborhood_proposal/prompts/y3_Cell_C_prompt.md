# Task: Propose One Bounded Neighborhood for Schedule Improvement

You are an optimization agent that sees the current concrete state of a parallel machine scheduling solver at a stagnation point. Your task is to propose exactly ONE bounded neighborhood of candidate moves that, when evaluated with exact DP, is likely to improve the current solution.

## Current Solver State

Below is the state trace for **Cell_C** (epsilon=195, tight regime, 25 machines, 350 jobs, job sizes 1-12). The solver has completed 5 rounds, accepted 5 moves, and TEC is 27031.

```json
{
  "trace_id": "Cell_C_r4",
  "cell_label": "Cell_C",
  "round": 4,
  "timestamp": "generated",
  "regime": {
    "cell_label": "Cell_C",
    "epsilon": 195,
    "num_machines": 25,
    "total_jobs": 350,
    "epsilon_regime": "tight",
    "job_size_range": [1, 12],
    "episode_epsilon_progression": [195]
  },
  "snapshot": {
    "current_tec": 27031,
    "best_tec_episode": 27031,
    "tec_improvement_last_n_rounds": 0.0,
    "no_hit_streak": 0,
    "total_rounds_completed": 5,
    "total_accepted_moves_so_far": 5,
    "exact_dp_evals_so_far": 51,
    "core_lane_stagnation_active": true,
    "exception_lane_active": false,
    "stop_reason_guard": "none"
  },
  "machines": [
    {"id":"M0","jobs":24,"load":184,"slack":11,"load_pressure":0.944,"exact_cost":2457,"relaxed_lb":2457,"gap":0,"cost_density":13.353,"small_jobs":1,"medium_jobs":12,"large_jobs":11,"core_source_hits":3,"core_target_hits":0,"rate":3,"starved":false},
    {"id":"M1","jobs":3,"load":27,"slack":168,"load_pressure":0.138,"exact_cost":504,"relaxed_lb":216,"gap":288,"cost_density":18.667,"small_jobs":0,"medium_jobs":1,"large_jobs":2,"core_source_hits":0,"core_target_hits":5,"rate":6,"starved":true},
    {"id":"M2","jobs":25,"load":184,"slack":11,"load_pressure":0.944,"exact_cost":2457,"relaxed_lb":2457,"gap":0,"cost_density":13.353,"small_jobs":1,"medium_jobs":13,"large_jobs":11,"core_source_hits":3,"core_target_hits":0,"rate":3,"starved":false},
    {"id":"M3","jobs":7,"load":21,"slack":174,"load_pressure":0.108,"exact_cost":225,"relaxed_lb":120,"gap":105,"cost_density":10.714,"small_jobs":5,"medium_jobs":2,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":5,"starved":true},
    {"id":"M4","jobs":8,"load":21,"slack":174,"load_pressure":0.108,"exact_cost":215,"relaxed_lb":120,"gap":95,"cost_density":10.238,"small_jobs":6,"medium_jobs":2,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":5,"starved":true},
    {"id":"M5","jobs":5,"load":17,"slack":178,"load_pressure":0.087,"exact_cost":190,"relaxed_lb":85,"gap":105,"cost_density":11.176,"small_jobs":3,"medium_jobs":2,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":5,"starved":true},
    {"id":"M6","jobs":3,"load":27,"slack":168,"load_pressure":0.138,"exact_cost":504,"relaxed_lb":216,"gap":288,"cost_density":18.667,"small_jobs":0,"medium_jobs":1,"large_jobs":2,"core_source_hits":0,"core_target_hits":5,"rate":6,"starved":true},
    {"id":"M7","jobs":23,"load":184,"slack":11,"load_pressure":0.944,"exact_cost":2457,"relaxed_lb":2457,"gap":0,"cost_density":13.353,"small_jobs":1,"medium_jobs":16,"large_jobs":6,"core_source_hits":0,"core_target_hits":0,"rate":3,"starved":true},
    {"id":"M8","jobs":5,"load":17,"slack":178,"load_pressure":0.087,"exact_cost":190,"relaxed_lb":85,"gap":105,"cost_density":11.176,"small_jobs":3,"medium_jobs":2,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":5,"starved":true},
    {"id":"M9","jobs":17,"load":195,"slack":0,"load_pressure":1,"exact_cost":1814,"relaxed_lb":1814,"gap":0,"cost_density":9.303,"small_jobs":0,"medium_jobs":1,"large_jobs":16,"core_source_hits":0,"core_target_hits":0,"rate":2,"starved":true},
    {"id":"M10","jobs":3,"load":11,"slack":184,"load_pressure":0.056,"exact_cost":135,"relaxed_lb":55,"gap":80,"cost_density":12.273,"small_jobs":2,"medium_jobs":1,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":5,"starved":true},
    {"id":"M11","jobs":3,"load":11,"slack":184,"load_pressure":0.056,"exact_cost":135,"relaxed_lb":55,"gap":80,"cost_density":12.273,"small_jobs":2,"medium_jobs":1,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":5,"starved":true},
    {"id":"M12","jobs":2,"load":16,"slack":179,"load_pressure":0.082,"exact_cost":276,"relaxed_lb":96,"gap":180,"cost_density":17.25,"small_jobs":0,"medium_jobs":1,"large_jobs":1,"core_source_hits":0,"core_target_hits":5,"rate":6,"starved":true},
    {"id":"M13","jobs":22,"load":184,"slack":11,"load_pressure":0.944,"exact_cost":2460,"relaxed_lb":2457,"gap":3,"cost_density":13.37,"small_jobs":1,"medium_jobs":11,"large_jobs":10,"core_source_hits":3,"core_target_hits":0,"rate":3,"starved":false},
    {"id":"M14","jobs":24,"load":184,"slack":11,"load_pressure":0.944,"exact_cost":2457,"relaxed_lb":2457,"gap":0,"cost_density":13.353,"small_jobs":0,"medium_jobs":19,"large_jobs":5,"core_source_hits":3,"core_target_hits":0,"rate":3,"starved":false},
    {"id":"M15","jobs":3,"load":11,"slack":184,"load_pressure":0.056,"exact_cost":135,"relaxed_lb":55,"gap":80,"cost_density":12.273,"small_jobs":2,"medium_jobs":1,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":5,"starved":true},
    {"id":"M16","jobs":1,"load":5,"slack":190,"load_pressure":0.026,"exact_cost":72,"relaxed_lb":30,"gap":42,"cost_density":14.4,"small_jobs":0,"medium_jobs":1,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":6,"starved":true},
    {"id":"M17","jobs":3,"load":11,"slack":184,"load_pressure":0.056,"exact_cost":135,"relaxed_lb":55,"gap":80,"cost_density":12.273,"small_jobs":2,"medium_jobs":1,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":5,"starved":true},
    {"id":"M18","jobs":37,"load":123,"slack":72,"load_pressure":0.631,"exact_cost":1664,"relaxed_lb":1620,"gap":44,"cost_density":13.528,"small_jobs":30,"medium_jobs":7,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":4,"starved":true},
    {"id":"M19","jobs":24,"load":195,"slack":0,"load_pressure":1,"exact_cost":2721,"relaxed_lb":2721,"gap":0,"cost_density":13.954,"small_jobs":1,"medium_jobs":11,"large_jobs":12,"core_source_hits":3,"core_target_hits":0,"rate":3,"starved":false},
    {"id":"M20","jobs":36,"load":123,"slack":72,"load_pressure":0.631,"exact_cost":1664,"relaxed_lb":1620,"gap":44,"cost_density":13.528,"small_jobs":24,"medium_jobs":12,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":4,"starved":true},
    {"id":"M21","jobs":18,"load":195,"slack":0,"load_pressure":1,"exact_cost":1814,"relaxed_lb":1814,"gap":0,"cost_density":9.303,"small_jobs":1,"medium_jobs":0,"large_jobs":17,"core_source_hits":0,"core_target_hits":0,"rate":2,"starved":true},
    {"id":"M22","jobs":18,"load":195,"slack":0,"load_pressure":1,"exact_cost":907,"relaxed_lb":907,"gap":0,"cost_density":4.651,"small_jobs":0,"medium_jobs":1,"large_jobs":17,"core_source_hits":0,"core_target_hits":0,"rate":1,"starved":true},
    {"id":"M23","jobs":3,"load":11,"slack":184,"load_pressure":0.056,"exact_cost":135,"relaxed_lb":55,"gap":80,"cost_density":12.273,"small_jobs":2,"medium_jobs":1,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":5,"starved":true},
    {"id":"M24","jobs":33,"load":104,"slack":91,"load_pressure":0.533,"exact_cost":1308,"relaxed_lb":1208,"gap":100,"cost_density":12.577,"small_jobs":26,"medium_jobs":7,"large_jobs":0,"core_source_hits":0,"core_target_hits":0,"rate":4,"starved":true}
  ],
  "recent": {
    "last_accepted_moves": [
      {"round":0,"source":"M0","target":"M1","job_size":11,"delta_tec":60,"was_exception":false},
      {"round":1,"source":"M2","target":"M1","job_size":11,"delta_tec":36,"was_exception":false},
      {"round":2,"source":"M7","target":"M6","job_size":11,"delta_tec":60,"was_exception":false},
      {"round":3,"source":"M13","target":"M6","job_size":11,"delta_tec":33,"was_exception":false},
      {"round":4,"source":"M14","target":"M12","job_size":11,"delta_tec":60,"was_exception":false}
    ],
    "failed_summary": {"evaluated_exact_this_round": 5, "no_improving_move_found": false},
    "core_shortlist_composition": {"distinct_sources_note": "not tracked per-round in trace probe"},
    "outside_pool_composition": {"total_candidates": 0, "distinct_sources": 0, "distinct_targets": 0, "source_coverage": 0},
    "next_round_budget": {"core_budget": 14}
  },
  "candidate_pools": {
    "top_sources_by_cost": [
      {"id":"M19","exact_cost":2721,"gap":0,"cost_density":13.954,"jobs":24},
      {"id":"M13","exact_cost":2460,"gap":3,"cost_density":13.37,"jobs":22},
      {"id":"M14","exact_cost":2457,"gap":0,"cost_density":13.353,"jobs":24},
      {"id":"M7","exact_cost":2457,"gap":0,"cost_density":13.353,"jobs":23},
      {"id":"M2","exact_cost":2457,"gap":0,"cost_density":13.353,"jobs":25}
    ],
    "top_sources_by_gap": [
      {"id":"M6","gap":288,"exact_cost":504},
      {"id":"M1","gap":288,"exact_cost":504},
      {"id":"M12","gap":180,"exact_cost":276},
      {"id":"M8","gap":105,"exact_cost":190},
      {"id":"M5","gap":105,"exact_cost":190}
    ],
    "top_targets_by_slack": [
      {"id":"M16","slack":190,"load_pressure":0.026,"jobs":1},
      {"id":"M23","slack":184,"load_pressure":0.056,"jobs":3},
      {"id":"M17","slack":184,"load_pressure":0.056,"jobs":3},
      {"id":"M15","slack":184,"load_pressure":0.056,"jobs":3},
      {"id":"M11","slack":184,"load_pressure":0.056,"jobs":3}
    ],
    "underexplored_sources": [
      {"id":"M7","exact_cost":2457,"gap":0,"cost_density":13.353,"core_hits":0},
      {"id":"M21","exact_cost":1814,"gap":0,"cost_density":9.303,"core_hits":0},
      {"id":"M9","exact_cost":1814,"gap":0,"cost_density":9.303,"core_hits":0},
      {"id":"M20","exact_cost":1664,"gap":44,"cost_density":13.528,"core_hits":0},
      {"id":"M18","exact_cost":1664,"gap":44,"cost_density":13.528,"core_hits":0}
    ],
    "underexplored_targets": [
      {"id":"M16","slack":190,"load_pressure":0.026,"jobs":1,"core_hits":0},
      {"id":"M23","slack":184,"load_pressure":0.056,"jobs":3,"core_hits":0},
      {"id":"M17","slack":184,"load_pressure":0.056,"jobs":3,"core_hits":0},
      {"id":"M15","slack":184,"load_pressure":0.056,"jobs":3,"core_hits":0},
      {"id":"M11","slack":184,"load_pressure":0.056,"jobs":3,"core_hits":0}
    ],
    "job_size_by_cost_quartile": {
      "q4_highest": {"small_pct":4,"medium_pct":51,"large_pct":45},
      "q3": {"small_pct":56,"medium_pct":20,"large_pct":24},
      "q2": {"small_pct":57,"medium_pct":33,"large_pct":10},
      "q1_lowest": {"small_pct":63,"medium_pct":38,"large_pct":0}
    }
  },
  "prior_arms": {
    "trimmed": 27031,
    "llm_exception": 26926,
    "random_best": 26262,
    "score_escape": 26470,
    "phaseX_random_best": 26263,
    "phaseX_llm_best": 26478
  }
}
```

## Reference Results

- Trace probe TEC: 27031
- Manual proposal: TEC=26715 (550 generated, 20 evaluated, 12 improvements, Δ=-316)
- Random baseline (5 seeds, only seed 1 passed): TEC=26947 (Δ=-84)
- Prior best from Phase S: 26262 (Δ=-769 from current!)
- Phase X best: 26263

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
