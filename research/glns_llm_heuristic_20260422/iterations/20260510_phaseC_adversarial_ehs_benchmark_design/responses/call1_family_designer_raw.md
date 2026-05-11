{
  "generator": "deepseek_v4_pro",
  "generator_call": "call1_family_designer",
  "generator_description": "LLM-designed adversarial instance families targeting EHS failure mechanisms M1-M8 based on B6 closure evidence.",
  "n_families": 8,
  "families": [
    {
      "family_name": "first_khat_dominance_giant",
      "hypothesis": "The first khat (SGH construction at khat=T) costs O(n*m*T) and dominates the runtime on large instances, consuming 100–400 s on VLS-scale problems. Under a strict time budget (60–120 s), EHS fails to complete even a single khat, resulting in zero Pareto front points. This family uses high n, high m, and large T to maximize SGH cost, making first‑khat domination the primary failure mechanism. Evidence: B6 surface ‘first_khat_dominance’ shows VLS instances have first‑khat cost >100 s, leaving no front points within short budget.",
      "description": "Massive instance with 800 jobs, 40 machines, horizon T=800. Processing times widely distributed (10–60) to stress feasibility checks. Machine rates span a 10× range. Dual‑peak TOU provides energy differentiation. The sheer size forces EHS to spend all its short‑budget time on the first SGH construction, producing an empty front.",
      "n_jobs_range": {
        "min": 700,
        "max": 900
      },
      "m_machines_range": {
        "min": 35,
        "max": 45
      },
      "horizon_T_range": {
        "min": 750,
        "max": 850
      },
      "processing_time_distribution": {
        "type": "uniform",
        "params": {
          "low": 10,
          "high": 60
        }
      },
      "machine_rate_distribution": {
        "type": "uniform",
        "params": {
          "low": 0.3,
          "high": 3.0
        }
      },
      "TOU_price_profile_type": "dual_peak",
      "TOU_price_profile_params": {
        "peak1_start": 0.1,
        "peak1_end": 0.3,
        "peak2_start": 0.6,
        "peak2_end": 0.8,
        "peak_multiplier": 2.5,
        "base_price": 1.0
      },
      "epsilon_regime": "medium",
      "expected_EHS_failure_mechanism": "first_khat_dominance",
      "expected_EHS_failure_mechanism_evidence": "B6.1: first_khat_dominance surface closed – VLS instances first khat cost >100 s, dominates short budget, 0 Pareto points.",
      "generated_instances_count": 8,
      "validity_constraints": {
        "feasibility_guarantee": true,
        "min_total_work": 24000,
        "n_per_machine_min": 10
      },
      "rejection_conditions": [
        "sum(p_j) > m * T",
        "all e[h] equal",
        "all ct[t] equal",
        "n < 2 * m"
      ],
      "seed_behavior": {
        "base_seed": 1000,
        "expected_seed_variance": "low"
      }
    },
    {
      "family_name": "asgh_trajectory_conflict",
      "hypothesis": "A‑SGH retains 96–98 % of jobs from the previous khat level. When the optimal assignment changes structurally at different khat values, this lock‑in prevents EHS from discovering better solutions at lower khats. Bimodal job sizes (small cheap‑to‑move, large load‑critical) combined with a dual‑peak TOU create different optimal machine‑job clusters at khat=T and khat≈T/2. Evidence: B6.11 – released‑jobs repair back to the same trajectory, A‑SGH retention saturates improvement.",
      "description": "80–120 jobs with 30 % small (p=2–5) and 70 % large (p=12–20), 6–12 machines, horizon 200–350. Machines split into low/high energy rates (step function). Dual‑peak TOU with compact price windows. Medium epsilon ensures multiple khats, amplifying lock‑in.",
      "n_jobs_range": {
        "min": 80,
        "max": 120
      },
      "m_machines_range": {
        "min": 6,
        "max": 12
      },
      "horizon_T_range": {
        "min": 200,
        "max": 350
      },
      "processing_time_distribution": {
        "type": "bimodal",
        "params": {
          "small_low": 2,
          "small_high": 5,
          "large_low": 12,
          "large_high": 20,
          "small_fraction": 0.3
        }
      },
      "machine_rate_distribution": {
        "type": "step",
        "params": {
          "low_rate": 0.4,
          "high_rate": 3.0,
          "step_fraction": 0.6
        }
      },
      "TOU_price_profile_type": "dual_peak",
      "TOU_price_profile_params": {
        "peak1_start": 0.2,
        "peak1_end": 0.35,
        "peak2_start": 0.6,
        "peak2_end": 0.75,
        "peak_multiplier": 2.0,
        "base_price": 1.0
      },
      "epsilon_regime": "medium",
      "expected_EHS_failure_mechanism": "asgh_lock_in",
      "expected_EHS_failure_mechanism_evidence": "B6.11: A‑SGH retains 96‑98 % jobs; released jobs repair back to same trajectory. Bimodal sizes force structurally different optimal assignments at different khats.",
      "generated_instances_count": 8,
      "validity_constraints": {
        "feasibility_guarantee": true,
        "min_total_work": 400,
        "n_per_machine_min": 4
      },
      "rejection_conditions": [
        "sum(p_j) > m * T",
        "all e[h] equal",
        "all ct[t] equal",
        "n < 2 * m"
      ],
      "seed_behavior": {
        "base_seed": 30000,
        "expected_seed_variance": "medium"
      }
    },
    {
      "family_name": "reinsertion_starvation_tight_epsilon",
      "hypothesis": "R‑ES reinsertion costs 1.81 s per khat but improves only 1.4 % of khats. In instances with many local optima and a tight epsilon (many khat steps), the reinsertion procedure would be extremely valuable but is starved of time under a short budget. The solver either never triggers it or runs it on too few khats, missing the improvements needed for a high‑quality front. Evidence: B6.5 – R‑ES/ESR improvement limited to 1.4 % of khats; fast_mode skips R‑ES after 75 % budget.",
      "description": "200 jobs, 10 machines, horizon 500. Processing times vary widely (1‑40) to create many feasible schedules. Machine rates differ by a factor of 5. TOU with high variance (frequent sharp peaks) produces many local minima. Tight epsilon forces hundreds of khats, starving R‑ES entirely under a 60‑s budget.",
      "n_jobs_range": {
        "min": 150,
        "max": 250
      },
      "m_machines_range": {
        "min": 8,
        "max": 12
      },
      "horizon_T_range": {
        "min": 450,
        "max": 550
      },
      "processing_time_distribution": {
        "type": "uniform",
        "params": {
          "low": 1,
          "high": 40
        }
      },
      "machine_rate_distribution": {
        "type": "uniform",
        "params": {
          "low": 0.5,
          "high": 2.5
        }
      },
      "TOU_price_profile_type": "high_variance",
      "TOU_price_profile_params": {
        "base_price": 1.0,
        "variation_amplitude": 3.0,
        "peak_frequency": 0.3
      },
      "epsilon_regime": "tight",
      "expected_EHS_failure_mechanism": "res_reinsertion_starvation",
      "expected_EHS_failure_mechanism_evidence": "B6.5: R‑ES/ESR improve only 1.4 % of khats, cost 1.81 s/khat. Tight epsilon multiplies khats, causing R‑ES starvation under short budget.",
      "generated_instances_count": 8,
      "validity_constraints": {
        "feasibility_guarantee": true,
        "min_total_work": 1500,
        "n_per_machine_min": 10
      },
      "rejection_conditions": [
        "sum(p_j) > m * T",
        "all e[h] equal",
        "all ct[t] equal",
        "n < 2 * m"
      ],
      "seed_behavior": {
        "base_seed": 40000,
        "expected_seed_variance": "high"
      }
    },
    {
      "family_name": "es_local_optima_trap_extreme_rates",
      "hypothesis": "ES (Exchange Procedure with Search) improves 36.6 % of khats but can become trapped in local minima, preventing the R‑ES reinsertion from escaping to globally better regions. Heterogeneous machine rates and sharp, high‑multiplier TOU peaks create many energy‑cost trade‑offs that act as deep local optima for a first‑improvement neighbourhood search. Evidence: B6.3 – EPS ordering saturated, local improvements cannot break out; ES non‑empty improvements cause lock‑in.",
      "description": "100 jobs, 10 machines, horizon 500. Machine rates differ by a factor of 50 (0.1 vs 5.0) and are assigned as a step function (half cheap, half expensive). TOU uses a dual‑peak profile with a 5× multiplier, creating extreme cost gradients. Bimodal processing times (10 small, 25 large) generate many swappable job pairs. ES easily gets trapped in configurations that are energy‑cheap but makespan‑poor.",
      "n_jobs_range": {
        "min": 80,
        "max": 120
      },
      "m_machines_range": {
        "min": 8,
        "max": 12
      },
      "horizon_T_range": {
        "min": 400,
        "max": 600
      },
      "processing_time_distribution": {
        "type": "bimodal",
        "params": {
          "small_low": 3,
          "small_high": 6,
          "large_low": 18,
          "large_high": 28,
          "small_fraction": 0.4
        }
      },
      "machine_rate_distribution": {
        "type": "step",
        "params": {
          "low_rate": 0.1,
          "high_rate": 5.0,
          "step_fraction": 0.5
        }
      },
      "TOU_price_profile_type": "dual_peak",
      "TOU_price_profile_params": {
        "peak1_start": 0.15,
        "peak1_end": 0.25,
        "peak2_start": 0.65,
        "peak2_end": 0.75,
        "peak_multiplier": 5.0,
        "base_price": 1.0
      },
      "epsilon_regime": "medium",
      "expected_EHS_failure_mechanism": "es_exploration_tension",
      "expected_EHS_failure_mechanism_evidence": "B6.3: EPS ordering saturated, local improvements cannot escape local minima; 36.6% khats improved but may trap solution.",
      "generated_instances_count": 8,
      "validity_constraints": {
        "feasibility_guarantee": true,
        "min_total_work": 600,
        "n_per_machine_min": 5
      },
      "rejection_conditions": [
        "sum(p_j) > m * T",
        "all e[h] equal",
        "all ct[t] equal",
        "n < 2 * m"
      ],
      "seed_behavior": {
        "base_seed": 50000,
        "expected_seed_variance": "medium"
      }
    },
    {
      "family_name": "front_coverage_gap_step_TOU",
      "hypothesis": "When the TOU price profile is a step function with large discrete jumps, the energy‑cost vs makespan trade‑off becomes discontinuous. EHS’s uniform khat descent (by epsilon) can miss the critical cmax values where the trade‑off changes, leaving gaps in the Pareto front. Coarse epsilon amplifies this effect. Evidence: B6.7 – front coverage is sensitive to TOU shape; step‑like cost functions cause missed intermediate points.",
      "description": "120 jobs, 12 machines, horizon 500. Processing times uniform (5‑30). Machine rates vary moderately (0.5–2.0). The TOU profile is a step function with base price 1.0, a high‑price zone (4.0) for t in [150,250] and [400,450], and zero elsewhere. This creates a staircase‑shaped cost curve. Loose epsilon causes skipped Pareto points.",
      "n_jobs_range": {
        "min": 100,
        "max": 140
      },
      "m_machines_range": {
        "min": 10,
        "max": 14
      },
      "horizon_T_range": {
        "min": 450,
        "max": 550
      },
      "processing_time_distribution": {
        "type": "uniform",
        "params": {
          "low": 5,
          "high": 30
        }
      },
      "machine_rate_distribution": {
        "type": "uniform",
        "params": {
          "low": 0.5,
          "high": 2.0
        }
      },
      "TOU_price_profile_type": "step_function",
      "TOU_price_profile_params": {
        "base_price": 1.0,
        "steps": [
          {"start": 0, "end": 150, "multiplier": 1.0},
          {"start": 150, "end": 250, "multiplier": 4.0},
          {"start": 250, "end": 400, "multiplier": 1.0},
          {"start": 400, "end": 450, "multiplier": 4.0},
          {"start": 450, "end": 500, "multiplier": 1.0}
        ]
      },
      "epsilon_regime": "loose",
      "expected_EHS_failure_mechanism": "front_coverage_gap",
      "expected_EHS_failure_mechanism_evidence": "B6.7: step‑function TOU creates discontinuous cost trades; loose epsilon exacerbates skipping of intermediate front points.",
      "generated_instances_count": 8,
      "validity_constraints": {
        "feasibility_guarantee": true,
        "min_total_work": 800,
        "n_per_machine_min": 5
      },
      "rejection_conditions": [
        "sum(p_j) > m * T",
        "all e[h] equal",
        "all ct[t] equal",
        "n < 2 * m"
      ],
      "seed_behavior": {
        "base_seed": 60000,
        "expected_seed_variance": "low"
      }
    },
    {
      "family_name": "short_budget_critical_size",
      "hypothesis": "EHS at 120 s reaches only 12.9–71.6 % of published HV, with the gap largest when the first khat cost equals the time budget. This family sizes the instance so that SGH construction at khat=T consumes ≈80 s, leaving minimal time for khat descent and thus producing a poor Pareto front. Evidence: B6.1 and B6.10 – first khat dominates on large instances; short budget yields low HV proportion.",
      "description": "500 jobs, 25 machines, horizon 400. Processing times uniform (5–20) give total work ~6250, well within m·T=10000. Machine rates 0.3–3.0, dual‑peak TOU. The O(n·m·T) cost is calibrated to consume most of a 120‑s budget in the first khat, preventing adequate khat descent and front expansion.",
      "n_jobs_range": {
        "min": 450,
        "max": 550
      },
      "m_machines_range": {
        "min": 20,
        "max": 30
      },
      "horizon_T_range": {
        "min": 350,
        "max": 450
      },
      "processing_time_distribution": {
        "type": "uniform",
        "params": {
          "low": 5,
          "high": 20
        }
      },
      "machine_rate_distribution": {
        "type": "uniform",
        "params": {
          "low": 0.3,
          "high": 3.0
        }
      },
      "TOU_price_profile_type": "dual_peak",
      "TOU_price_profile_params": {
        "peak1_start": 0.2,
        "peak1_end": 0.4,
        "peak2_start": 0.6,
        "peak2_end": 0.8,
        "peak_multiplier": 2.0,
        "base_price": 1.0
      },
      "epsilon_regime": "medium",
      "expected_EHS_failure_mechanism": "short_budget_pressure",
      "expected_EHS_failure_mechanism_evidence": "B6.1 + B6.10: short budget (120 s) yields 12.9‑71.6 % of published HV; first‑khat cost ≈ time budget causes severe front truncation.",
      "generated_instances_count": 8,
      "validity_constraints": {
        "feasibility_guarantee": true,
        "min_total_work": 5000,
        "n_per_machine_min": 10
      },
      "rejection_conditions": [
        "sum(p_j) > m * T",
        "all e[h] equal",
        "all ct[t] equal",
        "n < 2 * m"
      ],
      "seed_behavior": {
        "base_seed": 70000,
        "expected_seed_variance": "medium"
      }
    },
    {
      "family_name": "load_imbalance_narrow_jobs_wide_rates",
      "hypothesis": "SGH construction greedily places jobs on the cheapest‑energy machines, creating a severe load imbalance when machine energy rates differ widely and processing times have little variance. This inflates the makespan, and the subsequent khat descent struggles to redistribute nearly identical jobs. Evidence: B6.2 – SGH tie‑breaking rejection; B6.7 – load imbalance observed with heterogeneous rates.",
      "description": "200 jobs of nearly identical processing time (10–14), 20 machines with a 20× energy rate spread (0.1 to 2.0). Horizon 300. SGH will overload the cheapest machines, causing a makespan far above the theoretical optimum. The narrow job size range offers no natural structure to guide rebalancing, leaving high‑cmax points dominant on the front.",
      "n_jobs_range": {
        "min": 180,
        "max": 220
      },
      "m_machines_range": {
        "min": 15,
        "max": 25
      },
      "horizon_T_range": {
        "min": 250,
        "max": 350
      },
      "processing_time_distribution": {
        "type": "uniform",
        "params": {
          "low": 10,
          "high": 14
        }
      },
      "machine_rate_distribution": {
        "type": "uniform",
        "params": {
          "low": 0.1,
          "high": 2.0
        }
      },
      "TOU_price_profile_type": "single_peak",
      "TOU_price_profile_params": {
        "peak_start": 0.3,
        "peak_end": 0.7,
        "peak_multiplier": 1.8,
        "base_price": 1.0
      },
      "epsilon_regime": "medium",
      "expected_EHS_failure_mechanism": "load_imbalance",
      "expected_EHS_failure_mechanism_evidence": "B6.2: SGH tie‑breaking rejected; B6.7: heterogeneous rates cause load concentration on cheap machines, inflating cmax.",
      "generated_instances_count": 8,
      "validity_constraints": {
        "feasibility_guarantee": true,
        "min_total_work": 2000,
        "n_per_machine_min": 5
      },
      "rejection_conditions": [
        "sum(p_j) > m * T",
        "all e[h] equal",
        "all ct[t] equal",
        "n < 2 * m"
      ],
      "seed_behavior": {
        "base_seed": 80000,
        "expected_seed_variance": "low"
      }
    },
    {
      "family_name": "epsilon_skip_narrow_cost_steps",
      "hypothesis": "EHS descends khat by an epsilon step that may be larger than the cmax differences at which significant energy‑cost changes occur. When machine rates vary widely but the TOU price profile has narrow, small‑multiplier variations, the energy‑cost vs cmax curve exhibits step‑like behaviour with fine steps. A loose epsilon regime will skip over these critical cmax values, leaving intermediate Pareto points uncovered. Evidence: B6.8 – epsilon step size directly influences front coverage; coarse skipping observed with step‑like TOU.",
      "description": "150 jobs, 15 machines, horizon 500. Processing times uniform (8‑30). Machine rates span an 8× range (0.25 to 2.0) with a step distribution (half cheap, half expensive). TOU uses a single moderate peak (1.0 base, 2.0 peak) over 30–60 % of the horizon. Loose epsilon ensures only a handful of khats, overstepping the fine cost‑saving jumps.",
      "n_jobs_range": {
        "min": 120,
        "max": 180
      },
      "m_machines_range": {
        "min": 12,
        "max": 18
      },
      "horizon_T_range": {
        "min": 450,
        "max": 550
      },
      "processing_time_distribution": {
        "type": "uniform",
        "params": {
          "low": 8,
          "high": 30
        }
      },
      "machine_rate_distribution": {
        "type": "step",
        "params": {
          "low_rate": 0.25,
          "high_rate": 2.0,
          "step_fraction": 0.5
        }
      },
      "TOU_price_profile_type": "single_peak",
      "TOU_price_profile_params": {
        "peak_start": 0.3,
        "peak_end": 0.6,
        "peak_multiplier": 2.0,
        "base_price": 1.0
      },
      "epsilon_regime": "loose",
      "expected_EHS_failure_mechanism": "epsilon_skip",
      "expected_EHS_failure_mechanism_evidence": "B6.8: epsilon step size influences coverage; loose epsilon causes coarse descent, skipping intermediate cmax points where energy‑cost steps occur.",
      "generated_instances_count": 8,
      "validity_constraints": {
        "feasibility_guarantee": true,
        "min_total_work": 1200,
        "n_per_machine_min": 5
      },
      "rejection_conditions": [
        "sum(p_j) > m * T",
        "all e[h] equal",
        "all ct[t] equal",
        "n < 2 * m"
      ],
      "seed_behavior": {
        "base_seed": 90000,
        "expected_seed_variance": "low"
      }
    }
  ]
}