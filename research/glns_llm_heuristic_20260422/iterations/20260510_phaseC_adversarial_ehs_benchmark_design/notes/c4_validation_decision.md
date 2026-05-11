# C4 Validation Decision

**Date**: 2026-05-11
**Instances**: 12 families × 5 instances = 60 total
**Budgets**: 30s / 90s

## Summary Table

| Arm | N | Feasible | Evaluable | Gen% | HY | HY% | Mean Δfs | MC% | NT-MC-HY% |
|-----|--|----------|-----------|-------|----|------|----------|------|------------|
| LLM Call2 (L2) | 20 | 20 | 20 | 100% | 20 | 100% | 20.4 | 25% | 25% |
| Human sweep | 20 | 20 | 20 | 100% | 20 | 100% | 38.2 | 50% | 50% |
| Random | 20 | 15 | 15 | 75% | 14 | 93% | 2.6 | 0% | 0% |

## Gate: MODERATE

High-Yield:    LLM=100%  Human=100%  Random=93%
NT-MC-HY:      LLM=25%  Human=50%
Mean Δfs:      LLM=20.4  Human=38.2  Random=2.6

## Decision Questions

1. **Did L2 transfer beyond C3-L2?** C3-L2 had 6/6 (100%) HY with Δfs_mean=19.5. C4 validation: 100% HY with Δfs_mean=20.4. Transfer confirmed — yield rate maintained on 4× more families.
2. **Did L2 beat random?** Random=93% HY. Yes, narrow margin.
3. **Did L2 beat or tie human sweep?** Human=100% HY, MC=50%, NT-MC-HY=50%. Tied on HY rate but lost on NT-MC-HY.

4. **LLM advantage**: Generation quality=100% (vs Human=100%), Adversarial yield=100% (vs Human=100%), Mechanism specificity NT-MC-HY=25% (vs Human=50%).

5. **Strong enough for thesis?** Gate=MODERATE. See discussion below.

## Caveats

- **Budget too short**: All 60 instances timed out on 30s short budget. This means the Δfs metric is dominated by budget truncation, not mechanism-specific behavior.
- **Human sweep structural advantage**: Human families have n up to 150 and uniform p_j=(1,10), producing very large raw Δfs (+73 to +116 on loose_epsilon). This is a known structural property of the metric, not adversarial insight.
- **LLM L2 families**: 20/20 high-yield (100%) with moderate Δfs (+0 to +70). The heterogeneous/step rate mechanisms produce front growth but with smaller raw magnitude due to fewer uniform small jobs.

## Per-Family Breakdown

### LLM Call2 (L2)

- **hybrid_M1_tight_hetero_rates_firstkhat_dom** (first_khat_dominance): 5/5 HY, MC=0/5, NT=0/5, Δfs=[+8, +5, +28, +70, +7]
- **hybrid_M2_tight_steprates_asghlock** (asgh_lock_in): 5/5 HY, MC=5/5, NT=5/5, Δfs=[+30, +31, +38, +7, +19]
- **hybrid_M3_narrowrates_reinsert_starve** (res_reinsertion_starvation): 5/5 HY, MC=0/5, NT=0/5, Δfs=[+0, +4, +2, +0, +1]
- **hybrid_M4_dualpeak_exploration_tension** (es_exploration_tension): 5/5 HY, MC=0/5, NT=0/5, Δfs=[+7, +41, +32, +12, +66]

### Human sweep

- **human_loose_epsilon** (epsilon_skip): 5/5 HY, MC=0/5, NT=0/5, Δfs=[+73, +83, +63, +50, +116]
- **human_many_machines_sparse** (front_coverage_gap): 5/5 HY, MC=5/5, NT=5/5, Δfs=[+22, +43, +32, +28, +46]
- **human_mixed_job_sizes** (asgh_lock_in): 5/5 HY, MC=5/5, NT=5/5, Δfs=[+10, +5, +9, +8, +5]
- **human_tight_epsilon** (first_khat_dominance): 5/5 HY, MC=0/5, NT=0/5, Δfs=[+98, +28, +23, +4, +19]

### Random

- **random_000** (load_imbalance): 5/5 HY, MC=0/5, NT=0/5, Δfs=[+0, +0, +0, +0, +0]
- **random_001** (epsilon_skip): 5/5 HY, MC=0/5, NT=0/5, Δfs=[+0, +0, +0, +1, +0]
- **random_002** (asgh_lock_in): 4/5 HY, MC=0/5, NT=0/5, Δfs=[+1, +1, +0, +1, +2]
- **random_003** (es_exploration_tension): 5/5 HY, MC=0/5, NT=0/5, Δfs=[+6, +7, +6, +4, +10]

## Per-Instance Details

### LLM Call2 (L2)

- c4v_hybrid_M1_tight_hetero_rates_firstkhat_dom_180000: fs=6→14 cmax=160→160 rt=31.7s/97.2s TO/ HIGH [+8]
- c4v_hybrid_M1_tight_hetero_rates_firstkhat_dom_180001: fs=10→15 cmax=143→143 rt=31.1s/90.3s TO/ HIGH [+5]
- c4v_hybrid_M1_tight_hetero_rates_firstkhat_dom_180002: fs=7→35 cmax=150→150 rt=32.3s/90.8s TO/ HIGH [+28]
- c4v_hybrid_M1_tight_hetero_rates_firstkhat_dom_180003: fs=18→88 cmax=174→174 rt=30.1s/64.3s TO/ HIGH [+70]
- c4v_hybrid_M1_tight_hetero_rates_firstkhat_dom_180004: fs=6→13 cmax=167→167 rt=30.5s/95.8s TO/ HIGH [+7]
- c4v_hybrid_M2_tight_steprates_asghlock_180000: fs=17→47 cmax=169→169 rt=33.4s/90.7s TO/ HIGH [MC] ***NT*** [+30]
- c4v_hybrid_M2_tight_steprates_asghlock_180001: fs=26→57 cmax=179→163 rt=30.5s/65.7s TO/ HIGH [MC] ***NT*** [+31]
- c4v_hybrid_M2_tight_steprates_asghlock_180002: fs=14→52 cmax=177→177 rt=31.4s/90.7s TO/ HIGH [MC] ***NT*** [+38]
- c4v_hybrid_M2_tight_steprates_asghlock_180003: fs=32→39 cmax=136→125 rt=30.4s/56.6s TO/ HIGH [MC] ***NT*** [+7]
- c4v_hybrid_M2_tight_steprates_asghlock_180004: fs=9→28 cmax=146→146 rt=31.7s/91.1s TO/ HIGH [MC] ***NT*** [+19]
- c4v_hybrid_M3_narrowrates_reinsert_starve_180000: fs=1→1 cmax=185→185 rt=103.7s/104.4s TO/ HIGH [slow]
- c4v_hybrid_M3_narrowrates_reinsert_starve_180001: fs=1→5 cmax=179→179 rt=47.9s/101.8s TO/ HIGH [+4]
- c4v_hybrid_M3_narrowrates_reinsert_starve_180002: fs=1→3 cmax=156→156 rt=30.6s/99.9s TO/ HIGH [+2]
- c4v_hybrid_M3_narrowrates_reinsert_starve_180003: fs=1→1 cmax=190→184 rt=54.6s/92.8s TO/ HIGH [cmaxΔ=6]
- c4v_hybrid_M3_narrowrates_reinsert_starve_180004: fs=1→2 cmax=189→189 rt=53.4s/90.7s TO/ HIGH [slow]
- c4v_hybrid_M4_dualpeak_exploration_tension_180000: fs=11→18 cmax=177→177 rt=31.5s/90.4s TO/ HIGH [+7]
- c4v_hybrid_M4_dualpeak_exploration_tension_180001: fs=15→56 cmax=144→144 rt=30.3s/68.9s TO/ HIGH [+41]
- c4v_hybrid_M4_dualpeak_exploration_tension_180002: fs=17→49 cmax=172→172 rt=31.0s/90.4s TO/ HIGH [+32]
- c4v_hybrid_M4_dualpeak_exploration_tension_180003: fs=13→25 cmax=157→157 rt=31.9s/91.5s TO/ HIGH [+12]
- c4v_hybrid_M4_dualpeak_exploration_tension_180004: fs=18→84 cmax=174→174 rt=30.1s/81.7s TO/ HIGH [+66]

### Human sweep

- c4v_human_loose_epsilon_120000: fs=30→103 cmax=171→171 rt=30.7s/90.4s TO/ HIGH [+73]
- c4v_human_loose_epsilon_120001: fs=56→139 cmax=169→169 rt=31.1s/67.0s TO/ HIGH [+83]
- c4v_human_loose_epsilon_120002: fs=21→84 cmax=191→191 rt=30.9s/90.6s TO/ HIGH [+63]
- c4v_human_loose_epsilon_120003: fs=12→62 cmax=188→188 rt=30.5s/90.3s TO/ HIGH [+50]
- c4v_human_loose_epsilon_120004: fs=43→159 cmax=200→200 rt=30.1s/82.8s TO/ HIGH [+116]
- c4v_human_many_machines_sparse_120000: fs=10→32 cmax=150→150 rt=32.5s/92.1s TO/ HIGH [MC] ***NT*** [+22]
- c4v_human_many_machines_sparse_120001: fs=15→58 cmax=166→166 rt=31.6s/90.2s TO/ HIGH [MC] ***NT*** [+43]
- c4v_human_many_machines_sparse_120002: fs=16→48 cmax=135→135 rt=31.7s/90.9s TO/ HIGH [MC] ***NT*** [+32]
- c4v_human_many_machines_sparse_120003: fs=12→40 cmax=143→143 rt=32.5s/92.9s TO/ HIGH [MC] ***NT*** [+28]
- c4v_human_many_machines_sparse_120004: fs=20→66 cmax=196→196 rt=30.8s/90.4s TO/ HIGH [MC] ***NT*** [+46]
- c4v_human_mixed_job_sizes_120000: fs=3→13 cmax=200→200 rt=33.2s/90.4s TO/ HIGH [MC] ***NT*** [+10]
- c4v_human_mixed_job_sizes_120001: fs=7→12 cmax=200→200 rt=31.6s/94.3s TO/ HIGH [MC] ***NT*** [+5]
- c4v_human_mixed_job_sizes_120002: fs=2→11 cmax=200→200 rt=30.5s/96.0s TO/ HIGH [MC] ***NT*** [+9]
- c4v_human_mixed_job_sizes_120003: fs=3→11 cmax=200→200 rt=30.4s/90.5s TO/ HIGH [MC] ***NT*** [+8]
- c4v_human_mixed_job_sizes_120004: fs=2→7 cmax=200→200 rt=40.6s/93.5s TO/ HIGH [MC] ***NT*** [+5]
- c4v_human_tight_epsilon_120000: fs=40→138 cmax=188→188 rt=30.5s/66.1s TO/ HIGH [+98]
- c4v_human_tight_epsilon_120001: fs=12→40 cmax=169→169 rt=31.2s/92.1s TO/ HIGH [+28]
- c4v_human_tight_epsilon_120002: fs=9→32 cmax=163→163 rt=31.5s/91.2s TO/ HIGH [+23]
- c4v_human_tight_epsilon_120003: fs=26→30 cmax=199→199 rt=30.6s/90.9s TO/ HIGH [+4]
- c4v_human_tight_epsilon_120004: fs=5→24 cmax=193→193 rt=36.7s/91.1s TO/ HIGH [+19]

### Random

- c4v_random_000_110000: fs=0→0 cmax=200→200 rt=0.0s/0.0s TO/ HIGH [slow]
- c4v_random_000_110001: fs=0→0 cmax=200→200 rt=0.0s/0.0s TO/ HIGH [slow]
- c4v_random_000_110002: fs=0→0 cmax=200→200 rt=0.0s/0.0s TO/ HIGH [slow]
- c4v_random_000_110003: fs=0→0 cmax=200→200 rt=0.0s/0.0s TO/ HIGH [slow]
- c4v_random_000_110004: fs=0→0 cmax=200→200 rt=0.0s/0.0s TO/ HIGH [slow]
- c4v_random_001_110000: fs=1→1 cmax=200→200 rt=309.6s/318.0s TO/ HIGH [slow]
- c4v_random_001_110001: fs=1→1 cmax=200→200 rt=228.5s/227.8s TO/ HIGH [slow]
- c4v_random_001_110002: fs=1→1 cmax=200→200 rt=181.7s/182.3s TO/ HIGH [slow]
- c4v_random_001_110003: fs=1→2 cmax=200→200 rt=70.4s/99.8s TO/ HIGH [slow]
- c4v_random_001_110004: fs=1→1 cmax=200→200 rt=144.4s/146.1s TO/ HIGH [slow]
- c4v_random_002_110000: fs=2→3 cmax=200→200 rt=52.9s/671.5s TO/ low [Δ1]
- c4v_random_002_110001: fs=1→2 cmax=200→200 rt=51.2s/103.0s TO/ HIGH [slow]
- c4v_random_002_110002: fs=1→1 cmax=200→200 rt=367.0s/281.0s TO/ HIGH [slow]
- c4v_random_002_110003: fs=1→2 cmax=200→200 rt=63.3s/276.5s TO/ HIGH [slow]
- c4v_random_002_110004: fs=2→4 cmax=200→200 rt=32.4s/141.9s TO/ HIGH [+2]
- c4v_random_003_110000: fs=3→9 cmax=200→200 rt=31.2s/100.2s TO/ HIGH [+6]
- c4v_random_003_110001: fs=3→10 cmax=200→200 rt=34.7s/97.0s TO/ HIGH [+7]
- c4v_random_003_110002: fs=3→9 cmax=200→200 rt=35.6s/94.1s TO/ HIGH [+6]
- c4v_random_003_110003: fs=1→5 cmax=200→200 rt=32.2s/95.5s TO/ HIGH [+4]
- c4v_random_003_110004: fs=4→14 cmax=200→200 rt=32.0s/92.7s TO/ HIGH [+10]

