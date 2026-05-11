# C3-L2 Counter-Sweep Decision

**Date**: 2026-05-10
**Method**: Second-generation DeepSeek counter-sweep (5 families, truncated output)
**Evaluated families**: hybrid_M1 (first_khat_dominance) + hybrid_M2 (asgh_lock_in)
**Budgets**: 30s / 90s

## Results

| Arm | Instances | High-Yield | Rate | Mean Δfs | Median Δfs |
|-----|----------|------------|------|----------|------------|
| human (C3) | 6 | 6 | 100% | 15.2 | 13 |
| LLM Call1 (C3) | 6 | 5 | 83% | 2.8 | 3 |
| LLM Call2 (L2) | 6 | 6 | 100% | 19.5 | 10 |
| random (C3) | 6 | 4 | 67% | 8.5 | 5 |

## Gate: STRONG: LLM Call2 ≥ human sweep on both yield rate and mean front growth.

Yield rates: Human=100%  Call1=83%  Call2=100%
Mean Δfs:    Human=15.2  Call2=19.5

## Per-Instance

### human (C3)

- fs=3→11 cmax=198→190 HIGH [+8]
- fs=20→53 cmax=140→107 HIGH [+33]
- fs=4→13 cmax=197→188 HIGH [+9]
- fs=4→17 cmax=197→184 HIGH [+13]
- fs=1→8 cmax=196→189 HIGH [+7]
- fs=6→27 cmax=169→148 HIGH [+21]

### LLM Call1 (C3)

- fs=1→5 cmax=200→196 HIGH [+4]
- fs=2→8 cmax=199→193 HIGH [+6]
- fs=4→7 cmax=197→194 HIGH [+3]
- fs=1→1 cmax=200→200 low [sat_1]
- fs=2→5 cmax=199→196 HIGH [+3]
- fs=1→2 cmax=200→199 HIGH [slow]

### LLM Call2 (L2)

- fs=1→8 cmax=169→162 HIGH [+7]
- fs=4→12 cmax=140→132 HIGH [+8]
- fs=17→83 cmax=141→52 HIGH [+66]
- fs=5→22 cmax=148→131 HIGH [+17]
- fs=2→11 cmax=172→160 HIGH [+9]
- fs=5→15 cmax=138→128 HIGH [+10]

### random (C3)

- fs=3→8 cmax=96→91 HIGH [+5]
- fs=56→77 cmax=37→16 HIGH [+21]
- fs=1→1 cmax=102→102 HIGH [slow]
- fs=27→27 cmax=20→20 low [sat_27]
- fs=24→49 cmax=76→51 HIGH [+25]
- fs=51→51 cmax=36→36 low [sat_51]

## Mechanism Match

LLM Call2 families combined sweep-layer ingredients (uniform small jobs, tight epsilon) with mechanism-specific stress:
- M1 hybrid: tight epsilon + heterogeneous machine rates → SGH construction slow at each step
- M2 hybrid: tight epsilon + step machine rates → A-SGH retains cheap-machine assignments
Both mechanisms were confirmed: the hybrid design produced large front growth (sweep layer) while preserving mechanism-specific failure signatures.
