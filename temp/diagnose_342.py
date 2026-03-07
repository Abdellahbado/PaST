"""Diagnose the 304 vs 342 paper mismatch."""

import sys
import numpy as np

sys.path.insert(0, "/Users/mac/Documents/Study/PFE")
from PaST.solvers.machine_states import MachineStateConfig, compute_spaces
from PaST.solvers.optimal_benchmark_dp import solve_optimal_benchmark_dp

prices = np.array(
    [9, 7, 9, 13, 3, 11, 3, 13, 6, 7, 60, 4, 10, 6, 9, 3, 14, 0, 4, 6], dtype=np.float64
)
jobs = [1, 2, 4]

print("=" * 60)
print("Test 1: Current paper_nosby (P(proc,off)=0.0)")
cfg0 = MachineStateConfig.paper_nosby()
r0 = solve_optimal_benchmark_dp(jobs, prices, machine_config=cfg0, track_schedule=True)
print(f"  cost={r0.cost:.1f}  schedule={r0.schedule}")

print()
print("=" * 60)
print("Test 2: paper_nosby with P(proc,off)=1.0")
cfg1 = MachineStateConfig.custom(
    states=["off", "proc", "idle"],
    transitions=[
        ("off", "off", 1, 0.0),
        ("off", "proc", 2, 5.0),
        ("proc", "proc", 1, 4.0),
        ("proc", "off", 1, 1.0),  # P(proc,off)=1 instead of 0
        ("proc", "idle", 0, 0.0),
        ("idle", "idle", 1, 2.0),
        ("idle", "proc", 0, 0.0),
    ],
)
sp1 = compute_spaces(prices, cfg1)
r1 = solve_optimal_benchmark_dp(jobs, prices, machine_config=cfg1, track_schedule=True)
print(f"  cost={r1.cost:.1f}  schedule={r1.schedule}")
print(f"  early={sp1.early}  late={sp1.late}")

if r1.schedule:
    sched = sorted(r1.schedule, key=lambda x: x[1])
    cost_manual = 0
    prev_end = None
    for i, (jid, start, end) in enumerate(sched):
        if i == 0:
            cost_manual += sp1.c_start[start]
            print(f"  c_start[{start}]={sp1.c_start[start]:.1f}")
        else:
            g = sp1.gap_cost(prev_end, start)
            cost_manual += g
            print(f"  gap({prev_end}->{start})={g:.1f}")
        pc = sum(prices[t] * 4.0 for t in range(start, end))
        cost_manual += pc
        print(f"  J{jid+1}[{start}:{end}) proc_cost={pc:.1f}")
        prev_end = end
    cost_manual += sp1.c_end[prev_end]
    print(f"  c_end[{prev_end}]={sp1.c_end[prev_end]:.1f}")
    print(f"  Total manual: {cost_manual:.1f}")

print()
print("=" * 60)
print("Test 3: NOSBY 2-state (no idle) with P(proc,off)=1.0")
cfg2 = MachineStateConfig.custom(
    states=["off", "proc"],
    transitions=[
        ("off", "off", 1, 0.0),
        ("off", "proc", 2, 5.0),
        ("proc", "proc", 1, 4.0),
        ("proc", "off", 1, 1.0),
    ],
)
r2 = solve_optimal_benchmark_dp(jobs, prices, machine_config=cfg2, track_schedule=True)
print(f"  cost={r2.cost:.1f}  schedule={r2.schedule}")

print()
print("=" * 60)
print("Test 4: Check c_end values for cfg1 (to see shutdown costs)")
print("  c_end array (t=3..15):", [f"t{t}={sp1.c_end[t]:.1f}" for t in range(3, 16)])

print()
print("=" * 60)
print("Brute-force: manually compute cost for ALL perms with fixed schedule [3,5,6]")
# Manual verification: J2@[3,5), J1@[5,6), J3@[6,10) for P(proc,off)=1.0
# Startup: off@0, off[0] at P=0=0, startup[1,2] at P=5=7*5+9*5=80
# J2@[3,4,5]: prices[3]*4+prices[4]*4=13*4+3*4=52+12=64
# J1@[5]: prices[5]*4=11*4=44
# J3@[6,7,8,9]: prices[6:10]*4=116
# Shutdown: prices[10]*1=60
startup = prices[0] * 0 + prices[1] * 5 + prices[2] * 5
j2 = prices[3] * 4 + prices[4] * 4
j1 = prices[5] * 4
j3 = prices[6] * 4 + prices[7] * 4 + prices[8] * 4 + prices[9] * 4
shutdown_t10 = prices[10] * 1.0
total_t10 = startup + j2 + j1 + j3 + shutdown_t10
print(
    f"  startup={startup:.1f} j2={j2:.1f} j1={j1:.1f} j3={j3:.1f} shutdown@t10={shutdown_t10:.1f}"
)
print(f"  Total (shutdown@t10)={total_t10:.1f}")

# What if we delay processing to shut down at a cheaper interval?
# Try J2@[15,17), J1@[17,18) - but those are at late=17, need shutdown at 18, c[18]=4
# Actually late=17,  so last proc at 17 and then shutdown at 18 with c[18]*1=4
# But we need to find where those jobs would go. This is complex.
# Let's just check shutdown cost at various points
print()
for t_end in range(10, 20):
    print(f"  c_end[{t_end}] = {sp1.c_end[t_end]:.1f}")
