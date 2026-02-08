import sys
from pathlib import Path
import numpy as np

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(_REPO_ROOT))

from sandbox.neurols.env import NeuroLSEnv, EnvConfig
from sandbox.neurols.price_profile_analyzer import PriceProfileAnalyzer
from sandbox.neurols.operators import OperatorID, OPERATOR_BY_ID
from sandbox.neurols.solution import PMALNSSolution
from PaST.data.sm_benchmark_data import RawInstance

def create_dummy_instance():
    """Create a simple instance with 1 machine and clear Peak/Valley."""
    p = [2, 2, 2] # 3 jobs of length 2
    
    # 20-hour day
    # Valley: 0-5 (Price 1)
    # Peak: 5-10 (Price 10)
    K = 20
    ct = np.ones(K) * 5.0
    ct[0:5] = 1.0 # Valley
    ct[5:10] = 10.0 # Peak
    ct[10:] = 1.0 # Make the rest Valley too, so jobs landing later are Valley
    
    return RawInstance(
        instance_id="dummy",
        n=3,
        m=1, # 1 machine to force congestion
        e=[1.0],
        p=[2, 2, 2], # 3 jobs of length 2 -> Total 6

        ct=ct,
        scale=1.0,
        D_days=1,
        hours_per_day=20,
        T_max=20,
        Tk=20,
        ck=ct, # Duplicate ct as ck? Usually ck is discrete levels? No, ck is cost k?
               # Let's assume ck=ct for now.
        period_starts=[],
    )

def test_evacuate_peak():
    instance = create_dummy_instance()
    K = 20
    
    print("Initializing Env...")
    env = NeuroLSEnv(EnvConfig(action_space="STRUCTURE_AWARE"))
    state = env.reset(instance, K)
    
    # Force a solution where a job is in Peak
    # Job 0 at 10 (Peak is 10-14, Job len 2 -> 10-12)
    # Job 1 at 0 (Valley is 0-5, Job len 2 -> 0-2)
    # Job 2 at 15 (Standard)
    
    # Manually constructed solution?
    # PMALNSSolution stores sequence, not start times.
    # Start times come from DP.
    # We need to construct a sequence that naturally puts a job in peak.
    # With 1 machine, sequence [0, 1, 2] implies times 0, 2, 4 (if greedy/compact).
    # DP will optimize it.
    
    # To test EvacuatePeak, we need a setup where DP *might* put something in cost.
    # But DP is optimal for the sequence.
    # If we have 1 machine, DP finds optimal start times.
    # If optimal places it in Peak, then it's unavoidable (or necessary).
    
    # Let's use 2 machines to allow relocation.
    # Machine 1: High Rate?
    # Machine 2: Low Rate?
    
    # Use 1 machine setup from create_dummy_instance
    # instance.m = 2 ... removed
    
    # Update env
    state = env.reset(instance, K)
    
    # Machine 0: [Job 0, Job 1, Job 2]
    # Times: 0, 2, 4.
    # Job 2 (4-6) overlaps Peak (5-10). Midpoint 5 is Peak.
    # Should trigger EvacuatePeak.
    
    # Let's create a "Blocked" valley scenario.
    # We can't block easily without dummy fixed jobs (not supported).
    # But we can fill it.
    
    print("Testing Price Analyzer...")
    analyzer = env._price_analyzer
    print(f"Valleys: {analyzer.valleys}")
    print(f"Peaks: {analyzer.peaks}")
    
    # Just run the operator on the initial random solution
    # and print generated moves.
    
    print("\nInitial Solution:")
    print(state.solution.sequences)
    
    # Evaluate
    print(f"Initial Cost: {state.current_cost}")
    
    op = OPERATOR_BY_ID[OperatorID.EVACUATE_PEAK]
    
    print(f"\nGenerating moves for {op.name}...")
    move_result = env._candidate_gen.generate_best_move(
        state.solution,
        env._current_eval,
        op,
        context={"evaluation": env._current_eval, "processing_times": env._processing_times, "price_analyzer": env._price_analyzer}
    )
    
    if move_result:
        print(f"Found Move: {move_result.move}")
        print(f"Delta: {move_result.delta_cost}")
    else:
        print("No move found (maybe no jobs in peak?)")

    # Debug: Print start times to see where jobs are
    for m in range(instance.m):
        print(f"Machine {m} Start Times: {env._current_eval.per_machine[m].start_times}")
        
    print("\nDone.")

def test_swap_peak_valley():
    print("\n=== Testing SWAP_PEAK_VALLEY ===")
    instance = create_dummy_instance()
    # Modify instance to have distinguishable jobs
    # Job 0 (len 2) -> Valley (0-2)
    # Job 1 (len 2) -> Peak (6-8)
    instance.m = 1
    # Machine 0: [0, 1] -> 0, 2
    # If 0-5 is Valley, 5-10 is Peak.
    # Job 0 at 0-2 (Valley)
    # Job 1 at 2-4 (Valley)
    # Job 2 at 4-6 (Splits)
    # We need a clear Peak job.
    # Let's add more jobs to push later jobs into Peak.
    instance.p = [2, 2, 2, 2] # 4 jobs -> 0, 2, 4, 6.
    instance.n = 4
    
    K = 20
    env = NeuroLSEnv(EnvConfig(action_space="STRUCTURE_AWARE"))
    state = env.reset(instance, K)
    
    print("\nInitial Solution:")
    print(state.solution.sequences)
    
    # Machine 0: [0, 1, 2, 3]
    # Times: 0, 2, 4, 6
    # Job 0 (0-2): Valley
    # Job 3 (6-8): Peak (5-10 implies 6 is Peak)
    
    op = OPERATOR_BY_ID[OperatorID.SWAP_PEAK_VALLEY]
    
    print(f"\nGenerating moves for {op.name}...")
    
    # Debug: Print State
    eval_state = env._current_eval
    analyzer = env._price_analyzer
    m=0
    start_times = eval_state.per_machine[m].start_times
    print(f"Start Times (M0): {start_times}")
    
    for pos, start in enumerate(start_times):
        p_j = 2
        mid = start + p_j // 2
        tier = analyzer.get_tier_at(mid)
        print(f"Job {pos} @ {start}: mid={mid}, tier={tier}")

    # Create context manually or let env do it (env doesn't expose way to run 1 op with context easily from outside)
    # We need to construct context.
    context = {
        "evaluation": env._current_eval,
        "processing_times": env._processing_times,
        "price_analyzer": env._price_analyzer,
    }
    
    moves = op.enumerate_moves(state.solution, context=context)
    print(f"Proposed Moves: {len(moves)}")
    for m in moves:
        print(m)
        
    # Expect Swap(m0, pos0(Job0), m0, pos3(Job3))
    # m0=0. pos0=0 (Job0 is Valley). pos3=3 (Job3 is Peak).
    # Job 1 (2-4) Valley. Job 2 (4-6) Split.
    # Maybe Job 2 is Peak? Midpoint 5 is Peak.
    # So Job 2 is Peak.
    # Expect Swap(0, 0, 0, 3) and Swap(0, 1, 0, 3) etc.

if __name__ == "__main__":
    test_evacuate_peak()
    test_swap_peak_valley()
