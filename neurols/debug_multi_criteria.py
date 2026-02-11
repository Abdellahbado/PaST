
import numpy as np
import random
from PaST.neurols.solution import PMALNSSolution
from PaST.neurols.move_evaluator import MoveEvaluator
from PaST.neurols.candidate_generator import CandidateGenerator, CriterionID
from PaST.neurols.operators import OperatorID, OPERATOR_BY_ID

def debug_mc():
    seed = 42
    rng = np.random.default_rng(seed)
    n_jobs = 20
    n_machines = 4
    K = 100

    p = rng.integers(1, 10, size=n_jobs)
    ct = rng.uniform(0.5, 5.0, size=K)
    e = rng.integers(1, 4, size=n_machines)

    # Random solution
    sol = PMALNSSolution.from_random(n_jobs, n_machines, p, rng)
    
    # Evaluator
    evaluator = MoveEvaluator(p, ct, e, K, n_machines)
    full_eval = evaluator.evaluate_solution(sol)
    
    generator = CandidateGenerator(evaluator)
    
    print(f"Initial Cost: {full_eval.total_energy}")
    
    for op_id in [OperatorID.RELOCATE_1]:
        op = OPERATOR_BY_ID[op_id]
        print(f"\n--- Operator: {op.name} ---")
        
        multi = generator.generate_multi_criteria_moves(sol, full_eval, op)
        
        moves = {}
        for crit, ev in multi.items():
            if ev is None:
                print(f"  {crit.name}: None")
                moves[crit] = None
            else:
                print(f"  {crit.name}: DeltaCost={ev.delta_cost:.2f} Move={ev.move}")
                moves[crit] = ev.move
        
        # Check diversity
        unique_moves = set(m for m in moves.values() if m is not None)
        print(f"  Unique moves: {len(unique_moves)}")
        
        if len(unique_moves) < 3:
             print("  [WARNING] Collapse detected!")

if __name__ == "__main__":
    debug_mc()
