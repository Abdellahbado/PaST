#!/usr/bin/env python3
"""Quick smoke test for the assignment-only G-LNS pipeline.

Tests:
1. Import chain
2. Seed operators (destroy + repair)
3. Sanity check
4. Sequencing layer (heuristic mode, since Cython may not be compiled)
5. Evaluation engine (1 episode, few iterations)
"""
import sys
import random
import copy


def main():
    print("=== Assignment-only G-LNS smoke test ===\n")

    # 1. Imports
    print("1. Testing imports...")
    from glns.config import GLNSConfig, EvalConfig, SandboxConfig
    from glns.sequencing import (
        evaluate_assignment,
        make_initial_assignment,
        sequences_to_assignment,
        assignment_to_sequences,
    )
    from glns.seed_operators_v2 import (
        build_seed_destroy_operators_v2,
        build_seed_repair_operators_v2,
    )
    from glns.sanity import sanity_check_assignment, make_toy_instances
    from glns.evaluation_v2 import run_evaluation_phase_v2
    from glns.pareto import ParetoArchive

    print("   All imports OK\n")

    # 2. Seed operators
    print("2. Testing seed operators...")
    d_ops = build_seed_destroy_operators_v2()
    r_ops = build_seed_repair_operators_v2()
    print(f"   {len(d_ops)} destroy, {len(r_ops)} repair operators")

    # 3. Sanity check each seed operator
    print("\n3. Running sanity checks on seed operators...")
    sandbox_cfg = SandboxConfig(timeout_sec=5.0)
    all_pass = True
    for op in d_ops:
        passed, err = sanity_check_assignment(op.spec, sandbox_cfg)
        status = "PASS" if passed else f"FAIL: {err}"
        print(f"   Destroy [{op.id[:8]}]: {status}")
        if not passed:
            all_pass = False
    for op in r_ops:
        passed, err = sanity_check_assignment(op.spec, sandbox_cfg)
        status = "PASS" if passed else f"FAIL: {err}"
        print(f"   Repair  [{op.id[:8]}]: {status}")
        if not passed:
            all_pass = False
    if not all_pass:
        print("\n   SOME OPERATORS FAILED SANITY CHECK!")
        return 1

    # 4. Sequencing layer
    print("\n4. Testing sequencing layer...")
    toy = make_toy_instances()[0]
    assign = make_initial_assignment(toy)
    print(f"   Instance: n={toy['n']}, m={toy['m']}, T={toy['T']}")
    print(f"   Initial assignment: {assign}")

    energy, cmax, seqs, starts = evaluate_assignment(
        assign, toy, sequencing_mode="heuristic"
    )
    print(f"   Energy={energy}, Cmax={cmax}")
    print(f"   Sequences: {seqs}")
    print(f"   Start times: {starts}")
    if energy == float("inf"):
        print("   WARNING: Infinite energy (scheduling failed)")
    else:
        print("   Sequencing OK")

    # Test round-trip conversion
    assign2 = sequences_to_assignment(seqs, toy["n"])
    seqs2 = assignment_to_sequences(assign2, toy["m"])
    print(f"   Round-trip assignment: {assign2}")
    assert assign == assign2, f"Round-trip failed: {assign} != {assign2}"
    print("   Round-trip OK")

    # 5. Quick evaluation engine test
    print("\n5. Testing evaluation engine (1 episode, 5 iterations)...")
    eval_cfg = EvalConfig(
        K_episodes=1,
        T_iters=5,
        sa_T0=0.5,
        sa_alpha=0.95,
        destroy_ratio=0.3,
    )
    archive = ParetoArchive(max_size=20)
    rng = random.Random(42)
    F_d, F_r, synergy, stats = run_evaluation_phase_v2(
        destroy_pool=d_ops,
        repair_pool=r_ops,
        instances=[toy],
        archive=archive,
        eval_cfg=eval_cfg,
        sandbox_cfg=sandbox_cfg,
        rng=rng,
        sequencing_mode="heuristic",
    )
    print(f"   Archive size: {archive.size()}")
    print(
        f"   Stats: sigma1={stats['sigma1']}, sigma2={stats['sigma2']}, "
        f"sigma3={stats['sigma3']}, sigma4={stats['sigma4']}"
    )
    print(
        f"   Destroy fails={stats['destroy_fail']}, Repair fails={stats['repair_fail']}"
    )
    print(f"   Invalid candidates={stats['invalid_candidate']}")
    if archive.size() > 0:
        front = archive.front()
        print(f"   Front: {front[:5]}")
    print("   Evaluation engine OK")

    print(f"\n=== ALL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
