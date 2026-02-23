"""LLM-driven evolution strategies.

Orchestrates one **batched** LLM call per generation to fill ALL pruned slots
across both pools.  Strategies:
    m1 — Logic Evolution (mutation on lower-ranked elite)
    m2 — Parameter Calibration (mutation on top-ranked elite)
    c1 — Homogeneous Crossover (feature recombination within same pool)
    c2 — Synergistic Joint Crossover (co-evolve a destroy+repair pair)
"""

from __future__ import annotations

import logging
import random
import uuid
from typing import List

from glns.config import EvolutionConfig, SandboxConfig
from glns.llm_client import GroqOperatorClient
from glns.population import PopulationManager
from glns.sanity import sanity_check
from glns.schemas import OperatorRecord, OperatorSpec
from glns.seed_operators import (
    build_seed_destroy_operators,
    build_seed_repair_operators,
)

logger = logging.getLogger(__name__)

STRATEGY_NAMES = [
    "mutation_logic",
    "mutation_param",
    "homo_crossover",
    "joint_crossover",
]


def _sample_strategy(weights: tuple, rng: random.Random) -> str:
    """Weighted random choice among the 4 strategies."""
    total = sum(weights)
    probs = [w / total for w in weights]
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return STRATEGY_NAMES[i]
    return STRATEGY_NAMES[-1]


def _build_action_desc(
    strategy: str,
    pop: PopulationManager,
    pool_type: str,
    rng: random.Random,
    generation: int,
) -> dict:
    """Build the action descriptor dict for one slot to send to the LLM."""
    pool = pop.destroy_pool if pool_type == "destroy" else pop.repair_pool

    if strategy == "mutation_logic":
        parent = pop.select_for_mutation(pool, "logic", rng)
        return {
            "action": "mutation_logic",
            "type": pool_type,
            "parent_idea": parent.idea,
            "parent_code": parent.code,
            "hint": (
                "Generate a NEW operator with a significantly different algorithmic mechanism "
                "than the parent. Focus on structural innovation."
            ),
        }

    elif strategy == "mutation_param":
        parent = pop.select_for_mutation(pool, "param", rng)
        return {
            "action": "mutation_param",
            "type": pool_type,
            "parent_idea": parent.idea,
            "parent_code": parent.code,
            "hint": (
                "Refine this operator by calibrating its parameters (thresholds, ratios, "
                "randomisation factors). Keep the core logic but improve robustness."
            ),
        }

    elif strategy == "homo_crossover":
        a, b = pop.select_for_homo_crossover(pool, rng)
        return {
            "action": "homo_crossover",
            "type": pool_type,
            "parent_idea": a.idea,
            "parent_code": a.code,
            "parent2_idea": b.idea,
            "parent2_code": b.code,
            "hint": (
                "Fuse the strengths of both parents into a single new operator. "
                "The child should combine their complementary features."
            ),
        }

    else:  # joint_crossover
        d_op, r_op, syn_score = pop.select_for_joint_crossover()
        # Joint crossover produces one destroy AND one repair; we'll split
        # them into two slots. But since we're building one action per slot,
        # the caller should handle the pair.
        return {
            "action": "joint_crossover",
            "type": pool_type,  # will be overridden for the pair
            "parent_idea": d_op.idea,
            "parent_code": d_op.code,
            "parent2_idea": r_op.idea,
            "parent2_code": r_op.code,
            "synergy_score": syn_score,
            "hint": (
                "Co-evolve this synergistic destroy+repair pair. The new repair must be "
                "specifically tailored to reconstruct the structural defects introduced "
                "by the new destroy. Maximise their complementary effectiveness."
            ),
        }


def evolve_generation(
    pop: PopulationManager,
    llm: GroqOperatorClient,
    evo_cfg: EvolutionConfig,
    sandbox_cfg: SandboxConfig,
    generation: int,
    rng: random.Random,
) -> List[OperatorRecord]:
    """Fill all pruned slots with ONE batched LLM call per generation.

    Returns the list of newly created (and sanity-checked) OperatorRecords.
    """
    d_slots = pop.destroy_pool.empty_slots()
    r_slots = pop.repair_pool.empty_slots()
    total_needed = d_slots + r_slots
    if total_needed == 0:
        return []

    actions: List[dict] = []

    # Determine which slots need joint crossover (produces a pair).
    # Strategy: sample one strategy per slot, but if joint_crossover is sampled
    # we allocate one destroy + one repair slot together.
    remaining_d = d_slots
    remaining_r = r_slots

    while remaining_d + remaining_r > 0:
        strategy = _sample_strategy(evo_cfg.strategy_weights, rng)

        if strategy == "joint_crossover" and remaining_d > 0 and remaining_r > 0:
            # Produces BOTH a destroy and a repair from the SAME parent pair.
            # Select the synergy-best pair once, then ask the LLM to co-design a coupled child pair.
            d_op, r_op, syn_score = pop.select_for_joint_crossover()
            base = {
                "action": "joint_crossover",
                "parent_idea": d_op.idea,
                "parent_code": d_op.code,
                "parent2_idea": r_op.idea,
                "parent2_code": r_op.code,
                "synergy_score": syn_score,
                "hint": (
                    "Co-evolve this synergistic destroy+repair pair. The new repair must be "
                    "specifically tailored to reconstruct the structural defects introduced "
                    "by the new destroy. Maximise their complementary effectiveness."
                ),
            }
            actions.append({**base, "type": "destroy"})
            actions.append({**base, "type": "repair"})
            remaining_d -= 1
            remaining_r -= 1
        elif remaining_d > 0:
            actions.append(
                _build_action_desc(strategy, pop, "destroy", rng, generation)
            )
            remaining_d -= 1
        elif remaining_r > 0:
            actions.append(_build_action_desc(strategy, pop, "repair", rng, generation))
            remaining_r -= 1

    # ----- ONE batched LLM call ------------------------------------------
    reference_ops = pop.top_reference_ops(k=3)
    try:
        batch = llm.generate_evolution_batch(actions, reference_ops)
    except Exception as exc:
        logger.error("LLM evolution call failed: %s", exc)
        return _fallback_fill(pop, d_slots, r_slots, generation)

    # ----- Sanity check each returned operator ----------------------------
    new_ops: List[OperatorRecord] = []
    for idx, spec in enumerate(batch.operators):
        # Ensure type matches what we asked for.
        expected_type = actions[idx]["type"] if idx < len(actions) else spec.type
        if spec.type != expected_type:
            spec = OperatorSpec(type=expected_type, idea=spec.idea, code=spec.code)

        passed, err = sanity_check(spec, sandbox_cfg)
        if passed:
            rec = OperatorRecord(spec=spec, generation_born=generation)
            new_ops.append(rec)
            continue

        # Retry up to max_retries_per_operator times.
        logger.warning("Operator %d failed sanity: %s — retrying", idx, err)
        fixed = False
        for attempt in range(evo_cfg.max_retries_per_operator):
            try:
                retry_batch = llm.regenerate_with_error(spec, err or "Unknown error")
                retry_spec = retry_batch.operators[0]
                if retry_spec.type != expected_type:
                    retry_spec = OperatorSpec(
                        type=expected_type, idea=retry_spec.idea, code=retry_spec.code
                    )
                passed2, err2 = sanity_check(retry_spec, sandbox_cfg)
                if passed2:
                    rec = OperatorRecord(spec=retry_spec, generation_born=generation)
                    new_ops.append(rec)
                    fixed = True
                    break
                err = err2
            except Exception as retry_exc:
                logger.error(
                    "Retry %d failed for operator %d: %s", attempt, idx, retry_exc
                )
                err = str(retry_exc)

        if not fixed:
            logger.warning("All retries exhausted for slot %d; using fallback", idx)
            fb = _get_fallback(expected_type, generation)
            new_ops.append(fb)

    # ----- Insert into pools ---------------------------------------------
    inserted: List[OperatorRecord] = []
    for op in new_ops:
        pool = pop.destroy_pool if op.op_type == "destroy" else pop.repair_pool
        if pool.add(op):
            inserted.append(op)

    return inserted


# ---------------------------------------------------------------------------
# Fallback: hand-coded operators when LLM fails completely
# ---------------------------------------------------------------------------


def _get_fallback(op_type: str, generation: int) -> OperatorRecord:
    if op_type == "destroy":
        seeds = build_seed_destroy_operators()
    else:
        seeds = build_seed_repair_operators()
    fb = seeds[0]
    fb.generation_born = generation
    fb.id = f"fallback_{op_type}_{generation}_{uuid.uuid4().hex[:6]}"  # type: ignore[assignment]
    return fb  # type: ignore[return-value]


def _fallback_fill(
    pop: PopulationManager, d_slots: int, r_slots: int, generation: int
) -> List[OperatorRecord]:
    """Emergency fill with hand-coded operators when the entire LLM call fails."""
    ops: List[OperatorRecord] = []
    for _ in range(d_slots):
        fb = _get_fallback("destroy", generation)
        pop.destroy_pool.add(fb)
        ops.append(fb)
    for _ in range(r_slots):
        fb = _get_fallback("repair", generation)
        pop.repair_pool.add(fb)
        ops.append(fb)
    return ops
