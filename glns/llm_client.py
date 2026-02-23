"""Groq LLM client for G-LNS operator generation.

Wraps the official ``groq`` SDK with:
- A **fixed** system prompt (maximizes Groq prompt caching for Kimi-K2).
- All variable content (parents, action, N needed) in the user message.
- Built-in SDK retries (no extra tenacity layer).
- Client-side throttle to respect RPM limits.
- JSON extraction with Pydantic validation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import List, Optional

from groq import Groq

from glns.config import LLMConfig
from glns.schemas import OperatorBatch, OperatorRecord, OperatorSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constant system prompt — never changes across calls (prompt-cache friendly)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert algorithm designer specialising in combinatorial optimisation.

## Problem
Bi-objective green parallel-machine scheduling under Time-of-Use (TOU) electricity prices.
- n jobs, each with integer processing time p_j.
- m identical parallel machines, each with integer energy rate e_h.
- Horizon of T time slots with per-slot electricity price c_t.
- Objectives: minimise makespan (Cmax) AND total energy cost (TEC).
  TEC = sum over all machines h, for each job j on h: e_h * sum(c_t for t in processing slots of j).
- A separate DP layer computes optimal start times given a fixed job ordering per machine.
  Your operators only control ASSIGNMENT (which machine) and SEQUENCING (job order per machine).

## Solution representation
A solution is `sequences: List[List[int]]` of length m.
  sequences[h] = ordered list of global job indices assigned to machine h.
Every job index in 0..n-1 appears exactly once across all machines.

## Instance dict (available as `instance` argument)
{
  "m": int,               # number of machines
  "n": int,               # number of jobs
  "T": int,               # time horizon length
  "p": List[int],         # processing times, length n
  "e": List[int],         # machine energy rates, length m
  "ct": List[int],        # per-slot TOU prices, length T
}

## Operator interface contract
### Destroy operator
```python
def destroy(solution: List[List[int]],
            destroy_cnt: int,
            instance: dict,
            rng) -> tuple[list[int], list[list[int]]]:
    \"\"\"Remove destroy_cnt jobs from the solution.
    Args:
        solution: current sequences (list of m lists of job ids).
        destroy_cnt: number of jobs to remove.
        instance: problem data dict.
        rng: random.Random instance for stochastic decisions.
    Returns:
        (removed_jobs, partial_solution)
        - removed_jobs: list of removed job indices.
        - partial_solution: sequences with those jobs removed.
    \"\"\"
```

### Repair operator
```python
def repair(partial_solution: List[List[int]],
           removed_jobs: List[int],
           instance: dict,
           rng) -> list[list[int]]:
    \"\"\"Reinsert all removed jobs into the partial solution.
    Args:
        partial_solution: sequences with some jobs missing.
        removed_jobs: list of job indices to reinsert.
        instance: problem data dict.
        rng: random.Random instance.
    Returns:
        Complete sequences (every job 0..n-1 appears exactly once).
    \"\"\"
```

## Rules
1. Each operator is a SINGLE self-contained Python function. ALL helper functions and imports go INSIDE.
2. Only stdlib + `math` + `random` may be used (no numpy/scipy/torch).
3. Return ONLY valid JSON — no markdown fences, no prose outside the JSON.
4. Keep operators efficient: O(n*m) or O(n^2) at most; avoid exponential search.

## Domain hints (idea library)
Destroy ideas: random removal, worst-energy-contributor removal, most-loaded-machine removal,
peak-period job removal, slack-sensitive removal, energy-imbalance pair removal,
high-penalty-gap removal, critical-path tail removal, idle-gap cleanup, TOU-boundary removal.
Repair ideas: greedy TOU-aware insertion (try all machine+position, pick lowest energy estimate),
load-balancing assignment, random insertion, block-based insertion, energy-aware pairwise swap.
For bi-objective balance: sometimes focus destroy on Cmax-sensitive parts (tail of critical machine),
sometimes on TEC-sensitive parts (peak-period jobs), sometimes mix both.
"""


# ---------------------------------------------------------------------------
# Client wrapper
# ---------------------------------------------------------------------------


class GroqOperatorClient:
    """Thin wrapper over the Groq SDK for batched operator generation."""

    def __init__(self, cfg: LLMConfig, api_key: Optional[str] = None) -> None:
        self.cfg = cfg
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY not set.  Pass it via constructor or env var."
            )
        self.client = Groq(
            api_key=key,
            max_retries=cfg.max_retries,
            timeout=cfg.timeout_sec,
        )
        self._last_call_ts: float = 0.0

    # ----- low-level call -------------------------------------------------

    def _throttled_chat(self, user_message: str) -> str:
        """Call the Groq chat API with client-side rate limiting."""
        elapsed = time.monotonic() - self._last_call_ts
        wait = self.cfg.min_call_interval_sec - elapsed
        if wait > 0:
            logger.debug("LLM throttle: sleeping %.2fs", wait)
            time.sleep(wait)

        logger.info(
            "LLM call: model=%s temp=%.2f max_tokens=%d user_chars=%d",
            self.cfg.model,
            self.cfg.temperature,
            self.cfg.max_tokens,
            len(user_message),
        )
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        self._last_call_ts = time.monotonic()
        content = resp.choices[0].message.content or ""
        logger.info("LLM response: chars=%d", len(content))
        return content

    # ----- JSON parsing & validation --------------------------------------

    @staticmethod
    def _extract_json(raw: str) -> list:
        """Extract a JSON array from possibly markdown-fenced LLM output."""
        # Try direct parse first.
        raw_stripped = raw.strip()
        if raw_stripped.startswith("["):
            try:
                return json.loads(raw_stripped)
            except json.JSONDecodeError:
                pass

        # Try extracting from ```json ... ``` fences.
        m = re.search(r"```(?:json)?\s*(\[.*?])\s*```", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # Last resort: find outermost [ ... ].
        m = re.search(r"\[.*]", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"Could not extract JSON array from LLM response:\n{raw[:500]}"
        )

    def _parse_response(self, raw: str) -> OperatorBatch:
        items = self._extract_json(raw)
        return OperatorBatch(operators=[OperatorSpec(**item) for item in items])

    # ----- public generation methods --------------------------------------

    def generate_init_batch(self, n_destroy: int, n_repair: int) -> OperatorBatch:
        """Initialization: generate seed operators from scratch (1 LLM call)."""
        user_msg = (
            f"Action: INITIALIZATION\n"
            f"No reference operators; design from scratch.\n\n"
            f"Generate exactly {n_destroy} destroy operators and {n_repair} repair "
            f"operators for bi-objective green parallel-machine scheduling.\n\n"
            f"Return a JSON list of {n_destroy + n_repair} operator dicts:\n"
            f'[{{"type": "destroy", "idea": "...", "code": "def destroy(...): ..."}}, ...]\n'
        )
        raw = self._throttled_chat(user_msg)
        return self._parse_response(raw)

    def generate_evolution_batch(
        self,
        actions: List[dict],
        reference_ops: List[OperatorRecord],
    ) -> OperatorBatch:
        """One batched LLM call per generation to fill ALL pruned slots.

        Args:
            actions: list of dicts, each describing one needed operator, e.g.
                {"action": "mutation_logic", "type": "destroy",
                 "parent_code": "...", "parent_idea": "..."}
            reference_ops: top-performing operators shown as in-context examples
                (max 3 to stay within token budget).
        """
        # Build reference block (only top 3).
        ref_block_parts: list[str] = []
        for i, op in enumerate(reference_ops[:3]):
            ref_block_parts.append(
                f"### Reference operator {i+1} ({op.op_type})\n"
                f"Idea: {op.idea}\n```python\n{op.code}\n```"
            )
        ref_block = "\n\n".join(ref_block_parts) if ref_block_parts else "None."

        # Build per-slot action descriptions.
        slot_parts: list[str] = []
        for idx, a in enumerate(actions):
            slot_parts.append(
                f"--- Slot {idx+1} ---\n"
                f"Action: {a['action']}\n"
                f"Operator type needed: {a['type']}\n"
                + (
                    f"Parent idea: {a.get('parent_idea', 'N/A')}\n"
                    if "parent_idea" in a
                    else ""
                )
                + (
                    f"Parent code:\n```python\n{a['parent_code']}\n```\n"
                    if "parent_code" in a
                    else ""
                )
                + (
                    f"Second parent idea: {a.get('parent2_idea', 'N/A')}\n"
                    if "parent2_idea" in a
                    else ""
                )
                + (
                    f"Second parent code:\n```python\n{a['parent2_code']}\n```\n"
                    if "parent2_code" in a
                    else ""
                )
                + (
                    f"Synergy score: {a.get('synergy_score', 'N/A')}\n"
                    if "synergy_score" in a
                    else ""
                )
                + (f"Strategy hint: {a.get('hint', '')}\n" if "hint" in a else "")
            )
        slots_block = "\n".join(slot_parts)

        n_total = len(actions)
        user_msg = (
            f"Action: EVOLUTION (fill {n_total} pruned slots in one batch)\n\n"
            f"## Existing high-performing operators (for context)\n{ref_block}\n\n"
            f"## Slots to fill\n{slots_block}\n\n"
            f"Return a JSON list of exactly {n_total} operator dicts "
            f"(in the same order as the slots above):\n"
            f'[{{"type": "...", "idea": "...", "code": "def destroy/repair(...): ..."}}, ...]\n'
        )
        raw = self._throttled_chat(user_msg)
        return self._parse_response(raw)

    def regenerate_with_error(
        self, failed_spec: OperatorSpec, error_msg: str
    ) -> OperatorBatch:
        """Retry a single operator that failed sanity check."""
        user_msg = (
            f"Action: FIX OPERATOR (retry after failure)\n\n"
            f"The following {failed_spec.type} operator failed validation:\n"
            f"Idea: {failed_spec.idea}\n"
            f"```python\n{failed_spec.code}\n```\n\n"
            f"Error: {error_msg}\n\n"
            f"Fix the operator and return a JSON list with exactly 1 corrected operator dict:\n"
            f'[{{"type": "{failed_spec.type}", "idea": "...", "code": "def {failed_spec.type}(...): ..."}}]\n'
        )
        raw = self._throttled_chat(user_msg)
        return self._parse_response(raw)
