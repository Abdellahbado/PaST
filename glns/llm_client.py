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
from dataclasses import dataclass
from typing import List, Optional

from groq import Groq

from glns.config import LLMConfig
from glns.schemas import OperatorBatch, OperatorRecord, OperatorSpec

logger = logging.getLogger(__name__)


@dataclass
class GroqRateLimitError(RuntimeError):
    """Raised when the Groq API rate-limits the request.

    retry_after_sec is a best-effort parse from Groq's error message.
    """

    message: str
    retry_after_sec: float = 0.0

    def __str__(self) -> str:  # pragma: no cover
        if self.retry_after_sec > 0:
            return f"{self.message} (retry_after_sec={self.retry_after_sec:.1f})"
        return self.message


def _parse_retry_after_seconds(msg: str) -> float:
    """Extract wait time from Groq's error message, if present."""
    # Example: "Please try again in 3m48.672s"
    m = re.search(r"Please try again in\s+(\d+)m([0-9.]+)s", msg)
    if m:
        return 60.0 * float(m.group(1)) + float(m.group(2))
    m = re.search(r"Please try again in\s+([0-9.]+)s", msg)
    if m:
        return float(m.group(1))
    return 0.0


def _looks_like_rate_limit(exc: Exception) -> bool:
    s = str(exc)
    return ("Error code: 429" in s) or ("rate_limit" in s.lower())


def _parse_api_keys(api_key: Optional[str]) -> List[str]:
    if api_key:
        return [api_key.strip()]
    multi = os.environ.get("GROQ_API_KEYS", "").strip()
    if multi:
        # Allow comma / newline separation.
        parts = re.split(r"[\s,]+", multi)
        keys = [p.strip() for p in parts if p.strip()]
        return keys
    single = os.environ.get("GROQ_API_KEY", "").strip()
    return [single] if single else []


def _truncate_code(code: str, *, max_lines: int, max_chars: int) -> str:
    if not code:
        return code
    lines = code.splitlines()
    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[:max_lines] + [
            f"# ... truncated ({len(code.splitlines())-max_lines} more lines) ..."
        ]
    out = "\n".join(lines)
    if max_chars > 0 and len(out) > max_chars:
        out = (
            out[:max_chars] + f"\n# ... truncated ({len(out)-max_chars} more chars) ..."
        )
    return out


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
- **A separate optimal DP solver handles job ordering and start-time scheduling on each machine.**
  **Your operators ONLY control ASSIGNMENT: which machine gets which job.**

## Architecture (important!)
The system uses a decomposed approach:
1. YOUR operators decide the machine assignment (which jobs go to which machine).
2. An OPTIMAL dynamic programming solver then determines the best job ordering
   and start times on each machine — you do NOT need to handle sequencing.
This means: a good assignment = good solution. Focus all your intelligence on
balancing load, energy cost, and makespan through smart assignment decisions.

## Solution representation
A solution is `assignment: List[int]` of length n.
  assignment[j] = index of the machine (0 to m-1) that job j is assigned to.
Every job index 0..n-1 has exactly one machine assignment.

## Instance dict (available as `instance` argument)
{
  "m": int,               # number of machines
  "n": int,               # number of jobs
  "T": int,               # time horizon length
  "p": List[int],         # processing times, length n (integers, can be 1..12)
  "e": List[int],         # machine energy rates, length m
  "ct": List[int],        # per-slot TOU prices, length T
}

## Operator interface contract
### Destroy operator
```python
def destroy(assignment: List[int],
            destroy_cnt: int,
            instance: dict,
            rng) -> tuple[list[int], list[int]]:
    \"\"\"Unassign destroy_cnt jobs from the assignment.
    Args:
        assignment: current assignment vector (length n, values 0..m-1).
        destroy_cnt: number of jobs to unassign.
        instance: problem data dict.
        rng: random.Random instance for stochastic decisions.
    Returns:
        (removed_jobs, partial_assignment)
        - removed_jobs: list of unassigned job indices.
        - partial_assignment: copy of assignment with those jobs set to -1.
    \"\"\"
```

### Repair operator
```python
def repair(partial_assignment: List[int],
           removed_jobs: List[int],
           instance: dict,
           rng) -> list[int]:
    \"\"\"Assign all removed jobs to machines.
    Args:
        partial_assignment: assignment with some entries = -1 (unassigned).
        removed_jobs: list of job indices to assign.
        instance: problem data dict.
        rng: random.Random instance.
    Returns:
        Complete assignment (length n, every value in 0..m-1, no -1).
    \"\"\"
```

## Rules
1. Each operator is a SINGLE self-contained Python function. ALL helper functions and imports go INSIDE.
2. Only stdlib + `math` + `random` may be used (no numpy/scipy/torch).
3. ALWAYS include `import random` inside the function (and `import math` if you use math).
4. Return ONLY valid JSON — no markdown fences, no prose outside the JSON.
5. Keep operators efficient: O(n*m) or O(n^2) at most; avoid exponential search.
6. Keep the "idea" field SHORT (1-2 sentences, <120 words). Put detailed reasoning in code comments instead.

## Benchmark characteristics (Wang2018)
The benchmark has 90 instances across 3 scales — your operators must work well on ALL:
- **Small** (1-30): n=6-25 jobs, m=3-7 machines, T=50-80, processing times p ∈ {1..5}, K=3-5 distinct p values
- **Medium** (31-60): n=30-200 jobs, m=8-25 machines, T=100-300, p ∈ {1..4}, K=3-4
- **Large** (61-90): n=250-500 jobs, m=25-40 machines, T=350-500, p ∈ {1..12}, K=12
Key characteristics:
- Processing times are SHORT (often 1-5 slots) and HIGHLY VARIABLE within an instance.
- Many jobs with p=1 exist — these are trivially scheduled but their assignment still matters for load balance.
- Energy rates e vary across machines (some machines are 2-3x more expensive).
- TOU prices ct fluctuate significantly (peaks can be 4-5x valleys).
- The DP solver handles sequencing optimally, so your repair operator's main lever is:
  which machine to assign each job to, considering load balance AND energy cost.

## Domain hints (idea library for assignment operators)
Destroy ideas: random removal, worst-energy-contributor removal (remove jobs from expensive machines),
most-loaded-machine removal (remove from bottleneck), peak-period job removal (remove large jobs
that likely land in expensive time slots), energy-imbalance removal (swap between high/low-e machines),
load-variance removal (remove from machines with extreme loads), processing-time-class removal
(remove all jobs of a specific p value to enable rebalancing).
Repair ideas: greedy load-balance (assign to least-loaded machine by total processing time),
energy-aware assignment (assign to cheapest machine, accounting for e_h and likely slot costs),
hybrid score (assign to machine minimising w*load_increase + (1-w)*energy_estimate with random w),
makespan-aware (assign large-p jobs first to balance load, small-p jobs to cheap machines),
random diversification (pure random assignment for exploration),
regret-based (assign job to machine with largest gap between best and 2nd-best assignment cost).
For bi-objective balance: sometimes focus destroy on Cmax-sensitive parts (bottleneck machine),
sometimes on TEC-sensitive parts (jobs on expensive machines), sometimes both.
"""

# System prompt for sequence-based mode (legacy, kept for backward compatibility)
SYSTEM_PROMPT_SEQUENCE = SYSTEM_PROMPT  # TODO: keep old prompt if needed


# ---------------------------------------------------------------------------
# Client wrapper
# ---------------------------------------------------------------------------


class GroqOperatorClient:
    """Thin wrapper over the Groq SDK for batched operator generation."""

    def __init__(self, cfg: LLMConfig, api_key: Optional[str] = None) -> None:
        self.cfg = cfg
        keys = _parse_api_keys(api_key)
        keys = [k for k in keys if k]
        if not keys:
            raise RuntimeError(
                "GROQ_API_KEY not set.  Pass it via constructor or env var."
            )

        self._keys = keys
        self._clients = [
            Groq(api_key=k, max_retries=cfg.max_retries, timeout=cfg.timeout_sec)
            for k in keys
        ]
        # Per-key cooldown when Groq tells us to wait (TPD/RPM/etc).
        self._cooldown_until = [0.0 for _ in keys]  # monotonic timestamps
        self._rr_idx = 0

        self._last_call_ts: float = 0.0

    def _pick_client_index(self) -> int:
        now = time.monotonic()
        n = len(self._clients)
        for k in range(n):
            idx = (self._rr_idx + k) % n
            if now >= self._cooldown_until[idx]:
                self._rr_idx = (idx + 1) % n
                return idx
        # None available.
        soonest = min(self._cooldown_until)
        wait = max(0.0, soonest - now)
        raise GroqRateLimitError(
            "All configured Groq API keys are rate-limited", retry_after_sec=wait
        )

    # ----- low-level call -------------------------------------------------

    def _throttled_chat(self, user_message: str) -> str:
        """Call the Groq chat API with client-side rate limiting."""
        elapsed = time.monotonic() - self._last_call_ts
        wait = self.cfg.min_call_interval_sec - elapsed
        if wait > 0:
            logger.debug("LLM throttle: sleeping %.2fs", wait)
            time.sleep(wait)

        logger.info(
            "LLM call: model=%s temp=%.2f max_tokens=%d user_chars=%d keys=%d",
            self.cfg.model,
            self.cfg.temperature,
            self.cfg.max_tokens,
            len(user_message),
            len(self._clients),
        )
        last_exc: Optional[Exception] = None
        # Try each key at most once.
        for _attempt in range(len(self._clients)):
            idx = self._pick_client_index()
            try:
                logger.debug("LLM using key[%d]", idx)
                resp = self._clients[idx].chat.completions.create(
                    model=self.cfg.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                )
                break
            except Exception as exc:  # SDK raises a variety of exception types
                last_exc = exc
                if _looks_like_rate_limit(exc):
                    retry_after = _parse_retry_after_seconds(str(exc))
                    if retry_after <= 0:
                        retry_after = 60.0
                    self._cooldown_until[idx] = time.monotonic() + retry_after
                    logger.warning(
                        "Groq rate-limited key[%d]; cooling down %.1fs",
                        idx,
                        retry_after,
                    )
                    continue
                raise
        else:
            raise GroqRateLimitError(
                message=str(last_exc) if last_exc else "Groq rate-limited",
                retry_after_sec=(
                    _parse_retry_after_seconds(str(last_exc)) if last_exc else 0.0
                ),
            )

        self._last_call_ts = time.monotonic()
        content = resp.choices[0].message.content or ""
        logger.info("LLM response: chars=%d", len(content))
        return content

    # ----- JSON parsing & validation --------------------------------------

    @staticmethod
    def _extract_json(raw: str) -> list:
        """Extract a JSON array from possibly markdown-fenced LLM output.

        Includes a fallback that salvages individually-complete JSON objects
        when the response was truncated mid-array (common when max_tokens is
        tight).  This avoids falling back to seed operators for an entire
        generation just because the last operator was cut off.
        """
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

        # Try outermost [ ... ].
        m = re.search(r"\[.*]", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        # --- Truncation salvage -------------------------------------------
        # If we reach here the JSON array is likely truncated (max_tokens hit).
        # Try to recover individual complete {...} objects from the text.
        salvaged: list = []
        depth = 0
        start = -1
        for i, ch in enumerate(raw):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    candidate = raw[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and "code" in obj and "type" in obj:
                            salvaged.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = -1
        if salvaged:
            logger.warning(
                "JSON array was truncated; salvaged %d/%s complete operator(s)",
                len(salvaged),
                "?",
            )
            return salvaged

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
        search_context: Optional[str] = None,
    ) -> OperatorBatch:
        """One batched LLM call per generation to fill ALL pruned slots.

        Args:
            actions: list of dicts, each describing one needed operator, e.g.
                {"action": "mutation_logic", "type": "destroy",
                 "parent_code": "...", "parent_idea": "..."}
            reference_ops: top-performing operators shown as in-context examples
                (max 3 to stay within token budget).
        """
        # Build reference block (top performers; truncated for prompt budget).
        ref_block_parts: list[str] = []
        n_ref = max(0, int(getattr(self.cfg, "n_reference_ops", 2)))
        for i, op in enumerate(reference_ops[:n_ref]):
            code = _truncate_code(
                op.code,
                max_lines=int(getattr(self.cfg, "prompt_max_code_lines", 120)),
                max_chars=int(getattr(self.cfg, "prompt_max_code_chars", 3500)),
            )
            ref_block_parts.append(
                f"### Reference operator {i+1} ({op.op_type})\n"
                f"Idea: {op.idea}\n"
                f"Code:\n{code}"
            )
        ref_block = "\n\n".join(ref_block_parts) if ref_block_parts else "None."

        # Build per-slot action descriptions.
        slot_parts: list[str] = []
        max_lines = int(getattr(self.cfg, "prompt_max_code_lines", 120))
        max_chars = int(getattr(self.cfg, "prompt_max_code_chars", 3500))

        for idx, a in enumerate(actions):
            parent_code = (
                _truncate_code(
                    a["parent_code"], max_lines=max_lines, max_chars=max_chars
                )
                if "parent_code" in a and isinstance(a.get("parent_code"), str)
                else None
            )
            parent2_code = (
                _truncate_code(
                    a["parent2_code"], max_lines=max_lines, max_chars=max_chars
                )
                if "parent2_code" in a and isinstance(a.get("parent2_code"), str)
                else None
            )
            slot_parts.append(
                f"--- Slot {idx+1} ---\n"
                f"Action: {a['action']}\n"
                f"Operator type needed: {a['type']}\n"
                + (
                    f"Parent idea: {a.get('parent_idea', 'N/A')}\n"
                    if "parent_idea" in a
                    else ""
                )
                + (f"Parent code:\n{parent_code}\n" if parent_code else "")
                + (
                    f"Second parent idea: {a.get('parent2_idea', 'N/A')}\n"
                    if "parent2_idea" in a
                    else ""
                )
                + (f"Second parent code:\n{parent2_code}\n" if parent2_code else "")
                + (
                    f"Synergy score: {a.get('synergy_score', 'N/A')}\n"
                    if "synergy_score" in a
                    else ""
                )
                + (f"Strategy hint: {a.get('hint', '')}\n" if "hint" in a else "")
            )
        slots_block = "\n".join(slot_parts)

        ctx_block = ""
        if getattr(self.cfg, "include_search_context", True) and search_context:
            max_chars = int(getattr(self.cfg, "search_context_max_chars", 900))
            ctx = str(search_context).strip()
            if max_chars > 0 and len(ctx) > max_chars:
                ctx = ctx[:max_chars] + "..."
            ctx_block = f"## Search context (concise)\n{ctx}\n\n"

        n_total = len(actions)
        user_msg = (
            f"Action: EVOLUTION (fill {n_total} pruned slots in one batch)\n\n"
            + ctx_block
            + f"## Existing high-performing operators (for context)\n{ref_block}\n\n"
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
        code = _truncate_code(
            failed_spec.code,
            max_lines=int(getattr(self.cfg, "prompt_max_code_lines", 120)),
            max_chars=int(getattr(self.cfg, "prompt_max_code_chars", 3500)),
        )
        user_msg = (
            f"Action: FIX OPERATOR (retry after failure)\n\n"
            f"The following {failed_spec.type} operator failed validation:\n"
            f"Idea: {failed_spec.idea}\n"
            f"Code:\n{code}\n\n"
            f"Error: {error_msg}\n\n"
            f"Fix the operator and return a JSON list with exactly 1 corrected operator dict:\n"
            f'[{{"type": "{failed_spec.type}", "idea": "...", "code": "def {failed_spec.type}(...): ..."}}]\n'
        )
        raw = self._throttled_chat(user_msg)
        return self._parse_response(raw)
