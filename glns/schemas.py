"""Pydantic schemas for LLM-generated operator specifications.

Enforces correct structure and provides compile-time validation of
the code returned by the LLM.
"""

from __future__ import annotations

import ast
import re
import textwrap
import uuid
from typing import Callable, List, Literal, Optional

from pydantic import BaseModel, Field, PrivateAttr, field_validator


# ---------------------------------------------------------------------------
# LLM response schemas
# ---------------------------------------------------------------------------


class OperatorSpec(BaseModel):
    """One destroy or repair operator as returned by the LLM."""

    type: Literal["destroy", "repair"]
    idea: str = Field(..., min_length=3, max_length=1200)
    code: str = Field(..., min_length=20)

    @field_validator("code")
    @classmethod
    def code_must_parse(cls, v: str) -> str:
        """Ensure the code is syntactically valid Python."""
        try:
            ast.parse(v)
        except SyntaxError as exc:
            raise ValueError(f"Code has a syntax error: {exc}") from exc
        return v

    @field_validator("code")
    @classmethod
    def code_has_correct_signature(cls, v: str, info) -> str:
        """Check that the function signature matches the expected contract."""
        op_type = info.data.get("type")
        if op_type == "destroy":
            pattern = r"def\s+destroy\s*\("
        elif op_type == "repair":
            pattern = r"def\s+repair\s*\("
        else:
            return v  # can't validate without knowing type yet
        if not re.search(pattern, v):
            expected = "destroy" if op_type == "destroy" else "repair"
            raise ValueError(
                f"Code must define a top-level function named '{expected}'. "
                f"Expected signature: def {expected}(..."
            )
        return v


class OperatorBatch(BaseModel):
    """A batch of operators returned by one LLM call."""

    operators: List[OperatorSpec] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Internal bookkeeping record (not sent to/from the LLM)
# ---------------------------------------------------------------------------


class OperatorRecord(BaseModel):
    """Wraps an OperatorSpec with runtime metadata.

    The `id` is stable across pruning/insertion cycles so that the synergy
    matrix can use (destroy_id, repair_id) keys instead of positional indices.
    """

    model_config = {"arbitrary_types_allowed": True}

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    spec: OperatorSpec
    generation_born: int = 0
    # Compiled function reference is NOT serialized; stored via PrivateAttr.
    _fn: Optional[Callable] = PrivateAttr(default=None)

    # ------ helpers ---------------------------------------------------

    @property
    def op_type(self) -> str:
        return self.spec.type

    @property
    def idea(self) -> str:
        return self.spec.idea

    @property
    def code(self) -> str:
        return self.spec.code

    @property
    def fn(self) -> Optional[Callable]:
        return self._fn

    @fn.setter
    def fn(self, value: Callable) -> None:
        self._fn = value

    def short_label(self) -> str:
        return f"{self.op_type}:{self.id[:6]}|{self.idea[:40]}"
