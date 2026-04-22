"""Grounding checks for LLM-generated case briefs."""

from __future__ import annotations

import re
from dataclasses import dataclass

PROHIBITED_PHRASES = (
    "the llm detected fraud",
    "the language model detected fraud",
    "fraud is proven",
    "definitely fraud",
    "guaranteed fraud",
)


@dataclass(frozen=True)
class GroundingResult:
    ok: bool
    reasons: tuple[str, ...]


def validate_brief(brief: str, context_count: int) -> GroundingResult:
    reasons: list[str] = []
    if not brief.strip():
        reasons.append("brief is empty")
    if context_count > 0:
        cited = {int(match) for match in re.findall(r"\[(\d+)\]", brief)}
        valid = set(range(1, context_count + 1))
        if not cited:
            reasons.append("brief has no citations")
        elif not cited.issubset(valid):
            reasons.append("brief cites context outside retrieved policy documents")
    lowered = brief.lower()
    for phrase in PROHIBITED_PHRASES:
        if phrase in lowered:
            reasons.append(f"brief contains prohibited claim: {phrase}")
    return GroundingResult(ok=not reasons, reasons=tuple(reasons))

