"""Dataset and inference feature contract."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"
PCA_COLUMNS = tuple(f"V{i}" for i in range(1, 29))
FEATURE_COLUMNS = (TIME_COLUMN, *PCA_COLUMNS, AMOUNT_COLUMN)
LABEL_COLUMN = "Class"
TRAINING_COLUMNS = (*FEATURE_COLUMNS, LABEL_COLUMN)


@dataclass(frozen=True)
class SchemaCheck:
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.unexpected

    def message(self) -> str:
        parts: list[str] = []
        if self.missing:
            parts.append(f"missing columns: {', '.join(self.missing)}")
        if self.unexpected:
            parts.append(f"unexpected columns: {', '.join(self.unexpected)}")
        return "; ".join(parts) if parts else "schema ok"


def check_columns(columns: Iterable[str], *, include_label: bool) -> SchemaCheck:
    expected = set(TRAINING_COLUMNS if include_label else FEATURE_COLUMNS)
    seen = set(columns)
    return SchemaCheck(
        missing=tuple(sorted(expected - seen)),
        unexpected=tuple(sorted(seen - expected)),
    )


def coerce_transaction(payload: Mapping[str, Any], *, allow_label: bool = False) -> dict[str, float]:
    columns = set(payload.keys())
    if allow_label:
        columns.discard(LABEL_COLUMN)
    check = check_columns(columns, include_label=False)
    if not check.ok:
        raise ValueError(check.message())

    values: dict[str, float] = {}
    for column in FEATURE_COLUMNS:
        raw = payload[column]
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must be numeric") from exc
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{column} must be finite")
        values[column] = value
    return values


def ordered_feature_vector(transaction: Mapping[str, float]) -> list[float]:
    return [float(transaction[column]) for column in FEATURE_COLUMNS]
