from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Literal

import pandas as pd
import numpy as np
import re
import warnings


# -------------------------
# Data models
# -------------------------

FeatureKind = Literal["numeric", "categorical", "datetime", "id"]
FeatureRole = Literal["feature", "target", "ignored"]


@dataclass(frozen=True)
class FeatureStats:
    missing_pct: float
    unique_count: int
    sample_values: list[Any]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Feature:
    name: str
    kind: FeatureKind
    role: FeatureRole
    stats: FeatureStats
    flags: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "role": self.role,
            "stats": self.stats.to_dict(),
            "flags": sorted(self.flags),
        }


@dataclass(frozen=True)
class Schema:
    features: list[Feature]
    n_rows: int
    inferred_at: str
    version: str = "v1"

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "inferred_at": self.inferred_at,
            "n_rows": self.n_rows,
            "features": [f.to_dict() for f in self.features],
        }


# -------------------------
# Inference helpers
# -------------------------

_ID_NAME_REGEX = re.compile(
    r"(?:^|_)(id|uuid|guid|hash|user_id|txn_id)(?:$|_)",
    re.IGNORECASE,
)

_TIME_NAME_HINTS = re.compile(
    r"(date|time|timestamp|created|joined|signup)",
    re.IGNORECASE,
)


def _is_id_like_name(col: str) -> bool:
    return bool(_ID_NAME_REGEX.search(col))


def _looks_like_hash(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    return all(len(v) >= 16 and re.fullmatch(r"[a-fA-F0-9]+", v) for v in sample)


import warnings

def _can_parse_datetime(
    series: pd.Series,
    col_name: str,
    threshold: float = 0.8,
) -> bool:
    """
    Conservative datetime detection.
    - Never infer from numeric dtypes
    - Require name hint OR very strong parse signal
    - Suppress pandas inference warnings
    """

    # Never infer datetime from numeric columns
    if pd.api.types.is_numeric_dtype(series):
        return False

    non_null = series.dropna()
    if non_null.empty:
        return False

    # Suppress noisy pandas warnings during probing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(non_null, errors="coerce")

    success_ratio = parsed.notna().mean()

    if success_ratio < threshold:
        return False

    # Prevent constant / fake timestamps
    if parsed.nunique() <= 1:
        return False

    # Require name hint unless parsing is extremely strong
    has_name_hint = bool(_TIME_NAME_HINTS.search(col_name))

    return has_name_hint or success_ratio >= 0.95



def _numeric_as_categorical(series: pd.Series, n_rows: int) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False

    non_null = series.dropna()
    if non_null.empty:
        return False

    unique_count = non_null.nunique()
    max_unique = min(20, int(np.sqrt(n_rows)))

    if unique_count > max_unique:
        return False

    # must be integer-like
    if not np.all(np.equal(np.mod(non_null, 1), 0)):
        return False

    return True


# -------------------------
# Core inference function
# -------------------------

def infer_schema(
    df: pd.DataFrame,
    target: str | None = None,
) -> Schema:
    """
    Infer schema from a pandas DataFrame.
    Deterministic and side-effect free.
    """

    n_rows = len(df)
    features: list[Feature] = []

    for col in sorted(df.columns):
        series = df[col]

        missing_pct = float(series.isna().mean())
        unique_count = int(series.nunique(dropna=True))
        sample_values = series.dropna().head(5).tolist()

        flags: set[str] = set()

        # -------------------------
        # Role pre-check
        # -------------------------
        is_target = target is not None and col == target

        # -------------------------
        # ID detection (highest priority)
        # -------------------------
        uniqueness_ratio = unique_count / max(n_rows, 1)

        if (
            uniqueness_ratio >= 0.95
            and not is_target
            and (_is_id_like_name(col) or _looks_like_hash(series))
        ):
            kind: FeatureKind = "id"
            flags.add("id_like")

        # -------------------------
        # Datetime detection (conservative)
        # -------------------------
        elif (
            not is_target
            and (
                pd.api.types.is_datetime64_any_dtype(series)
                or _can_parse_datetime(series, col)
            )
        ):
            kind = "datetime"
            flags.add("potential_time")

        # -------------------------
        # Numeric vs categorical
        # -------------------------
        elif pd.api.types.is_numeric_dtype(series):
            if _numeric_as_categorical(series, n_rows):
                kind = "categorical"
            else:
                kind = "numeric"

        else:
            kind = "categorical"

        # -------------------------
        # Flags
        # -------------------------
        if unique_count == 1:
            flags.add("constant")

        if kind == "categorical" and unique_count > 50:
            flags.add("high_cardinality")

        if kind == "numeric" and series.dropna().std() == 0:
            flags.add("low_variance")

        # -------------------------
        # Role assignment
        # -------------------------
        if is_target:
            role: FeatureRole = "target"
        elif kind == "id" or "constant" in flags:
            role = "ignored"
        else:
            role = "feature"

        features.append(
            Feature(
                name=col,
                kind=kind,
                role=role,
                stats=FeatureStats(
                    missing_pct=missing_pct,
                    unique_count=unique_count,
                    sample_values=sample_values,
                ),
                flags=flags,
            )
        )

    return Schema(
        features=features,
        n_rows=n_rows,
        inferred_at=datetime.utcnow().isoformat(),
    )
