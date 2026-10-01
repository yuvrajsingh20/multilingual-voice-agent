"""Small, dependency-free statistics used by every Track 1 scorer."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """95% Wilson score interval for a proportion. ``(None, None)`` when n == 0."""
    if n == 0:
        return None, None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def rate(flags: Sequence[bool]) -> dict[str, float | int | None]:
    n = len(flags)
    k = sum(1 for f in flags if f)
    low, high = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": [low, high]}


def mcnemar_exact(b: int, c: int) -> float | None:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return round(min(1.0, 2 * tail), 4)


def cohen_kappa(a: Sequence[object], b: Sequence[object]) -> float | None:
    """Cohen's kappa for two raters over the same items. None if undefined."""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    if expected == 1.0:
        return None
    return round((observed - expected) / (1 - expected), 4)


def weighted_kappa(a: Sequence[int], b: Sequence[int], categories: Sequence[int]) -> float | None:
    """Quadratically weighted kappa for ordinal ratings (the 1-5 register rubric)."""
    if len(a) != len(b) or not a:
        return None
    k = len(categories)
    index = {c: i for i, c in enumerate(categories)}
    n = len(a)
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        observed[index[x]][index[y]] += 1
    ra = [sum(row) for row in observed]
    rb = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            num += w * observed[i][j]
            den += w * ra[i] * rb[j] / n
    if den == 0:
        return None
    return round(1 - num / den, 4)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]."""
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    if len(data) == 1:
        return round(data[0], 1)
    pos = (len(data) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    value = data[lo] + (data[hi] - data[lo]) * (pos - lo)
    return round(value, 1)


def latency_summary(values: Sequence[float | None]) -> dict[str, float | int | None]:
    data = [v for v in values if v is not None]
    return {
        "n": len(data),
        "p50": percentile(data, 50),
        "p95": percentile(data, 95),
        "p99": percentile(data, 99),
        "max": round(max(data), 1) if data else None,
    }
