"""Explicit, injectable clock.

Calling-hour compliance depends on the *customer's* local time, not on whatever
timezone the host machine happens to be configured with. Every component that
needs "now" takes a :class:`Clock` so that tests state the instant explicitly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol
from zoneinfo import ZoneInfo


class Clock(Protocol):
    """Source of the current instant in a known timezone."""

    @property
    def tzinfo(self) -> ZoneInfo: ...

    def now(self) -> datetime:
        """Return a timezone-aware ``datetime`` in :attr:`tzinfo`."""


class SystemClock:
    """Real time, converted into a configured IANA zone."""

    def __init__(self, tz_name: str) -> None:
        self._tz = ZoneInfo(tz_name)

    @property
    def tzinfo(self) -> ZoneInfo:
        return self._tz

    def now(self) -> datetime:
        return datetime.now(timezone.utc).astimezone(self._tz)


class FixedClock:
    """Deterministic clock for tests and replay.

    The supplied instant must be timezone-aware; a naive datetime is rejected
    rather than silently assumed to be UTC.
    """

    def __init__(self, instant: datetime) -> None:
        if instant.tzinfo is None:
            raise ValueError("FixedClock requires a timezone-aware datetime")
        self._instant = instant

    @property
    def tzinfo(self) -> ZoneInfo:
        tz = self._instant.tzinfo
        if isinstance(tz, ZoneInfo):
            return tz
        raise TypeError("FixedClock was built with a non-ZoneInfo tzinfo")

    def now(self) -> datetime:
        return self._instant
