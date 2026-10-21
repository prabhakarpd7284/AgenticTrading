"""In-sample / out-of-sample split and the OOS guard.

OOS is touched ONCE, by the OOS-validator role, with an explicit
``allow_oos=True`` (CLI: ``--allow-oos``). Everything else must stay inside IS.
The guard is enforced in three places: ``resolve_period`` (here), the
``CandleSource`` (refuses any query past its ``max_date``) and the trading
calendar (never reads sessions past ``max_date``).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

DATA_START = date(2026, 4, 20)
IS_START = date(2026, 4, 20)
IS_END = date(2026, 8, 31)
OOS_START = date(2026, 9, 1)
OOS_END = date(2026, 10, 19)


class OOSAccessError(RuntimeError):
    """Raised when research code tries to read out-of-sample dates without --allow-oos."""


@dataclass(frozen=True)
class Period:
    name: str
    start: date
    end: date
    allow_oos: bool = False

    @property
    def touches_oos(self) -> bool:
        return self.end > IS_END

    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end


def check_date(d: date, allow_oos: bool) -> None:
    if d > IS_END and not allow_oos:
        raise OOSAccessError(
            f"{d.isoformat()} is out-of-sample (> {IS_END.isoformat()}). "
            "Only the OOS-validator role may read it, with --allow-oos.")


def resolve_period(name: str = "is", start: date | None = None, end: date | None = None,
                   allow_oos: bool = False) -> Period:
    """``name`` is ``is`` | ``oos`` | ``all``. ``start``/``end`` narrow it (never widen
    an IS run into OOS: any end past IS_END needs ``allow_oos``)."""
    name = (name or "is").lower()
    if name == "is":
        s, e = IS_START, IS_END
    elif name == "oos":
        s, e = OOS_START, OOS_END
    elif name == "all":
        s, e = IS_START, OOS_END
    else:
        raise ValueError(f"unknown period {name!r} (is | oos | all)")
    s = start or s
    e = end or e
    if e < s:
        raise ValueError(f"end {e} before start {s}")
    if name == "is" and e > IS_END:
        raise OOSAccessError(f"--period is cannot end after {IS_END} (got {e}); "
                             "use --period oos --allow-oos")
    check_date(e, allow_oos)
    return Period(name=name, start=s, end=e, allow_oos=allow_oos)
