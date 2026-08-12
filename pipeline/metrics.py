"""Run instrumentation (PRD §8).

``run_id`` is **derived from the date**, not from wall-clock time, so re-running
the same date lands in the same run directory and produces byte-identical
content. Idempotency (PRD §9) is worth more here than a unique timestamp.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from . import paths
from .models import Metrics


def run_id_for(d: date | str | None = None) -> str:
    if d is None:
        return "run_" + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"run_{d}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


class Run:
    """A run directory plus its metrics.json. Loaded and re-saved by each stage."""

    def __init__(self, run_id: str, d: Optional[date] = None):
        self.run_id = run_id
        self.dir = paths.run_dir(run_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "raw").mkdir(exist_ok=True)
        self.metrics = self._load(d)

    @classmethod
    def for_date(cls, d: date | str | None) -> "Run":
        dd = date.fromisoformat(d) if isinstance(d, str) else d
        return cls(run_id_for(dd), dd)

    @property
    def metrics_path(self) -> Path:
        return self.dir / "metrics.json"

    @property
    def raw_dir(self) -> Path:
        return self.dir / "raw"

    def _load(self, d: Optional[date]) -> Metrics:
        if self.metrics_path.exists():
            try:
                m = Metrics.model_validate_json(
                    self.metrics_path.read_text(encoding="utf-8")
                )
                if d and not m.date:
                    m.date = d
                return m
            except Exception:
                pass
        return Metrics(run_id=self.run_id, date=d)

    def save(self) -> None:
        self.metrics_path.write_text(
            json.dumps(
                self.metrics.model_dump(mode="json", by_alias=True),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    # -- accumulators ----------------------------------------------------

    def count(self, key: str, value: int) -> None:
        setattr(self.metrics.counts, key, value)

    def add_cost(self, kind: str, usd: float) -> None:
        setattr(self.metrics.cost, kind, getattr(self.metrics.cost, kind) + usd)
        self.metrics.cost.total_usd = round(
            self.metrics.cost.embedding_usd
            + self.metrics.cost.llm_usd
            + self.metrics.cost.openalex_usd,
            6,
        )

    def add_tokens(self, tin: int, tout: int) -> None:
        self.metrics.tokens.input += tin
        self.metrics.tokens.output += tout

    def stage(self, name: str, status: str) -> None:
        self.metrics.stages[name] = status

    def error(self, msg: str) -> None:
        if msg not in self.metrics.errors:
            self.metrics.errors.append(msg)

    def write_raw(self, name: str, text: str) -> Path:
        """Raw API responses are preserved verbatim (PRD §5.1, non-negotiable)."""
        p = self.raw_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
        return p

    def append_jsonl(self, name: str, obj: Any) -> None:
        with (self.dir / name).open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")

    @contextmanager
    def timed(self, key: str) -> Iterator[None]:
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.metrics.timing[key] = round(
                self.metrics.timing.get(key, 0.0) + time.monotonic() - t0, 2
            )


class BudgetExceeded(RuntimeError):
    """Raised when a stage hits its share of the OpenAlex daily budget."""


class OpenAlexBudget:
    """Accumulates ``meta.cost_usd`` and stops the stage at 80% of the daily cap."""

    def __init__(self, daily_usd: float = 1.0, stop_fraction: float = 0.8):
        self.daily_usd = daily_usd
        self.stop_fraction = stop_fraction
        self.spent = 0.0
        self.calls = 0

    @property
    def limit(self) -> float:
        return self.daily_usd * self.stop_fraction

    def charge(self, usd: float) -> None:
        self.spent += usd or 0.0
        self.calls += 1
        if self.spent >= self.limit:
            raise BudgetExceeded(
                f"OpenAlex spend ${self.spent:.4f} reached {self.stop_fraction:.0%} "
                f"of the ${self.daily_usd:.2f} daily budget after {self.calls} calls"
            )
