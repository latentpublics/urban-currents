"""Telling someone when nobody is watching (phase 0k, X7).

The real output of automation is not the issue — it is knowing whether there was
one. Three things live here:

**The failure alert.** A `not_published` day mails `UC_ALERT_RECIPIENT`, and the
subject carries the date and the reason, because the decision "do I need to look
at this now" should be answerable from a notification list without opening
anything.

**The escalation.** Five identical alerts carry less information than one. A run
of consecutive failures says so in the subject and stops repeating the same
sentence — the second day of an outage is a different fact from the first.

**The weekly summary.** One mail: what happened on each of seven days, what it
cost, how many sends went out.

Alerts are best-effort by construction. A failure to notify is recorded in the
run log and never propagates: the pipeline's job is the issue, and an unreachable
mail server must not turn a published day into a failed one.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any, Optional

from . import paths
from .config import cfg
from .outcome import NOT_PUBLISHED, PUBLISHED, QUIET, all_logs, load_log


def alert_recipients() -> list[str]:
    """Where alerts go. Never the reader list — this is operational mail."""
    value = os.environ.get("UC_ALERT_RECIPIENT", "").strip()
    return [value] if value else []


def consecutive_failures(upto: date, limit: int = 30) -> int:
    """How many days in a row ending at `upto` we could not see."""
    logs = {row["date"]: row for row in all_logs()}
    streak = 0
    day = upto
    for _ in range(limit):
        row = logs.get(str(day))
        if not row or row.get("status") != NOT_PUBLISHED:
            break
        streak += 1
        day -= timedelta(days=1)
    return streak


def failure_subject(d: date, reasons: list[str], streak: int) -> str:
    """Readable without opening the mail.

    The date and the cause come first because that is what decides whether
    someone stops what they are doing. The streak is in the subject too — the
    fourth morning of an outage should not look like the first.
    """
    reason = (reasons[0] if reasons else "unknown reason").rstrip(".")
    if streak >= 2:
        return f"Urban Currents: no issue for {streak} days — {d}, {reason}"
    return f"Urban Currents: no issue {d} — {reason}"


def failure_body(d: date, reasons: list[str], streak: int) -> str:
    lines = [
        f"No issue was published for {d}.",
        "",
        "Why:",
    ]
    lines += [f"  - {r}" for r in (reasons or ["unknown"])]
    if streak >= 2:
        lines += [
            "",
            f"This is day {streak} in a row. Each scheduled run retries missed "
            f"dates for up to {int(cfg('daily.catch_up_days', 7))} days, so these "
            f"have already been attempted more than once — whatever is wrong is "
            f"not something another attempt will fix.",
        ]
    lines += [
        "",
        "Nothing was sent to readers. No issue file was written, so the archive "
        "shows this date as a day we could not see rather than as a quiet one.",
        "",
        "  uc status              — what the pipeline thinks the situation is",
        f"  uc daily --date {d}  — retry this date now",
    ]
    return "\n".join(lines) + "\n"


def notify_failure(
    d: date, reasons: list[str], backend=None, run=None
) -> dict[str, Any]:
    """Mail the alert. Never raises — a silent alert is better than a dead run."""
    from .deliver import Message, get_backend

    recipients = alert_recipients()
    if not recipients:
        return {"status": "no_alert_recipient"}

    streak = consecutive_failures(d)
    subject = failure_subject(d, reasons, streak)
    body = failure_body(d, reasons, streak)
    message = Message(
        subject=subject,
        html=f"<pre>{body}</pre>",
        text=body,
        issue_date=d,
        recipients=recipients,
    )
    try:
        backend = backend or get_backend()
        result = backend.send(message)
        return {"status": "alerted", "subject": subject, "streak": streak, **result}
    except Exception as e:  # noqa: BLE001 - notification failure is never fatal
        if run is not None:
            run.error(f"notify: {type(e).__name__}: {e}")
        return {"status": "alert_failed", "error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------
# Weekly summary
# --------------------------------------------------------------------------


def weekly_summary(end: Optional[date] = None, days: int = 7) -> dict[str, Any]:
    """The last seven days as one object: outcomes, spend, sends."""
    from .deliver import already_delivered
    from .llm import UsageState

    end = end or date.today()
    start = end - timedelta(days=days - 1)

    rows = []
    counts = {PUBLISHED: 0, QUIET: 0, NOT_PUBLISHED: 0}
    published_items = 0
    sends = 0
    for i in range(days):
        day = start + timedelta(days=i)
        log = load_log(day)
        status = (log or {}).get("status")
        if status in counts:
            counts[status] += 1
        published_items += int((log or {}).get("published", 0) or 0)
        ledger = already_delivered(day)
        day_sends = len((ledger or {}).get("sends") or [])
        sends += day_sends
        rows.append({
            "date": str(day),
            "status": status or "no record",
            "published": (log or {}).get("published", 0),
            "reasons": (log or {}).get("reasons") or [],
            "sends": day_sends,
        })

    usage = UsageState.load()
    return {
        "from": str(start),
        "to": str(end),
        "outcomes": counts,
        "days_without_a_record": sum(1 for r in rows if r["status"] == "no record"),
        "items_published": published_items,
        "emails_sent": sends,
        "llm_cost_total_usd": round(usage.cost_usd, 6),
        "llm_calls_total": usage.calls,
        "days": rows,
    }


def weekly_body(summary: dict[str, Any]) -> str:
    lines = [
        f"Urban Currents — {summary['from']} to {summary['to']}",
        "",
        f"  published      {summary['outcomes'][PUBLISHED]}",
        f"  quiet          {summary['outcomes'][QUIET]}",
        f"  not published  {summary['outcomes'][NOT_PUBLISHED]}",
        f"  no record      {summary['days_without_a_record']}",
        "",
        f"  items published  {summary['items_published']}",
        f"  emails sent      {summary['emails_sent']}",
        f"  LLM spend total  ${summary['llm_cost_total_usd']:.4f} "
        f"({summary['llm_calls_total']} calls, cumulative)",
        "",
    ]
    for row in summary["days"]:
        detail = f" — {row['reasons'][0]}" if row["reasons"] and row["status"] == NOT_PUBLISHED else ""
        lines.append(f"  {row['date']}  {row['status']:<14} {row['published']:>3}{detail}")
    return "\n".join(lines) + "\n"


def notify_weekly(end: Optional[date] = None, backend=None) -> dict[str, Any]:
    from .deliver import Message, get_backend

    recipients = alert_recipients()
    summary = weekly_summary(end)
    if not recipients:
        return {"status": "no_alert_recipient", "summary": summary}

    body = weekly_body(summary)
    message = Message(
        subject=(
            f"Urban Currents weekly — {summary['outcomes'][PUBLISHED]} published, "
            f"{summary['outcomes'][QUIET]} quiet, "
            f"{summary['outcomes'][NOT_PUBLISHED]} missed"
        ),
        html=f"<pre>{body}</pre>",
        text=body,
        issue_date=date.fromisoformat(summary["to"]),
        recipients=recipients,
    )
    try:
        backend = backend or get_backend()
        result = backend.send(message)
        return {"status": "sent", "summary": summary, **result}
    except Exception as e:  # noqa: BLE001 - a summary is never worth a crash
        return {"status": "failed", "error": f"{type(e).__name__}: {e}", "summary": summary}


# --------------------------------------------------------------------------
# `uc status`
# --------------------------------------------------------------------------


def status() -> dict[str, Any]:
    """What a person needs on the morning they come back.

    Deliberately answers "is anything wrong" before "what happened": the last
    success, the days still missing, what has been spent, and what runs next.
    """
    from .daily import lock_path, target_window
    from .deliver import get_backend, ledger_dir, recipients
    from .llm import UsageState
    from .outcome import unpublished_dates

    logs = all_logs()
    successes = [r for r in logs if r.get("status") in (PUBLISHED, QUIET)]
    successes.sort(key=lambda r: r["date"])
    missing = unpublished_dates()

    # The run log and the archive answer different questions, and right now they
    # disagree: every issue published before X3 exists without a log row. Showing
    # only `last_success` would read as "nothing has ever worked" to someone
    # looking at a working archive. Both, so the gap is visible instead of
    # alarming.
    # Filenames, not parsed models: status must stay answerable on the day a
    # bad file would stop it from being answered.
    issue_dir = paths.CONTENT / "issues"
    issues = sorted(p.stem for p in issue_dir.glob("*.json")) if issue_dir.exists() else []

    covers_from, covers_to = target_window()
    usage = UsageState.load()
    sends = len(list(ledger_dir().glob("*.json"))) if ledger_dir().exists() else 0

    lock_held = None
    if lock_path().exists():
        import json as _json

        try:
            lock_held = _json.loads(lock_path().read_text(encoding="utf-8"))
        except _json.JSONDecodeError:
            lock_held = {"unreadable": True}

    return {
        "last_success": successes[-1]["date"] if successes else None,
        "last_success_status": successes[-1]["status"] if successes else None,
        "last_issue": issues[-1] if issues else None,
        "issues_published": len(issues),
        "unpublished_dates": [r["date"] for r in missing],
        # A source that reports OK and returns nothing is the failure that does
        # not look like one. It belongs next to the missed days, not buried in a
        # run file nobody opens.
        "silent_sources_last_run": (
            sorted(logs, key=lambda r: r["date"])[-1].get("silent_sources") or []
            if logs
            else []
        ),
        "consecutive_failures": consecutive_failures(date.today() - timedelta(days=1)),
        "next_window": {"from": str(covers_from), "to": str(covers_to)},
        "llm_cost_total_usd": round(usage.cost_usd, 6),
        "llm_calls_total": usage.calls,
        "days_delivered": sends,
        "delivery_backend": get_backend().name,
        "reader_recipients": len(recipients()),
        "alert_recipients": len(alert_recipients()),
        "lock": lock_held,
    }
