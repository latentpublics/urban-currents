"""End-to-end vertical slice on the built-in fixtures — no network, no keys."""

from __future__ import annotations

from html.parser import HTMLParser

from pipeline import run_stages, store
from pipeline.validate import validate_content


class _CardCounter(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.cards = 0
        self.external_srcs: list[str] = []
        self.title_seen = False

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag == "article" and "uc-card" in (d.get("class") or ""):
            self.cards += 1
        if tag in ("img", "script", "iframe", "link"):
            src = d.get("src") or d.get("href") or ""
            if src.startswith(("http://", "https://", "//")):
                self.external_srcs.append(src)
        if tag == "title":
            self.title_seen = True


def test_fixture_run_produces_valid_content(repo, sample_date):
    run = run_stages.run_all(sample_date, fixture=True, use_llm=False)

    assert run.metrics.stages["collect"] == "OK"
    assert run.metrics.stages["issue"] == "OK"
    # summarize must report SKIPPED rather than silently claiming success.
    assert run.metrics.stages["summarize"] == "SKIPPED"

    issue = store.load_issue(sample_date)
    assert issue is not None
    assert len(issue.items) == 3
    assert issue.scan_meta.items_published == 3

    result = validate_content()
    assert result.ok, result.errors


def test_preview_is_self_contained_and_card_count_matches(repo, sample_date):
    run = run_stages.run_all(sample_date, fixture=True, use_llm=False)
    html = (run.dir / "preview.html").read_text(encoding="utf-8")

    parser = _CardCounter()
    parser.feed(html)

    issue = store.load_issue(sample_date)
    assert parser.cards == issue.scan_meta.items_published
    # PRD §5.7: no external dependencies of any kind, and no paper figures.
    assert parser.external_srcs == []
    assert parser.title_seen


def test_pipeline_is_idempotent(repo, sample_date):
    """Running the same date twice must leave content/ byte-identical (PRD §9)."""
    run_stages.run_all(sample_date, fixture=True, use_llm=False)
    before = {
        p.relative_to(repo).as_posix(): p.read_bytes()
        for p in sorted((repo / "content").rglob("*.json"))
    }

    run_stages.run_all(sample_date, fixture=True, use_llm=False)
    after = {
        p.relative_to(repo).as_posix(): p.read_bytes()
        for p in sorted((repo / "content").rglob("*.json"))
    }

    assert set(before) == set(after)
    for k in before:
        assert before[k] == after[k], f"{k} changed on the second run"


def test_second_run_does_not_republish_into_a_later_issue(repo, sample_date):
    """A paper carried by an earlier issue is not re-published on a later date."""
    from datetime import timedelta

    run_stages.run_all(sample_date, fixture=True, use_llm=False)
    next_day = sample_date + timedelta(days=1)
    run_stages.run_all(next_day, fixture=True, use_llm=False)

    first = store.load_issue(sample_date)
    second = store.load_issue(next_day)
    assert len(first.items) == 3
    assert second.items == []
    assert second.quiet_day is True
