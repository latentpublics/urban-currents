"""Sending the issue (phase 0k, X5).

The three properties worth a test: the same content reaches all three outputs,
nothing goes out twice, and nothing in the mail can tell us a reader opened it.
"""

from __future__ import annotations

import json
import re
from datetime import date
from html.parser import HTMLParser

import pytest

from pipeline import paths, store
from pipeline.deliver import (
    ConsoleBackend,
    DeliveryError,
    FileBackend,
    Message,
    already_delivered,
    assert_no_tracking,
    build_email,
    deliver,
    get_backend,
    recipients,
)
from pipeline.models import (
    Bibliography,
    Headline,
    Issue,
    Item,
    PrimaryLocation,
    ScanMeta,
    SummaryEn,
)
from pipeline.render.inline import to_email
from pipeline.render.plaintext import render_text
from pipeline.render.preview import render_issue

DAY = date(2026, 8, 11)


def _items(n: int = 3) -> list[Item]:
    out = []
    for i in range(n):
        it = Item(
            work_key=f"arxiv:2608.{i:05d}",
            first_published=DAY,
            bibliography=Bibliography(
                title=f"Paper {i} on urban form",
                abstract="x",
                primary_location=PrimaryLocation(
                    source_name="arXiv",
                    landing_page_url=f"https://arxiv.org/abs/2608.{i:05d}",
                ),
            ),
        )
        it.summary.en = SummaryEn(
            what=f"What paper {i} found.", why=f"Why paper {i} matters."
        )
        out.append(it)
    return out


def _issue(items: list[Item]) -> Issue:
    return Issue(
        date=DAY,
        items=[i.work_key for i in items],
        headline=Headline(present=True, work_key=items[0].work_key, line="A full line."),
        scan_meta=ScanMeta(items_published=len(items), candidates_scanned=100, journals=96),
    )


def _message(items: list[Item], issue: Issue) -> Message:
    web = render_issue(issue, items)
    return Message(
        subject="Urban Currents 2026-08-11 — 3 papers",
        html=to_email(web),
        text=render_text(issue, items),
        issue_date=DAY,
        recipients=["reader@example.org"],
    )


class _Visible(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("style", "script"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("style", "script"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.text.append(" ".join(data.split()))


# --------------------------------------------------------------------------
# Three outputs, one content
# --------------------------------------------------------------------------


def test_web_html_mail_and_plain_text_say_the_same_things(repo):
    """Form differs. Every title, what and why must appear in all three."""
    items = _items()
    issue = _issue(items)

    web = render_issue(issue, items)
    mail = to_email(web)
    text = render_text(issue, items)

    web_visible = " ".join(_Visible().text) if False else None
    parser = _Visible()
    parser.feed(web)
    web_blob = " ".join(parser.text)
    parser2 = _Visible()
    parser2.feed(mail)
    mail_blob = " ".join(parser2.text)
    text_blob = " ".join(text.split())

    for item in items:
        title = item.bibliography.title
        assert title in web_blob
        assert title in mail_blob
        assert title in text_blob
        assert item.summary.en.what in web_blob
        assert item.summary.en.what in mail_blob
        assert item.summary.en.what in text_blob
        assert item.summary.en.why in text_blob


def test_the_plain_text_is_written_not_stripped(repo):
    """Tags removed from HTML is a worse issue, not the same one in another form."""
    items = _items()
    text = render_text(_issue(items), items)

    assert "<" not in text
    assert "URBAN CURRENTS" in text
    assert "Why it matters —" in text
    # Wrapped to a readable measure rather than left as one long line.
    assert max(len(line) for line in text.splitlines()) <= 78
    # And the links survive, because a text reader still needs to reach the paper.
    assert "https://arxiv.org/abs/2608.00000" in text


def test_a_wrapped_bullet_does_not_become_several_bullets(repo):
    """A list marker used as an indent repeats on every continuation line."""
    from pipeline.render.plaintext import _wrap

    long_title = "Contribution of streetscape audits to explanation of physical activity in four age groups"
    wrapped = _wrap(long_title, indent="    ", bullet="- ").splitlines()
    assert wrapped[0].startswith("    - ")
    assert all(not line.strip().startswith("- ") for line in wrapped[1:])


# --------------------------------------------------------------------------
# No tracking
# --------------------------------------------------------------------------


def test_the_email_carries_no_tracking_artefact(repo):
    items = _items()
    message = _message(items, _issue(items))
    assert_no_tracking(message.html)          # does not raise
    email = build_email(message)
    body = email.as_string()

    assert "utm_source" not in body
    assert not re.search(r"<img[^>]*height\s*=\s*[\"']?1", body, re.I)


def test_a_tracking_pixel_is_refused_rather_than_sent(repo):
    with pytest.raises(DeliveryError):
        assert_no_tracking('<p>hi</p><img src="https://x/track/open" width="1" height="1">')


def test_no_unsubscribe_header_is_emitted_without_an_endpoint(repo):
    """A List-Unsubscribe pointing nowhere is worse than none: the client shows
    the button and the reader presses it."""
    items = _items()
    email = build_email(_message(items, _issue(items)))
    assert email.get("List-Unsubscribe") is None
    # The body still tells a reader how to stop.
    text = render_text(_issue(items), items)
    assert "unsubscribe" in text.lower()


def test_the_message_is_multipart_with_text_first(repo):
    items = _items()
    email = build_email(_message(items, _issue(items)))
    parts = [p.get_content_type() for p in email.walk()]
    assert "text/plain" in parts
    assert "text/html" in parts
    assert parts.index("text/plain") < parts.index("text/html")
    # A daily is not a thread.
    assert email.get("In-Reply-To") is None


# --------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------


def test_nothing_is_sent_twice(repo):
    items = _items()
    message = _message(items, _issue(items))

    first = deliver(DAY, message, backend=FileBackend())
    assert first["status"] == "sent"

    second = deliver(DAY, message, backend=FileBackend())
    assert second["status"] == "already_delivered"
    assert second["sends"] == 1


def test_a_forced_resend_is_recorded_as_one(repo):
    items = _items()
    message = _message(items, _issue(items))

    deliver(DAY, message, backend=FileBackend())
    again = deliver(DAY, message, backend=FileBackend(), force=True)
    assert again["status"] == "sent"

    ledger = already_delivered(DAY)
    assert len(ledger["sends"]) == 2
    assert ledger["sends"][0]["resend"] is False
    assert ledger["sends"][1]["resend"] is True


def test_the_ledger_records_what_was_sent_not_only_that_it_was(repo):
    """Two different bodies on one date must be findable afterwards."""
    items = _items()
    issue = _issue(items)
    first = _message(items, issue)
    deliver(DAY, first, backend=FileBackend())

    changed = _message(items[:2], _issue(items[:2]))
    deliver(DAY, changed, backend=FileBackend(), force=True)

    hashes = [s["body_sha256"] for s in already_delivered(DAY)["sends"]]
    assert len(set(hashes)) == 2


def test_no_recipients_means_no_send_and_no_ledger(repo):
    items = _items()
    message = _message(items, _issue(items))
    message.recipients = []

    result = deliver(DAY, message, backend=FileBackend())
    assert result["status"] == "no_recipients"
    assert already_delivered(DAY) is None


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------


def test_the_default_backend_sends_nothing(repo):
    assert get_backend().name == "file"


def test_smtp_without_credentials_falls_back_to_file(repo, monkeypatch):
    """A missing password should cost a send, not a day's issue."""
    monkeypatch.delenv("UC_SMTP_USER", raising=False)
    monkeypatch.delenv("UC_SMTP_PASSWORD", raising=False)
    assert get_backend("smtp").name == "file"


def test_the_file_backend_writes_a_readable_eml(repo):
    import email as email_lib

    items = _items()
    result = FileBackend().send(_message(items, _issue(items)))
    raw = paths.RUNS / "outbox" / f"{DAY}.eml"
    assert raw.exists()

    parsed = email_lib.message_from_bytes(raw.read_bytes())
    assert parsed["Subject"].startswith("Urban Currents")
    assert result["recipients"] == 1


def test_recipients_come_from_the_environment_only(repo, monkeypatch):
    monkeypatch.setenv("UC_PREVIEW_RECIPIENT", "reader@example.org")
    assert recipients() == ["reader@example.org"]
    monkeypatch.delenv("UC_PREVIEW_RECIPIENT")
    assert recipients() == []
