"""Alerting reaches a person (phase 1P).

`deliver.backend` is `smtp` now (Resend). What is pinned here is everything a
switch like that could get wrong without anybody noticing:

- the fallback to `file` when a secret is missing is **said**, not silent;
- the From is the configured sender, never the SMTP user name (`resend`);
- `uc status` says "reach a person" only when a person can actually be reached;
- 🔴 the issue itself still goes to nobody without `UC_PREVIEW_RECIPIENT`;
- credentials in an SMTP failure never reach `content/runs_log/`.

No network and no real credentials: `conftest.no_real_mail` refuses every SMTP
connection, and the ones here are fakes.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

DAY = date(2026, 10, 5)
FAKE_PASSWORD = "re_FAKE0000000000000000000000000000"


@pytest.fixture
def smtp_env(repo, monkeypatch):
    monkeypatch.setenv("UC_SMTP_USER", "resend")
    monkeypatch.setenv("UC_SMTP_PASSWORD", FAKE_PASSWORD)
    return repo


def _message(recipients=None):
    from pipeline.deliver import Message

    return Message(
        subject="Urban Currents: test",
        html="<pre>body</pre>",
        text="body\n",
        issue_date=DAY,
        recipients=["ops@example.org"] if recipients is None else recipients,
    )


class _FakeSMTP:
    """Records what would have been sent; can be told to fail like Resend."""

    sent: list = []
    fail_with: Exception | None = None

    def __init__(self, host, port, timeout=None):
        self.host, self.port = host, port

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        pass

    def login(self, user, password):
        if _FakeSMTP.fail_with is not None:
            raise _FakeSMTP.fail_with

    def send_message(self, email):
        _FakeSMTP.sent.append(email)


@pytest.fixture
def fake_smtp(monkeypatch):
    import smtplib

    _FakeSMTP.sent = []
    _FakeSMTP.fail_with = None
    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)
    return _FakeSMTP


# --------------------------------------------------------------------------
# The configuration itself
# --------------------------------------------------------------------------


def test_the_committed_config_is_resend(repo):
    from pipeline.config import cfg

    assert cfg("deliver.backend") == "smtp"
    assert cfg("deliver.smtp.host") == "smtp.resend.com"
    assert int(cfg("deliver.smtp.port")) == 587
    assert cfg("deliver.sender").endswith("@send.latentpublics.com>")
    # Untouched, on purpose (1P, "하지 않을 것" 6).
    assert cfg("deliver.unsubscribe_url") is None
    assert cfg("deliver.unsubscribe_mailto") is None


# --------------------------------------------------------------------------
# P5-3 — the fallback is not silent
# --------------------------------------------------------------------------


@pytest.mark.parametrize("drop", ["UC_SMTP_USER", "UC_SMTP_PASSWORD"])
def test_a_missing_secret_falls_back_and_says_which(smtp_env, monkeypatch, capsys, drop):
    from pipeline.deliver import FileBackend, get_backend

    monkeypatch.delenv(drop)
    backend = get_backend()

    assert isinstance(backend, FileBackend)
    assert backend.fell_back_from == "smtp"
    assert backend.missing == (drop,)
    err = capsys.readouterr().err
    assert drop in err and "falling back to 'file'" in err
    # Names, never values.
    assert FAKE_PASSWORD not in err

    result = backend.send(_message())
    assert result["fell_back_from"] == "smtp" and result["missing"] == [drop]


def test_a_missing_host_falls_back_and_says_which(smtp_env, monkeypatch):
    from pipeline import config
    from pipeline.deliver import FileBackend, get_backend

    real = config.cfg

    def cfg(path, default=None):
        return None if path == "deliver.smtp.host" else real(path, default)

    monkeypatch.setattr("pipeline.deliver.cfg", cfg)
    backend = get_backend()
    assert isinstance(backend, FileBackend)
    assert backend.missing == ("deliver.smtp.host",)


def test_all_three_present_is_smtp(smtp_env):
    from pipeline.deliver import SmtpBackend, get_backend

    assert isinstance(get_backend(), SmtpBackend)


# --------------------------------------------------------------------------
# P3 — From is the configured sender
# --------------------------------------------------------------------------


def test_from_is_the_configured_sender_not_the_user_name(smtp_env, fake_smtp):
    from pipeline.deliver import SmtpBackend

    SmtpBackend().send(_message())
    (email,) = fake_smtp.sent
    assert email["From"] == "Urban Currents <no-reply@send.latentpublics.com>"
    assert email["From"] != "resend"


def test_no_sender_refuses_rather_than_using_the_user_name(smtp_env, fake_smtp, monkeypatch):
    from pipeline import config
    from pipeline.deliver import DeliveryError, SmtpBackend

    real = config.cfg
    monkeypatch.setattr(
        "pipeline.deliver.cfg",
        lambda p, d=None: None if p == "deliver.sender" else real(p, d),
    )
    with pytest.raises(DeliveryError):
        SmtpBackend().send(_message())
    assert fake_smtp.sent == []


# --------------------------------------------------------------------------
# P5-1 — what `uc status` says
# --------------------------------------------------------------------------


def test_smtp_with_a_recipient_reaches_a_person(smtp_env, monkeypatch):
    from pipeline.notify import alerting_state

    monkeypatch.setenv("UC_ALERT_RECIPIENT", "ops@example.org")
    state = alerting_state()
    assert state["backend"] == "smtp" and state["reaches_a_person"] is True
    assert state["fallback_missing"] == []


def test_smtp_without_a_recipient_reaches_nobody(smtp_env):
    from pipeline.notify import alerting_state

    state = alerting_state()
    assert state["backend"] == "smtp"
    assert state["reaches_a_person"] is False


def test_a_fallback_reaches_nobody_and_status_says_why(smtp_env, monkeypatch):
    from typer.testing import CliRunner

    from pipeline.cli import app
    from pipeline.notify import alerting_state

    monkeypatch.setenv("UC_ALERT_RECIPIENT", "ops@example.org")
    monkeypatch.delenv("UC_SMTP_PASSWORD")
    state = alerting_state()
    assert state["reaches_a_person"] is False
    assert state["configured_backend"] == "smtp"
    assert state["fallback_missing"] == ["UC_SMTP_PASSWORD"]

    out = CliRunner().invoke(app, ["status"]).output
    # The exact string `daily.yml` greps for.
    assert "alerts reach nobody" in out
    assert "UC_SMTP_PASSWORD not set" in out


def test_status_says_reach_a_person_when_it_can(smtp_env, monkeypatch):
    from typer.testing import CliRunner

    from pipeline.cli import app

    monkeypatch.setenv("UC_ALERT_RECIPIENT", "ops@example.org")
    out = CliRunner().invoke(app, ["status"]).output
    assert "alerts reach a person" in out
    assert "reach nobody" not in out


# --------------------------------------------------------------------------
# 🔴 P2 — reader mail stays off
# --------------------------------------------------------------------------


def test_no_preview_recipient_means_no_reader_mail(smtp_env, fake_smtp):
    """The regression guard for P2. `backend: smtp` with every SMTP secret set
    still sends the issue to nobody while `UC_PREVIEW_RECIPIENT` is absent."""
    from pipeline.deliver import deliver, ledger_path, recipients

    assert recipients() == []
    result = deliver(DAY, _message(recipients=recipients()))
    assert result["status"] == "no_recipients"
    assert fake_smtp.sent == []
    assert not ledger_path(DAY).exists()


def test_the_alert_goes_to_the_alert_recipient_only(smtp_env, fake_smtp, monkeypatch):
    from pipeline.notify import notify_failure

    monkeypatch.setenv("UC_ALERT_RECIPIENT", "ops@example.org")
    result = notify_failure(DAY, ["test reason"])
    assert result["status"] == "alerted" and result["reached_a_person"] is True
    (email,) = fake_smtp.sent
    assert email["To"] == "ops@example.org"


# --------------------------------------------------------------------------
# 🔴 P6 — a failing SMTP send does not carry a credential anywhere durable
# --------------------------------------------------------------------------


def test_an_smtp_failure_with_the_password_in_it_is_scrubbed(smtp_env, fake_smtp, monkeypatch):
    """The Springer shape (1L): an exception whose text carries the secret.

    smtplib's own errors carry the server's reply, not the password, but the
    test assumes the worst so the guarantee does not depend on that.
    """
    import smtplib

    from pipeline.metrics import Run
    from pipeline.notify import notify_failure
    from pipeline.outcome import decide, log_dir, record

    monkeypatch.setenv("UC_ALERT_RECIPIENT", "ops@example.org")
    fake_smtp.fail_with = smtplib.SMTPAuthenticationError(
        535, f"bad credentials password={FAKE_PASSWORD} for ops@example.org".encode()
    )

    run = Run.for_date(DAY)
    run.metrics.stages.update({"collect": "FAILED"})
    outcome = decide(run, DAY, published_count=0)
    record(outcome)
    result = notify_failure(DAY, outcome.reasons, run=run)
    run.save()

    assert result["status"] == "alert_failed"
    # What `uc daily` prints to the Actions log.
    assert FAKE_PASSWORD not in json.dumps(result)
    assert "ops@example.org" not in json.dumps(result)
    # What goes to the artifact.
    assert FAKE_PASSWORD not in json.dumps(run.metrics.errors)
    assert any("notify:" in e for e in run.metrics.errors)
    # What goes to the public repository.
    committed = (log_dir() / f"{DAY}.json").read_text(encoding="utf-8")
    assert FAKE_PASSWORD not in committed
    assert "ops@example.org" not in committed


def test_the_user_name_is_below_the_value_scrub_and_that_is_known(smtp_env):
    """`resend` is six characters, under `MIN_SECRET_LEN`, so the value-based
    scrub does not touch it — deliberately: it is not a secret, and scrubbing
    every "resend" would corrupt ordinary log lines. The parameter-name pattern
    still catches `password=`."""
    from pipeline.redact import MIN_SECRET_LEN, SECRET_ENV, redact

    for name in ("UC_SMTP_PASSWORD", "UC_SMTP_USER", "UC_SMTP_HOST",
                 "UC_ALERT_RECIPIENT", "CONTACT_EMAIL"):
        assert name in SECRET_ENV
    assert len("resend") < MIN_SECRET_LEN
    assert redact("login as resend") == "login as resend"
    assert FAKE_PASSWORD not in redact(f"password={FAKE_PASSWORD}")
    assert FAKE_PASSWORD not in redact(f"key is {FAKE_PASSWORD}")


def test_no_test_can_open_a_real_smtp_connection():
    import smtplib

    with pytest.raises(AssertionError):
        smtplib.SMTP("smtp.resend.com", 587)


# --------------------------------------------------------------------------
# The workflow
# --------------------------------------------------------------------------


def _step(text: str, name: str) -> str:
    start = text.index(f"- name: {name}")
    nxt = text.find("\n      - ", start + 1)
    return text[start: nxt if nxt != -1 else None]


@pytest.mark.parametrize(
    "name",
    ["Run the day", "Catch up on missed days", "Report whether alerting can reach anyone"],
)
def test_every_step_that_can_alert_has_the_smtp_secrets(name):
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / ".github/workflows/daily.yml").read_text(
        encoding="utf-8"
    )
    step = _step(text, name)
    for var in ("UC_SMTP_USER", "UC_SMTP_PASSWORD", "UC_ALERT_RECIPIENT"):
        assert f"{var}: ${{{{ secrets.{var} }}}}" in step, f"{name} lacks {var}"
