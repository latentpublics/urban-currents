"""Sending the issue (phase 0k, X5).

Three backends behind one interface, and **the default sends nothing**:

- `file` — writes a `.eml` to disk. What runs when no credentials exist, which
  is now and until YJUN chooses a provider.
- `smtp` — SMTP with STARTTLS. Every provider worth using speaks it, and an app
  password is enough, so nothing here needs a vendor SDK.
- `console` — prints the headers. For tests and for eyeballing a subject line.

No HTTP API backend. SMTP covers the requirement, and picking a provider is a
decision with a bill and a domain attached to it.

**Two things this module refuses to do.**

It will not send twice. `content/deliveries/YYYY-MM-DD.json` is written when a
send succeeds and is checked before the next one starts. The ledger carries a
hash of the body, because "we sent something on the 14th" is a weaker fact than
"we sent *this* on the 14th" — if a day ever goes out twice with different
contents, the hashes are how anyone finds out.

And it will not carry a tracking pixel, a redirect wrapper, or an open beacon.
The home page says "no ads, no engagement metrics"; that sentence is a
constraint on this file, and a test enforces it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import date
from email.message import EmailMessage
from email.utils import formatdate
from pathlib import Path
from typing import Any, Optional, Protocol

from .. import paths
from ..config import cfg, secret
from ..metrics import utcnow


class DeliveryError(RuntimeError):
    """A send failed. Never fatal to the pipeline — the issue is already written."""


@dataclass
class Message:
    """One issue, ready to send, in both forms it has to exist in."""

    subject: str
    html: str
    text: str
    issue_date: date
    recipients: list[str] = field(default_factory=list)

    @property
    def body_hash(self) -> str:
        """Content identity. What was sent, not merely that something was."""
        digest = hashlib.sha256()
        digest.update(self.subject.encode("utf-8"))
        digest.update(b"\0")
        digest.update(self.html.encode("utf-8"))
        digest.update(b"\0")
        digest.update(self.text.encode("utf-8"))
        return digest.hexdigest()


class Backend(Protocol):
    name: str

    def send(self, message: Message) -> dict[str, Any]:
        ...


# --------------------------------------------------------------------------
# Building the message
# --------------------------------------------------------------------------

# Anything that would let us learn a reader opened the mail. The home page
# promises none of it, so the promise is enforced here rather than remembered.
TRACKING_PATTERNS = (
    re.compile(r"<img[^>]*\b(?:width|height)\s*=\s*[\"']?1[\"']?", re.I),
    re.compile(r"\bopen\.(?:track|pixel)", re.I),
    re.compile(r"/track(?:ing)?/(?:open|pixel|beacon)", re.I),
    re.compile(r"utm_(?:source|medium|campaign)", re.I),
)


def assert_no_tracking(html: str) -> None:
    for pattern in TRACKING_PATTERNS:
        if pattern.search(html):
            raise DeliveryError(f"tracking artefact in the email body: {pattern.pattern}")


def unsubscribe_headers(message: EmailMessage) -> None:
    """RFC 8058 one-click, when there is somewhere for it to point.

    With no domain there is no endpoint, and a `List-Unsubscribe` header that
    goes nowhere is worse than none: a mail client will show the button and the
    reader will press it. So the mailto form is used when an address exists, and
    otherwise the header is left off entirely and the body still carries the
    plain instruction.
    """
    address = cfg("deliver.unsubscribe_mailto", None)
    url = cfg("deliver.unsubscribe_url", None)
    parts = []
    if url:
        parts.append(f"<{url}>")
    if address:
        parts.append(f"<mailto:{address}?subject=unsubscribe>")
    if not parts:
        return
    message["List-Unsubscribe"] = ", ".join(parts)
    if url:
        message["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"


def build_email(message: Message, sender: Optional[str] = None) -> EmailMessage:
    """A multipart/alternative message: plain text first, HTML second."""
    assert_no_tracking(message.html)

    email = EmailMessage()
    email["Subject"] = message.subject
    email["From"] = sender or cfg("deliver.sender", "urban-currents@localhost")
    email["To"] = ", ".join(message.recipients) or "undisclosed-recipients:;"
    email["Date"] = formatdate(localtime=False)
    # No Message-ID and no In-Reply-To: a daily is not a thread, and threading
    # eight issues together makes the eighth unreadable in most clients.
    email.set_content(message.text)
    email.add_alternative(message.html, subtype="html")
    unsubscribe_headers(email)
    return email


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


@dataclass
class FileBackend:
    """Writes the `.eml` and sends nothing. The default, and the safe one."""

    name: str = "file"
    directory: Optional[Path] = None

    def send(self, message: Message) -> dict[str, Any]:
        target = self.directory or (paths.RUNS / "outbox")
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{message.issue_date}.eml"
        path.write_bytes(bytes(build_email(message)))
        return {"backend": self.name, "path": str(path), "recipients": len(message.recipients)}


@dataclass
class ConsoleBackend:
    name: str = "console"
    printer: Any = print

    def send(self, message: Message) -> dict[str, Any]:
        self.printer(f"To: {', '.join(message.recipients)}")
        self.printer(f"Subject: {message.subject}")
        self.printer(f"({len(message.html)} bytes html, {len(message.text)} bytes text)")
        return {"backend": self.name, "recipients": len(message.recipients)}


@dataclass
class SmtpBackend:
    """Standard SMTP over STARTTLS. Credentials come from the environment only."""

    name: str = "smtp"

    def send(self, message: Message) -> dict[str, Any]:
        import smtplib

        host = cfg("deliver.smtp.host", None)
        port = int(cfg("deliver.smtp.port", 587))
        user = secret("UC_SMTP_USER")
        password = secret("UC_SMTP_PASSWORD")
        if not (host and user and password):
            raise DeliveryError("smtp backend selected but host/user/password are not set")

        email = build_email(message, sender=cfg("deliver.sender", user))
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(email)
        return {
            "backend": self.name,
            "host": host,
            "recipients": len(message.recipients),
        }


def get_backend(name: Optional[str] = None) -> Backend:
    """The configured backend, falling back to `file` when it cannot run.

    Falling back rather than failing is deliberate: a missing password should
    cost a send, not a day's issue. The fallback is recorded in the ledger so
    "why did nobody get it" has an answer.
    """
    name = name or cfg("deliver.backend", "file")
    if name == "console":
        return ConsoleBackend()
    if name == "smtp":
        if secret("UC_SMTP_USER") and secret("UC_SMTP_PASSWORD") and cfg("deliver.smtp.host", None):
            return SmtpBackend()
        return FileBackend()
    return FileBackend()


# --------------------------------------------------------------------------
# Recipients
# --------------------------------------------------------------------------


def recipients() -> list[str]:
    """Who gets it. One address, from the environment, never from a file in git.

    A subscriber list is personal data. `subscribers/` is ignored at the top of
    `.gitignore` and a test asserts it stays there, but the real protection is
    that nothing reads a list yet: there is one address and it lives in `.env`.
    """
    single = os.environ.get("UC_PREVIEW_RECIPIENT", "").strip()
    return [single] if single else []


# --------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------


def ledger_dir() -> Path:
    return paths.CONTENT / "deliveries"


def ledger_path(d: date) -> Path:
    return ledger_dir() / f"{d}.json"


def already_delivered(d: date) -> Optional[dict]:
    path = ledger_path(d)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def record_delivery(
    d: date, message: Message, result: dict[str, Any], resend: bool = False
) -> Path:
    """Append-only in spirit: a resend adds to the record, never replaces it."""
    ledger_dir().mkdir(parents=True, exist_ok=True)
    path = ledger_path(d)

    existing = already_delivered(d)
    entry = {
        "sent_at": utcnow().isoformat(),
        "backend": result.get("backend"),
        "recipients": result.get("recipients", 0),
        "body_sha256": message.body_hash,
        "subject": message.subject,
        "message_id": result.get("message_id"),
        "resend": resend,
    }
    if existing:
        sends = list(existing.get("sends") or [])
        sends.append(entry)
        doc = {**existing, "sends": sends}
    else:
        doc = {"date": str(d), "sends": [entry]}

    path.write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return path


def deliver(
    d: date,
    message: Message,
    backend: Optional[Backend] = None,
    force: bool = False,
) -> dict[str, Any]:
    """Send one issue, once.

    `force` exists for the day something has to go out again, and it is recorded
    as a resend rather than allowed to look like the first attempt.
    """
    previous = already_delivered(d)
    if previous and not force:
        return {
            "status": "already_delivered",
            "date": str(d),
            "sends": len(previous.get("sends") or []),
            "body_sha256": (previous.get("sends") or [{}])[-1].get("body_sha256"),
        }

    if not message.recipients:
        return {"status": "no_recipients", "date": str(d)}

    backend = backend or get_backend()
    result = backend.send(message)
    record_delivery(d, message, result, resend=bool(previous))
    return {"status": "sent", "date": str(d), **result, "body_sha256": message.body_hash}
