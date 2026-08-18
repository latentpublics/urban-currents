"""OpenAlex requests have a deadline (phase 0N, P0).

The bug that cost forty days of backfill. pyalex 0.21 calls
`session.get(url, auth=...)` without a `timeout`, and `requests` waits forever
by default, so a single stalled connection held the process indefinitely:
`runs/run_2026-07-02/` recorded `collect.arxiv: OK`, an empty `raw/openalex/`,
and then nothing — it stopped before the first page was written.

The sibling collectors already had deadlines (arXiv 60s, abstracts 30s), which
is why neither of them has ever hung. This is the missing third.

What matters is not the number but the property: **collection ends in finite
time and records a failure, rather than waiting forever.** A date that cannot be
collected becomes `not_published` and the backfill moves to the next one — one
bad date must not stop sixty.
"""

from __future__ import annotations

import socket
import threading
import time
from datetime import date

import pytest

from pipeline.collectors.openalex import _install_http_policy


class _BlackHole:
    """A socket that accepts connections and then never answers.

    Exactly the failure mode observed: the TCP handshake completes, so a connect
    timeout does not fire, and then nothing is ever sent. Only a *read* timeout
    ends it.
    """

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.held: list = []
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                self.sock.settimeout(0.5)
                conn, _ = self.sock.accept()
                self.held.append(conn)   # accepted, never answered
            except (OSError, socket.timeout):
                continue

    def close(self):
        self._stop.set()
        for c in self.held:
            try:
                c.close()
            except OSError:
                pass
        try:
            self.sock.close()
        except OSError:
            pass


@pytest.fixture
def black_hole():
    server = _BlackHole()
    yield server
    server.close()


def test_a_stalled_endpoint_fails_in_finite_time(repo, black_hole, monkeypatch):
    """The whole point: it ends, and it ends as a failure rather than a hang."""
    import requests

    monkeypatch.setenv("OPENALEX_KEY", "test-key")
    monkeypatch.setenv("CONTACT_EMAIL", "test@example.org")
    monkeypatch.setattr("pipeline.config.cfg", lambda key, default=None: {
        "openalex.connect_timeout_s": 2.0,
        "openalex.read_timeout_s": 2.0,
    }.get(key, default))

    import pyalex

    _install_http_policy(pyalex)
    session = pyalex.api._get_requests_session()

    started = time.monotonic()
    with pytest.raises(requests.exceptions.RequestException):
        session.get(f"http://127.0.0.1:{black_hole.port}/works")
    elapsed = time.monotonic() - started

    # Generous ceiling: the assertion is "finite", not a stopwatch.
    assert elapsed < 30, f"took {elapsed:.1f}s — the deadline did not fire"


def test_the_timeout_is_applied_to_every_request(repo, monkeypatch):
    """Bound on the session, so `_get_from_url` and the paginator both get it.

    Captures what actually reaches `requests.Session.request` rather than
    asserting on a value the test itself supplies.
    """
    import requests

    import pyalex

    monkeypatch.setattr("pipeline.config.cfg", lambda key, default=None: {
        "openalex.connect_timeout_s": 11.0,
        "openalex.read_timeout_s": 22.0,
    }.get(key, default))

    seen: dict = {}

    def recorder(self, method, url, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("captured")

    # Patched before the factory runs, so the wrapper closes over the recorder.
    monkeypatch.setattr(requests.Session, "request", recorder)
    monkeypatch.setattr(pyalex.api, "_uc_timeout_installed", False, raising=False)
    _install_http_policy(pyalex)

    session = pyalex.api._get_requests_session()
    with pytest.raises(RuntimeError):
        session.get("https://api.openalex.org/works")

    assert seen.get("timeout") == (11.0, 22.0)


def test_pagination_is_bounded(repo):
    """`n_max=None` is unbounded. Not the cause of the hang, but the next one."""
    from pipeline.collectors.openalex import OpenAlexCollector
    from pipeline.metrics import Run

    collector = OpenAlexCollector(Run.for_date(date(2026, 8, 20)))

    assert collector.max_records > 0
    assert collector.max_records is not None
