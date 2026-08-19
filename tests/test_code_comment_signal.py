"""The arXiv comment is where the repository link actually is (0P, Q0).

0k added `bibliography.comment` to the model for exactly this purpose and then
nothing read it. The path reported no yield because it was never connected —
the same shape as a measured zero that was never measured.

Wired here, and deliberately last: the abstract is still the first answer, and
a paper that says "neither the data nor the code" is not overruled by a link in
a comment.
"""

from __future__ import annotations

from pipeline.models import Bibliography, Item
from pipeline.signals import apply_badges, code_signal


def _item(abstract=None, comment=None, title="A paper about cities"):
    return Item(
        work_key="arxiv:2607.04330",
        bibliography=Bibliography(title=title, abstract=abstract, comment=comment),
    )


def test_a_repository_in_the_comment_is_found(repo):
    sig = code_signal(_item(
        abstract="We study kerbside parking in three cities.",
        comment="Accepted to NeurIPS 2026. Code: https://github.com/chrisyan/RZDG",
    ))

    assert sig.value is True
    assert sig.url == "https://github.com/chrisyan/RZDG"
    assert sig.detail == "from the arXiv comment"


def test_the_abstract_still_answers_first(repo):
    """When both carry a link the abstract's wins, and the badge is not marked
    as coming from the comment."""
    sig = code_signal(_item(
        abstract="Code available at https://github.com/real/repo",
        comment="Also at https://github.com/mirror/repo",
    ))

    assert sig.url == "https://github.com/real/repo"
    assert sig.detail is None


def test_an_explicit_refusal_is_not_overruled_by_a_comment(repo):
    """"Neither the data nor the code" means what it says."""
    sig = code_signal(_item(
        abstract="Neither the data nor the code are publicly available.",
        comment="See https://github.com/some/repo for the paper source",
    ))

    assert sig.value is False


def test_no_comment_is_still_no_signal(repo):
    """Absence of a comment must read as 'nothing found', not as False-with-
    confidence. Most journal items have no comment field at all."""
    sig = code_signal(_item(abstract="A study of bus headways."))

    assert sig.value is False
    assert sig.detail is None


def test_the_badge_follows(repo):
    item = _item(abstract="A study.", comment="Code: https://github.com/a/b")
    item.signals.code_available = code_signal(item)
    apply_badges(item)

    assert "code" in item.badges


def test_a_bare_url_that_is_not_a_repository_does_not_count(repo):
    sig = code_signal(_item(
        abstract="A study.",
        comment="Accepted at TRB. Slides: https://example.com/talk.pdf",
    ))

    assert sig.value is False
