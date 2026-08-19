"""A subfield we have barely seen cannot be excluded (phase 0Q, R2).

Four subfields were excluded on three to six labelled papers each, and five
targeted judgements per subfield overturned all four (0P Q3). The bar that let
that happen was `MIN_OBSERVED = 3` together with a test on the **point
estimate** of the keep rate.

YJUN, after labelling them: "서브필드에서 몇 편 되지 않더라도 내용상 urban에
가까우면 검토대상으로 두어야 할 것 같습니다."

Two conditions now, and a subfield is excluded only if **both** hold. These
tests exist so that a future edit cannot quietly lower either one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from paper_subfields import (  # noqa: E402
    MIN_KEEP_RATE,
    MIN_OBSERVED,
    derive,
    wilson_upper,
)


def _rows(subfield: str, n: int, keeps: int):
    return [
        {"work_key": f"{subfield}:{i}", "subfield": subfield, "keep": i < keeps,
         "title": f"paper {i}", "date": "2026-08-05", "rank": i, "found": True}
        for i in range(n)
    ]


def test_the_bar_is_ten_observations():
    """Three was the bar that produced four wrong exclusions."""
    assert MIN_OBSERVED >= 10
    assert MIN_KEEP_RATE == 0.50


@pytest.mark.parametrize("n,keeps", [(1, 0), (3, 0), (3, 1), (6, 1), (9, 0), (9, 1)])
def test_a_thin_subfield_is_always_included(n, keeps):
    """Whatever its keep rate. Zero keeps in nine is still not enough to judge."""
    included, table = derive(_rows("9999", n, keeps))

    assert "9999" in included
    assert table[0]["thin"] is True
    assert table[0]["included"] is True


@pytest.mark.parametrize("n,keeps", [
    (10, 0),   # the four real cases, scaled up to where they would be decidable
    (12, 1),
    (20, 3),
])
def test_a_well_observed_and_clearly_low_subfield_is_excluded(n, keeps):
    """The rule still has teeth. It is a higher bar, not an absent one."""
    included, table = derive(_rows("9999", n, keeps))

    assert "9999" not in included
    assert table[0]["confidently_below_half"] is True


def test_a_low_point_estimate_is_not_enough_on_its_own():
    """The heart of it. 4 keeps in 12 is a point estimate of 0.33 — below the
    coin flip — but the interval reaches 0.61, so it is not a low rate."""
    included, table = derive(_rows("9999", 12, 4))

    assert table[0]["keep_rate"] < MIN_KEEP_RATE
    assert table[0]["keep_rate_upper_95"] > MIN_KEEP_RATE
    assert "9999" in included, "a low estimate is not a low rate"


def test_the_four_that_were_wrongly_excluded_would_now_pass():
    """Checked at their real sizes, without using the outcome we now know."""
    for subfield, n, keeps in (("1408", 3, 1), ("2208", 5, 2),
                               ("2306", 4, 1), ("3312", 6, 1)):
        included, table = derive(_rows(subfield, n, keeps))
        assert subfield in included, f"{subfield} would still be excluded"
        assert table[0]["keep_rate"] < MIN_KEEP_RATE, "on the old rule it was out"


def test_wilson_stays_inside_the_unit_interval():
    """The normal approximation does not, at these sizes, which is why it is not
    used: 1 keep in 3 gives an upper bound above 1.0."""
    assert 0.0 <= wilson_upper(1, 3) <= 1.0
    assert 0.0 <= wilson_upper(0, 4) <= 1.0
    assert wilson_upper(0, 4) > 0.0, "zero keeps in four is not certainty"
    assert wilson_upper(0, 0) == 1.0, "nothing observed means nothing is ruled out"


def test_more_evidence_narrows_the_bound():
    """The same rate, seen more often, is known better."""
    assert wilson_upper(1, 3) > wilson_upper(10, 30) > wilson_upper(100, 300)


def test_unclassified_is_never_added_to_the_list():
    """"We could not read the subfield" is not a subfield."""
    included, _ = derive(_rows("unclassified", 40, 1))

    assert "unclassified" not in included
