"""Canon affinity: the binary flag and the normalisation the labels chose (V2).

The probe's 30 band-stratified labels are the evidence, and they say two things:
the zero line carries the `not_our_kind` signal, and the grades above zero do
not. What is pinned here is the arithmetic those findings rest on.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from journal_metrics import canon_affinity, cites_canon  # noqa: E402

# Two real papers, with the numbers that started this. Weights are the canon's
# own `weighted_score`, rounded; what matters is the ordering they produce.
WALKABILITY = ["c1", "c2", "c3", "c4", "c5", "c6", "c7"] + [f"x{i}" for i in range(56)]
WESTERN = ["c1", "c2"] + ["x100", "x101", "x102"]
CANON = {f"c{i}": 20.0 for i in range(1, 8)}


def test_seven_canon_hits_in_sixty_three_beats_two_in_five():
    """The inversion that moved the default off `sqrt`.

    A paper citing seven foundational works out of 63 references is more
    embedded in the field than one citing two out of five. Under the square root
    it scored lower, which is the wrong way round.
    """
    assert canon_affinity(WALKABILITY, CANON) > canon_affinity(WESTERN, CANON)


def test_the_square_root_normalisation_inverts_that_pair():
    # Kept as a mode, and kept honest: this is why it is no longer the default.
    assert canon_affinity(WALKABILITY, CANON, "sqrt") < canon_affinity(WESTERN, CANON, "sqrt")


def test_the_default_is_the_unnormalised_weighted_sum():
    assert canon_affinity(WALKABILITY, CANON) == canon_affinity(WALKABILITY, CANON, "none")
    assert canon_affinity(WALKABILITY, CANON) == 140.0  # 7 hits x 20.0


def test_weights_matter_not_just_hit_counts():
    central = {"c1": 100.0, "c2": 1.0}
    one_central = canon_affinity(["c1", "x", "y"], central)
    one_marginal = canon_affinity(["c2", "x", "y"], central)
    assert one_central > one_marginal


def test_every_normalisation_agrees_that_no_hits_is_zero():
    for mode in ("none", "sqrt", "linear", "log"):
        assert canon_affinity(["x1", "x2"], CANON, mode) == 0.0


def test_an_empty_reference_list_is_zero_not_an_error():
    for mode in ("none", "sqrt", "linear", "log"):
        assert canon_affinity([], CANON, mode) == 0.0


def test_cites_canon_is_the_zero_line_and_nothing_else():
    assert cites_canon(WESTERN, CANON) is True
    assert cites_canon(["x1", "x2", "x3"], CANON) is False
    # And it is deliberately blind to how much: one hit and seven read the same.
    assert cites_canon(WESTERN, CANON) == cites_canon(WALKABILITY, CANON)


def test_no_references_and_no_canon_hits_are_indistinguishable_here():
    """The ambiguity the flag cannot resolve, pinned so it stays visible.

    A paper with no reference list on file and a paper that cites 60 works and
    none of them foundational both come back False. The pool that feeds the
    probe excludes the first kind for exactly this reason; anything else using
    this flag has to make the same exclusion itself.
    """
    assert cites_canon([], CANON) is False
    assert cites_canon(["x1"] * 60, CANON) is False
