"""Repository links must be the repository's claim, not ours (phase 0Q, R3).

The gate said do not build the linking feature: 7% recall, and one link in three
would have been wrong. These tests pin the reasoning so that a later attempt has
to argue with it rather than rediscover it.

Nothing here makes a network call.
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from repo_backlink_probe import SUPPLEMENT, identifiers  # noqa: E402


def test_a_citation_is_not_a_supplement():
    """A repository citing a paper is not the paper's repository. That is the
    whole distinction the search rule rests on."""
    assert "IsCitedBy" not in SUPPLEMENT
    assert "Cites" not in SUPPLEMENT
    assert "References" not in SUPPLEMENT
    assert "IsSupplementTo" in SUPPLEMENT


def test_identifiers_come_from_the_work_key(repo):
    from pipeline.models import Bibliography, Item

    arx = Item(work_key="arxiv:2607.03443", bibliography=Bibliography(title="x"))
    doi = Item(work_key="doi:10.1016/j.trd.2026.105522",
               bibliography=Bibliography(title="x"))

    assert identifiers(arx)["arxiv"] == "2607.03443"
    assert identifiers(doi)["doi"] == "10.1016/j.trd.2026.105522"


def test_a_preprint_record_is_not_a_repository(repo, monkeypatch):
    """The first pass counted arXiv's own DataCite record — `IsVersionOf`,
    resourceType `Preprint`, URL arxiv.org — as a hit. That is the paper
    pointing at itself."""
    import repo_backlink_probe as probe

    payload = {"data": [{"attributes": {
        "doi": "10.48550/arxiv.2607.24795",
        "url": "https://arxiv.org/abs/2607.24795",
        "types": {"resourceTypeGeneral": "Preprint"},
        "relatedIdentifiers": [
            {"relatedIdentifier": "10.1/paper", "relationType": "IsVersionOf"}
        ],
    }}]}
    monkeypatch.setattr(probe, "_get", lambda *a, **k: (200, payload))

    assert probe.datacite_related("10.1/paper") == []


def test_a_dataset_deposit_is_a_repository(repo, monkeypatch):
    import repo_backlink_probe as probe

    payload = {"data": [{"attributes": {
        "doi": "10.13012/IDB-2103837",
        "url": "https://databank.illinois.edu/datasets/IDB-2103837",
        "types": {"resourceTypeGeneral": "Dataset"},
        "relatedIdentifiers": [
            {"relatedIdentifier": "10.1/paper", "relationType": "IsSupplementTo"}
        ],
    }}]}
    monkeypatch.setattr(probe, "_get", lambda *a, **k: (200, payload))

    hits = probe.datacite_related("10.1/paper")

    assert len(hits) == 1
    assert hits[0]["relation"] == "IsSupplementTo"


def test_software_counts_too(repo, monkeypatch):
    import repo_backlink_probe as probe

    payload = {"data": [{"attributes": {
        "doi": "10.5281/zenodo.1", "url": "https://zenodo.org/records/1",
        "types": {"resourceTypeGeneral": "Software"},
        "relatedIdentifiers": [
            {"relatedIdentifier": "10.1/paper", "relationType": "IsSupplementTo"}
        ],
    }}]}
    monkeypatch.setattr(probe, "_get", lambda *a, **k: (200, payload))

    assert len(probe.datacite_related("10.1/paper")) == 1


def test_the_match_count_travels_with_the_hits(repo, monkeypatch):
    """`total_count` is what tells a real repository from a reading list.
    Searching for *Attention Is All You Need* returns 4,147 repositories and
    essentially none of them is that paper's code."""
    import repo_backlink_probe as probe

    payload = {"total_count": 4147, "items": [
        {"full_name": "someone/awesome-papers", "html_url": "https://x",
         "description": "a reading list", "stargazers_count": 900},
    ]}
    monkeypatch.setattr(probe, "_get", lambda *a, **k: (200, payload))

    hits, total = probe.github_backlink("1706.03762", None)

    assert total == 4147
    assert len(hits) == 1, "the count is reported, not silently thresholded"


def test_a_refused_request_is_reported_not_swallowed(repo, monkeypatch):
    """`/search/code` returns 401 without a token. A blocked request must not
    look like 'no repository found' — that is a zero we did not measure."""
    import repo_backlink_probe as probe

    monkeypatch.setattr(probe, "_get", lambda *a, **k: (401, {}))

    hits, total = probe.github_backlink("2607.03443", None)

    assert hits == [{"_status": 401}]
    assert total == 0


def test_the_probe_writes_no_badge_and_no_link(repo):
    """It is a gate, not a feature. Nothing in it may touch content/."""
    source = (Path(__file__).resolve().parent.parent
              / "scripts" / "repo_backlink_probe.py").read_text(encoding="utf-8")

    assert "save_item" not in source
    assert "apply_badges" not in source
    assert "content" not in source.replace("content/", "").replace("# ", "")


def test_the_optional_token_is_optional(repo, monkeypatch):
    """An absent GITHUB_TOKEN is a supported state, not a degraded one."""
    from pipeline.config import github_token

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    assert github_token() is None
