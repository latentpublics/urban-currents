"""One switch, both directions (phase 0X, X5).

The site goes up at its real address before it is meant to be found, so it has
to be reachable and not indexed at the same time. Two things enforce that — the
`robots.txt` a crawler reads before anything else, and the `noindex` meta a
crawler reads when it arrived from somewhere other than `robots.txt` — and they
are driven by one value, `site.published`.

One value because two is how a repository ends up half-published: a
`robots.txt` that says no over pages that say nothing, or the reverse, because
someone flipped the switch they remembered. This file is what makes that
impossible to do by accident — it asserts both states, both artefacts, at both
depths of the site.

G5 flips one line in `config/pipeline.yaml` and these tests change sides.
"""

from __future__ import annotations

import re

import pytest

from pipeline.render import site as site_mod

BASE = "https://latentpublics.com/urban-currents"


@pytest.fixture
def unpublished(monkeypatch):
    monkeypatch.setattr(
        site_mod, "cfg",
        lambda k, d=None: {"site.published": False, "site.base_url": BASE}.get(k, d),
    )


@pytest.fixture
def published(monkeypatch):
    monkeypatch.setattr(
        site_mod, "cfg",
        lambda k, d=None: {"site.published": True, "site.base_url": BASE}.get(k, d),
    )


# --------------------------------------------------------------------------
# Off — the state this repository is actually in
# --------------------------------------------------------------------------


def test_unpublished_robots_refuses_everything(repo, unpublished, tmp_path):
    text = site_mod.build_robots(tmp_path / "robots.txt").read_text(encoding="utf-8")

    assert "Disallow: /" in text
    assert "Allow: /" not in text
    # A sitemap is an invitation. Publishing one beside `Disallow: /` states
    # two intentions in one directory.
    assert "Sitemap:" not in text


def test_unpublished_pages_carry_noindex(repo, unpublished):
    for root in ("", "../"):
        extras = site_mod.head_extras(root)
        assert 'name="robots"' in extras
        assert "noindex" in extras


def test_the_feed_link_is_there_either_way(repo, unpublished):
    """Not indexing is not the same as not being readable.

    The feed is the one way to follow this without giving anyone an address,
    and `site.py` has said so since 0k while no page linked to it.
    """
    assert 'rel="alternate"' in site_mod.head_extras("")
    assert 'href="feed.xml"' in site_mod.head_extras("")
    assert 'href="../feed.xml"' in site_mod.head_extras("../")


# --------------------------------------------------------------------------
# On — what G5 will do
# --------------------------------------------------------------------------


def test_published_robots_invites_and_points_at_the_sitemap(repo, published, tmp_path):
    text = site_mod.build_robots(tmp_path / "robots.txt").read_text(encoding="utf-8")

    assert "Allow: /" in text
    assert "Disallow: /" not in text
    assert f"Sitemap: {BASE}/sitemap.xml" in text


def test_published_pages_drop_the_noindex(repo, published):
    for root in ("", "../"):
        assert "noindex" not in site_mod.head_extras(root)
        assert 'rel="alternate"' in site_mod.head_extras(root)


def test_the_two_move_together(repo, monkeypatch, tmp_path):
    """The assertion the whole file exists for.

    Not "each is correct in isolation" — that was true of the two-switch
    version too. This walks the one value through both states and checks that
    `robots.txt` and the page meta never disagree about it.
    """
    for value in (False, True):
        monkeypatch.setattr(
            site_mod, "cfg",
            lambda k, d=None, v=value: (
                {"site.published": v, "site.base_url": BASE}.get(k, d)
            ),
        )
        robots = site_mod.build_robots(tmp_path / f"robots-{value}.txt").read_text(
            encoding="utf-8"
        )
        blocked_by_robots = "Disallow: /" in robots
        blocked_by_meta = "noindex" in site_mod.head_extras("")

        assert blocked_by_robots == blocked_by_meta == (not value), (
            f"site.published={value}: robots blocks={blocked_by_robots}, "
            f"meta blocks={blocked_by_meta} — the two disagree"
        )


# --------------------------------------------------------------------------
# The sub-path (X2)
# --------------------------------------------------------------------------


def test_absolute_urls_are_used_only_where_the_format_demands_them(
    repo, published, tmp_path
):
    """`<loc>` must be a full URL; a link between two pages must not be.

    The deployed address is a sub-path, so a root-absolute `/issues/…` would
    resolve against the organisation's domain root. Relative is also what lets
    the built site open from the filesystem.
    """
    sitemap = site_mod.build_sitemap(tmp_path / "sitemap.xml").read_text(encoding="utf-8")

    locs = re.findall(r"<loc>([^<]+)</loc>", sitemap)
    assert locs, "no urls in the sitemap"
    assert all(u.startswith(f"{BASE}/") for u in locs), locs
    # And nothing root-absolute anywhere in it.
    assert not re.search(r"<loc>/", sitemap)


def test_the_feed_names_an_author(repo, published, tmp_path):
    """RFC 4287 §4.1.1 requires it; without one the document is not a feed."""
    xml = site_mod.build_feed(tmp_path / "feed.xml").read_text(encoding="utf-8")

    assert "<author>" in xml and "<name>" in xml
    assert f'<link rel="self" href="{BASE}/feed.xml"/>' in xml
