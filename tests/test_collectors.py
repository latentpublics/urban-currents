"""Collector parsing and identifier normalisation. No network."""

from __future__ import annotations

from datetime import date

from pipeline.collectors.arxiv import ArxivCollector, entry_to_item, parse_atom
from pipeline.collectors.base import (
    normalize_arxiv_id,
    normalize_doi,
    normalize_openalex_id,
    normalize_title,
    invert_abstract,
)
from pipeline.collectors.openalex import _arxiv_id_from_work, work_to_item

ATOM_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>2</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <entry>
    <id>http://arxiv.org/abs/2608.01234v2</id>
    <updated>2026-08-12T09:00:00Z</updated>
    <published>2026-08-11T17:31:00Z</published>
    <title>Street-View Imagery and
      Pedestrian Volume</title>
    <summary>  We train a model on 3.4M images across 12 cities.
    </summary>
    <author><name>Rui Alvarez</name><arxiv:affiliation>TU Delft</arxiv:affiliation></author>
    <author><name>Mina Park</name></author>
    <link href="http://arxiv.org/abs/2608.01234v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2608.01234v2" rel="related"/>
    <arxiv:primary_category term="cs.CV"/>
    <category term="cs.CY"/>
    <category term="cs.CV"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2608.02345v1</id>
    <published>2026-08-11T04:00:00Z</published>
    <title>Transit Accessibility</title>
    <summary>Panel of 43 metropolitan areas.</summary>
    <author><name>Hana Oyelaran</name></author>
    <arxiv:primary_category term="cs.CY"/>
    <category term="cs.CY"/>
    <arxiv:doi>10.1016/j.cities.2026.104999</arxiv:doi>
  </entry>
</feed>
"""


def test_parse_atom_reads_totals_and_entries():
    page = parse_atom(ATOM_SAMPLE)
    assert page.total_results == 2
    assert len(page.entries) == 2


def test_entry_to_item_normalises_metadata():
    page = parse_atom(ATOM_SAMPLE)
    item = entry_to_item(page.entries[0])

    assert item.work_key == "arxiv:2608.01234"  # version stripped
    assert item.ids.arxiv == "2608.01234"
    assert item.ids.doi == "10.48550/arxiv.2608.01234"
    # Whitespace and newlines inside title/abstract are collapsed.
    assert item.bibliography.title == "Street-View Imagery and Pedestrian Volume"
    assert item.bibliography.abstract.startswith("We train a model")
    assert item.bibliography.publication_date == date(2026, 8, 11)
    # Primary category comes first; cross-lists follow.
    assert item.bibliography.categories[0] == "cs.CV"
    assert "cs.CY" in item.bibliography.categories
    assert item.bibliography.authors[0].institutions[0].name == "TU Delft"
    assert item.bibliography.primary_location.source_id == "S4306400194"


def test_entry_to_item_prefers_a_real_journal_doi():
    page = parse_atom(ATOM_SAMPLE)
    item = entry_to_item(page.entries[1])
    assert item.ids.doi == "10.1016/j.cities.2026.104999"


def test_date_range_query_is_inclusive_of_both_endpoints():
    q = ArxivCollector.date_range_query(["cs.CY", "cs.SI"], date(2026, 8, 1), date(2026, 8, 11))
    assert "cat:cs.CY OR cat:cs.SI" in q
    assert "submittedDate:[202608010000 TO 202608112359]" in q


def test_normalize_arxiv_id_handles_every_shape():
    assert normalize_arxiv_id("2608.01234v3") == "2608.01234"
    assert normalize_arxiv_id("arXiv:2608.01234") == "2608.01234"
    assert normalize_arxiv_id("http://arxiv.org/abs/2608.01234v2") == "2608.01234"
    assert normalize_arxiv_id("10.48550/arXiv.2608.01234") == "2608.01234"
    assert normalize_arxiv_id("https://doi.org/10.1016/j.cities.2026.1") is None
    assert normalize_arxiv_id(None) is None


def test_normalize_doi_strips_url_prefixes():
    assert normalize_doi("https://doi.org/10.1016/J.CITIES.1") == "10.1016/j.cities.1"
    assert normalize_openalex_id("https://openalex.org/W4392") == "W4392"


def test_normalize_title_is_punctuation_and_case_insensitive():
    a = normalize_title("Street-View Imagery: A 12-City Model!")
    b = normalize_title("street view imagery  a 12 city model")
    assert a == b


def test_invert_abstract_rebuilds_word_order():
    inv = {"Cities": [0], "are": [1], "complex": [2]}
    assert invert_abstract(inv) == "Cities are complex"
    assert invert_abstract(None) is None


# --------------------------------------------------------------------------
# OpenAlex Work shape
# --------------------------------------------------------------------------

WORK_JOURNAL = {
    "id": "https://openalex.org/W4392000001",
    "doi": "https://doi.org/10.1016/j.cities.2026.104999",
    "display_name": "Transit Accessibility and Residential Sorting",
    "type": "article",
    "publication_date": "2026-08-11",
    "cited_by_count": 4,
    "referenced_works": ["https://openalex.org/W2145", "https://openalex.org/W3011"],
    "related_works": ["https://openalex.org/W4123"],
    "primary_location": {
        "source": {
            "id": "https://openalex.org/S12345",
            "display_name": "Cities",
            "type": "journal",
        },
        "version": "publishedVersion",
        "landing_page_url": "https://example.org/article",
    },
    "locations": [
        {
            "source": {"id": "https://openalex.org/S4306400194", "display_name": "arXiv"},
            "landing_page_url": "https://arxiv.org/abs/2608.02345",
        }
    ],
    "authorships": [
        {
            "author": {
                "id": "https://openalex.org/A5023",
                "display_name": "Hana Oyelaran",
                "orcid": "https://orcid.org/0000-0002-1825-0097",
            },
            "institutions": [
                {"id": "https://openalex.org/I98", "ror": "https://ror.org/03dbr7087",
                 "display_name": "University of Toronto"}
            ],
        }
    ],
    "topics": [
        {"id": "https://openalex.org/T10746", "display_name": "Urban Transport Systems",
         "score": 0.82, "subfield": {"id": "https://openalex.org/subfields/3322"}}
    ],
    "primary_topic": {"id": "https://openalex.org/T10746"},
    "abstract_inverted_index": {"We": [0], "study": [1], "sorting": [2]},
}


def test_arxiv_id_is_recovered_from_a_journal_work_location():
    """PRD §5.2 rule 2 — the preprint may only be visible in locations[]."""
    assert _arxiv_id_from_work(WORK_JOURNAL) == "2608.02345"


def test_work_to_item_prefers_the_arxiv_work_key():
    item = work_to_item(WORK_JOURNAL)
    assert item.work_key == "arxiv:2608.02345"
    assert item.ids.openalex == "W4392000001"
    assert item.ids.doi == "10.1016/j.cities.2026.104999"


def test_work_to_item_takes_openalex_entities_verbatim():
    item = work_to_item(WORK_JOURNAL)
    assert item.entities.topics[0].id == "openalex:T10746"
    assert item.entities.topics[0].is_primary is True
    assert item.entities.topics[0].subfield == "3322"
    assert item.entities.people[0].id == "orcid:0000-0002-1825-0097"
    assert item.entities.orgs[0].id == "ror:https://ror.org/03dbr7087"
    # OpenAlex field names are preserved, not renamed (PRD §12).
    assert item.graph.referenced_works == ["openalex:W2145", "openalex:W3011"]
    assert item.graph.cited_by_count == 4


def test_work_to_item_marks_journal_publication_status():
    item = work_to_item(WORK_JOURNAL)
    assert item.publication_status.state == "published"
    assert item.publication_status.journal == "Cities"


def test_preprint_only_work_stays_a_preprint():
    work = dict(WORK_JOURNAL)
    work = {
        **work,
        "type": "preprint",
        "primary_location": {
            "source": {"id": "https://openalex.org/S4306400194",
                       "display_name": "arXiv", "type": "repository"},
            "landing_page_url": "https://arxiv.org/abs/2608.02345",
        },
    }
    item = work_to_item(work)
    assert item.publication_status.state == "preprint"
