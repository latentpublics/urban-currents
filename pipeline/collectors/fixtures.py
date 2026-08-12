"""Three hand-written papers used to exercise the pipeline without any network.

They exist so ``uc run --date … --fixture`` proves the whole chain end to end,
and so tests have deterministic input. They are not real papers.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from ..models import (
    Author,
    Bibliography,
    Ids,
    Institution,
    Item,
    PrimaryLocation,
    Provenance,
    PublicationStatus,
)
from .base import ARXIV_SOURCE_ID, arxiv_doi

_RAW = [
    {
        "arxiv": "2608.01234",
        "title": "Street-View Imagery and Pedestrian Volume: A Twelve-City Model",
        "authors": [("Rui Alvarez", "Delft University of Technology"),
                    ("Mina Park", "Seoul National University")],
        "date": "2026-08-11",
        "categories": ["cs.CV", "cs.CY"],
        "abstract": (
            "We train a convolutional model on 3.4M street-view images across 12 cities "
            "to predict pedestrian volume at 15 m resolution, using 2019-2023 automated "
            "counter data as ground truth. The model reaches R2 = 0.71 out of sample and "
            "transfers to two held-out cities with a 9-point drop. We release code and "
            "the trained weights at https://github.com/example/streetcount."
        ),
    },
    {
        "arxiv": "2608.02345",
        "title": "Transit Accessibility and Residential Sorting in Mid-Sized Metropolitan Areas",
        "authors": [("Hana Oyelaran", "University of Toronto")],
        "date": "2026-08-11",
        "categories": ["cs.CY", "stat.AP"],
        "abstract": (
            "Using a panel of 43 mid-sized metropolitan areas from 2010 to 2024, we "
            "estimate how changes in transit accessibility relate to residential sorting "
            "by income. A one standard deviation increase in accessibility is associated "
            "with a 3.1 percentage point rise in the share of high-income households "
            "within 800 m of a station. Data are drawn from public census tabulations."
        ),
    },
    {
        "arxiv": "2608.03456",
        "title": "A Graph Neural Network for Origin-Destination Flow Imputation",
        "authors": [("Tomas Weiss", "ETH Zurich"), ("Li Chen", None)],
        "date": "2026-08-10",
        "categories": ["cs.LG", "cs.SI"],
        "abstract": (
            "Origin-destination matrices from mobile phone data are sparse. We propose a "
            "graph neural network that imputes missing flows using road network structure "
            "and land use as node features. On a benchmark of four cities the method "
            "reduces mean absolute error by 18% against the strongest baseline. "
            "Neither the data nor the code is publicly released."
        ),
    },
]


def _mk(rec: dict) -> Item:
    aid = rec["arxiv"]
    return Item(
        work_key=f"arxiv:{aid}",
        first_published=date.fromisoformat(rec["date"]),
        updated=date.fromisoformat(rec["date"]),
        ids=Ids(arxiv=aid, doi=arxiv_doi(aid)),
        bibliography=Bibliography(
            title=rec["title"],
            authors=[
                Author(
                    name=n,
                    institutions=[Institution(name=inst)] if inst else [],
                )
                for n, inst in rec["authors"]
            ],
            publication_date=date.fromisoformat(rec["date"]),
            primary_location=PrimaryLocation(
                source_id=ARXIV_SOURCE_ID,
                source_name="arXiv",
                type="repository",
                version="submittedVersion",
                landing_page_url=f"https://arxiv.org/abs/{aid}",
                pdf_url=f"https://arxiv.org/pdf/{aid}",
            ),
            abstract=rec["abstract"],
            categories=rec["categories"],
        ),
        publication_status=PublicationStatus(state="preprint"),
        provenance=Provenance(collectors=["fixture"]),
    )


def fixture_items(for_date: Optional[date] = None) -> list[Item]:
    items = [_mk(r) for r in _RAW]
    if for_date is not None:
        for it in items:
            it.first_published = for_date
            it.updated = for_date
            it.bibliography.publication_date = for_date
    return items

