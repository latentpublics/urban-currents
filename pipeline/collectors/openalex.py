"""OpenAlex collector (PRD §5.1, §4.3).

Uses ``pyalex`` rather than hand-rolled HTTP: cursor paging and retries are
already solved there.

Two jobs:

1. **Journal incremental** — Works published in whitelist journals since a date.
2. **Enrichment** — for arXiv Items already collected, find the matching Work and
   fill ``ids.openalex``, ``graph.*`` and the OpenAlex-native entities. This is
   best-effort: arXiv preprints are often not indexed for days, and a miss just
   queues the Item in ``openalex_enrich_pending`` for a later attempt.

Budget: every response's ``meta.cost_usd`` is accumulated and the stage stops at
80% of the daily budget (PRD §10). Charging is per page, which is where the cost
actually lands.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any, Iterable, Iterator, Optional

from ..config import cfg, contact_email, journals_vocab, openalex_key
from ..metrics import BudgetExceeded, OpenAlexBudget, Run
from ..models import Bibliography, Ids, Item, PrimaryLocation, Provenance, PublicationStatus
from .base import (
    repository_urls_from_work,
    ARXIV_SOURCE_ID,
    clean_text,
    invert_abstract,
    normalize_arxiv_id,
    normalize_doi,
    normalize_openalex_id,
)


class OpenAlexUnavailable(RuntimeError):
    """No API key — the daily budget without one is $0."""


def configure_pyalex():
    key = openalex_key()
    if not key:
        raise OpenAlexUnavailable(
            "OPENALEX_KEY is not set; the no-key daily budget is $0 as of 2026-08"
        )
    import pyalex

    pyalex.config.api_key = key
    pyalex.config.email = contact_email()
    pyalex.config.max_retries = 3
    pyalex.config.retry_backoff_factor = 0.5
    pyalex.config.user_agent = "urban-currents/0.2"
    return pyalex


class OpenAlexCollector:
    def __init__(self, run: Run, budget: Optional[OpenAlexBudget] = None):
        self.run = run
        self.budget = budget or OpenAlexBudget(
            daily_usd=float(cfg("openalex.daily_budget_usd", 1.0)),
            stop_fraction=float(cfg("openalex.budget_stop_fraction", 0.8)),
        )
        self.per_page = int(cfg("openalex.per_page", 100))
        self._pyalex = None

    def _api(self):
        if self._pyalex is None:
            self._pyalex = configure_pyalex()
        return self._pyalex

    # -- budget ----------------------------------------------------------

    def _charge(self, meta: dict[str, Any] | None) -> None:
        cost = float((meta or {}).get("cost_usd") or 0.0)
        self.run.add_cost("openalex_usd", cost)
        self.budget.charge(cost)

    # -- journal incremental --------------------------------------------

    def whitelist_source_ids(self) -> list[str]:
        doc = journals_vocab()
        return [
            s["id"]
            for s in (doc.get("sources") or [])
            if s.get("id") and s.get("include", True)
        ]

    def collect_journals(
        self,
        d: date,
        backfill_from: Optional[date] = None,
        max_pages: int = 20,
    ) -> list[Item]:
        source_ids = self.whitelist_source_ids()
        if not source_ids:
            self.run.error(
                "collect.openalex: vocab/sources/journals.yaml is empty; "
                "run scripts/build_journal_whitelist.py"
            )
            return []

        pyalex = self._api()
        start = backfill_from or d
        items: list[Item] = []

        # OpenAlex allows ~50 OR'd IDs per filter comfortably; chunk to stay safe.
        for chunk_no, chunk in enumerate(_chunks(source_ids, 40)):
            query = (
                pyalex.Works()
                .filter(
                    primary_location={"source": {"id": "|".join(chunk)}},
                    from_publication_date=str(start),
                    to_publication_date=str(d),
                )
                .sort(publication_date="desc")
            )
            try:
                for page_no, page in enumerate(
                    query.paginate(per_page=self.per_page, n_max=None)
                ):
                    self._charge(getattr(page, "meta", None))
                    self.run.write_raw(
                        f"openalex/journals_{start}_{d}_c{chunk_no:02d}_p{page_no:03d}.json",
                        json.dumps(list(page), ensure_ascii=False, indent=1),
                    )
                    for work in page:
                        try:
                            item = work_to_item(work)
                        except Exception as e:  # noqa: BLE001
                            # One malformed Work must not cost the whole page.
                            self.run.error(
                                f"collect.openalex: skipped {work.get('id')}: "
                                f"{type(e).__name__}"
                            )
                            continue
                        if item is not None:
                            items.append(item)
                    if page_no + 1 >= max_pages:
                        break
            except BudgetExceeded as e:
                self.run.error(f"collect.openalex: {e}")
                self.run.stage("collect.openalex", "PARTIAL")
                break

        return items

    # -- enrichment ------------------------------------------------------

    def find_work_for_arxiv(
        self, arxiv_id: str, title: Optional[str] = None, deep: bool = False
    ) -> Optional[dict]:
        """DOI singleton lookup; optionally fall back to a title search.

        The title fallback is **off by default** and it is a cost decision, not a
        quality one. Measured on 2026-08-11: 100 same-day arXiv items, 7 hits, and
        $0.089 spent — 9% of the daily budget — because every miss falls through
        to a full-text search at search rates. The DOI singleton is free and finds
        everything OpenAlex has already indexed. Anything it misses is simply not
        indexed yet, which the pending queue retries on later days for free.
        """
        pyalex = self._api()
        doi = f"10.48550/arxiv.{arxiv_id}"
        try:
            work = pyalex.Works()[f"doi:{doi}"]
            if work:
                return dict(work)
        except Exception:
            pass

        if not deep or not title:
            return None
        try:
            results = (
                pyalex.Works()
                .search_filter(title=title)
                .get(per_page=5)
            )
            self._charge(getattr(results, "meta", None))
        except BudgetExceeded:
            raise
        except Exception:
            return None

        from .base import normalize_title

        want = normalize_title(title)
        for work in results:
            if normalize_title(work.get("display_name") or "") == want:
                return dict(work)
            for loc in work.get("locations") or []:
                src = (loc.get("source") or {}).get("id") or ""
                if normalize_openalex_id(src) == ARXIV_SOURCE_ID:
                    if normalize_arxiv_id(loc.get("landing_page_url")) == arxiv_id:
                        return dict(work)
        return None

    def enrich(
        self,
        items: Iterable[Item],
        max_lookups: int = 400,
        deep: bool = False,
        retry_archive: bool = True,
    ) -> dict[str, Any]:
        """Fill OpenAlex fields on arXiv Items. A miss is normal, not an error.

        Today's items are tried first, then Items from earlier days still sitting
        in the pending queue — arXiv preprints commonly show up in OpenAlex days
        after they appear, so a single attempt on the day of collection would
        leave most Items permanently without a citation graph.
        """
        from .. import store
        from ..linking.openalex_passthrough import apply_passthrough

        queue = EnrichQueue.load()
        enriched = 0
        attempted = 0
        targets: list[Item] = [
            it for it in items if it.ids.arxiv and not it.ids.openalex
        ]
        fresh_keys = {it.work_key for it in targets}

        if retry_archive:
            for work_key in queue.due(exclude=fresh_keys):
                archived = store.load_item(work_key)
                if archived is None:
                    queue.drop(work_key)
                elif archived.ids.openalex:
                    queue.drop(work_key)
                else:
                    targets.append(archived)

        updated_archive: list[Item] = []
        for item in targets:
            if attempted >= max_lookups:
                queue.defer(item.work_key)
                continue
            attempted += 1
            try:
                work = self.find_work_for_arxiv(
                    item.ids.arxiv, item.bibliography.title, deep=deep
                )
            except BudgetExceeded as e:
                self.run.error(f"enrich: {e}")
                queue.defer(item.work_key)
                break
            except Exception as e:  # noqa: BLE001
                self.run.error(f"enrich {item.work_key}: {type(e).__name__}: {e}")
                queue.defer(item.work_key)
                continue

            if not work:
                queue.defer(item.work_key)
                continue

            apply_passthrough(item, work)
            _apply_publication_status(item, work)
            queue.drop(item.work_key)
            enriched += 1
            if item.work_key not in fresh_keys:
                updated_archive.append(item)

        # Items pulled back out of the archive are saved here; today's items flow
        # on through the stages and are saved by `uc issue`.
        for item in updated_archive:
            store.save_item(item)

        queue.save()
        self.run.metrics.linking.openalex_enrich_pending = len(queue.entries)
        return {
            "enriched": enriched,
            "pending": len(queue.entries),
            "attempted": attempted,
            "revived": len(updated_archive),
        }


class EnrichQueue:
    """Work keys awaiting an OpenAlex match, with an attempt count.

    Persisted outside ``content/`` because it is run state, not published data.
    Attempts are capped: a preprint that has not appeared after this many tries
    probably never will, and retrying it forever would quietly consume budget.
    """

    MAX_ATTEMPTS = 6

    def __init__(self, entries: dict[str, int]):
        self.entries = entries

    @staticmethod
    def path():
        from ..paths import STATE

        return STATE / "openalex_enrich_pending.json"

    @classmethod
    def load(cls) -> "EnrichQueue":
        p = cls.path()
        if p.exists():
            try:
                return cls(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
        return cls({})

    def save(self) -> None:
        p = self.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.entries, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def due(self, exclude: set[str], limit: int = 200) -> list[str]:
        """Least-attempted first, so nothing starves behind a stuck batch."""
        ready = [
            (n, k)
            for k, n in self.entries.items()
            if k not in exclude and n < self.MAX_ATTEMPTS
        ]
        ready.sort()
        return [k for _, k in ready[:limit]]

    def defer(self, work_key: str) -> None:
        self.entries[work_key] = self.entries.get(work_key, 0) + 1
        if self.entries[work_key] >= self.MAX_ATTEMPTS:
            self.entries.pop(work_key, None)

    def drop(self, work_key: str) -> None:
        self.entries.pop(work_key, None)


def _chunks(seq: list[str], n: int) -> Iterator[list[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# --------------------------------------------------------------------------
# Work → Item
# --------------------------------------------------------------------------


def _primary_location(work: dict) -> PrimaryLocation:
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    return PrimaryLocation(
        source_id=normalize_openalex_id(src.get("id")),
        source_name=src.get("display_name"),
        type=src.get("type") or loc.get("version"),
        version=loc.get("version"),
        landing_page_url=loc.get("landing_page_url"),
        pdf_url=loc.get("pdf_url"),
    )


def _arxiv_id_from_work(work: dict) -> Optional[str]:
    """Rule 2 of the merge keys: an arXiv location inside the Work (PRD §5.2)."""
    ids = work.get("ids") or {}
    from_doi = normalize_arxiv_id(ids.get("doi") or work.get("doi"))
    if from_doi:
        return from_doi
    for loc in work.get("locations") or []:
        src = (loc.get("source") or {}).get("id") or ""
        if normalize_openalex_id(src) == ARXIV_SOURCE_ID:
            aid = normalize_arxiv_id(loc.get("landing_page_url")) or normalize_arxiv_id(
                loc.get("pdf_url")
            )
            if aid:
                return aid
    return None


def _apply_publication_status(item: Item, work: dict) -> None:
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    src_id = normalize_openalex_id(src.get("id"))
    wtype = (work.get("type") or "").lower()
    is_preprint = wtype == "preprint" or src_id == ARXIV_SOURCE_ID
    if is_preprint:
        return
    if src.get("type") == "journal" or wtype in ("article", "book-chapter", "review"):
        item.publication_status = PublicationStatus(
            state="published",
            journal=src.get("display_name"),
            source_id=src_id,
            doi=normalize_doi(work.get("doi")),
            detected_at=datetime.now(timezone.utc).replace(microsecond=0),
        )


def work_to_item(work: dict) -> Optional[Item]:
    from ..linking.openalex_passthrough import (
        authors_from_work,
        graph_from_work,
        orgs_from_work,
        people_from_work,
        topics_from_work,
    )

    # OpenAlex passes publisher markup through: JATS `<scp>`, `<i>`, `<sub>`
    # and friends turn up inside `display_name` (P4-1).
    title = clean_text(work.get("display_name") or work.get("title"))
    if not title:
        return None

    oa_id = normalize_openalex_id(work.get("id"))
    doi = normalize_doi(work.get("doi"))
    arxiv_id = _arxiv_id_from_work(work)

    # work_key priority: arXiv ID → DOI → OpenAlex ID (PRD §5.2).
    if arxiv_id:
        work_key = f"arxiv:{arxiv_id}"
    elif doi:
        work_key = f"doi:{doi}"
    elif oa_id:
        work_key = f"openalex:{oa_id}"
    else:
        return None

    pub_date = work.get("publication_date")
    item = Item(
        work_key=work_key,
        first_published=date.fromisoformat(pub_date) if pub_date else None,
        updated=date.fromisoformat(pub_date) if pub_date else None,
        ids=Ids(openalex=oa_id, doi=doi, arxiv=arxiv_id),
        bibliography=Bibliography(
            title=title,
            authors=authors_from_work(work),
            publication_date=date.fromisoformat(pub_date) if pub_date else None,
            primary_location=_primary_location(work),
            abstract=invert_abstract(work.get("abstract_inverted_index")),
            repository_urls=repository_urls_from_work(work),
        ),
        graph=graph_from_work(work),
        provenance=Provenance(
            collected_at=datetime.now(timezone.utc).replace(microsecond=0),
            collectors=["openalex"],
        ),
    )
    item.entities.topics = topics_from_work(work)
    item.entities.people = people_from_work(work)
    item.entities.orgs = orgs_from_work(work)
    _apply_publication_status(item, work)
    if work.get("is_retracted"):
        from ..models import Signal

        item.signals.is_retracted = Signal(value=True, confidence="high", basis="openalex")
    return item
