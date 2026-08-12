"""Summarisation (PRD §5.5).

Input is the abstract plus bibliographic metadata, nothing more — no PDFs in
Phase 0. Bibliography, links, publication status and authors are written by
collectors and never accepted from the model, which will otherwise invent an
author list and a year without hesitation.

One call per item produces both the two-layer summary and the overlay entity
candidates (methods / data / tools / places). Splitting them would double the
call count against a 60-summary budget for no gain: both need the same abstract
and the same reading.

Schema violation → one retry → ``review.status = "pending"`` and the run
continues. One item failing must never stop the day's issue.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from ..config import cfg
from ..llm import LLMBudgetExceeded, LLMClient, LLMUnavailable, parse_json
from ..metrics import Run
from ..models import Item, LlmProvenance, SummaryEn
from ..signals import Signal, apply_badges, apply_rule_signals, geographic_scope_from_llm

PROMPT_PATH = Path(__file__).parent / "prompts" / "papers.md"

REQUIRED_KEYS = ("what", "why")


def system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def user_prompt(item: Item) -> str:
    return json.dumps(
        {
            "title": item.bibliography.title,
            "abstract": item.bibliography.abstract or "",
        },
        ensure_ascii=False,
        indent=2,
    )


def cache_key(item: Item) -> str:
    return item.work_key


def validate_payload(payload: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Minimal contract check: both layers present and non-empty strings."""
    if not isinstance(payload, dict):
        return None
    for k in REQUIRED_KEYS:
        v = payload.get(k)
        if not isinstance(v, str) or not v.strip():
            return None
    return payload


def apply_payload(item: Item, payload: dict[str, Any], model: str, prompt_version: str) -> None:
    caveats = payload.get("caveats")
    item.summary.en = SummaryEn(
        what=payload["what"].strip(),
        why=payload["why"].strip(),
        caveats=caveats.strip() if isinstance(caveats, str) and caveats.strip() else None,
    )
    scope = geographic_scope_from_llm(payload.get("geographic_scope"))
    if scope is not None:
        item.signals.geographic_scope = scope
    if isinstance(payload.get("data_available"), bool):
        # Only let the model overrule a low-confidence rule verdict.
        existing = item.signals.data_available
        if existing is None or existing.confidence == "low":
            item.signals.data_available = Signal(
                value=payload["data_available"], confidence="medium", basis="llm"
            )
    item.provenance.llm = LlmProvenance(model=model, prompt_version=prompt_version)
    apply_badges(item)


def overlay_candidates(payload: dict[str, Any]) -> dict[str, list[str]]:
    """Raw strings from the model. They are only ever *candidates*; nothing
    reaches ``entities`` until it matches controlled vocabulary (PRD §9)."""
    out = {}
    for facet in ("methods", "data", "tools", "places"):
        vals = payload.get(facet) or []
        if isinstance(vals, list):
            out[facet] = [str(v).strip().lower() for v in vals if str(v).strip()][:6]
        else:
            out[facet] = []
    return out


def stash_path(run: Run) -> Path:
    return run.dir / "overlay_candidates.json"


def load_overlay_stash(run: Run) -> dict[str, dict[str, list[str]]]:
    p = stash_path(run)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_overlay_stash(run: Run, stash: dict[str, dict[str, list[str]]]) -> None:
    stash_path(run).write_text(
        json.dumps(stash, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def summarize_items(
    items: Sequence[Item],
    run: Run,
    use_llm: bool = True,
    limit: Optional[int] = None,
    client: Optional[LLMClient] = None,
) -> dict[str, Any]:
    """Summarise in place. Returns stage stats; never raises for one bad item."""
    prompt_version = cfg("llm.prompt_version", "summarize/papers@0.2.0")
    for it in items:
        apply_rule_signals(it)
        apply_badges(it)

    if not use_llm:
        return {"status": "SKIPPED", "summarized": 0, "reason": "use_llm=False"}

    client = client or LLMClient()
    if not client.available():
        run.error("summarize: ANTHROPIC_API_KEY unavailable; items left unsummarized")
        return {"status": "SKIPPED", "summarized": 0, "reason": "no_api_key"}

    system = system_prompt()
    stash = load_overlay_stash(run)
    n = 0
    failures = 0
    budget_stop: Optional[str] = None

    targets = list(items)[: limit] if limit else list(items)
    for item in targets:
        if not item.bibliography.abstract:
            item.review.status = "pending"
            continue
        try:
            resp = client.complete(
                system=system,
                user=user_prompt(item),
                cache_key=cache_key(item),
                prompt_version=prompt_version,
            )
        except LLMBudgetExceeded as e:
            budget_stop = str(e)
            run.error(f"summarize: {e}")
            break
        except LLMUnavailable as e:
            run.error(f"summarize: {e}")
            budget_stop = str(e)
            break

        payload = validate_payload(parse_json(resp.text))
        if payload is None:
            # One retry with an explicit correction, per the output contract.
            try:
                retry = client.complete(
                    system=system,
                    user=user_prompt(item)
                    + "\n\nYour previous response was not a valid JSON object with "
                    "non-empty 'what' and 'why' strings. Respond with JSON only.",
                    cache_key=cache_key(item) + ".retry",
                    prompt_version=prompt_version,
                )
                payload = validate_payload(parse_json(retry.text))
                resp = retry
            except (LLMBudgetExceeded, LLMUnavailable) as e:
                run.error(f"summarize retry: {e}")
                payload = None

        if payload is None:
            failures += 1
            item.review.status = "pending"
            run.error(f"summarize: schema violation for {item.work_key}")
            continue

        apply_payload(item, payload, model=resp.model, prompt_version=prompt_version)
        stash[item.work_key] = overlay_candidates(payload)
        item.provenance.cost_usd = round(item.provenance.cost_usd + resp.cost_usd, 6)
        item.provenance.tokens.input += resp.input_tokens
        item.provenance.tokens.output += resp.output_tokens
        run.add_cost("llm_usd", resp.cost_usd)
        run.add_tokens(resp.input_tokens, resp.output_tokens)
        n += 1

    save_overlay_stash(run, stash)
    status = "OK"
    if budget_stop:
        status = "PARTIAL"
    return {
        "status": status,
        "summarized": n,
        "failures": failures,
        "budget_stop": budget_stop,
    }
