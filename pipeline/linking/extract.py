"""LLM overlay extraction (methods / data / tools / places).

Its own prompt, its own ``prompt_version``, its own model slot. This was folded
into the summarize call in Phase 0 (D8) to save calls against a 60-call budget;
the measured workload cost is a few dollars a month, so the saving was not worth
the coupling (D24).

What the model returns here are **candidates**, never tags. Nothing reaches
``entities`` until it matches controlled vocabulary — that gate lives in
``vocab_match`` and is what keeps ``entities`` free of free strings (PRD §9).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from ..llm import (
    LLMBudgetExceeded,
    LLMClient,
    LLMQuotaExhausted,
    LLMUnavailable,
    parse_json,
)
from ..metrics import Run
from ..models import Item

PROMPT_PATH = Path(__file__).parent / "prompts" / "overlay.md"

FACETS = ("methods", "data", "tools", "places")
MAX_PER_FACET = 6

_STRING_LIST = {"type": "array", "items": {"type": "string"}, "maxItems": MAX_PER_FACET}

OVERLAY_SCHEMA = {
    "type": "object",
    "properties": {facet: dict(_STRING_LIST) for facet in FACETS},
    "required": list(FACETS),
    "additionalProperties": False,
}


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


def normalize_payload(payload: Optional[dict[str, Any]]) -> Optional[dict[str, list[str]]]:
    """Clean the model's lists. Returns None if the response is unusable."""
    if not isinstance(payload, dict):
        return None
    out: dict[str, list[str]] = {}
    for facet in FACETS:
        vals = payload.get(facet) or []
        if not isinstance(vals, list):
            vals = []
        seen: set[str] = set()
        cleaned: list[str] = []
        for v in vals:
            s = str(v).strip().lower()
            if s and s not in seen:
                seen.add(s)
                cleaned.append(s)
        out[facet] = cleaned[:MAX_PER_FACET]
    return out


def extract_overlay(
    items: Sequence[Item],
    run: Run,
    client: Optional[LLMClient] = None,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    """Extract candidates for each item. Returns (by_work_key, stats).

    Never raises for one bad item: an extraction failure leaves that item with
    whatever the rule-based scan found, which is a smaller loss than a stopped run.
    """
    client = client or LLMClient(task="extract")
    if not client.available():
        return {}, {
            "status": "SKIPPED",
            "extracted": 0,
            "reason": "no_api_key",
        }

    system = system_prompt()
    results: dict[str, dict[str, list[str]]] = {}
    failures = 0
    stop_reason: Optional[str] = None

    for item in items:
        if not item.bibliography.abstract:
            continue
        try:
            resp = client.complete(
                system=system,
                user=user_prompt(item),
                cache_key=item.work_key,
                schema=OVERLAY_SCHEMA,
            )
        except LLMQuotaExhausted as e:
            stop_reason = str(e)
            run.error(f"extract: {e}")
            break
        except LLMBudgetExceeded as e:
            stop_reason = str(e)
            run.error(f"extract: {e}")
            break
        except LLMUnavailable as e:
            stop_reason = str(e)
            run.error(f"extract: {e}")
            break

        payload = normalize_payload(parse_json(resp.text))
        if payload is None:
            failures += 1
            run.error(f"extract: unusable response for {item.work_key}")
            continue

        results[item.work_key] = payload
        item.provenance.cost_usd = round(item.provenance.cost_usd + resp.cost_usd, 6)
        item.provenance.tokens.input += resp.input_tokens
        item.provenance.tokens.output += resp.output_tokens
        run.add_cost("llm_usd", resp.cost_usd)
        run.add_tokens(resp.input_tokens, resp.output_tokens)

    status = "OK"
    if stop_reason:
        status = "PARTIAL" if results else "SKIPPED"
    return results, {
        "status": status,
        "extracted": len(results),
        "failures": failures,
        "stop_reason": stop_reason,
        "prompt_version": client.prompt_version,
        "model": client.model,
    }
