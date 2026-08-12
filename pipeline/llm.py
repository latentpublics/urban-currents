"""Claude API client with an on-disk response cache.

Every LLM call goes through here, and every call is cached at
``runs/cache/{prompt_version}/{work_key}.json``. Same item + same prompt version
→ no second call. Editing a prompt bumps its version, which is what makes the
cache safe rather than merely cheap.

Two hard stops exist because a development loop leaks budget quietly:
a per-run cap and a cumulative cap tracked in ``runs/state/llm_usage.json``.

Unit tests never reach this module's network path — they inject a fake caller.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from . import paths
from .config import anthropic_key, cfg

# Published per-million-token prices for the summarize model, used only to
# estimate spend for the report. Update alongside `llm.model`.
PRICE_PER_MTOK = {
    "claude-sonnet-5": {"in": 3.0, "out": 15.0},
    "claude-opus-5": {"in": 15.0, "out": 75.0},
    "claude-haiku-4-5-20251001": {"in": 1.0, "out": 5.0},
}


class LLMUnavailable(RuntimeError):
    """No API key, or the SDK is not installed."""


class LLMBudgetExceeded(RuntimeError):
    """A call was refused because a configured cap was reached."""


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    model: str = ""

    @property
    def cost_usd(self) -> float:
        price = PRICE_PER_MTOK.get(self.model)
        if not price or self.cached:
            return 0.0
        return round(
            self.input_tokens / 1e6 * price["in"] + self.output_tokens / 1e6 * price["out"], 6
        )


@dataclass
class UsageState:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @classmethod
    def path(cls) -> Path:
        return paths.STATE / "llm_usage.json"

    @classmethod
    def load(cls) -> "UsageState":
        p = cls.path()
        if p.exists():
            try:
                return cls(**json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
        return cls()

    def save(self) -> None:
        p = self.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.__dict__, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )


def _cache_path(prompt_version: str, key: str) -> Path:
    safe_version = re.sub(r"[^A-Za-z0-9._@-]", "_", prompt_version)
    safe_key = re.sub(r"[^A-Za-z0-9._-]", "_", key)
    return paths.LLM_CACHE / safe_version / f"{safe_key}.json"


def cache_get(prompt_version: str, key: str) -> Optional[LLMResponse]:
    p = _cache_path(prompt_version, key)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return LLMResponse(
        text=data.get("text", ""),
        input_tokens=data.get("input_tokens", 0),
        output_tokens=data.get("output_tokens", 0),
        cached=True,
        model=data.get("model", ""),
    )


def cache_put(prompt_version: str, key: str, resp: LLMResponse) -> None:
    p = _cache_path(prompt_version, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "text": resp.text,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "model": resp.model,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


@dataclass
class LLMClient:
    model: str = field(default_factory=lambda: cfg("llm.model", "claude-sonnet-5"))
    max_tokens: int = field(default_factory=lambda: int(cfg("llm.max_tokens", 1200)))
    cache_enabled: bool = field(default_factory=lambda: bool(cfg("llm.cache_enabled", True)))
    max_calls_this_run: int = field(
        default_factory=lambda: int(cfg("llm.max_summaries_per_run", 30))
    )
    max_calls_total: int = field(default_factory=lambda: int(cfg("llm.max_summaries_total", 60)))
    # Injection point for tests; when set, no network call is made.
    caller: Optional[Callable[[str, str], LLMResponse]] = None

    calls_this_run: int = 0

    def _client(self):
        key = anthropic_key()
        if not key:
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic
        except ImportError as e:
            raise LLMUnavailable("anthropic SDK is not installed") from e
        return anthropic.Anthropic(api_key=key)

    def available(self) -> bool:
        if self.caller is not None:
            return True
        try:
            self._client()
            return True
        except LLMUnavailable:
            return False

    def complete(
        self,
        system: str,
        user: str,
        cache_key: str,
        prompt_version: str,
        retries: int = 2,
    ) -> LLMResponse:
        if self.cache_enabled:
            hit = cache_get(prompt_version, cache_key)
            if hit is not None:
                return hit

        if self.calls_this_run >= self.max_calls_this_run:
            raise LLMBudgetExceeded(
                f"per-run LLM cap reached ({self.max_calls_this_run} calls)"
            )
        usage = UsageState.load()
        if usage.calls >= self.max_calls_total:
            raise LLMBudgetExceeded(
                f"cumulative LLM cap reached ({usage.calls}/{self.max_calls_total} calls)"
            )

        if self.caller is not None:
            resp = self.caller(system, user)
            resp.model = resp.model or self.model
        else:
            resp = self._call_api(system, user, retries=retries)

        self.calls_this_run += 1
        usage.calls += 1
        usage.input_tokens += resp.input_tokens
        usage.output_tokens += resp.output_tokens
        usage.cost_usd = round(usage.cost_usd + resp.cost_usd, 6)
        usage.save()

        if self.cache_enabled:
            cache_put(prompt_version, cache_key, resp)
        return resp

    def _call_api(self, system: str, user: str, retries: int = 2) -> LLMResponse:
        client = self._client()
        last: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                msg = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                text = "".join(
                    block.text for block in msg.content if getattr(block, "type", "") == "text"
                )
                return LLMResponse(
                    text=text,
                    input_tokens=msg.usage.input_tokens,
                    output_tokens=msg.usage.output_tokens,
                    model=self.model,
                )
            except Exception as e:  # noqa: BLE001
                last = e
                if attempt < retries:
                    time.sleep(2**attempt)
        raise LLMUnavailable(f"Claude API call failed after {retries + 1} attempts: {last}")


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse_json(text: str) -> Optional[dict[str, Any]]:
    """Parse a JSON object out of a model response, tolerating code fences."""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-z]*\n?|```$", "", stripped, flags=re.M).strip()
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK.search(stripped)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None
