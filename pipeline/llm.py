"""Provider-agnostic LLM client with an on-disk response cache.

Every LLM call goes through here, and every call is cached at
``runs/cache/{prompt_version}/{key}.json``. Same input + same prompt version →
no second call. Editing a prompt means bumping its version, which is what makes
the cache safe rather than merely cheap.

Two providers are supported behind one interface (``llm.provider``): Gemini and
Anthropic. The split exists because the three LLM jobs in this pipeline have
different needs — prose quality for summaries, schema compliance for extraction
— and pinning them to one vendor closes that choice off.

**Thinking is disabled explicitly.** Gemini 3.x bills reasoning tokens as
output. Measured on a one-word prompt: ``thinking_level="low"`` produced 95
thought tokens, ``thinking_budget=0`` produced none. At summarize volume that is
the difference between an estimate and a surprise, so the client sets it to 0
and records ``thinking_tokens`` on every response so the setting can be audited
rather than trusted.

Three hard stops exist because a development loop leaks budget quietly: a
per-run call cap, a cumulative call cap, and a cumulative spend cap, all tracked
in ``runs/state/llm_usage.json``.

Unit tests never reach the network path — they inject a fake caller.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from . import paths
from .config import anthropic_key, cfg, google_key

# Published per-million-token prices, used to estimate spend for the report.
# Gemini's free tier bills $0, but the paid-equivalent figure is still recorded:
# it is the only basis for projecting a monthly cost after a paid switch.
PRICE_PER_MTOK = {
    "claude-sonnet-5": {"in": 3.0, "out": 15.0},
    "claude-opus-5": {"in": 15.0, "out": 75.0},
    "claude-haiku-4-5-20251001": {"in": 1.0, "out": 5.0},
    "gemini-3.5-flash": {"in": 1.50, "out": 9.00},
    "gemini-3.5-flash-lite": {"in": 0.30, "out": 2.50},
    "gemini-2.5-flash-lite": {"in": 0.10, "out": 0.40},
}


class LLMUnavailable(RuntimeError):
    """No API key, or the SDK is not installed."""


class LLMBudgetExceeded(RuntimeError):
    """A call was refused because a configured cap was reached."""


class LLMQuotaExhausted(RuntimeError):
    """The provider's own quota is gone (free-tier daily limit).

    Distinct from ``LLMBudgetExceeded``: that is our own cap and resets when we
    say so. This one means stop, save what exists, and report — partial
    completion is not failure.
    """


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached: bool = False
    model: str = ""

    @property
    def cost_usd(self) -> float:
        """Paid-equivalent cost. Thinking tokens bill as output where they occur."""
        price = PRICE_PER_MTOK.get(self.model)
        if not price or self.cached:
            return 0.0
        out = self.output_tokens + self.thinking_tokens
        return round(self.input_tokens / 1e6 * price["in"] + out / 1e6 * price["out"], 6)


# --------------------------------------------------------------------------
# Usage state
# --------------------------------------------------------------------------


@dataclass
class UsageState:
    """Cumulative LLM usage, split by task.

    The per-task split matters because the caps are stated per task: "no more
    than 200 summaries" is a different sentence from "no more than 200 API
    calls", and extraction runs one call per item alongside summarize. Sharing
    one counter would make a summary budget silently half a summary budget.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cost_usd: float = 0.0
    by_task: dict = field(default_factory=dict)

    def task(self, name: str) -> dict:
        return self.by_task.setdefault(name, {"calls": 0, "cost_usd": 0.0})

    @classmethod
    def path(cls) -> Path:
        return paths.STATE / "llm_usage.json"

    @classmethod
    def load(cls) -> "UsageState":
        p = cls.path()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
                return cls(**known)
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


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


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
        thinking_tokens=data.get("thinking_tokens", 0),
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
                "thinking_tokens": resp.thinking_tokens,
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


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------

_RATE_LIMIT_MARKERS = ("429", "resource_exhausted", "rate limit", "quota")
_QUOTA_MARKERS = ("quota", "resource_exhausted", "exceeded your current quota")


def _looks_rate_limited(err: Exception) -> bool:
    s = f"{type(err).__name__} {err}".lower()
    return any(m in s for m in _RATE_LIMIT_MARKERS)


def _looks_quota_exhausted(err: Exception) -> bool:
    s = str(err).lower()
    # A per-minute limit says "retry in Ns"; a daily limit does not.
    if "perday" in s.replace(" ", "") or "per day" in s or "daily" in s:
        return True
    return any(m in s for m in _QUOTA_MARKERS) and "retry" not in s


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, max_tokens: int, thinking: bool = False):
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking

    def available(self) -> bool:
        if not google_key():
            return False
        try:
            import google.genai  # noqa: F401

            return True
        except ImportError:
            return False

    def _client(self):
        key = google_key()
        if not key:
            raise LLMUnavailable("GOOGLE_API_KEY is not set")
        try:
            from google import genai
        except ImportError as e:
            raise LLMUnavailable("google-genai is not installed; `uv add google-genai`") from e
        return genai.Client(api_key=key)

    def complete(self, system: str, user: str, schema: Optional[dict] = None) -> LLMResponse:
        from google.genai import types

        client = self._client()
        cfg_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": self.max_tokens,
            "temperature": 0.2,
            # thinking_budget=0 is the only setting that actually produces zero
            # thought tokens; thinking_level="minimal"/"low" still bills them.
            "thinking_config": types.ThinkingConfig(
                thinking_budget=-1 if self.thinking else 0
            ),
        }
        if schema is not None:
            cfg_kwargs["response_mime_type"] = "application/json"
            cfg_kwargs["response_json_schema"] = schema

        resp = client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        usage = resp.usage_metadata
        return LLMResponse(
            text=resp.text or "",
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            thinking_tokens=int(getattr(usage, "thoughts_token_count", 0) or 0),
            model=self.model,
        )


class AnthropicProvider:
    """Kept alongside Gemini rather than replaced — the account is out of credit,
    not the code out of date, and the two vendors suit different jobs."""

    name = "anthropic"

    def __init__(self, model: str, max_tokens: int, thinking: bool = False):
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking

    def available(self) -> bool:
        if not anthropic_key():
            return False
        try:
            import anthropic  # noqa: F401

            return True
        except ImportError:
            return False

    def _client(self):
        key = anthropic_key()
        if not key:
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic
        except ImportError as e:
            raise LLMUnavailable("anthropic SDK is not installed") from e
        return anthropic.Anthropic(api_key=key)

    def complete(self, system: str, user: str, schema: Optional[dict] = None) -> LLMResponse:
        client = self._client()
        prompt = user
        if schema is not None:
            # Anthropic's schema constraints are the tightest of the three
            # providers, so the schema is expressed in the prompt and enforced
            # client-side instead (the JSON contract is validated either way).
            prompt = (
                f"{user}\n\nRespond with a single JSON object matching this schema, "
                f"and nothing else:\n{json.dumps(schema, indent=2)}"
            )
        msg = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
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


def build_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    thinking: Optional[bool] = None,
):
    name = (provider or cfg("llm.provider", "gemini")).lower()
    model = model or cfg("llm.model", "gemini-3.5-flash")
    max_tokens = int(max_tokens or cfg("llm.max_tokens", 1200))
    think = bool(cfg("llm.thinking", False) if thinking is None else thinking)
    if name == "gemini":
        return GeminiProvider(model, max_tokens, think)
    if name == "anthropic":
        return AnthropicProvider(model, max_tokens, think)
    raise LLMUnavailable(f"unknown llm.provider {name!r}")


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


@dataclass
class LLMClient:
    """One task's LLM access: a provider, a cache, and the spend guards.

    ``task`` selects a config sub-block (``llm.summarize`` / ``llm.extract``) so
    the two jobs can move to different models and prompts independently.
    """

    task: str = "summarize"
    provider_name: Optional[str] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    cache_enabled: Optional[bool] = None
    max_calls_this_run: Optional[int] = None
    max_calls_total: Optional[int] = None
    max_spend_usd: Optional[float] = None
    # Injection point for tests; when set, no network call is made.
    caller: Optional[Callable[[str, str], LLMResponse]] = None

    calls_this_run: int = 0
    _provider: Any = field(default=None, repr=False)

    def _task_cfg(self, key: str, default: Any) -> Any:
        return cfg(f"llm.{self.task}.{key}", cfg(f"llm.{key}", default))

    def __post_init__(self) -> None:
        self.provider_name = self.provider_name or cfg("llm.provider", "gemini")
        self.model = self.model or self._task_cfg("model", "gemini-3.5-flash")
        self.max_tokens = int(self.max_tokens or self._task_cfg("max_tokens", 1200))
        if self.cache_enabled is None:
            self.cache_enabled = bool(cfg("llm.cache_enabled", True))
        if self.max_calls_this_run is None:
            self.max_calls_this_run = int(cfg("llm.max_calls_per_run", 60))
        if self.max_calls_total is None:
            self.max_calls_total = int(cfg("llm.max_calls_total", 200))
        if self.max_spend_usd is None:
            self.max_spend_usd = float(cfg("llm.max_spend_usd", 10.0))

    @property
    def prompt_version(self) -> str:
        return self._task_cfg("prompt_version", f"{self.task}@0.0.0")

    def provider(self):
        if self._provider is None:
            self._provider = build_provider(
                self.provider_name, self.model, self.max_tokens
            )
        return self._provider

    def available(self) -> bool:
        if self.caller is not None:
            return True
        try:
            return self.provider().available()
        except LLMUnavailable:
            return False

    def _check_caps(self) -> None:
        if self.calls_this_run >= self.max_calls_this_run:
            raise LLMBudgetExceeded(
                f"per-run LLM cap reached ({self.max_calls_this_run} calls)"
            )
        usage = UsageState.load()
        task_calls = usage.task(self.task)["calls"]
        if task_calls >= self.max_calls_total:
            raise LLMBudgetExceeded(
                f"cumulative {self.task} cap reached "
                f"({task_calls}/{self.max_calls_total} calls)"
            )
        if usage.cost_usd >= self.max_spend_usd:
            raise LLMBudgetExceeded(
                f"cumulative LLM spend cap reached "
                f"(${usage.cost_usd:.4f}/${self.max_spend_usd:.2f})"
            )

    def complete(
        self,
        system: str,
        user: str,
        cache_key: str,
        prompt_version: Optional[str] = None,
        schema: Optional[dict] = None,
        retries: int = 3,
    ) -> LLMResponse:
        version = prompt_version or self.prompt_version
        if self.cache_enabled:
            hit = cache_get(version, cache_key)
            if hit is not None:
                return hit

        self._check_caps()

        if self.caller is not None:
            resp = self.caller(system, user)
            resp.model = resp.model or self.model or ""
        else:
            resp = self._call_with_backoff(system, user, schema, retries)

        self.calls_this_run += 1
        usage = UsageState.load()
        usage.calls += 1
        usage.input_tokens += resp.input_tokens
        usage.output_tokens += resp.output_tokens
        usage.thinking_tokens += resp.thinking_tokens
        usage.cost_usd = round(usage.cost_usd + resp.cost_usd, 6)
        bucket = usage.task(self.task)
        bucket["calls"] += 1
        bucket["cost_usd"] = round(bucket["cost_usd"] + resp.cost_usd, 6)
        usage.save()

        if self.cache_enabled:
            cache_put(version, cache_key, resp)
        return resp

    def _call_with_backoff(
        self, system: str, user: str, schema: Optional[dict], retries: int
    ) -> LLMResponse:
        provider = self.provider()
        last: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                return provider.complete(system, user, schema)
            except Exception as e:  # noqa: BLE001
                last = e
                if _looks_quota_exhausted(e):
                    # A daily free-tier limit does not clear by waiting a few
                    # seconds. Stop and let the caller save what it has.
                    raise LLMQuotaExhausted(
                        f"{provider.name} quota exhausted: {str(e)[:200]}"
                    ) from e
                if _looks_rate_limited(e) and attempt < retries:
                    time.sleep(min(60.0, 5.0 * (2**attempt)))
                    continue
                if attempt < retries:
                    time.sleep(2**attempt)
        raise LLMUnavailable(
            f"{self.provider_name} call failed after {retries + 1} attempts: {last}"
        )


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse_json(text: str) -> Optional[dict[str, Any]]:
    """Parse a JSON object out of a model response, tolerating code fences.

    Constrained decoding makes this nearly always trivial, but the Anthropic
    path is prompt-constrained rather than schema-constrained, so the tolerant
    parser stays.
    """
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
