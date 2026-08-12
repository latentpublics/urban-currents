"""Provider abstraction, cost accounting, and the spend guards.

No test here reaches a network. The provider classes are exercised through
stubs; the one thing that cannot be faked — whether thinking is actually off —
is asserted on the config the client builds.
"""

from __future__ import annotations

import json

import pytest

from pipeline.llm import (
    AnthropicProvider,
    GeminiProvider,
    LLMBudgetExceeded,
    LLMClient,
    LLMQuotaExhausted,
    LLMResponse,
    LLMUnavailable,
    UsageState,
    _looks_quota_exhausted,
    _looks_rate_limited,
    build_provider,
    parse_json,
)


# --------------------------------------------------------------------------
# Cost accounting
# --------------------------------------------------------------------------


def test_gemini_cost_uses_published_prices():
    r = LLMResponse(text="x", input_tokens=1_000_000, output_tokens=1_000_000,
                    model="gemini-3.5-flash")
    assert r.cost_usd == pytest.approx(1.50 + 9.00)


def test_thinking_tokens_are_billed_as_output():
    """They are the actual cost risk, so they cannot be quietly excluded."""
    without = LLMResponse(text="x", input_tokens=0, output_tokens=1000,
                          model="gemini-3.5-flash")
    with_thoughts = LLMResponse(text="x", input_tokens=0, output_tokens=1000,
                                thinking_tokens=3000, model="gemini-3.5-flash")
    assert with_thoughts.cost_usd == pytest.approx(without.cost_usd * 4)


def test_cached_responses_cost_nothing():
    r = LLMResponse(text="x", input_tokens=10_000, output_tokens=10_000,
                    cached=True, model="gemini-3.5-flash")
    assert r.cost_usd == 0.0


def test_unknown_model_reports_zero_rather_than_guessing():
    r = LLMResponse(text="x", input_tokens=10_000, output_tokens=10_000,
                    model="some-model-we-have-no-price-for")
    assert r.cost_usd == 0.0


# --------------------------------------------------------------------------
# Provider selection and thinking
# --------------------------------------------------------------------------


def test_provider_is_selected_from_config(repo):
    assert isinstance(build_provider("gemini"), GeminiProvider)
    assert isinstance(build_provider("anthropic"), AnthropicProvider)
    with pytest.raises(LLMUnavailable):
        build_provider("some-other-vendor")


def test_thinking_is_off_by_default(repo):
    """thinking_budget=0 is the only setting that yields zero thought tokens;
    `thinking_level` values still bill them."""
    p = build_provider("gemini")
    assert p.thinking is False


def test_gemini_request_config_disables_thinking_and_sets_the_schema(repo, monkeypatch):
    """Assert on the request actually handed to the SDK, not on intent."""
    from google.genai import types

    captured = {}

    class FakeModels:
        def generate_content(self, model, contents, config):
            captured["model"] = model
            captured["config"] = config
            return type(
                "R",
                (),
                {
                    "text": '{"ok": true}',
                    "usage_metadata": type(
                        "U", (), {"prompt_token_count": 5, "candidates_token_count": 7,
                                  "thoughts_token_count": 0},
                    )(),
                },
            )()

    provider = GeminiProvider("gemini-3.5-flash", 900, thinking=False)
    monkeypatch.setattr(provider, "_client", lambda: type("C", (), {"models": FakeModels()})())

    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    resp = provider.complete("sys", "user", schema)

    cfg = captured["config"]
    assert isinstance(cfg.thinking_config, types.ThinkingConfig)
    assert cfg.thinking_config.thinking_budget == 0
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema == schema
    assert cfg.max_output_tokens == 900
    assert resp.input_tokens == 5 and resp.output_tokens == 7
    assert resp.thinking_tokens == 0


def test_anthropic_puts_the_schema_in_the_prompt(repo, monkeypatch):
    """Anthropic's schema constraints are the tightest of the three providers,
    so the contract is carried in the prompt and validated client-side."""
    captured = {}

    class FakeMessages:
        def create(self, **kw):
            captured.update(kw)
            return type(
                "M", (), {
                    "content": [type("B", (), {"type": "text", "text": "{}"})()],
                    "usage": type("U", (), {"input_tokens": 1, "output_tokens": 2})(),
                },
            )()

    provider = AnthropicProvider("claude-haiku-4-5-20251001", 500)
    monkeypatch.setattr(
        provider, "_client", lambda: type("C", (), {"messages": FakeMessages()})()
    )
    provider.complete("sys", "user", {"type": "object"})
    assert "type" in captured["messages"][0]["content"]
    assert captured["system"] == "sys"


# --------------------------------------------------------------------------
# Rate limits vs quota exhaustion
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "429 RESOURCE_EXHAUSTED: rate limit exceeded, retry in 12s",
        "Rate limit reached for model",
    ],
)
def test_transient_rate_limits_are_retried(message):
    err = RuntimeError(message)
    assert _looks_rate_limited(err)
    assert not _looks_quota_exhausted(err)


@pytest.mark.parametrize(
    "message",
    [
        "429 RESOURCE_EXHAUSTED: quota exceeded for GenerateRequestsPerDay",
        "You exceeded your current quota (daily limit)",
    ],
)
def test_daily_quota_is_not_retried(message):
    """Waiting five seconds does not restore a daily free-tier limit; the caller
    has to save progress and stop."""
    assert _looks_quota_exhausted(RuntimeError(message))


def test_quota_exhaustion_surfaces_as_its_own_exception(repo, monkeypatch):
    client = LLMClient(task="summarize")

    class Boom:
        name = "gemini"

        def complete(self, *a, **kw):
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded per day")

    monkeypatch.setattr(client, "provider", lambda: Boom())
    with pytest.raises(LLMQuotaExhausted):
        client.complete("s", "u", cache_key="k", prompt_version="v@1")


# --------------------------------------------------------------------------
# Spend guards
# --------------------------------------------------------------------------


def _ok_caller(system, user):
    return LLMResponse(text=json.dumps({"what": "a", "why": "b"}), input_tokens=10,
                       output_tokens=5, model="gemini-3.5-flash")


def test_cumulative_spend_cap_blocks_further_calls(repo):
    usage = UsageState.load()
    usage.cost_usd = 99.0
    usage.save()

    client = LLMClient(task="summarize", caller=_ok_caller, max_spend_usd=10.0)
    with pytest.raises(LLMBudgetExceeded, match="spend cap"):
        client.complete("s", "u", cache_key="k", prompt_version="v@1")


def test_usage_state_accumulates_thinking_tokens(repo):
    def thinky(system, user):
        return LLMResponse(text="{}", input_tokens=100, output_tokens=50,
                           thinking_tokens=200, model="gemini-3.5-flash")

    client = LLMClient(task="summarize", caller=thinky)
    client.complete("s", "u", cache_key="k1", prompt_version="v@1")

    usage = UsageState.load()
    assert usage.calls == 1
    assert usage.thinking_tokens == 200
    assert usage.cost_usd > 0


def test_per_task_config_gives_each_task_its_own_prompt_version(repo):
    assert LLMClient(task="summarize").prompt_version.startswith("summarize/")
    assert LLMClient(task="extract").prompt_version.startswith("extract/")


def test_parse_json_still_tolerates_fences():
    assert parse_json('```json\n{"a":1}\n```') == {"a": 1}
    assert parse_json("nope") is None
