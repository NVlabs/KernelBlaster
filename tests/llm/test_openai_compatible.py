from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

from src.kernelblaster.llm import LLMBudgetExceeded
from src.kernelblaster.llm.openai_compatible import (
    OpenAICompatibleProvider,
    OpenAICompatibleSettings,
)
import src.kernelblaster.llm.openai_compatible as provider_module


def _response(
    content: str,
    *,
    request_id: str = "request-1",
    completion_id: str = "completion-1",
    model: str = "gpt-5.6-terra",
    usage: dict | None = None,
):
    return SimpleNamespace(
        id=completion_id,
        _request_id=request_id,
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=usage,
    )


def _status_error(status_code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return openai.APIStatusError("test status", response=response, body=None)


class FakeCompletions:
    def __init__(self, behavior):
        self.behavior = behavior
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        return await self.behavior(index, kwargs)


def _client(behavior):
    completions = FakeCompletions(behavior)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        completions=completions,
    )


def _settings(**overrides) -> OpenAICompatibleSettings:
    values = {
        "base_url": "https://example.test/v1?api_key=must-not-leak",
        "api_key": "unit-test-key",
        "max_retries": 0,
        "max_completion_tokens": 32,
        "max_total_tokens": 100_000,
        "reasoning_effort": "low",
    }
    values.update(overrides)
    return OpenAICompatibleSettings(**values)


@pytest.mark.asyncio
async def test_client_side_fanout_preserves_order_and_uses_n_one():
    async def behavior(index, _kwargs):
        await asyncio.sleep((3 - index) * 0.002)
        return _response(
            f"candidate-{index}",
            request_id=f"request-{index}",
            usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        )

    fake = _client(behavior)
    provider = OpenAICompatibleProvider(_settings(), client=fake)
    result = await provider.generate(
        [{"role": "user", "content": "generate"}],
        model="gpt-5.6-terra",
        n=4,
    )

    assert result.generations == [f"candidate-{index}" for index in range(4)]
    assert result.response_models == ["gpt-5.6-terra"] * 4
    assert result.request_ids == [f"request-{index}" for index in range(4)]
    assert result.usage["total_tokens"] == 20
    assert len(fake.completions.calls) == 4
    assert all(call["n"] == 1 for call in fake.completions.calls)
    assert all(call["reasoning_effort"] == "low" for call in fake.completions.calls)
    assert all(call["max_completion_tokens"] == 32 for call in fake.completions.calls)


@pytest.mark.asyncio
async def test_completion_id_is_not_reported_as_http_request_id():
    async def behavior(_index, _kwargs):
        response = _response(
            "ok",
            request_id="",
            completion_id="chatcmpl-not-a-request-id",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
        del response._request_id
        return response

    provider = OpenAICompatibleProvider(_settings(), client=_client(behavior))
    result = await provider.generate([{"role": "user", "content": "x"}], "model")

    assert result.request_ids == []


@pytest.mark.asyncio
async def test_concurrency_never_exceeds_configured_limit():
    active = 0
    maximum = 0
    release = asyncio.Event()

    async def behavior(index, _kwargs):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await release.wait()
        active -= 1
        return _response(
            f"candidate-{index}",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    fake = _client(behavior)
    provider = OpenAICompatibleProvider(
        _settings(max_concurrency=2),
        client=fake,
    )
    task = asyncio.create_task(
        provider.generate([{"role": "user", "content": "x"}], "model", n=4)
    )
    while len(fake.completions.calls) < 2:
        await asyncio.sleep(0)
    assert maximum == 2
    release.set()
    await task
    assert maximum == 2


@pytest.mark.parametrize("status_code", [429, 500, 503])
@pytest.mark.asyncio
async def test_retryable_statuses_are_retried(monkeypatch, status_code):
    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(provider_module.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(provider_module.random, "uniform", lambda *_args: 0)

    async def behavior(index, _kwargs):
        if index == 0:
            raise _status_error(status_code)
        return _response(
            "ok",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    fake = _client(behavior)
    provider = OpenAICompatibleProvider(_settings(max_retries=1), client=fake)
    result = await provider.generate([{"role": "user", "content": "x"}], "model")
    assert result.response == "ok"
    assert len(fake.completions.calls) == 2
    assert result.attempts == 2


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 408, 409, 422])
@pytest.mark.asyncio
async def test_non_retryable_statuses_fail_immediately(status_code):
    async def behavior(_index, _kwargs):
        raise _status_error(status_code)

    fake = _client(behavior)
    provider = OpenAICompatibleProvider(_settings(max_retries=3), client=fake)
    with pytest.raises(openai.APIStatusError):
        await provider.generate([{"role": "user", "content": "x"}], "model")
    assert len(fake.completions.calls) == 1


@pytest.mark.parametrize("error_type", [openai.APITimeoutError, openai.APIConnectionError])
@pytest.mark.asyncio
async def test_transport_errors_are_retried(monkeypatch, error_type):
    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(provider_module.asyncio, "sleep", no_sleep)
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")

    async def behavior(index, _kwargs):
        if index == 0:
            raise error_type(request=request)
        return _response(
            "ok",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    fake = _client(behavior)
    provider = OpenAICompatibleProvider(_settings(max_retries=1), client=fake)
    await provider.generate([{"role": "user", "content": "x"}], "model")
    assert len(fake.completions.calls) == 2


@pytest.mark.asyncio
async def test_request_budget_stops_new_requests():
    async def behavior(index, _kwargs):
        return _response(
            str(index),
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    fake = _client(behavior)
    provider = OpenAICompatibleProvider(_settings(max_requests=2), client=fake)
    with pytest.raises(LLMBudgetExceeded):
        await provider.generate([{"role": "user", "content": "x"}], "model", n=3)
    assert len(fake.completions.calls) == 2


@pytest.mark.asyncio
async def test_concurrent_token_reservations_enforce_hard_cap():
    release = asyncio.Event()

    async def behavior(index, _kwargs):
        await release.wait()
        return _response(
            str(index),
            usage={"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        )

    fake = _client(behavior)
    provider = OpenAICompatibleProvider(
        _settings(max_completion_tokens=20, max_total_tokens=70),
        client=fake,
    )
    task = asyncio.create_task(
        provider.generate([{"role": "user", "content": "x"}], "model", n=3)
    )
    while len(fake.completions.calls) < 2:
        await asyncio.sleep(0)
    release.set()
    with pytest.raises(LLMBudgetExceeded):
        await task
    assert len(fake.completions.calls) == 2


@pytest.mark.asyncio
async def test_missing_usage_is_estimated():
    async def behavior(_index, _kwargs):
        return _response("estimated response", usage=None)

    provider = OpenAICompatibleProvider(_settings(), client=_client(behavior))
    result = await provider.generate(
        [{"role": "user", "content": "estimate me"}], "model"
    )
    assert result.usage_source == "estimated"
    assert result.usage["prompt_tokens"] > 0
    assert result.usage["completion_tokens"] > 0


@pytest.mark.asyncio
async def test_partial_fanout_failure_never_returns_partial_success():
    async def behavior(index, _kwargs):
        if index == 1:
            raise _status_error(400)
        return _response(
            f"candidate-{index}",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    fake = _client(behavior)
    provider = OpenAICompatibleProvider(_settings(), client=fake)
    with pytest.raises(openai.APIStatusError):
        await provider.generate([{"role": "user", "content": "x"}], "model", n=3)
    assert len(fake.completions.calls) == 3


def test_public_config_never_contains_api_key():
    async def behavior(_index, _kwargs):
        raise AssertionError("not called")

    provider = OpenAICompatibleProvider(_settings(), client=_client(behavior))
    config = provider.public_config()
    assert "unit-test-key" not in repr(config)
    assert config["base_url"] == "https://example.test/v1"
    assert config["api_key_configured"] is True


def test_provider_instances_keep_api_keys_isolated(monkeypatch):
    constructed: list[dict] = []

    def fake_openai(**kwargs):
        constructed.append(kwargs)
        return _client(lambda *_args: None)

    monkeypatch.setattr(provider_module.openai, "AsyncOpenAI", fake_openai)
    first = OpenAICompatibleProvider(_settings(api_key="key-for-first"))
    second = OpenAICompatibleProvider(_settings(api_key="key-for-second"))

    assert [item["api_key"] for item in constructed] == [
        "key-for-first",
        "key-for-second",
    ]
    assert "key-for-first" not in repr(first.public_config())
    assert "key-for-second" not in repr(second.public_config())
