import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.exceptions import RateLimitError, Timeout

from llm_utils import MAX_ATTEMPTS, acompletion_with_retry


def _rate_limit():
    return RateLimitError("rate limited", "openrouter", "m")


def _timeout():
    return Timeout("timed out", "m", "openrouter")


@pytest.mark.asyncio
async def test_retries_rate_limit_then_succeeds():
    ok = MagicMock()
    api = AsyncMock(side_effect=[_rate_limit(), ok])
    with patch("llm_utils.litellm.acompletion", api), \
         patch("llm_utils.asyncio.sleep", AsyncMock()):
        result = await acompletion_with_retry(model="m", messages=[])
    assert result is ok
    assert api.call_count == 2


@pytest.mark.asyncio
async def test_raises_after_max_attempts():
    api = AsyncMock(side_effect=[_rate_limit() for _ in range(MAX_ATTEMPTS)])
    with patch("llm_utils.litellm.acompletion", api), \
         patch("llm_utils.asyncio.sleep", AsyncMock()):
        with pytest.raises(RateLimitError):
            await acompletion_with_retry(model="m", messages=[])
    assert api.call_count == MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_rate_limit_backs_off_longer_than_other_errors():
    sleep = AsyncMock()
    ok = MagicMock()
    with patch("llm_utils.litellm.acompletion", AsyncMock(side_effect=[_rate_limit(), ok])), \
         patch("llm_utils.asyncio.sleep", sleep):
        await acompletion_with_retry(model="m", messages=[])
    rate_limit_delay = sleep.call_args_list[0].args[0]

    sleep.reset_mock()
    with patch("llm_utils.litellm.acompletion", AsyncMock(side_effect=[_timeout(), ok])), \
         patch("llm_utils.asyncio.sleep", sleep):
        await acompletion_with_retry(model="m", messages=[])
    timeout_delay = sleep.call_args_list[0].args[0]

    assert rate_limit_delay >= 5.0
    assert timeout_delay < rate_limit_delay


def _content_response(content):
    r = MagicMock()
    r.choices[0].message.content = content
    r.choices[0].message.tool_calls = None  # a plain completion has no tool calls
    return r


@pytest.mark.asyncio
async def test_retry_on_blank_retries_empty_content():
    """Provider failures (finish_reason 'error') come back as a 200 with blank
    content, never an exception — opt-in retry must catch them."""
    blank, ok = _content_response(""), _content_response("{}")
    api = AsyncMock(side_effect=[blank, ok])
    with patch("llm_utils.litellm.acompletion", api), \
         patch("llm_utils.asyncio.sleep", AsyncMock()):
        result = await acompletion_with_retry(model="m", messages=[], retry_on_blank=True)
    assert result is ok
    assert api.call_count == 2


@pytest.mark.asyncio
async def test_retry_on_blank_returns_last_response_when_all_blank():
    """Callers keep their own parse-failure handling — exhausting attempts
    must return the response, not raise."""
    api = AsyncMock(side_effect=[_content_response("") for _ in range(MAX_ATTEMPTS)])
    with patch("llm_utils.litellm.acompletion", api), \
         patch("llm_utils.asyncio.sleep", AsyncMock()):
        result = await acompletion_with_retry(model="m", messages=[], retry_on_blank=True)
    assert result.choices[0].message.content == ""
    assert api.call_count == MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_blank_content_not_retried_by_default():
    api = AsyncMock(return_value=_content_response(""))
    with patch("llm_utils.litellm.acompletion", api):
        await acompletion_with_retry(model="m", messages=[])
    assert api.call_count == 1


@pytest.mark.asyncio
async def test_non_retryable_error_raises_immediately():
    api = AsyncMock(side_effect=ValueError("bad request"))
    with patch("llm_utils.litellm.acompletion", api):
        with pytest.raises(ValueError):
            await acompletion_with_retry(model="m", messages=[])
    assert api.call_count == 1
