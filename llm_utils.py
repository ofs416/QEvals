import asyncio
import random

import litellm
from litellm.exceptions import APIConnectionError, APIError, RateLimitError, Timeout

RETRYABLE = (Timeout, APIConnectionError, APIError, RateLimitError)
MAX_ATTEMPTS = 4


def _blank_content(response) -> bool:
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        return True
    # A tool-call turn legitimately carries blank content (the payload is in
    # tool_calls), so it must not count as blank or the generation tool loop
    # would retry every round.
    if getattr(message, "tool_calls", None):
        return False
    return not (getattr(message, "content", None) or "").strip()


async def acompletion_with_retry(*, retry_on_blank: bool = False, **kwargs):
    """litellm.acompletion with exponential backoff on transient errors.

    Rate limits get a 5s base delay (5/10/20s) vs 1s for other transient
    errors — a 429 storm at the upstream provider doesn't clear in a second.

    retry_on_blank=True also retries responses with blank message content:
    upstream provider failures (finish_reason 'error') surface as a 200 with
    empty content after LiteLLM's finish_reason mapping, so they never raise.
    The last response is returned even if still blank, so callers keep their
    own parse-failure handling. _blank_content only inspects message.content, so
    a response carrying reasoning_content but blank content is still retried;
    callers that salvage reasoning-only output (generate falls back to
    reasoning_content) keep that fallback on the final returned response.
    """
    response = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = await litellm.acompletion(**kwargs)
        except RETRYABLE as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            base = 5.0 if isinstance(e, RateLimitError) else 1.0
            delay = base * 2**attempt + random.uniform(0, 1)
            print(
                f"[retry] {type(e).__name__} for {kwargs.get('model')} -- "
                f"retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_ATTEMPTS})"
            )
            await asyncio.sleep(delay)
            continue
        if retry_on_blank and _blank_content(response) and attempt < MAX_ATTEMPTS - 1:
            delay = 1.0 * 2**attempt + random.uniform(0, 1)
            print(
                f"[retry] blank completion from {kwargs.get('model')} -- "
                f"retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_ATTEMPTS})"
            )
            await asyncio.sleep(delay)
            continue
        return response
    return response
