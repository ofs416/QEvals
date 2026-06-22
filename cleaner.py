"""Shared malformed-JSON repair with cross-provider fallback.

The primary cleaner (CLEANER_MODEL, Haiku via OpenRouter) occasionally hits a
transient OpenRouter outage — instant connection-timeouts (Timeout with
time-taken 0s) or blank finish_reason 'error' responses that exhaust every
retry. When that happens, or the primary returns text that still won't parse,
the repair falls through to CLEANER_FALLBACK_MODEL on a *different* provider
rather than giving up. Used by the generation stage and both solver stages
(blind solve + reconcile).
"""
from typing import Callable

from config import CLEANER_FALLBACK_MODEL, CLEANER_MODEL, request_kwargs, response_cost
from llm_utils import RETRYABLE, acompletion_with_retry
from parse_utils import parse_json_robust


async def clean_json(
    raw: str, build_prompt: Callable[[str], str]
) -> tuple[dict | None, float]:
    """Repair malformed JSON (syntax only) via the cleaner model, then a backup.

    Tries CLEANER_MODEL first; if its route is down (retries exhausted) or it
    returns unparseable text, falls through to CLEANER_FALLBACK_MODEL on another
    provider. `build_prompt` turns the raw text into a schema-specific repair
    instruction, so the same machinery serves the generation and solver schemas.

    Returns (parsed_dict_or_None, total_cost_usd); cost accumulates across both
    attempts.
    """
    total_cost = 0.0
    candidates = [CLEANER_MODEL, CLEANER_FALLBACK_MODEL]
    for i, model in enumerate(candidates):
        is_last = i == len(candidates) - 1
        next_step = "giving up" if is_last else "falling back to backup provider"
        try:
            response = await acompletion_with_retry(
                model=model["id"],
                messages=[{"role": "user", "content": build_prompt(raw)}],
                max_tokens=16384,
                timeout=120,
                **request_kwargs(model, None),
                retry_on_blank=True,
            )
        except RETRYABLE as e:
            print(f"[cleaner] {type(e).__name__} for {model['short']} -- {next_step}")
            continue
        total_cost += response_cost(model, response)
        parsed, _ = parse_json_robust(response.choices[0].message.content)
        if parsed is not None:
            return parsed, total_cost
        print(f"[cleaner] {model['short']} could not repair -- {next_step}")
    return None, total_cost
