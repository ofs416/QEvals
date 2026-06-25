"""Stage 2: the maths optimiser.

Each drafter batch is handed to OPTIMISER_MODEL (Gemini, direct to Google AI
Studio) running its server-side code-execution sandbox. The optimiser solves
every draft, tweaks the constants until the numbers are neat, builds the mark
scheme, and emits a `values` map of verified latex-ready quantities — the
deterministic-oracle stage and the anti-hallucination source for the typesetter.
"""
import asyncio
import time
from datetime import datetime, timezone
from functools import partial

from cleaner import clean_json
from config import OPTIMISER_MODEL, OPTIMISER_REASONING, request_kwargs, response_cost
from llm_utils import acompletion_with_retry
from parse_utils import parse_json_robust, valid_generation
from prompts import cleaner_prompt_for, optimiser_prompt

PER_MODEL_CONCURRENCY = 4

_OPT_CLEANER = partial(cleaner_prompt_for, shape='with a top-level "questions" array')


def _skipped_record(generation: dict) -> dict:
    return {
        "generation_id": generation["generation_id"],
        "topic": generation.get("topic", ""),
        "board": generation.get("board", ""),
        "optimiser_model": OPTIMISER_MODEL["id"],
        "optimiser_short": OPTIMISER_MODEL["short"],
        "questions": [],
        "skipped": True,
        "parse_ok": False,
        "opt_input_tokens": 0,
        "opt_output_tokens": 0,
        "opt_cost_usd": 0.0,
        "latency_ms": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def optimise_one(generation: dict) -> dict:
    if not generation.get("json_parse_ok") or not generation.get("drafts"):
        return _skipped_record(generation)

    drafts = generation["drafts"]
    strand = generation.get("strand", "")
    board = generation["board"]
    start = time.time()
    response = await acompletion_with_retry(
        model=OPTIMISER_MODEL["id"],
        messages=[{"role": "user", "content": optimiser_prompt(drafts, strand, board)}],
        max_tokens=32768,
        timeout=300,
        tools=[{"code_execution": {}}],
        retry_on_blank=True,
        **request_kwargs(OPTIMISER_MODEL, OPTIMISER_REASONING),
    )
    latency_ms = int((time.time() - start) * 1000)

    usage = getattr(response, "usage", None)
    input_tokens = (usage.prompt_tokens or 0) if usage else 0
    output_tokens = (usage.completion_tokens or 0) if usage else 0
    cost = response_cost(OPTIMISER_MODEL, response)

    msg = response.choices[0].message
    content = msg.content or getattr(msg, "reasoning_content", None) or ""

    data, err = parse_json_robust(content)
    if data is not None and not valid_generation(data):
        data, err = None, "parsed JSON lacks a non-empty 'questions' list"
    cleaner_cost = 0.0
    cleaned = False
    if data is None and content.strip():
        print(f"[optimise] JSON parse error for {generation['generation_id']}: {err} -- sending to cleaner")
        data, cleaner_cost = await clean_json(content, _OPT_CLEANER)
        if data is not None and not valid_generation(data):
            data = None
        cleaned = data is not None
        print(f"[optimise] cleaner {'recovered' if cleaned else 'could not repair'} {generation['generation_id']}")
    parse_ok = data is not None

    record = {
        "generation_id": generation["generation_id"],
        "topic": generation.get("topic", ""),
        "board": board,
        "optimiser_model": OPTIMISER_MODEL["id"],
        "optimiser_short": OPTIMISER_MODEL["short"],
        "questions": data["questions"] if parse_ok else [],
        "skipped": False,
        "parse_ok": parse_ok,
        "cleaned": cleaned,
        "opt_input_tokens": input_tokens,
        "opt_output_tokens": output_tokens,
        "opt_cost_usd": cost + cleaner_cost,
        "latency_ms": latency_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if not parse_ok:
        record["optimiser_raw"] = content
    return record


async def optimise_all(generations: list[dict]) -> list[dict]:
    semaphore = asyncio.Semaphore(PER_MODEL_CONCURRENCY)

    async def _with_sem(gen):
        async with semaphore:
            try:
                return await optimise_one(gen)
            except Exception as e:
                print(f"[optimise] {type(e).__name__} for {gen['generation_id']}: {e}")
                rec = _skipped_record(gen)
                rec["error"] = str(e)[:200]
                return rec

    return await asyncio.gather(*[_with_sem(g) for g in generations])
