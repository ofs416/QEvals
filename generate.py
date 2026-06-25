import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from functools import partial

from cleaner import clean_json
from config import (
    BOARDS,
    GENERATOR_REASONING,
    MODELS,
    QUESTIONS_PER_TOPIC,
    TOPICS,
    request_kwargs,
    response_cost,
    strand_for_topic,
)
from llm_utils import acompletion_with_retry
from parse_utils import parse_json_robust, valid_drafts
from prompts import cleaner_prompt_for, drafter_system_prompt

PER_MODEL_CONCURRENCY = 4  # max in-flight requests per model, so one provider never sees the full burst

_DRAFTS_CLEANER = partial(cleaner_prompt_for, shape='with a top-level "drafts" array of question strings')


async def generate_one(model: dict, topic: str, board: str, run_id: str, n: int) -> dict:
    """The drafter: one model brainstorms n question scenarios for (topic, board).

    A single completion — no code, no mark scheme. The optimiser stage solves
    each draft and owns the maths.
    """
    start = time.time()
    strand = strand_for_topic(topic)
    reasoning = model.get("reasoning", GENERATOR_REASONING)

    messages = [
        {"role": "system", "content": drafter_system_prompt(strand, board, n)},
        {"role": "user", "content": f"Topic: {topic}"},
    ]
    response = await acompletion_with_retry(
        model=model["id"],
        messages=messages,
        max_tokens=8192,
        timeout=120,
        **request_kwargs(model, reasoning),
        retry_on_blank=True,
    )
    latency_ms = int((time.time() - start) * 1000)

    usage = getattr(response, "usage", None)
    input_tokens = (usage.prompt_tokens or 0) if usage else 0
    output_tokens = (usage.completion_tokens or 0) if usage else 0
    total_cost = response_cost(model, response)

    msg = response.choices[0].message
    content = msg.content or getattr(msg, "reasoning_content", None) or ""

    parsed, err = parse_json_robust(content)
    if parsed is not None and not valid_drafts(parsed):
        parsed, err = None, "parsed JSON lacks a non-empty 'drafts' list"
    json_parse_ok_raw = parsed is not None
    cleaned = False
    cleaner_cost = 0.0
    if parsed is None and content.strip():
        print(f"[generate] JSON parse error for {model['short']}/{topic}/{board}: {err} -- sending to cleaner")
        try:
            parsed, cleaner_cost = await clean_json(content, _DRAFTS_CLEANER)
        except Exception as exc:
            print(f"[generate] cleaner call failed for {model['short']}/{topic}/{board}: {exc}")
        if parsed is not None and not valid_drafts(parsed):
            parsed = None
        cleaned = parsed is not None
        print(f"[generate] cleaner {'recovered' if cleaned else 'could not repair'} {model['short']}/{topic}/{board}")
    json_parse_ok = parsed is not None

    return {
        "run_id": run_id,
        "generation_id": str(uuid.uuid4()),
        "model_id": model["id"],
        "model_short": model["short"],
        "topic": topic,
        "board": board,
        "strand": strand,
        "drafts": parsed["drafts"] if json_parse_ok else [],
        "output_raw": content,
        "json_parse_ok": json_parse_ok,
        "json_parse_ok_raw": json_parse_ok_raw,
        "cleaned": cleaned,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": total_cost + cleaner_cost,
        "latency_ms": latency_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _error_record(model: dict, topic: str, board: str, run_id: str, error: Exception) -> dict:
    return {
        "run_id": run_id,
        "generation_id": str(uuid.uuid4()),
        "model_id": model["id"],
        "model_short": model["short"],
        "topic": topic,
        "board": board,
        "strand": strand_for_topic(topic),
        "drafts": [],
        "output_raw": "",
        "json_parse_ok": False,
        "json_parse_ok_raw": False,
        "cleaned": False,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "latency_ms": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": str(error)[:200],
    }


async def generate_all(run_id: str, models: list[dict] | None = None, n: int = QUESTIONS_PER_TOPIC) -> list[dict]:
    """Model-ranking run: every model drafts n questions across TOPICS x BOARDS."""
    semaphore = asyncio.Semaphore(10)
    targets = models or MODELS
    model_sems = {m["short"]: asyncio.Semaphore(PER_MODEL_CONCURRENCY) for m in targets}

    async def _with_sem(model, topic, board):
        async with model_sems[model["short"]], semaphore:
            try:
                return await generate_one(model, topic, board, run_id, n)
            except Exception as e:
                print(f"[generate] {type(e).__name__} for {model['short']}/{topic}/{board}: {e}")
                return _error_record(model, topic, board, run_id, e)

    tasks = [
        _with_sem(model, topic, board)
        for model in targets
        for topic in TOPICS
        for board in BOARDS
    ]
    return await asyncio.gather(*tasks)


def save_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
