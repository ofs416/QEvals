"""Stage 3: the typesetter.

Each optimiser batch is handed to TYPESETTER_MODEL (OpenRouter, no sandbox),
which renders the finalised question and mark scheme as HTML. It is told to use
the optimiser's verified `values` verbatim, so it formats rather than computes —
the anti-hallucination contract. The per-question marks/difficulty/commandWord
are carried over from the optimiser record (the typesetter only returns HTML).
"""
import asyncio
import time
from datetime import datetime, timezone
from functools import partial

from cleaner import clean_json
from config import TYPESETTER_MODEL, TYPESETTER_REASONING, request_kwargs, response_cost
from llm_utils import acompletion_with_retry
from parse_utils import parse_json_robust, valid_generation
from prompts import cleaner_prompt_for, typesetter_prompt

PER_MODEL_CONCURRENCY = 4

_TS_CLEANER = partial(cleaner_prompt_for, shape='with a top-level "questions" array of HTML objects')


def _skipped_record(optimisation: dict) -> dict:
    return {
        "generation_id": optimisation["generation_id"],
        "topic": optimisation.get("topic", ""),
        "board": optimisation.get("board", ""),
        "typesetter_model": TYPESETTER_MODEL["id"],
        "typesetter_short": TYPESETTER_MODEL["short"],
        "questions": [],
        "skipped": True,
        "parse_ok": False,
        "ts_input_tokens": 0,
        "ts_output_tokens": 0,
        "ts_cost_usd": 0.0,
        "latency_ms": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _merge(opt_questions: list[dict], html_questions: list[dict]) -> list[dict]:
    """Pair each typeset HTML object with its optimiser question's metadata, by
    position. Extra/missing HTML entries are tolerated — only paired indices render.
    """
    merged = []
    for i, q in enumerate(opt_questions):
        html = html_questions[i] if i < len(html_questions) and isinstance(html_questions[i], dict) else {}
        merged.append({
            "question_html": html.get("question_html", ""),
            "mark_scheme_html": html.get("mark_scheme_html", ""),
            "commandWord": q.get("commandWord", ""),
            "marks": q.get("marks"),
            "difficulty": q.get("difficulty", ""),
        })
    return merged


async def typeset_one(optimisation: dict) -> dict:
    if optimisation.get("skipped") or not optimisation.get("parse_ok") or not optimisation.get("questions"):
        return _skipped_record(optimisation)

    opt_questions = optimisation["questions"]
    board = optimisation.get("board") or "Edexcel"
    start = time.time()
    response = await acompletion_with_retry(
        model=TYPESETTER_MODEL["id"],
        messages=[{"role": "user", "content": typesetter_prompt(opt_questions, board)}],
        max_tokens=16384,
        timeout=120,
        **request_kwargs(TYPESETTER_MODEL, TYPESETTER_REASONING),
        retry_on_blank=True,
    )
    latency_ms = int((time.time() - start) * 1000)

    usage = getattr(response, "usage", None)
    input_tokens = (usage.prompt_tokens or 0) if usage else 0
    output_tokens = (usage.completion_tokens or 0) if usage else 0
    cost = response_cost(TYPESETTER_MODEL, response)

    content = response.choices[0].message.content or ""
    data, err = parse_json_robust(content)
    if data is not None and not valid_generation(data):
        data, err = None, "parsed JSON lacks a non-empty 'questions' list"
    cleaner_cost = 0.0
    cleaned = False
    if data is None and content.strip():
        print(f"[typeset] JSON parse error for {optimisation['generation_id']}: {err} -- sending to cleaner")
        data, cleaner_cost = await clean_json(content, _TS_CLEANER)
        if data is not None and not valid_generation(data):
            data = None
        cleaned = data is not None
        print(f"[typeset] cleaner {'recovered' if cleaned else 'could not repair'} {optimisation['generation_id']}")
    parse_ok = data is not None

    record = {
        "generation_id": optimisation["generation_id"],
        "topic": optimisation.get("topic", ""),
        "board": board,
        "typesetter_model": TYPESETTER_MODEL["id"],
        "typesetter_short": TYPESETTER_MODEL["short"],
        "questions": _merge(opt_questions, data["questions"]) if parse_ok else [],
        "skipped": False,
        "parse_ok": parse_ok,
        "cleaned": cleaned,
        "ts_input_tokens": input_tokens,
        "ts_output_tokens": output_tokens,
        "ts_cost_usd": cost + cleaner_cost,
        "latency_ms": latency_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if not parse_ok:
        record["typesetter_raw"] = content
    return record


async def typeset_all(optimisations: list[dict]) -> list[dict]:
    semaphore = asyncio.Semaphore(PER_MODEL_CONCURRENCY)

    async def _with_sem(opt):
        async with semaphore:
            try:
                return await typeset_one(opt)
            except Exception as e:
                print(f"[typeset] {type(e).__name__} for {opt['generation_id']}: {e}")
                rec = _skipped_record(opt)
                rec["error"] = str(e)[:200]
                return rec

    return await asyncio.gather(*[_with_sem(o) for o in optimisations])
