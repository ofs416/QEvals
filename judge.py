"""Stage 4: the judge panel — two gates, both must pass.

Each finalised question is checked by two independent judges:
  * MATHS_JUDGE_MODEL (Gemini, server-side sandbox) — re-derives the maths and
    returns a hard maths_correct verdict.
  * STYLE_JUDGE_MODEL (Opus) — scores command word, difficulty and style.

A question passes only if it is maths_correct AND its suitability total meets
SUITABILITY_THRESHOLD. A judge call that fails to parse fails its gate (the
conservative direction — an unverified question does not pass).
"""
import asyncio
import time
from datetime import datetime, timezone

from config import (
    JUDGE_REASONING,
    MATHS_JUDGE_MODEL,
    STYLE_JUDGE_MODEL,
    SUITABILITY_THRESHOLD,
    request_kwargs,
    response_cost,
)
from llm_utils import acompletion_with_retry
from parse_utils import parse_json_robust
from prompts import maths_judge_prompt, style_judge_prompt


async def _judge_call(model: dict, prompt: str, *, sandbox: bool) -> dict:
    """One judge call. Returns parsed per-question entries keyed by index plus
    token/cost/parse_ok. A non-parsing response yields an empty map (gate fails).
    """
    kwargs = dict(
        model=model["id"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=32768 if sandbox else 16384,
        timeout=300 if sandbox else 60,
        retry_on_blank=True,
        **request_kwargs(model, JUDGE_REASONING),
    )
    if sandbox:
        kwargs["tools"] = [{"code_execution": {}}]
    response = await acompletion_with_retry(**kwargs)

    usage = getattr(response, "usage", None)
    msg = response.choices[0].message
    content = msg.content or getattr(msg, "reasoning_content", None) or ""
    parsed, err = parse_json_robust(content)
    by_index = {}
    if parsed is not None:
        for q in parsed.get("questions") or []:
            if isinstance(q, dict) and "question_index" in q:
                by_index[q["question_index"]] = q
    return {
        "by_index": by_index,
        "parse_ok": parsed is not None,
        "input_tokens": (usage.prompt_tokens or 0) if usage else 0,
        "output_tokens": (usage.completion_tokens or 0) if usage else 0,
        "cost": response_cost(model, response),
    }


def _base(generation_id: str) -> dict:
    return {
        "generation_id": generation_id,
        "maths_judge_model": MATHS_JUDGE_MODEL["id"],
        "style_judge_model": STYLE_JUDGE_MODEL["id"],
        "questions": [],
        "flags": [],
        "maths_judge_input_tokens": 0,
        "maths_judge_output_tokens": 0,
        "maths_judge_cost_usd": 0.0,
        "style_judge_input_tokens": 0,
        "style_judge_output_tokens": 0,
        "style_judge_cost_usd": 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def judge_one(optimisation: dict) -> dict:
    base = _base(optimisation["generation_id"])
    questions = optimisation.get("questions") or []
    if optimisation.get("skipped") or not optimisation.get("parse_ok") or not questions:
        base["flags"].append("optimiser_failure")
        return base

    board = optimisation.get("board") or "Edexcel"
    start = time.time()
    maths, style = await asyncio.gather(
        _judge_call(MATHS_JUDGE_MODEL, maths_judge_prompt(questions, board), sandbox=True),
        _judge_call(STYLE_JUDGE_MODEL, style_judge_prompt(questions, board), sandbox=False),
    )
    latency_ms = int((time.time() - start) * 1000)

    flags = []
    if not maths["parse_ok"]:
        flags.append("maths_judge_parse_failure")
    if not style["parse_ok"]:
        flags.append("style_judge_parse_failure")

    merged = []
    for i in range(len(questions)):
        m = maths["by_index"].get(i, {})
        s = style["by_index"].get(i, {})
        scores = s.get("scores") if isinstance(s.get("scores"), dict) else {}
        suitability_total = s.get("suitability_total")
        if not isinstance(suitability_total, (int, float)):
            suitability_total = sum(v for v in scores.values() if isinstance(v, (int, float)))
        maths_correct = m.get("maths_correct") is True
        merged.append({
            "question_index": i,
            "maths_correct": maths_correct,
            "maths_note": m.get("note", ""),
            "scores": scores,
            "suitability_total": suitability_total,
            "passed": maths_correct and suitability_total >= SUITABILITY_THRESHOLD,
            "flags": s.get("flags") or [],
            "notes": s.get("notes") or {},
        })

    return {
        **base,
        "questions": merged,
        "flags": flags,
        "maths_judge_input_tokens": maths["input_tokens"],
        "maths_judge_output_tokens": maths["output_tokens"],
        "maths_judge_cost_usd": maths["cost"],
        "style_judge_input_tokens": style["input_tokens"],
        "style_judge_output_tokens": style["output_tokens"],
        "style_judge_cost_usd": style["cost"],
        "latency_ms": latency_ms,
    }


async def judge_all(optimisations: list[dict]) -> list[dict]:
    semaphore = asyncio.Semaphore(10)

    async def _with_sem(opt):
        async with semaphore:
            try:
                return await judge_one(opt)
            except Exception as e:
                print(f"[judge] {type(e).__name__} for {opt['generation_id']}: {e}")
                rec = _base(opt["generation_id"])
                rec["flags"].append("api_error")
                rec["error"] = str(e)[:200]
                return rec

    return await asyncio.gather(*[_with_sem(o) for o in optimisations])
