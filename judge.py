import asyncio, json, time
from collections import defaultdict
from datetime import datetime, timezone
from config import JUDGE_MODEL, JUDGE_REASONING, PASS_THRESHOLD, SOLVER_MODELS, request_kwargs, response_cost
from llm_utils import acompletion_with_retry
from prompts import judge_prompt
from parse_utils import parse_json_robust

# Minimum number of independent solvers reconciling the same question as
# scheme_wrong before the hard fail-rule fires.
SCHEME_WRONG_CONSENSUS_MIN = 2


def group_solutions_by_generation(solutions: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for sol in solutions:
        grouped[sol["generation_id"]].append(sol)
    return dict(grouped)


def apply_scheme_wrong_consensus(judgement: dict, solutions: list[dict]) -> dict:
    """Hard rule, not judge discretion: when >= SCHEME_WRONG_CONSENSUS_MIN
    solvers independently reconcile the SAME question as scheme_wrong, that
    question must not pass — cap its correctness at 2, its total below the pass
    threshold, and flag it. Other questions in the batch are unaffected.
    """
    by_question: dict = defaultdict(set)
    for sol in solutions:
        for rec in sol.get("reconciliation") or []:
            if rec.get("verdict") == "scheme_wrong" and "question_index" in rec:
                by_question[rec["question_index"]].add(sol.get("solver_short"))
    consensus = {
        qi for qi, solvers in by_question.items()
        if len(solvers) >= SCHEME_WRONG_CONSENSUS_MIN
    }
    if not consensus:
        return judgement

    new_questions = []
    for q in judgement.get("questions") or []:
        if q.get("question_index") in consensus:
            scores = dict(q.get("scores") or {})
            correctness = scores.get("correctness")
            if isinstance(correctness, (int, float)) and correctness > 2:
                scores["correctness"] = 2
            total = sum(v for v in scores.values() if isinstance(v, (int, float)))
            flags = list(q.get("flags") or [])
            if "scheme_wrong_consensus" not in flags:
                flags.append("scheme_wrong_consensus")
            q = {**q, "scores": scores, "total": min(total, PASS_THRESHOLD - 1), "flags": flags}
        new_questions.append(q)
    return {**judgement, "questions": new_questions}


async def judge_one(generation: dict, solutions: list[dict]) -> dict:
    base = {
        "generation_id": generation["generation_id"],
        "judge_model": JUDGE_MODEL["id"],
        "questions": [],
        "flags": [],
        "solver_agreement": {f"{s['short']}_agrees": None for s in SOLVER_MODELS},
        "judge_input_tokens": 0,
        "judge_output_tokens": 0,
        "judge_cost_usd": 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if not generation["json_parse_ok"]:
        base["flags"].append("json_parse_failure")
        return base

    start = time.time()
    response = await acompletion_with_retry(
        model=JUDGE_MODEL["id"],
        messages=[{"role": "user", "content": judge_prompt(generation, solutions)}],
        max_tokens=16384,
        timeout=60,
        **request_kwargs(JUDGE_MODEL, JUDGE_REASONING),
        retry_on_blank=True,
    )
    latency_ms = int((time.time() - start) * 1000)

    if response.choices[0].finish_reason == "length":
        print(f"[judge] output truncated at max_tokens for {generation['generation_id']}")
    content = response.choices[0].message.content or ""
    input_tokens = response.usage.prompt_tokens or 0
    output_tokens = response.usage.completion_tokens or 0

    parsed, err = parse_json_robust(content)
    if err:
        print(f"[judge] JSON parse error for {generation['generation_id']}: {err}")

    base["judge_input_tokens"] = input_tokens
    base["judge_output_tokens"] = output_tokens
    base["judge_cost_usd"] = response_cost(JUDGE_MODEL, response)

    if parsed is None:
        base["flags"].append("judge_parse_failure")
        return base

    questions = parsed.get("questions") or []
    for q in questions:
        scores = q.get("scores")
        if "total" not in q and isinstance(scores, dict):
            q["total"] = sum(v for v in scores.values() if isinstance(v, (int, float)))

    judgement = {
        **base,
        "questions": questions,
        "solver_agreement": parsed.get("solver_agreement", base["solver_agreement"]),
        "latency_ms": latency_ms,
    }
    return apply_scheme_wrong_consensus(judgement, solutions)


async def judge_all(generations: list[dict], solutions: list[dict]) -> list[dict]:
    semaphore = asyncio.Semaphore(10)
    grouped = group_solutions_by_generation(solutions)

    async def _with_sem(gen):
        async with semaphore:
            try:
                return await judge_one(gen, grouped.get(gen["generation_id"], []))
            except Exception as e:
                print(f"[judge] {type(e).__name__} for {gen['generation_id']}: {e}")
                return {
                    "generation_id": gen["generation_id"],
                    "judge_model": JUDGE_MODEL["id"],
                    "questions": [],
                    "flags": ["api_error"],
                    "solver_agreement": {f"{s['short']}_agrees": None for s in SOLVER_MODELS},
                    "judge_input_tokens": 0,
                    "judge_output_tokens": 0,
                    "judge_cost_usd": 0.0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": str(e)[:200],
                }

    return await asyncio.gather(*[_with_sem(g) for g in generations])
