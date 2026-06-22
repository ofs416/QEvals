import asyncio
import time
from datetime import datetime, timezone

from cleaner import clean_json
from config import (
    GATE_FALLBACK_MODEL,
    GATE_MODEL,
    SOLVER_MODELS,
    SOLVER_REASONING,
    request_kwargs,
    response_cost,
)
from llm_utils import acompletion_with_retry
from parse_utils import parse_json_robust
from prompts import (
    match_gate_prompt,
    reconcile_prompt,
    solver_cleaner_prompt,
    solver_prompt,
)


def _answers_cleaner(raw: str) -> str:
    return solver_cleaner_prompt(
        raw, 'with a top-level "answers" array of per-question results'
    )


def _reconcile_cleaner(raw: str) -> str:
    return solver_cleaner_prompt(
        raw, 'with a top-level "reconciliations" array of per-question verdicts'
    )


async def solve_one(generation: dict, solver: dict) -> dict:
    if not generation["json_parse_ok"]:
        return {
            "generation_id": generation["generation_id"],
            "solver_model": solver["id"],
            "solver_short": solver["short"],
            "answers": [],
            "skipped": True,
            "parse_ok": False,
            "solver_input_tokens": 0,
            "solver_output_tokens": 0,
            "solver_cost_usd": 0.0,
            "latency_ms": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    questions = generation["output_parsed"]["questions"]
    start = time.time()
    response = await acompletion_with_retry(
        model=solver["id"],
        messages=[
            {"role": "user", "content": solver_prompt(questions, generation["board"])}
        ],
        max_tokens=16384,  # headroom for high-effort reasoning before the JSON answer
        timeout=60,
        **request_kwargs(solver, solver.get("reasoning", SOLVER_REASONING)),
        retry_on_blank=True,
    )
    latency_ms = int((time.time() - start) * 1000)

    content = response.choices[0].message.content or ""
    input_tokens = response.usage.prompt_tokens or 0
    output_tokens = response.usage.completion_tokens or 0

    answers = []
    data, err = parse_json_robust(content)
    cleaner_cost = 0.0
    cleaned = False
    if data is None and content.strip():
        print(
            f"[solve] JSON parse error for {solver['short']}/{generation['generation_id']}: {err} -- sending to cleaner"
        )
        data, cleaner_cost = await clean_json(content, _answers_cleaner)
        cleaned = data is not None
        print(
            f"[solve] cleaner {'recovered' if cleaned else 'could not repair'} "
            f"{solver['short']}/{generation['generation_id']}"
        )
    parse_ok = data is not None
    if parse_ok:
        # Drop non-dict entries: a solver occasionally emits a bare string in
        # the answers array, which would later AttributeError on .get() in the
        # gate/reconcile path.
        answers = [a for a in data.get("answers", []) if isinstance(a, dict)]

    record = {
        "generation_id": generation["generation_id"],
        "solver_model": solver["id"],
        "solver_short": solver["short"],
        "answers": answers,
        "skipped": False,
        "parse_ok": parse_ok,
        "cleaned": cleaned,
        "solver_input_tokens": input_tokens,
        "solver_output_tokens": output_tokens,
        "solver_cost_usd": response_cost(solver, response) + cleaner_cost,
        "latency_ms": latency_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if not parse_ok:
        # keep the raw text so failures can be diagnosed post-run
        record["solver_raw"] = content
    return record


async def _gate_matches(
    generation: dict, answers: list[dict]
) -> tuple[set, int, int, float]:
    """Cheap equivalence gate: which blind answers already match the mark
    scheme? Fails open — any gate failure means "no match", so the answer
    still reaches full reconciliation. Returns (matched_indices, in_tokens,
    out_tokens, cost).

    Tries GATE_MODEL first, then GATE_FALLBACK_MODEL on another provider if the
    primary's route is down or returns unparseable text, so a transient Haiku
    outage doesn't push every answer into the expensive reconcile call. Only
    fails open once both have been tried.
    """
    gateable = [a for a in answers if not a.get("unsolvable")]
    if not gateable:
        return set(), 0, 0, 0.0

    questions = generation["output_parsed"]["questions"]
    prompt = match_gate_prompt(questions, gateable)
    candidates = [GATE_MODEL, GATE_FALLBACK_MODEL]
    for i, model in enumerate(candidates):
        is_last = i == len(candidates) - 1
        next_step = "failing open" if is_last else "falling back to backup provider"
        try:
            response = await acompletion_with_retry(
                model=model["id"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                timeout=60,
                **request_kwargs(model, None),
                retry_on_blank=True,
            )
        except Exception as e:
            print(
                f"[gate] {type(e).__name__} for {generation['generation_id']} "
                f"via {model['short']}: {e} -- {next_step}"
            )
            continue

        data, err = parse_json_robust(response.choices[0].message.content or "")
        if data is None:
            print(
                f"[gate] JSON parse error for {generation['generation_id']} "
                f"via {model['short']}: {err} -- {next_step}"
            )
            continue

        matched = {
            m["question_index"]
            for m in data.get("matches", [])
            if m.get("match") is True and "question_index" in m
        }
        return (
            matched,
            response.usage.prompt_tokens or 0,
            response.usage.completion_tokens or 0,
            response_cost(model, response),
        )

    return set(), 0, 0, 0.0


async def reconcile_one(generation: dict, solution: dict, solver: dict) -> dict:
    """Stage 2: show the solver its own blind answer next to the mark scheme
    and have it adjudicate the conflict. The blind stage stays untouched —
    this only adds reconciliation fields to the solution record.

    A cheap gate model first checks which answers already match the scheme;
    only mismatches (and declared-unsolvable answers) get the full solver
    reconcile call.
    """
    if (
        solution.get("skipped")
        or not solution.get("parse_ok")
        or not solution.get("answers")
    ):
        return {
            **solution,
            "reconciliation": [],
            "reconcile_ok": None,
            "reconcile_input_tokens": 0,
            "reconcile_output_tokens": 0,
            "reconcile_cost_usd": 0.0,
        }

    answers = solution["answers"]
    matched, gate_in, gate_out, gate_cost = await _gate_matches(generation, answers)

    agreed = [
        {
            "question_index": i,
            "verdict": "agree",
            "reason": "blind answer matches the mark scheme (gate)",
        }
        for i in sorted(matched)
    ]
    pending = [a for a in answers if a.get("question_index") not in matched]

    if not pending:
        return {
            **solution,
            "reconciliation": agreed,
            "reconcile_ok": True,
            "reconcile_input_tokens": gate_in,
            "reconcile_output_tokens": gate_out,
            "reconcile_cost_usd": gate_cost,
        }

    questions = generation["output_parsed"]["questions"]
    response = await acompletion_with_retry(
        model=solver["id"],
        messages=[
            {
                "role": "user",
                "content": reconcile_prompt(questions, pending, generation["board"]),
            }
        ],
        max_tokens=16384,  # reasoning solvers burn thinking tokens before the JSON; high effort needs more headroom (4096 truncated deepseek-pro, 8192 truncated gemini at high effort)
        timeout=60,
        **request_kwargs(solver, solver.get("reasoning", SOLVER_REASONING)),
        retry_on_blank=True,
    )

    content = response.choices[0].message.content or ""
    adjudicated = []
    data, err = parse_json_robust(content)
    cleaner_cost = 0.0
    if data is None and content.strip():
        print(
            f"[reconcile] JSON parse error for {solver['short']}/{generation['generation_id']}: {err} -- sending to cleaner"
        )
        data, cleaner_cost = await clean_json(content, _reconcile_cleaner)
        print(
            f"[reconcile] cleaner {'recovered' if data is not None else 'could not repair'} "
            f"{solver['short']}/{generation['generation_id']}"
        )
    reconcile_ok = data is not None
    if reconcile_ok:
        adjudicated = data.get("reconciliations", [])

    reconciliation = sorted(
        agreed + adjudicated, key=lambda r: r.get("question_index", 0)
    )
    record = {
        **solution,
        "reconciliation": reconciliation,
        "reconcile_ok": reconcile_ok,
        "reconcile_input_tokens": gate_in + (response.usage.prompt_tokens or 0),
        "reconcile_output_tokens": gate_out + (response.usage.completion_tokens or 0),
        "reconcile_cost_usd": gate_cost + response_cost(solver, response) + cleaner_cost,
    }
    if not reconcile_ok:
        # keep the raw text so failures can be diagnosed post-run
        record["reconcile_raw"] = content
    return record


async def solve_all(generations: list[dict]) -> list[dict]:
    # Global cap of 16 with at most 4 in-flight per solver (mirrors
    # generate.py's PER_MODEL_CONCURRENCY): tasks spread across four provider
    # families, so no single provider sees more concurrency than the judge
    # stage already sends to one.
    semaphore = asyncio.Semaphore(12)
    solver_sems = {s["short"]: asyncio.Semaphore(3) for s in SOLVER_MODELS}

    async def _with_sem(gen, solver):
        async with solver_sems[solver["short"]], semaphore:
            try:
                solution = await solve_one(gen, solver)
            except Exception as e:
                print(
                    f"[solve] {type(e).__name__} for {solver['short']}/{gen['generation_id']}: {e}"
                )
                return {
                    "generation_id": gen["generation_id"],
                    "solver_model": solver["id"],
                    "solver_short": solver["short"],
                    "answers": [],
                    "skipped": True,
                    "parse_ok": False,
                    "solver_input_tokens": 0,
                    "solver_output_tokens": 0,
                    "solver_cost_usd": 0.0,
                    "latency_ms": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": str(e)[:200],
                    "reconciliation": [],
                    "reconcile_ok": None,
                    "reconcile_input_tokens": 0,
                    "reconcile_output_tokens": 0,
                    "reconcile_cost_usd": 0.0,
                }
            try:
                return await reconcile_one(gen, solution, solver)
            except Exception as e:
                # keep the blind solution; reconciliation is additive
                print(
                    f"[reconcile] {type(e).__name__} for {solver['short']}/{gen['generation_id']}: {e}"
                )
                return {
                    **solution,
                    "reconciliation": [],
                    "reconcile_ok": False,
                    "reconcile_input_tokens": 0,
                    "reconcile_output_tokens": 0,
                    "reconcile_cost_usd": 0.0,
                    "reconcile_error": str(e)[:200],
                }

    # Skip self-solving: a model that is both generator and solver would grade
    # its own questions, agreeing with its own mark-scheme errors and weakening
    # the independent scheme_wrong_consensus signal. Each such generation is
    # still solved by the remaining independent solvers.
    tasks = [
        _with_sem(gen, solver)
        for gen in generations
        for solver in SOLVER_MODELS
        if solver["short"] != gen["model_short"]
    ]
    return await asyncio.gather(*tasks)
