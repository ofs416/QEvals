import asyncio
import json
import time
import uuid
from datetime import datetime, timezone

from cleaner import clean_json
from config import (
    BOARDS,
    CODE_LOOP_MAX_ITERS,
    GENERATOR_REASONING,
    MODELS,
    TOPICS,
    request_kwargs,
    response_cost,
    strand_for_topic,
)
from llm_utils import acompletion_with_retry
from parse_utils import parse_json_robust, valid_generation
from prompts import NATIVE_PROMPT_VARIANTS, PROMPT_VARIANTS, cleaner_prompt, specialist_system_prompt
from sandbox import RUN_PYTHON_TOOL, format_tool_result, run_python

PER_MODEL_CONCURRENCY = 4  # max in-flight requests per model, so one provider never sees the full burst


def _assistant_turn(msg, tool_calls) -> dict:
    """Echo the model's tool-call turn back into the message list verbatim."""
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ],
    }


async def _run_tool_call(tc) -> dict:
    """Execute one tool call and return its tool-role reply message.

    run_python is blocking (subprocess), so it runs in a thread to keep the
    async generation fan-out from serialising on each code execution.
    """
    if tc.function.name != "run_python":
        return {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": f"ERROR: unknown tool {tc.function.name!r}",
        }
    try:
        args = json.loads(tc.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    result = await asyncio.to_thread(run_python, args.get("code", ""))
    return {
        "role": "tool",
        "tool_call_id": tc.id,
        "content": format_tool_result(result),
    }


async def generate_one(model: dict, topic: str, board: str, run_id: str, *, prompt_builder=specialist_system_prompt, prompt_variant: str | None = None) -> dict:
    start = time.time()
    strand = strand_for_topic(topic)
    reasoning = model.get("reasoning", GENERATOR_REASONING)
    req = request_kwargs(model, reasoning)
    native = bool(model.get("native_code_exec"))

    if native:
        # The native arm picks its prompt from NATIVE_PROMPT_VARIANTS by the same
        # variant name as the local A/B, so --compare-prompts varies only the
        # prompt while both arms keep Gemini's server-side sandbox. A plain run
        # (prompt_variant=None) gets the native baseline; the local prompt_builder
        # is intentionally ignored here — it carries no native code-exec form.
        native_builder = NATIVE_PROMPT_VARIANTS.get(prompt_variant or "baseline")
        if native_builder is None:
            raise ValueError(
                f"prompt variant {prompt_variant!r} has no native code-exec equivalent; "
                f"available: {sorted(NATIVE_PROMPT_VARIANTS)}"
            )
        system_prompt = native_builder(strand, board)
    else:
        system_prompt = prompt_builder(strand, board)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Topic: {topic}"},
    ]

    input_tokens = output_tokens = 0
    total_cost = 0.0
    python_runs = 0
    iterations = 0
    content = ""

    if native:
        # Server-side code execution: Gemini runs Python in Google's sandbox and
        # returns the final content. No harness tool, no loop, no tool withdrawal.
        response = await acompletion_with_retry(
            model=model["id"],
            messages=messages,
            max_tokens=32768,
            timeout=300,
            tools=[{"code_execution": {}}],
            retry_on_blank=True,
            **req,
        )
        usage = getattr(response, "usage", None)
        if usage:
            input_tokens += usage.prompt_tokens or 0
            output_tokens += usage.completion_tokens or 0
        total_cost += response_cost(model, response)
        msg = response.choices[0].message
        content = msg.content or getattr(msg, "reasoning_content", None) or ""
        iterations = 1
    else:
        # Code-in-loop: the specialist may call run_python to compute/verify values
        # and assert its invariants. On the final permitted iteration the tools are
        # withdrawn so the model is forced to emit the answer rather than loop forever.
        for iterations in range(1, CODE_LOOP_MAX_ITERS + 1):
            last = iterations == CODE_LOOP_MAX_ITERS
            kwargs = dict(
                model=model["id"],
                messages=messages,
                max_tokens=32768,  # headroom for models that ignore the reasoning cap
                timeout=300,
                **req,
                # Provider errors (finish_reason 'error') arrive as a 200 with blank
                # content -- retry before falling back to reasoning_content below.
                retry_on_blank=True,
            )
            if not last:
                kwargs["tools"] = [RUN_PYTHON_TOOL]
                kwargs["tool_choice"] = "auto"

            response = await acompletion_with_retry(**kwargs)

            usage = getattr(response, "usage", None)
            if usage:
                input_tokens += usage.prompt_tokens or 0
                output_tokens += usage.completion_tokens or 0
            total_cost += response_cost(model, response)

            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)
            if not last and tool_calls:
                messages.append(_assistant_turn(msg, tool_calls))
                for tc in tool_calls:
                    messages.append(await _run_tool_call(tc))
                    python_runs += 1
                continue

            content = msg.content or getattr(msg, "reasoning_content", None) or ""
            break

    latency_ms = int((time.time() - start) * 1000)

    output_parsed, err = parse_json_robust(content)
    if output_parsed is not None and not valid_generation(output_parsed):
        output_parsed, err = None, "parsed JSON lacks a non-empty 'questions' list"
    json_parse_ok_raw = output_parsed is not None
    cleaned = False
    cleaner_cost = 0.0
    if output_parsed is None and content.strip():
        print(
            f"[generate] JSON parse error for {model['short']}/{topic}/{board}: {err} -- sending to cleaner"
        )
        try:
            output_parsed, cleaner_cost = await clean_json(content, cleaner_prompt)
        except Exception as exc:
            print(
                f"[generate] cleaner call failed for {model['short']}/{topic}/{board}: {exc}"
            )
        if output_parsed is not None and not valid_generation(output_parsed):
            output_parsed = None
        cleaned = output_parsed is not None
        status = "recovered" if cleaned else "could not repair"
        print(f"[generate] cleaner {status} {model['short']}/{topic}/{board}")
    json_parse_ok = output_parsed is not None

    return {
        "run_id": run_id,
        "generation_id": str(uuid.uuid4()),
        "model_id": model["id"],
        "model_short": model["short"],
        "topic": topic,
        "board": board,
        "strand": strand,
        "prompt_variant": prompt_variant,
        "output_raw": content,
        "output_parsed": output_parsed,
        "json_parse_ok": json_parse_ok,
        "json_parse_ok_raw": json_parse_ok_raw,
        "cleaned": cleaned,
        "code_loop_iters": iterations,
        "python_runs": python_runs,
        "native_code_exec": native,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": total_cost + cleaner_cost,
        "latency_ms": latency_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _error_record(model: dict, topic: str, board: str, run_id: str, error: Exception, prompt_variant: str | None = None) -> dict:
    return {
        "run_id": run_id,
        "generation_id": str(uuid.uuid4()),
        "model_id": model["id"],
        "model_short": model["short"],
        "topic": topic,
        "board": board,
        "strand": strand_for_topic(topic),
        "prompt_variant": prompt_variant,
        "output_raw": "",
        "output_parsed": None,
        "json_parse_ok": False,
        "json_parse_ok_raw": False,
        "cleaned": False,
        "code_loop_iters": 0,
        "python_runs": 0,
        "native_code_exec": bool(model.get("native_code_exec")),
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "latency_ms": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": str(error)[:200],
    }


async def generate_all(run_id: str, models: list[dict] | None = None, *, prompt_builder=specialist_system_prompt) -> list[dict]:
    """Model-ranking run: every model generates across TOPICS x BOARDS.

    ``prompt_builder`` swaps the generator system prompt for the whole run (the
    --prompt selector). It is NOT tagged as a ``prompt_variant`` on the records,
    so the report still groups by model — a single-prompt run ranks models
    exactly as a default run does, just with a different prompt. (Tagging would
    collapse every model into one variant row; that is what --compare-prompts is
    for.)"""
    semaphore = asyncio.Semaphore(10)
    targets = models or MODELS
    model_sems = {m["short"]: asyncio.Semaphore(PER_MODEL_CONCURRENCY) for m in targets}

    async def _with_sem(model, topic, board):
        async with model_sems[model["short"]], semaphore:
            try:
                return await generate_one(model, topic, board, run_id, prompt_builder=prompt_builder)
            except Exception as e:
                print(
                    f"[generate] {type(e).__name__} for {model['short']}/{topic}/{board}: {e}"
                )
                return _error_record(model, topic, board, run_id, e)

    tasks = [
        _with_sem(model, topic, board)
        for model in targets
        for topic in TOPICS
        for board in BOARDS
    ]
    return await asyncio.gather(*tasks)


async def generate_all_variants(run_id: str, model: dict, variant_names: list[str]) -> list[dict]:
    """A/B run: one fixed model generates across variant_names x TOPICS x BOARDS,
    each generation tagged with its prompt_variant. Caps in-flight requests with
    a single semaphore (one model, so no separate per-model cap is needed).

    Validates variant_names up front so a typo fails fast rather than turning
    into an all-failed run via the per-task except (the CLI also validates, but
    this guards direct callers). A native model is validated against
    NATIVE_PROMPT_VARIANTS, since generate_one resolves its prompt from there — a
    local-only variant (e.g. no_guards) on the native arm fails here rather than
    silently producing the native baseline for every variant."""
    registry = NATIVE_PROMPT_VARIANTS if model.get("native_code_exec") else PROMPT_VARIANTS
    unknown = [v for v in variant_names if v not in registry]
    if unknown:
        raise ValueError(
            f"Unknown prompt variant(s) {unknown} for {model['short']}; "
            f"available: {sorted(registry)}"
        )
    semaphore = asyncio.Semaphore(PER_MODEL_CONCURRENCY)

    async def _with_sem(variant, topic, board):
        async with semaphore:
            try:
                return await generate_one(
                    model, topic, board, run_id,
                    prompt_builder=PROMPT_VARIANTS[variant], prompt_variant=variant,
                )
            except Exception as e:
                print(f"[generate] {type(e).__name__} for {variant}/{topic}/{board}: {e}")
                return _error_record(model, topic, board, run_id, e, prompt_variant=variant)

    tasks = [
        _with_sem(variant, topic, board)
        for variant in variant_names
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
