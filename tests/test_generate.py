# evals/tests/test_generate.py
import asyncio
import json, os, tempfile
from collections import defaultdict
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from generate import PER_MODEL_CONCURRENCY, generate_one, generate_all, generate_all_variants, save_jsonl, load_jsonl
from config import GENERATOR_REASONING, MODELS


def _mock_response(content: str, prompt_tokens=100, completion_tokens=200):
    r = MagicMock()
    r.choices[0].message.content = content
    r.choices[0].message.reasoning_content = None
    r.choices[0].message.tool_calls = None  # no code-in-loop call: this is the final turn
    r.usage.prompt_tokens = prompt_tokens
    r.usage.completion_tokens = completion_tokens
    r.usage.cost = 0.001  # OpenRouter-reported cost (usage.include)
    return r


def _tool_call_response(code: str, call_id="call_1", prompt_tokens=100, completion_tokens=50):
    """A turn where the model calls run_python instead of answering."""
    r = MagicMock()
    r.choices[0].message.content = None
    r.choices[0].message.reasoning_content = None
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = "run_python"
    tc.function.arguments = json.dumps({"code": code})
    r.choices[0].message.tool_calls = [tc]
    r.usage.prompt_tokens = prompt_tokens
    r.usage.completion_tokens = completion_tokens
    r.usage.cost = 0.001
    return r


VALID_OUTPUT = json.dumps({
    "subtopics": ["limits", "chain rule"],
    "questions": [{
        "text": "Find dy/dx when y = x^2",
        "commandWord": "Find",
        "marks": 2,
        "difficulty": "foundation",
        "markScheme": [{"tag": "M1", "text": "2x"}, {"tag": "A1", "text": "correct"}]
    }]
})


@pytest.mark.asyncio
async def test_generate_one_success():
    model = MODELS[0]
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response(VALID_OUTPUT)):
        record = await generate_one(model, "Integration", "AQA", "test-run")

    assert record["json_parse_ok"] is True
    assert record["topic"] == "Integration"
    assert record["board"] == "AQA"
    assert record["model_short"] == model["short"]
    assert len(record["output_parsed"]["questions"]) == 1
    assert record["input_tokens"] == 100
    assert record["output_tokens"] == 200
    assert record["cost_usd"] > 0
    assert "generation_id" in record


@pytest.mark.asyncio
async def test_generate_one_runs_code_then_answers():
    """The specialist may call run_python before answering; the tool result is
    fed back and the subsequent JSON turn is parsed as the generation."""
    model = MODELS[0]
    responses = [_tool_call_response("print(42)"), _mock_response(VALID_OUTPUT)]
    sandbox_result = {"ok": True, "timed_out": False, "returncode": 0, "stdout": "42", "stderr": ""}
    with patch("llm_utils.litellm.acompletion", AsyncMock(side_effect=responses)), \
         patch("generate.run_python", return_value=sandbox_result) as run:
        record = await generate_one(model, "Integration", "Edexcel", "test-run")

    run.assert_called_once_with("print(42)")
    assert record["json_parse_ok"] is True
    assert record["python_runs"] == 1
    assert record["code_loop_iters"] == 2
    assert record["strand"] == "pure"
    # cost accumulates across both the tool turn and the answer turn
    assert record["cost_usd"] == pytest.approx(0.002)


@pytest.mark.asyncio
async def test_generate_one_records_strand_for_topic():
    model = MODELS[0]
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response(VALID_OUTPUT)):
        record = await generate_one(model, "Kinematics", "Edexcel", "test-run")
    assert record["strand"] == "mechanics"


@pytest.mark.asyncio
async def test_generate_one_forces_answer_at_iteration_cap():
    """A model that never stops calling the tool is cut off at the cap: the
    final iteration withdraws the tools so it cannot loop forever."""
    from config import CODE_LOOP_MAX_ITERS

    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _tool_call_response("print(1)")

    with patch("llm_utils.litellm.acompletion", side_effect=fake_acompletion), \
         patch("generate.run_python", return_value={"ok": True, "stdout": "1", "stderr": "", "returncode": 0, "timed_out": False}):
        record = await generate_one(MODELS[0], "Integration", "Edexcel", "test-run")

    assert len(calls) == CODE_LOOP_MAX_ITERS
    assert "tools" in calls[0]            # tools offered early
    assert "tools" not in calls[-1]       # withdrawn on the final iteration
    assert record["code_loop_iters"] == CODE_LOOP_MAX_ITERS


@pytest.mark.asyncio
async def test_generate_one_json_parse_failure():
    model = MODELS[0]
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response("Sorry, I cannot do that.")):
        record = await generate_one(model, "Integration", "AQA", "test-run")

    assert record["json_parse_ok"] is False
    assert record["output_parsed"] is None


@pytest.mark.asyncio
async def test_generate_one_rejects_json_fragment_without_questions():
    """Chain-of-thought containing a small JSON fragment (e.g. a mark-scheme
    item) must not be marked parse-ok: downstream stages require 'questions'."""
    model = MODELS[0]
    rumination = (
        'Let me draft the mark scheme first:\n'
        '{"tag": "M1", "text": "Uses correct formula for nth term"}\n'
        "Hmm, I am still not sure about the escaping..."
    )
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response(rumination)):
        record = await generate_one(model, "Sequences & Series", "Edexcel", "test-run")

    assert record["json_parse_ok"] is False
    assert record["json_parse_ok_raw"] is False
    assert record["output_parsed"] is None


@pytest.mark.asyncio
async def test_generate_one_rejects_invalid_cleaner_output():
    """If the cleaner returns syntactically valid JSON that still lacks a
    non-empty 'questions' list, the record must stay parse-failed."""
    model = MODELS[0]
    # Both the generator and cleaner calls hit the same mock: malformed
    # output, then a cleaner "repair" that parses but has no questions.
    responses = [
        _mock_response('{"questions": [unclosed'),
        _mock_response('{"questions": []}'),
    ]
    mock = AsyncMock(side_effect=responses)
    with patch("llm_utils.litellm.acompletion", mock):
        record = await generate_one(model, "Integration", "AQA", "test-run")

    assert record["json_parse_ok"] is False
    assert record["cleaned"] is False
    assert record["output_parsed"] is None


@pytest.mark.asyncio
async def test_generate_one_json_wrapped_in_prose():
    model = MODELS[0]
    content = f"Here is the output:\n```json\n{VALID_OUTPUT}\n```"
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response(content)):
        record = await generate_one(model, "Integration", "AQA", "test-run")

    assert record["json_parse_ok"] is True


@pytest.mark.asyncio
async def test_generate_one_caps_reasoning_by_default():
    """Generators under test get the global reasoning cap so thinking-heavy
    models don't dominate the cost comparison."""
    model = {"id": "openrouter/test/reasoner", "short": "reasoner"}
    mock = AsyncMock(return_value=_mock_response(VALID_OUTPUT))
    with patch("llm_utils.litellm.acompletion", mock):
        await generate_one(model, "Integration", "AQA", "test-run")

    extra_body = mock.call_args.kwargs["extra_body"]
    # Assert the global default is wired through, whatever it's currently set to,
    # rather than a hardcoded literal that breaks when the knob is tuned.
    assert extra_body["reasoning"] == GENERATOR_REASONING


@pytest.mark.asyncio
async def test_generate_one_model_reasoning_override():
    model = {"id": "openrouter/test/reasoner", "short": "reasoner",
             "reasoning": {"max_tokens": 2048}}
    mock = AsyncMock(return_value=_mock_response(VALID_OUTPUT))
    with patch("llm_utils.litellm.acompletion", mock):
        await generate_one(model, "Integration", "AQA", "test-run")

    assert mock.call_args.kwargs["extra_body"]["reasoning"] == {"max_tokens": 2048}


@pytest.mark.asyncio
async def test_generate_one_reasoning_none_omits_param():
    """Non-reasoning models opt out with "reasoning": None — the param must be
    absent, not null, or their endpoints may reject the request."""
    model = {"id": "openrouter/test/instruct", "short": "instruct", "reasoning": None}
    mock = AsyncMock(return_value=_mock_response(VALID_OUTPUT))
    with patch("llm_utils.litellm.acompletion", mock):
        await generate_one(model, "Integration", "AQA", "test-run")

    assert "reasoning" not in mock.call_args.kwargs["extra_body"]


@pytest.mark.asyncio
async def test_generate_all_caps_per_model_concurrency():
    """No single model (= provider) may see more than PER_MODEL_CONCURRENCY
    requests in flight at once, even when the global semaphore allows more."""
    in_flight = defaultdict(int)
    peak = defaultdict(int)

    async def fake_acompletion(**kwargs):
        mid = kwargs["model"]
        in_flight[mid] += 1
        peak[mid] = max(peak[mid], in_flight[mid])
        await asyncio.sleep(0.005)
        in_flight[mid] -= 1
        return _mock_response(VALID_OUTPUT)

    with patch("llm_utils.litellm.acompletion", side_effect=fake_acompletion):
        await generate_all("test-run", models=[MODELS[0], MODELS[1]])

    assert peak, "no requests recorded"
    for mid, p in peak.items():
        assert p <= PER_MODEL_CONCURRENCY, f"{mid} peaked at {p} concurrent requests"


@pytest.mark.asyncio
async def test_generate_one_uses_prompt_builder_and_tags_variant():
    captured = {}

    async def fake_acompletion(**kwargs):
        captured["system"] = kwargs["messages"][0]["content"]
        return _mock_response(VALID_OUTPUT)

    def builder(strand, board):
        return "CUSTOM_PROMPT_XYZ"

    with patch("llm_utils.litellm.acompletion", side_effect=fake_acompletion):
        rec = await generate_one(
            MODELS[0], "Integration", "Edexcel", "r",
            prompt_builder=builder, prompt_variant="vX",
        )

    assert captured["system"] == "CUSTOM_PROMPT_XYZ"
    assert rec["prompt_variant"] == "vX"


@pytest.mark.asyncio
async def test_generate_all_uses_prompt_builder_without_tagging_variant():
    """--prompt swaps the system prompt for the whole run but leaves
    prompt_variant unset, so the report still groups by model."""
    seen = []

    async def fake_acompletion(**kwargs):
        seen.append(kwargs["messages"][0]["content"])
        return _mock_response(VALID_OUTPUT)

    def builder(strand, board):
        return "PROMPT_FROM_SELECTOR"

    with patch("llm_utils.litellm.acompletion", side_effect=fake_acompletion):
        recs = await generate_all("test-run", models=[MODELS[0]], prompt_builder=builder)

    assert seen and all(s == "PROMPT_FROM_SELECTOR" for s in seen)
    assert all(r["prompt_variant"] is None for r in recs)


@pytest.mark.asyncio
async def test_generate_all_variants_covers_variants_topics_boards(monkeypatch):
    import generate as gen
    from config import TOPICS, BOARDS

    def builder_a(s, b):
        return "A"

    def builder_b(s, b):
        return "B"

    seen = []

    async def fake_one(model, topic, board, run_id, *, prompt_builder, prompt_variant):
        seen.append((prompt_variant, prompt_builder))
        return {"prompt_variant": prompt_variant, "topic": topic, "board": board}

    monkeypatch.setattr(gen, "generate_one", fake_one)
    monkeypatch.setattr(gen, "PROMPT_VARIANTS", {"a": builder_a, "b": builder_b})

    recs = await gen.generate_all_variants("r", MODELS[0], ["a", "b"])

    assert len(recs) == 2 * len(TOPICS) * len(BOARDS)
    assert {r["prompt_variant"] for r in recs} == {"a", "b"}
    # each variant must be wired to ITS OWN builder, else the A/B is meaningless
    assert all(b is builder_a for v, b in seen if v == "a")
    assert all(b is builder_b for v, b in seen if v == "b")


@pytest.mark.asyncio
async def test_generate_all_variants_rejects_unknown_variant():
    # uses the real PROMPT_VARIANTS: baseline exists, the typo does not
    with pytest.raises(ValueError):
        await generate_all_variants("r", MODELS[0], ["baseline", "nope_xyz"])


def test_save_and_load_jsonl():
    records = [{"a": 1}, {"b": 2}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        save_jsonl(records, path)
        loaded = load_jsonl(path)
        assert loaded == records
    finally:
        os.unlink(path)


NATIVE_MODEL = {"id": "gemini/gemini-3.1-flash-lite", "short": "gl-native", "native_code_exec": True}


@pytest.mark.asyncio
async def test_generate_one_native_single_call_no_local_sandbox():
    """The native arm makes ONE completion call carrying the code_execution
    tool, never invokes the harness run_python sandbox, and records the native
    bookkeeping fields."""
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _mock_response(VALID_OUTPUT)

    with patch("llm_utils.litellm.acompletion", side_effect=fake_acompletion), \
         patch("generate.run_python") as run, \
         patch("config.litellm.completion_cost", return_value=0.002):
        rec = await generate_one(NATIVE_MODEL, "Integration", "Edexcel", "r")

    run.assert_not_called()
    assert captured["tools"] == [{"code_execution": {}}]
    assert "tool_choice" not in captured
    assert rec["json_parse_ok"] is True
    assert rec["python_runs"] == 0
    assert rec["code_loop_iters"] == 1
    assert rec["native_code_exec"] is True
    assert rec["cost_usd"] == 0.002


@pytest.mark.asyncio
async def test_generate_one_native_uses_native_prompt():
    """The native arm sends the native (code-execution) system prompt, not the
    baseline run_python one."""
    captured = {}

    async def fake_acompletion(**kwargs):
        captured["system"] = kwargs["messages"][0]["content"]
        return _mock_response(VALID_OUTPUT)

    with patch("llm_utils.litellm.acompletion", side_effect=fake_acompletion), \
         patch("config.litellm.completion_cost", return_value=0.0):
        await generate_one(NATIVE_MODEL, "Integration", "Edexcel", "r")

    assert "built-in Python code-execution environment" in captured["system"]
    assert "You have a run_python tool" not in captured["system"]


@pytest.mark.asyncio
async def test_generate_one_local_model_records_native_false():
    """A normal (non-native) model still runs the local loop and records
    native_code_exec=False, so every record carries the field."""
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response(VALID_OUTPUT)):
        rec = await generate_one(MODELS[0], "Integration", "Edexcel", "r")
    assert rec["native_code_exec"] is False


@pytest.mark.asyncio
async def test_generate_one_native_honours_work_backwards_variant():
    """In a --compare-prompts A/B on the native arm, prompt_variant selects the
    NATIVE work-backwards prompt (not the local builder, which is ignored for
    native), so the only varied thing is the prompt while both arms keep the
    server-side sandbox."""
    captured = {}

    async def fake_acompletion(**kwargs):
        captured["system"] = kwargs["messages"][0]["content"]
        captured["tools"] = kwargs.get("tools")
        return _mock_response(VALID_OUTPUT)

    # pass the LOCAL baseline builder as prompt_builder to prove it is ignored
    from prompts import specialist_system_prompt
    with patch("llm_utils.litellm.acompletion", side_effect=fake_acompletion), \
         patch("config.litellm.completion_cost", return_value=0.0):
        rec = await generate_one(
            NATIVE_MODEL, "Integration", "Edexcel", "r",
            prompt_builder=specialist_system_prompt, prompt_variant="work_backwards",
        )

    assert captured["tools"] == [{"code_execution": {}}]   # still the native sandbox
    assert "built-in Python code-execution environment" in captured["system"]
    assert "Answer first" in captured["system"]            # work-backwards protocol
    assert "You have a run_python tool" not in captured["system"]
    assert rec["prompt_variant"] == "work_backwards"


@pytest.mark.asyncio
async def test_generate_one_native_rejects_local_only_variant():
    """A variant with no native form (e.g. no_guards) raises rather than silently
    falling back to the native baseline."""
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response(VALID_OUTPUT)):
        with pytest.raises(ValueError):
            await generate_one(
                NATIVE_MODEL, "Integration", "Edexcel", "r", prompt_variant="no_guards",
            )


@pytest.mark.asyncio
async def test_generate_all_variants_rejects_local_only_variant_on_native():
    """generate_all_variants validates a native model against the native registry,
    so a local-only variant fails fast instead of producing an all-failed arm."""
    with pytest.raises(ValueError):
        await generate_all_variants("r", NATIVE_MODEL, ["baseline", "no_guards"])
