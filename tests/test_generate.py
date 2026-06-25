import asyncio
import json, os, tempfile
from collections import defaultdict
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from generate import PER_MODEL_CONCURRENCY, generate_one, generate_all, save_jsonl, load_jsonl
from config import GENERATOR_REASONING, MODELS, QUESTIONS_PER_TOPIC


def _mock_response(content: str, prompt_tokens=100, completion_tokens=200):
    r = MagicMock()
    r.choices[0].message.content = content
    r.choices[0].message.reasoning_content = None
    r.choices[0].message.tool_calls = None
    r.usage.prompt_tokens = prompt_tokens
    r.usage.completion_tokens = completion_tokens
    r.usage.cost = 0.001  # OpenRouter-reported cost (usage.include)
    return r


VALID_DRAFTS = json.dumps({"drafts": ["Find dy/dx when y = x^2", "Show that x^2 + 1 > 0"]})


@pytest.mark.asyncio
async def test_generate_one_success():
    # explicit OpenRouter model so the mocked usage.cost path is exercised
    # (MODELS[0] may be a direct-Gemini candidate whose cost comes from the
    # price table instead).
    model = {"id": "openrouter/openai/gpt-5.4-nano", "short": "gpt-nano"}
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response(VALID_DRAFTS)):
        record = await generate_one(model, "Integration", "Edexcel", "test-run", 2)

    assert record["json_parse_ok"] is True
    assert record["topic"] == "Integration"
    assert record["drafts"] == ["Find dy/dx when y = x^2", "Show that x^2 + 1 > 0"]
    assert record["model_short"] == model["short"]
    assert record["cost_usd"] > 0
    assert "generation_id" in record


@pytest.mark.asyncio
async def test_generate_one_records_strand_for_topic():
    model = MODELS[0]
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response(VALID_DRAFTS)):
        record = await generate_one(model, "Kinematics", "Edexcel", "test-run", 1)
    assert record["strand"] == "mechanics"


@pytest.mark.asyncio
async def test_generate_one_json_parse_failure():
    model = MODELS[0]
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response("Sorry, I cannot.")):
        record = await generate_one(model, "Integration", "Edexcel", "test-run", 2)

    assert record["json_parse_ok"] is False
    assert record["drafts"] == []


@pytest.mark.asyncio
async def test_generate_one_rejects_json_without_drafts():
    """A JSON fragment lacking a non-empty 'drafts' list must stay parse-failed."""
    model = MODELS[0]
    # generator emits malformed text; cleaner 'repairs' to JSON with no drafts
    responses = [_mock_response('{"drafts": [unclosed'), _mock_response('{"drafts": []}')]
    with patch("llm_utils.litellm.acompletion", AsyncMock(side_effect=responses)):
        record = await generate_one(model, "Integration", "Edexcel", "test-run", 2)

    assert record["json_parse_ok"] is False
    assert record["cleaned"] is False


@pytest.mark.asyncio
async def test_generate_one_json_wrapped_in_prose():
    model = MODELS[0]
    content = f"Here is the output:\n```json\n{VALID_DRAFTS}\n```"
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response(content)):
        record = await generate_one(model, "Integration", "Edexcel", "test-run", 2)
    assert record["json_parse_ok"] is True


@pytest.mark.asyncio
async def test_generate_one_caps_reasoning_by_default():
    model = {"id": "openrouter/test/reasoner", "short": "reasoner"}
    mock = AsyncMock(return_value=_mock_response(VALID_DRAFTS))
    with patch("llm_utils.litellm.acompletion", mock):
        await generate_one(model, "Integration", "Edexcel", "test-run", 1)
    assert mock.call_args.kwargs["extra_body"]["reasoning"] == GENERATOR_REASONING


@pytest.mark.asyncio
async def test_generate_one_reasoning_none_omits_param():
    model = {"id": "openrouter/test/instruct", "short": "instruct", "reasoning": None}
    mock = AsyncMock(return_value=_mock_response(VALID_DRAFTS))
    with patch("llm_utils.litellm.acompletion", mock):
        await generate_one(model, "Integration", "Edexcel", "test-run", 1)
    assert "reasoning" not in mock.call_args.kwargs["extra_body"]


@pytest.mark.asyncio
async def test_generate_one_passes_question_count_to_prompt():
    captured = {}

    async def fake_acompletion(**kwargs):
        captured["system"] = kwargs["messages"][0]["content"]
        return _mock_response(VALID_DRAFTS)

    with patch("llm_utils.litellm.acompletion", side_effect=fake_acompletion):
        await generate_one(MODELS[0], "Integration", "Edexcel", "r", 1)
    assert "a single well-judged question" in captured["system"]


@pytest.mark.asyncio
async def test_generate_all_caps_per_model_concurrency():
    in_flight = defaultdict(int)
    peak = defaultdict(int)

    async def fake_acompletion(**kwargs):
        mid = kwargs["model"]
        in_flight[mid] += 1
        peak[mid] = max(peak[mid], in_flight[mid])
        await asyncio.sleep(0.005)
        in_flight[mid] -= 1
        return _mock_response(VALID_DRAFTS)

    with patch("llm_utils.litellm.acompletion", side_effect=fake_acompletion):
        await generate_all("test-run", models=[MODELS[0], MODELS[1]])

    assert peak, "no requests recorded"
    for mid, p in peak.items():
        assert p <= PER_MODEL_CONCURRENCY, f"{mid} peaked at {p} concurrent requests"


@pytest.mark.asyncio
async def test_generate_all_defaults_question_count():
    seen = {}

    async def fake_acompletion(**kwargs):
        seen["system"] = kwargs["messages"][0]["content"]
        return _mock_response(VALID_DRAFTS)

    with patch("llm_utils.litellm.acompletion", side_effect=fake_acompletion):
        await generate_all("r", models=[MODELS[0]])
    # default n flows through to the prompt
    expected = "exactly 3 questions" if QUESTIONS_PER_TOPIC == 3 else str(QUESTIONS_PER_TOPIC)
    assert expected in seen["system"]


def test_save_and_load_jsonl():
    records = [{"a": 1}, {"b": 2}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        save_jsonl(records, path)
        assert load_jsonl(path) == records
    finally:
        os.unlink(path)
