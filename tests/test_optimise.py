import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from optimise import optimise_one


def _mock_response(content):
    r = MagicMock()
    r.choices[0].message.content = content
    r.choices[0].message.reasoning_content = None
    r.usage.prompt_tokens = 100
    r.usage.completion_tokens = 200
    r.usage.cost = 0.002
    return r


VALID = json.dumps({"questions": [{
    "text": "Find dy/dx of y=x^2", "commandWord": "Find", "marks": 2,
    "difficulty": "foundation", "markScheme": [{"tag": "M1", "text": "2x"}],
    "values": {"dydx": "2x"},
}]})


@pytest.mark.asyncio
async def test_optimise_one_skips_failed_draft():
    gen = {"generation_id": "g1", "json_parse_ok": False, "drafts": [], "board": "Edexcel"}
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock) as call:
        rec = await optimise_one(gen)
        call.assert_not_called()
    assert rec["skipped"] is True
    assert rec["questions"] == []


@pytest.mark.asyncio
async def test_optimise_one_uses_native_sandbox_and_parses_values():
    gen = {"generation_id": "g1", "json_parse_ok": True, "drafts": ["draft a"],
           "strand": "pure", "board": "Edexcel", "topic": "Differentiation"}
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _mock_response(VALID)

    with patch("llm_utils.litellm.acompletion", side_effect=fake_acompletion), \
         patch("config.litellm.completion_cost", return_value=0.002):
        rec = await optimise_one(gen)

    assert captured["tools"] == [{"code_execution": {}}]   # Google native sandbox
    assert rec["parse_ok"] is True
    assert rec["questions"][0]["values"] == {"dydx": "2x"}
    assert rec["topic"] == "Differentiation"
    assert rec["board"] == "Edexcel"
