import pytest
from unittest.mock import patch
from config import MATHS_JUDGE_MODEL, SUITABILITY_THRESHOLD
from judge import judge_one


def _opt(n_questions=1, skipped=False, parse_ok=True):
    return {
        "generation_id": "gen-abc",
        "board": "Edexcel",
        "skipped": skipped,
        "parse_ok": parse_ok,
        "questions": [
            {"text": f"Q{i}", "marks": 2, "markScheme": [], "values": {}}
            for i in range(n_questions)
        ],
    }


def _result(by_index, parse_ok=True):
    return {"by_index": by_index, "parse_ok": parse_ok,
            "input_tokens": 10, "output_tokens": 20, "cost": 0.001}


def _patched(maths_by_index, style_by_index, maths_ok=True, style_ok=True):
    async def fake_judge_call(model, prompt, *, sandbox):
        if model is MATHS_JUDGE_MODEL:
            return _result(maths_by_index, maths_ok)
        return _result(style_by_index, style_ok)
    return patch("judge._judge_call", side_effect=fake_judge_call)


@pytest.mark.asyncio
async def test_judge_one_passes_only_when_both_gates_pass():
    """The both-gates rule: q0 passes (maths ok + high suit), q1 fails (maths
    wrong despite high suit), q2 fails (maths ok but low suit)."""
    high = SUITABILITY_THRESHOLD + 2
    low = SUITABILITY_THRESHOLD - 2
    maths = {0: {"maths_correct": True}, 1: {"maths_correct": False}, 2: {"maths_correct": True}}
    style = {
        0: {"suitability_total": high, "scores": {}},
        1: {"suitability_total": high, "scores": {}},
        2: {"suitability_total": low, "scores": {}},
    }
    with _patched(maths, style):
        rec = await judge_one(_opt(3))

    passed = [q["passed"] for q in rec["questions"]]
    assert passed == [True, False, False]


@pytest.mark.asyncio
async def test_judge_one_derives_suitability_total_from_scores():
    """A missing suitability_total falls back to the sum of the three scores."""
    maths = {0: {"maths_correct": True}}
    style = {0: {"scores": {"command_word": 5, "difficulty": 5, "style": 4}}}
    with _patched(maths, style):
        rec = await judge_one(_opt(1))
    assert rec["questions"][0]["suitability_total"] == 14
    assert rec["questions"][0]["passed"] is True  # 14 >= threshold 11


@pytest.mark.asyncio
async def test_judge_one_skips_optimiser_failure():
    opt = _opt(skipped=True)
    # no judge calls should fire
    with patch("judge._judge_call") as call:
        rec = await judge_one(opt)
        call.assert_not_called()
    assert rec["questions"] == []
    assert "optimiser_failure" in rec["flags"]


@pytest.mark.asyncio
async def test_judge_one_maths_parse_failure_fails_the_gate():
    """A maths judge that doesn't parse flags the batch and fails every maths
    gate (conservative: an unverified question does not pass)."""
    style = {0: {"suitability_total": 15, "scores": {}}}
    with _patched({}, style, maths_ok=False):
        rec = await judge_one(_opt(1))
    assert "maths_judge_parse_failure" in rec["flags"]
    assert rec["questions"][0]["maths_correct"] is False
    assert rec["questions"][0]["passed"] is False


@pytest.mark.asyncio
async def test_judge_one_accumulates_panel_costs():
    with _patched({0: {"maths_correct": True}}, {0: {"suitability_total": 12, "scores": {}}}):
        rec = await judge_one(_opt(1))
    assert rec["maths_judge_cost_usd"] == 0.001
    assert rec["style_judge_cost_usd"] == 0.001
