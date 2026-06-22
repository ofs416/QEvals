# evals/tests/test_judge.py
import pytest, json
from unittest.mock import AsyncMock, MagicMock, patch
from config import PASS_THRESHOLD, SOLVER_MODELS
from judge import apply_scheme_wrong_consensus, judge_one, group_solutions_by_generation

GENERATION = {
    "generation_id": "gen-abc",
    "board": "AQA",
    "topic": "Integration",
    "json_parse_ok": True,
    "output_parsed": {
        "questions": [{"text": "Find ∫x dx", "marks": 2, "commandWord": "Find",
                       "difficulty": "foundation", "markScheme": [{"tag": "M1", "text": "x²/2"}, {"tag": "A1", "text": "+ C"}]}]
    }
}

SOLUTIONS = [
    {"generation_id": "gen-abc", "solver_model": "openrouter/qwen/qwq-32b",
     "answers": [{"question_index": 0, "answer": "x²/2 + C", "key_steps": ["integrate"]}],
     "skipped": False, "parse_ok": True},
    {"generation_id": "gen-abc", "solver_model": "openrouter/deepseek/deepseek-r1",
     "answers": [{"question_index": 0, "answer": "x²/2 + C", "key_steps": ["integrate"]}],
     "skipped": False, "parse_ok": True},
]

JUDGE_RESPONSE = json.dumps({
    "questions": [
        {"question_index": 0,
         "scores": {"correctness": 5, "mark_scheme": 5, "command_word": 4,
                    "difficulty": 4, "style": 4},
         "total": 22,
         "flags": [],
         "notes": {"correctness": "Both solvers confirm.",
                   "command_word": "Find is correct for AQA."}},
    ],
    "solver_agreement": {"qwq_agrees": True, "r1_agrees": True},
})


def _mock_response(content):
    r = MagicMock()
    r.choices[0].message.content = content
    r.usage.prompt_tokens = 400
    r.usage.completion_tokens = 120
    r.usage.cost = 0.001  # OpenRouter-reported cost (usage.include)
    return r


@pytest.mark.asyncio
async def test_judge_one_success():
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response(JUDGE_RESPONSE)):
        record = await judge_one(GENERATION, SOLUTIONS)

    assert record["generation_id"] == "gen-abc"
    assert len(record["questions"]) == 1
    q = record["questions"][0]
    assert q["question_index"] == 0
    assert q["total"] == 22
    assert q["scores"]["correctness"] == 5
    assert record["solver_agreement"]["qwq_agrees"] is True
    assert record["flags"] == []


@pytest.mark.asyncio
async def test_judge_one_skips_parse_failure():
    bad_gen = {**GENERATION, "json_parse_ok": False, "output_parsed": None}
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock) as mock_api:
        record = await judge_one(bad_gen, SOLUTIONS)
        mock_api.assert_not_called()

    assert record["questions"] == []
    assert "json_parse_failure" in record["flags"]
    assert set(record["solver_agreement"]) == {f"{s['short']}_agrees" for s in SOLVER_MODELS}


@pytest.mark.asyncio
async def test_judge_one_handles_malformed_judge_response():
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response("oops")):
        record = await judge_one(GENERATION, SOLUTIONS)

    assert "judge_parse_failure" in record["flags"]
    assert record["questions"] == []


def _sol_with_verdict(short, verdict, question_index=0):
    return {
        "generation_id": "gen-abc", "solver_short": short,
        "answers": [], "skipped": False, "parse_ok": True,
        "reconciliation": [
            {"question_index": question_index, "verdict": verdict, "reason": "..."}
        ],
    }


def _judgement_with_questions(*totals):
    """One question per total, each starting at correctness 4 (total given)."""
    return {
        "questions": [
            {"question_index": i,
             "scores": {"correctness": 4, "mark_scheme": 4, "command_word": 4,
                        "difficulty": 4, "style": t - 16},
             "total": t,
             "flags": []}
            for i, t in enumerate(totals)
        ],
        "flags": [],
    }


def test_scheme_wrong_consensus_caps_the_flagged_question():
    """Two solvers agreeing a question's scheme is wrong caps THAT question
    below pass, even if the judge scored it high."""
    judgement = _judgement_with_questions(22, 22)  # q0, q1 both high
    sols = [_sol_with_verdict("opus48", "scheme_wrong", question_index=0),
            _sol_with_verdict("gpt55", "scheme_wrong", question_index=0),
            _sol_with_verdict("gemini-35-flash", "agree", question_index=0)]
    out = apply_scheme_wrong_consensus(judgement, sols)
    q0 = out["questions"][0]
    q1 = out["questions"][1]
    assert q0["total"] < PASS_THRESHOLD
    assert q0["scores"]["correctness"] == 2
    assert "scheme_wrong_consensus" in q0["flags"]
    # the other question is untouched
    assert q1["total"] == 22
    assert "scheme_wrong_consensus" not in q1["flags"]


def test_scheme_wrong_consensus_leaves_already_low_correctness():
    """If the judge already scored correctness at or below the cap, consensus
    must not raise it — but the flag is still added and the total clamped."""
    judgement = {
        "questions": [
            {"question_index": 0,
             "scores": {"correctness": 1, "mark_scheme": 4, "command_word": 4,
                        "difficulty": 4, "style": 4},
             "total": 17,
             "flags": []}
        ],
        "flags": [],
    }
    sols = [_sol_with_verdict("opus48", "scheme_wrong", question_index=0),
            _sol_with_verdict("gpt55", "scheme_wrong", question_index=0)]
    out = apply_scheme_wrong_consensus(judgement, sols)
    q0 = out["questions"][0]
    assert q0["scores"]["correctness"] == 1  # untouched, already below cap
    assert q0["total"] < PASS_THRESHOLD
    assert "scheme_wrong_consensus" in q0["flags"]


def test_single_scheme_wrong_verdict_is_not_consensus():
    judgement = _judgement_with_questions(20)
    sols = [_sol_with_verdict("opus48", "scheme_wrong", question_index=0),
            _sol_with_verdict("gpt55", "agree", question_index=0)]
    out = apply_scheme_wrong_consensus(judgement, sols)
    assert out["questions"][0]["total"] == 20
    assert out["questions"][0]["flags"] == []


def test_scheme_wrong_on_different_questions_is_not_consensus():
    """Consensus means the SAME question — two solvers each flagging a
    different question is not convergence."""
    judgement = _judgement_with_questions(20, 20, 20)
    sols = [_sol_with_verdict("opus48", "scheme_wrong", question_index=1),
            _sol_with_verdict("gpt55", "scheme_wrong", question_index=2)]
    out = apply_scheme_wrong_consensus(judgement, sols)
    assert all(q["total"] == 20 for q in out["questions"])
    assert all(q["flags"] == [] for q in out["questions"])


@pytest.mark.asyncio
async def test_judge_one_applies_consensus_rule():
    sols = SOLUTIONS + [_sol_with_verdict("opus48", "scheme_wrong", question_index=0),
                        _sol_with_verdict("gpt55", "scheme_wrong", question_index=0)]
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response(JUDGE_RESPONSE)):
        record = await judge_one(GENERATION, sols)

    q0 = record["questions"][0]
    assert q0["total"] < PASS_THRESHOLD
    assert "scheme_wrong_consensus" in q0["flags"]


def test_group_solutions_by_generation():
    sols = [
        {"generation_id": "a", "solver_model": "qwq"},
        {"generation_id": "b", "solver_model": "qwq"},
        {"generation_id": "a", "solver_model": "r1"},
    ]
    grouped = group_solutions_by_generation(sols)
    assert len(grouped["a"]) == 2
    assert len(grouped["b"]) == 1
