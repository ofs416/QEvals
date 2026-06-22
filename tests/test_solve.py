# evals/tests/test_solve.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
from litellm.exceptions import Timeout
from solve import solve_one, solve_all, reconcile_one
from config import (
    CLEANER_MODEL,
    GATE_FALLBACK_MODEL,
    GATE_MODEL,
    SOLVER_MODELS,
)

GENERATION = {
    "generation_id": "gen-abc",
    "board": "AQA",
    "model_short": "test-generator",  # differs from every solver, so none self-solves
    "json_parse_ok": True,
    "output_parsed": {
        "questions": [
            {"text": "Find dy/dx when y = x^2", "marks": 2, "commandWord": "Find",
             "difficulty": "foundation", "markScheme": [{"tag": "M1", "text": "secret"}]},
        ]
    }
}

SOLVER_RESPONSE = json.dumps({
    "answers": [{"question_index": 0, "answer": "2x", "key_steps": ["differentiate"]}]
})


def _mock_response(content):
    r = MagicMock()
    r.choices[0].message.content = content
    r.usage.prompt_tokens = 80
    r.usage.completion_tokens = 120
    r.usage.cost = 0.001  # OpenRouter-reported cost (usage.include)
    return r


@pytest.mark.asyncio
async def test_solve_one_success():
    solver = SOLVER_MODELS[0]
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response(SOLVER_RESPONSE)):
        record = await solve_one(GENERATION, solver)

    assert record["generation_id"] == "gen-abc"
    assert record["skipped"] is False
    assert record["parse_ok"] is True
    assert len(record["answers"]) == 1
    assert record["answers"][0]["answer"] == "2x"


@pytest.mark.asyncio
async def test_solve_one_skips_failed_generation():
    bad_generation = {**GENERATION, "json_parse_ok": False, "output_parsed": None}
    solver = SOLVER_MODELS[0]
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock) as mock_api:
        record = await solve_one(bad_generation, solver)
        mock_api.assert_not_called()

    assert record["skipped"] is True
    assert record["answers"] == []


@pytest.mark.asyncio
async def test_solve_one_marks_parse_failure():
    solver = SOLVER_MODELS[0]
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response("I cannot solve this.")):
        record = await solve_one(GENERATION, solver)

    assert record["parse_ok"] is False
    assert record["answers"] == []
    # raw text is kept so failures can be diagnosed post-run
    assert record["solver_raw"] == "I cannot solve this."


@pytest.mark.asyncio
async def test_solve_one_omits_raw_on_success():
    solver = SOLVER_MODELS[0]
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response(SOLVER_RESPONSE)):
        record = await solve_one(GENERATION, solver)

    assert record["parse_ok"] is True
    assert "solver_raw" not in record


RECONCILE_RESPONSE = json.dumps({
    "reconciliations": [{"question_index": 0, "verdict": "scheme_wrong",
                         "reason": "the scheme drops a factor of 2"}]
})

GATE_MATCH = json.dumps({"matches": [{"question_index": 0, "match": True}]})
GATE_MISMATCH = json.dumps({"matches": [{"question_index": 0, "match": False}]})

SOLUTION = {
    "generation_id": "gen-abc",
    "solver_model": SOLVER_MODELS[0]["id"],
    "solver_short": SOLVER_MODELS[0]["short"],
    "answers": [{"question_index": 0, "answer": "2x", "key_steps": ["differentiate"]}],
    "skipped": False,
    "parse_ok": True,
    "solver_cost_usd": 0.001,
}


def _route_by_model(gate_content, reconcile_content, calls=None):
    """Mock router: the cheap gate model gets gate_content, solvers get
    reconcile_content. Records which model ids were called."""
    async def route(*args, **kwargs):
        if calls is not None:
            calls.append(kwargs["model"])
        if kwargs["model"] == GATE_MODEL["id"]:
            return _mock_response(gate_content)
        return _mock_response(reconcile_content)
    return route


@pytest.mark.asyncio
async def test_reconcile_one_attaches_verdicts_on_gate_mismatch():
    solver = SOLVER_MODELS[0]
    with patch("llm_utils.litellm.acompletion", side_effect=_route_by_model(GATE_MISMATCH, RECONCILE_RESPONSE)):
        record = await reconcile_one(GENERATION, dict(SOLUTION), solver)

    assert record["reconcile_ok"] is True
    assert record["reconciliation"] == [
        {"question_index": 0, "verdict": "scheme_wrong", "reason": "the scheme drops a factor of 2"}
    ]
    assert record["reconcile_cost_usd"] == pytest.approx(0.002)  # gate + reconcile
    # blind-stage fields must survive
    assert record["answers"][0]["answer"] == "2x"
    assert record["solver_cost_usd"] == 0.001


@pytest.mark.asyncio
async def test_reconcile_one_gate_match_skips_solver_call():
    """When the cheap gate confirms the blind answer matches the mark scheme,
    no expensive solver reconcile call is made."""
    solver = SOLVER_MODELS[0]
    calls = []
    with patch("llm_utils.litellm.acompletion", side_effect=_route_by_model(GATE_MATCH, RECONCILE_RESPONSE, calls)):
        record = await reconcile_one(GENERATION, dict(SOLUTION), solver)

    assert calls == [GATE_MODEL["id"]]  # gate only, no solver call
    assert record["reconcile_ok"] is True
    assert record["reconciliation"][0]["verdict"] == "agree"
    assert record["reconcile_cost_usd"] == pytest.approx(0.001)  # gate only


@pytest.mark.asyncio
async def test_reconcile_one_gate_failure_fails_open():
    """An unparseable gate response must not silently skip reconciliation —
    every question goes to the full reconcile call instead."""
    solver = SOLVER_MODELS[0]
    calls = []
    with patch("llm_utils.litellm.acompletion", side_effect=_route_by_model("not json", RECONCILE_RESPONSE, calls)):
        record = await reconcile_one(GENERATION, dict(SOLUTION), solver)

    assert GATE_MODEL["id"] in calls
    assert solver["id"] in calls
    assert record["reconcile_ok"] is True
    assert record["reconciliation"][0]["verdict"] == "scheme_wrong"


@pytest.mark.asyncio
async def test_reconcile_one_unsolvable_bypasses_gate():
    """A solver that declared a question unsolvable disagrees with the scheme
    by definition — it must go straight to adjudication, not the match gate."""
    solver = SOLVER_MODELS[0]
    unsolvable_solution = {
        **SOLUTION,
        "answers": [{"question_index": 0, "answer": "", "unsolvable": True,
                     "key_steps": ["no real roots"]}],
    }
    calls = []
    with patch("llm_utils.litellm.acompletion", side_effect=_route_by_model(GATE_MATCH, RECONCILE_RESPONSE, calls)):
        record = await reconcile_one(GENERATION, unsolvable_solution, solver)

    assert calls == [solver["id"]]  # no gate call at all
    assert record["reconciliation"][0]["verdict"] == "scheme_wrong"


@pytest.mark.asyncio
async def test_reconcile_one_shows_mark_scheme_and_answer():
    """Unlike the blind solve, the reconcile prompt must contain the mark
    scheme and the solver's own stage-1 answer."""
    solver = SOLVER_MODELS[0]
    captured = []

    async def capture(*args, **kwargs):
        captured.append((kwargs["model"], kwargs["messages"][0]["content"]))
        if kwargs["model"] == GATE_MODEL["id"]:
            return _mock_response(GATE_MISMATCH)
        return _mock_response(RECONCILE_RESPONSE)

    with patch("llm_utils.litellm.acompletion", side_effect=capture):
        await reconcile_one(GENERATION, dict(SOLUTION), solver)

    reconcile_prompts = [c for m, c in captured if m == solver["id"]]
    assert len(reconcile_prompts) == 1
    assert "secret" in reconcile_prompts[0]  # the mark scheme text
    assert "2x" in reconcile_prompts[0]  # the solver's blind answer


@pytest.mark.asyncio
async def test_reconcile_one_skips_unparsed_solution():
    solver = SOLVER_MODELS[0]
    bad_solution = {**SOLUTION, "parse_ok": False, "answers": []}
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock) as mock_api:
        record = await reconcile_one(GENERATION, bad_solution, solver)
        mock_api.assert_not_called()

    assert record["reconcile_ok"] is None
    assert record["reconciliation"] == []
    assert record["reconcile_cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_reconcile_one_marks_parse_failure():
    solver = SOLVER_MODELS[0]
    with patch("llm_utils.litellm.acompletion", new_callable=AsyncMock, return_value=_mock_response("not json")):
        record = await reconcile_one(GENERATION, dict(SOLUTION), solver)

    assert record["reconcile_ok"] is False
    assert record["reconciliation"] == []
    assert record["reconcile_raw"] == "not json"


@pytest.mark.asyncio
async def test_solve_all_runs_reconciliation_pass():
    """solve_all output records must carry reconciliation results for every
    successfully parsed blind solution."""
    async def route(*args, **kwargs):
        if kwargs["model"] == GATE_MODEL["id"]:
            return _mock_response(GATE_MISMATCH)
        prompt = kwargs["messages"][0]["content"]
        if "YOUR ANSWER" in prompt:
            return _mock_response(RECONCILE_RESPONSE)
        return _mock_response(SOLVER_RESPONSE)

    with patch("llm_utils.litellm.acompletion", side_effect=route):
        records = await solve_all([GENERATION])

    assert len(records) == len(SOLVER_MODELS)
    for record in records:
        assert record["parse_ok"] is True
        assert record["reconcile_ok"] is True
        assert record["reconciliation"][0]["verdict"] == "scheme_wrong"


@pytest.mark.asyncio
async def test_solve_one_cleans_malformed_json():
    """A malformed blind-solve response is sent to the cleaner; a successful
    repair yields a parsed record rather than a parse failure."""
    solver = SOLVER_MODELS[0]

    async def route(*args, **kwargs):
        if kwargs["model"] == CLEANER_MODEL["id"]:
            return _mock_response(SOLVER_RESPONSE)
        return _mock_response('{"answers": [ truncated')

    with patch("llm_utils.litellm.acompletion", side_effect=route):
        record = await solve_one(GENERATION, solver)

    assert record["parse_ok"] is True
    assert record["cleaned"] is True
    assert record["answers"][0]["answer"] == "2x"
    assert "solver_raw" not in record


@pytest.mark.asyncio
async def test_reconcile_one_cleans_malformed_json():
    """A malformed reconcile response is repaired by the cleaner."""
    solver = SOLVER_MODELS[0]
    # GATE_MODEL and CLEANER_MODEL share the same Haiku id, so route by call
    # order: the gate fires before the cleaner. 1st Haiku call = gate, 2nd =
    # cleaner.
    haiku_calls = {"n": 0}

    async def route(*args, **kwargs):
        if kwargs["model"] == solver["id"]:
            return _mock_response('{"reconciliations": [ truncated')
        haiku_calls["n"] += 1
        if haiku_calls["n"] == 1:
            return _mock_response(GATE_MISMATCH)  # gate: mismatch -> reconcile
        return _mock_response(RECONCILE_RESPONSE)  # cleaner repairs the reconcile

    with patch("llm_utils.litellm.acompletion", side_effect=route):
        record = await reconcile_one(GENERATION, dict(SOLUTION), solver)

    assert record["reconcile_ok"] is True
    assert record["reconciliation"][0]["verdict"] == "scheme_wrong"
    assert "reconcile_raw" not in record


@pytest.mark.asyncio
async def test_gate_falls_back_to_backup_provider():
    """When the primary gate model's route is down, the gate tries the backup
    provider before failing open."""
    solver = SOLVER_MODELS[0]
    calls = []

    async def route(*args, **kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == GATE_MODEL["id"]:
            raise Timeout("down", "m", "openrouter")
        if kwargs["model"] == GATE_FALLBACK_MODEL["id"]:
            return _mock_response(GATE_MATCH)
        return _mock_response(RECONCILE_RESPONSE)

    with patch("llm_utils.litellm.acompletion", side_effect=route), \
         patch("llm_utils.asyncio.sleep", new_callable=AsyncMock):
        record = await reconcile_one(GENERATION, dict(SOLUTION), solver)

    assert GATE_MODEL["id"] in calls  # primary tried (and exhausted retries)
    assert GATE_FALLBACK_MODEL["id"] in calls  # backup provider tried
    # backup gate confirmed the match -> no expensive solver reconcile call
    assert solver["id"] not in calls
    assert record["reconciliation"][0]["verdict"] == "agree"


@pytest.mark.asyncio
async def test_solve_one_strips_mark_scheme_from_prompt():
    solver = SOLVER_MODELS[0]
    captured_prompt = []

    async def capture(*args, **kwargs):
        captured_prompt.append(kwargs["messages"][0]["content"])
        return _mock_response(SOLVER_RESPONSE)

    with patch("llm_utils.litellm.acompletion", side_effect=capture):
        await solve_one(GENERATION, solver)

    assert "secret" not in captured_prompt[0]
