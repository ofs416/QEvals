from unittest.mock import MagicMock, patch

import pytest

from config import (
    MODELS,
    STRANDS,
    TOPICS,
    USAGE_INCLUDE,
    model_by_short,
    request_kwargs,
    response_cost,
    route_of,
    strand_for_topic,
)


def _response(cost=None, prompt_tokens=100, completion_tokens=200):
    r = MagicMock()
    r.usage = MagicMock(spec=["cost", "prompt_tokens", "completion_tokens"])
    r.usage.cost = cost
    r.usage.prompt_tokens = prompt_tokens
    r.usage.completion_tokens = completion_tokens
    return r


def test_every_topic_routes_to_a_known_strand():
    """No topic may fall through to the wrong specialist — every TOPICS entry
    must map to one of the three strands."""
    for topic in TOPICS:
        assert strand_for_topic(topic) in STRANDS


def test_strand_for_topic_raises_on_unmapped_topic():
    with pytest.raises(ValueError):
        strand_for_topic("Astrophysics")


def test_response_cost_uses_openrouter_reported_cost():
    r = _response(cost=8.684e-05)
    assert response_cost(MODELS[0], r) == 8.684e-05


def test_response_cost_accepts_zero_cost():
    # Free-tier models legitimately report 0
    r = _response(cost=0.0)
    assert response_cost(MODELS[0], r) == 0.0


def test_response_cost_returns_zero_and_warns_when_missing(capsys):
    r = _response(cost=None)
    assert response_cost(MODELS[0], r) == 0.0
    assert "no OpenRouter cost" in capsys.readouterr().out


def test_response_cost_returns_zero_on_non_numeric_cost(capsys):
    # Unexpected provider payloads must not leak into cost_usd
    r = _response(cost="0.001")
    assert response_cost(MODELS[0], r) == 0.0
    assert "no OpenRouter cost" in capsys.readouterr().out


def test_response_cost_returns_zero_when_usage_lacks_cost_attr(capsys):
    r = MagicMock()
    r.usage = MagicMock(spec=["prompt_tokens", "completion_tokens"])
    r.usage.prompt_tokens = 100
    r.usage.completion_tokens = 200
    assert response_cost(MODELS[0], r) == 0.0
    assert "no OpenRouter cost" in capsys.readouterr().out


# --- routing: OpenRouter vs direct provider --------------------------------

def test_route_of_defaults_to_openrouter_for_openrouter_ids():
    assert route_of({"id": "openrouter/openai/gpt-5.4-nano", "short": "x"}) == "openrouter"


def test_route_of_detects_direct_gemini_by_prefix():
    assert route_of({"id": "gemini/gemini-2.5-flash", "short": "g"}) == "gemini"


def test_route_of_honours_explicit_route_override():
    # An explicit route wins over the id prefix, so a model can be forced.
    assert route_of({"id": "openrouter/google/gemini-2.5-flash", "short": "g", "route": "gemini"}) == "gemini"


def test_request_kwargs_openrouter_carries_usage_reasoning_and_provider():
    model = {"id": "openrouter/google/gemini-2.5-flash", "short": "g",
             "provider": {"only": ["google-ai-studio"]}}
    kw = request_kwargs(model, {"effort": "medium"})
    assert kw == {"extra_body": {**USAGE_INCLUDE,
                                 "reasoning": {"effort": "medium"},
                                 "provider": {"only": ["google-ai-studio"]}}}


def test_request_kwargs_openrouter_omits_reasoning_when_none():
    kw = request_kwargs({"id": "openrouter/openai/gpt-5.4-nano", "short": "x"}, None)
    assert kw == {"extra_body": dict(USAGE_INCLUDE)}
    assert "reasoning" not in kw["extra_body"]


def test_request_kwargs_gemini_maps_effort_to_reasoning_effort():
    # Direct providers take litellm's portable reasoning_effort, NOT the
    # OpenRouter extra_body reasoning block or usage.include.
    kw = request_kwargs({"id": "gemini/gemini-2.5-flash", "short": "g"}, {"effort": "low"})
    assert kw == {"reasoning_effort": "low"}


def test_request_kwargs_gemini_omits_reasoning_effort_when_none():
    kw = request_kwargs({"id": "gemini/gemini-2.5-flash", "short": "g"}, None)
    assert kw == {}


def test_response_cost_direct_route_uses_litellm_completion_cost():
    model = {"id": "gemini/gemini-2.5-flash", "short": "g"}
    r = _response(cost=None)  # direct providers don't report a billed cost
    with patch("config.litellm.completion_cost", return_value=0.0021) as cc:
        assert response_cost(model, r) == 0.0021
    cc.assert_called_once()


def test_response_cost_direct_route_returns_zero_when_pricing_unknown(capsys):
    model = {"id": "gemini/gemini-2.5-flash", "short": "g"}
    r = _response(cost=None)
    with patch("config.litellm.completion_cost", side_effect=Exception("no price")):
        assert response_cost(model, r) == 0.0
    assert "completion_cost" in capsys.readouterr().out


def test_gemini_lite_native_entry_is_native_and_gemini_routed():
    m = model_by_short("gemini-lite-native")
    assert m["native_code_exec"] is True
    assert m["id"] == "gemini/gemini-3.1-flash-lite"
    assert route_of(m) == "gemini"
