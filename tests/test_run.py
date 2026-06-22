import pytest
from run import resolve_prompt_comparison, resolve_prompt
from prompts import specialist_system_prompt, specialist_work_backwards


def test_resolve_prompt_defaults_to_baseline_builder():
    assert resolve_prompt("baseline") is specialist_system_prompt


def test_resolve_prompt_returns_selected_variant_builder():
    assert resolve_prompt("work_backwards") is specialist_work_backwards


def test_resolve_prompt_rejects_unknown_variant():
    with pytest.raises(ValueError):
        resolve_prompt("does_not_exist")


def test_resolve_returns_model_and_names():
    model, names = resolve_prompt_comparison("gpt55", "baseline,no_guards")
    assert model["short"] == "gpt55"
    assert names == ["baseline", "no_guards"]


def test_resolve_rejects_unknown_variant():
    with pytest.raises(ValueError):
        resolve_prompt_comparison("gpt55", "baseline,does_not_exist")


def test_resolve_rejects_multiple_models():
    with pytest.raises(ValueError):
        resolve_prompt_comparison("gpt55,gpt-nano", "baseline,no_guards")


def test_resolve_rejects_single_variant():
    with pytest.raises(ValueError):
        resolve_prompt_comparison("gpt55", "baseline")


def test_resolve_rejects_no_model():
    with pytest.raises(ValueError):
        resolve_prompt_comparison(None, "baseline,no_guards")
