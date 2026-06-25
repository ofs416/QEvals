import pytest
from prompts import (
    STRAND_GUARDS,
    cleaner_prompt_for,
    drafter_system_prompt,
    maths_judge_prompt,
    optimiser_prompt,
    style_judge_prompt,
    typesetter_prompt,
)


# --- drafter -----------------------------------------------------------------

def test_drafter_prompt_asks_for_n_drafts_and_no_scheme():
    p = drafter_system_prompt("pure", "Edexcel", 3)
    assert "Edexcel" in p
    assert '"drafts"' in p
    assert "do not write a mark scheme" in p.lower()
    assert "exactly 3 questions" in p


def test_drafter_prompt_single_question_phrasing():
    p = drafter_system_prompt("pure", "Edexcel", 1)
    assert "a single well-judged question" in p


def test_drafter_prompt_carries_strand_design_intent():
    p = drafter_system_prompt("pure", "Edexcel", 2)
    assert STRAND_GUARDS["pure"] in p
    # drafter runs no code — it must not advertise a sandbox/oracle
    assert "run_python" not in p
    assert "code-execution" not in p


def test_drafter_prompt_rejects_unknown_strand():
    with pytest.raises(ValueError):
        drafter_system_prompt("calculus", "Edexcel", 2)


# --- optimiser ---------------------------------------------------------------

def test_optimiser_prompt_uses_native_sandbox_and_emits_values():
    p = optimiser_prompt(["Find dy/dx of y=x^2"], "pure", "Edexcel")
    assert "code-execution environment" in p   # Google native sandbox
    assert '"values"' in p
    assert '"markScheme"' in p
    assert "Find dy/dx of y=x^2" in p
    assert STRAND_GUARDS["pure"] in p


def test_optimiser_prompt_lists_each_draft():
    p = optimiser_prompt(["A", "B"], "pure", "Edexcel")
    assert "DRAFT 0: A" in p
    assert "DRAFT 1: B" in p


# --- typesetter --------------------------------------------------------------

def test_typesetter_prompt_demands_verbatim_values_and_html():
    questions = [{"text": "Find x", "values": {"x": "2"}, "markScheme": []}]
    p = typesetter_prompt(questions, "Edexcel")
    assert "VERBATIM" in p
    assert "question_html" in p
    assert "mark_scheme_html" in p
    assert "Do NOT recompute" in p


# --- judges ------------------------------------------------------------------

def test_maths_judge_prompt_uses_sandbox_and_asks_hard_verdict():
    p = maths_judge_prompt([{"text": "Find x", "markScheme": []}], "Edexcel")
    assert "code-execution environment" in p
    assert "maths_correct" in p
    assert "do not do the arithmetic in your head" in p.lower()


def test_style_judge_prompt_scores_three_dims_no_correctness():
    p = style_judge_prompt([{"text": "Find x"}], "Edexcel")
    assert "command_word" in p
    assert "difficulty" in p
    assert "style" in p
    assert "suitability_total" in p
    # correctness/mark_scheme are the maths judge's job, not this one
    assert "correctness" not in p


# --- shared cleaner ----------------------------------------------------------

def test_cleaner_prompt_for_embeds_shape_and_raw():
    p = cleaner_prompt_for("BROKEN{", shape='with a top-level "drafts" array')
    assert 'with a top-level "drafts" array' in p
    assert "BROKEN{" in p
    assert "syntax repairer" in p
