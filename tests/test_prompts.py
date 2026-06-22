import json
import pytest
from prompts import (
    specialist_system_prompt,
    specialist_no_guards,
    specialist_work_backwards,
    specialist_native_system_prompt,
    specialist_native_work_backwards,
    RUN_PYTHON_ORACLE,
    NATIVE_CODE_EXEC_ORACLE,
    WORK_BACKWARDS_PROTOCOL,
    WORK_BACKWARDS_PROTOCOL_NATIVE,
    PROMPT_VARIANTS,
    NATIVE_PROMPT_VARIANTS,
    STRAND_GUARDS,
    solver_prompt,
    judge_prompt,
    reconcile_prompt,
    match_gate_prompt,
)


def test_specialist_prompt_contains_board_schema_and_tool():
    p = specialist_system_prompt("pure", "Edexcel")
    assert "Edexcel" in p
    assert "JSON" in p
    assert "run_python" in p  # code-in-loop tool is advertised


def test_specialist_prompt_pure_guard():
    p = specialist_system_prompt("pure", "Edexcel")
    assert "0 < r < 1" in p          # geometric-series monotonicity guard
    assert "difficulty-collapse" in p  # parametric over-cancel guard


def test_specialist_prompt_mechanics_guard():
    p = specialist_system_prompt("mechanics", "Edexcel")
    assert "9.8" in p                 # gravity convention
    assert "friction" in p.lower()
    assert "R >= 0" in p              # physical-validity guard


def test_specialist_prompt_statistics_guard():
    p = specialist_system_prompt("statistics", "Edexcel")
    assert "Fraction" in p            # exact discrete-RV moments
    assert "Sxy**2 <= Sxx*Syy" in p   # regression |r|<=1 guard


def test_specialist_prompt_rejects_unknown_strand():
    with pytest.raises(ValueError):
        specialist_system_prompt("calculus", "Edexcel")


def test_solver_prompt_lists_questions():
    questions = [
        {"text": "Find dy/dx when y = x^2", "marks": 2},
        {"text": "Show that x^2 + 1 > 0", "marks": 3},
    ]
    p = solver_prompt(questions, "AQA")
    assert "Q1" in p
    assert "Q2" in p
    assert "[2 marks]" in p
    assert "markScheme" not in p  # mark scheme must be stripped


def test_solver_prompt_strips_mark_scheme():
    questions = [{"text": "Find x", "marks": 2, "markScheme": [{"tag": "M1", "text": "secret"}]}]
    p = solver_prompt(questions, "AQA")
    assert "secret" not in p
    assert "markScheme" not in p


def test_judge_prompt_contains_board_and_solver_answers():
    generation = {
        "board": "AQA",
        "topic": "Integration",
        "output_parsed": {
            "questions": [{"text": "Find ∫x dx", "marks": 2, "commandWord": "Find",
                           "difficulty": "foundation", "markScheme": [{"tag": "M1", "text": "x²/2"}, {"tag": "A1", "text": "+ C"}]}]
        }
    }
    solutions = [
        {"solver_model": "openrouter/qwen/qwq-32b", "answers": [{"question_index": 0, "answer": "x²/2 + C", "key_steps": ["integrate"]}], "skipped": False, "parse_ok": True},
        {"solver_model": "openrouter/deepseek/deepseek-r1", "answers": [{"question_index": 0, "answer": "x²/2 + C", "key_steps": ["integrate"]}], "skipped": False, "parse_ok": True},
    ]
    p = judge_prompt(generation, solutions)
    assert "AQA" in p
    assert "Integration" in p
    assert "x²/2 + C" in p
    assert "correctness" in p
    assert "style" in p


def test_solver_prompt_allows_unsolvable_flag():
    """Solvers must be able to declare a question unsolvable instead of
    being forced to emit some answer."""
    questions = [{"text": "Find x", "marks": 2}]
    p = solver_prompt(questions, "AQA")
    assert "unsolvable" in p


def test_reconcile_prompt_shows_answer_and_mark_scheme():
    questions = [
        {"text": "Find dy/dx when y = x^2", "marks": 2, "commandWord": "Find",
         "difficulty": "foundation",
         "markScheme": [{"tag": "M1", "text": "differentiate term by term"},
                        {"tag": "A1", "text": "dy/dx = 2x"}]},
    ]
    answers = [{"question_index": 0, "answer": "2x", "key_steps": ["power rule"]}]
    p = reconcile_prompt(questions, answers, "AQA")
    assert "Find dy/dx when y = x^2" in p
    assert "dy/dx = 2x" in p  # mark scheme is deliberately shown at this stage
    assert "2x" in p  # solver's own answer


def test_reconcile_prompt_requests_verdict_schema():
    questions = [{"text": "Find x", "marks": 2, "markScheme": [{"tag": "B1", "text": "x=1"}]}]
    answers = [{"question_index": 0, "answer": "x=2", "key_steps": []}]
    p = reconcile_prompt(questions, answers, "AQA")
    assert "agree" in p
    assert "solver_wrong" in p
    assert "scheme_wrong" in p
    assert "question_index" in p


def test_match_gate_prompt_compares_answer_with_scheme():
    questions = [{"text": "Differentiate $y = x^2$", "marks": 2,
                  "markScheme": [{"tag": "M1", "text": "power rule"},
                                 {"tag": "A1", "text": "$2x$"}]}]
    answers = [{"question_index": 0, "answer": "2x", "key_steps": []}]
    p = match_gate_prompt(questions, answers)
    assert "2x" in p
    assert "power rule" in p
    assert "question_index" in p
    assert "match" in p


def test_match_gate_prompt_tells_gate_to_fail_open():
    """When unsure, the gate must report a mismatch so the question still
    reaches full reconciliation."""
    questions = [{"text": "Find x", "marks": 2, "markScheme": [{"tag": "B1", "text": "x=1"}]}]
    answers = [{"question_index": 0, "answer": "x=1", "key_steps": []}]
    p = match_gate_prompt(questions, answers)
    assert "unsure" in p.lower()
    assert "false" in p


def test_judge_prompt_includes_reconciliation_verdicts():
    generation = {
        "board": "AQA",
        "topic": "Integration",
        "output_parsed": {"questions": [{"text": "Find x", "marks": 2, "commandWord": "Find",
                                         "difficulty": "foundation", "markScheme": []}]},
    }
    solutions = [
        {"solver_model": "openrouter/openai/gpt-5.4", "solver_short": "gpt54",
         "answers": [{"question_index": 0, "answer": "x=2", "key_steps": []}],
         "skipped": False, "parse_ok": True,
         "reconciliation": [{"question_index": 0, "verdict": "scheme_wrong",
                             "reason": "the scheme drops a factor of 2"}]},
    ]
    p = judge_prompt(generation, solutions)
    assert "scheme_wrong" in p
    assert "the scheme drops a factor of 2" in p


def test_judge_prompt_explains_disagreement_patterns():
    """The judge must be told to read the pattern of solver disagreement:
    convergence on one alternative answer implicates the scheme, scatter
    implicates the question (flagged as ambiguous_question)."""
    generation = {
        "board": "AQA",
        "topic": "Integration",
        "output_parsed": {"questions": [{"text": "Find x", "marks": 2, "commandWord": "Find",
                                         "difficulty": "foundation", "markScheme": []}]},
    }
    solutions = [
        {"solver_model": "openrouter/openai/gpt-5.4", "solver_short": "gpt54",
         "answers": [{"question_index": 0, "answer": "x=2", "key_steps": []}],
         "skipped": False, "parse_ok": True},
    ]
    p = judge_prompt(generation, solutions)
    assert "ambiguous_question" in p
    assert "SAME alternative answer" in p


def test_judge_prompt_shows_unsolvable_declarations():
    """A solver that declared a question unsolvable must surface that to the
    judge, distinct from an ordinary answer."""
    generation = {
        "board": "AQA",
        "topic": "Proof",
        "output_parsed": {"questions": [{"text": "Prove it", "marks": 4, "commandWord": "Prove that",
                                         "difficulty": "higher", "markScheme": []}]},
    }
    solutions = [
        {"solver_model": "openrouter/openai/gpt-5.4", "solver_short": "gpt54",
         "answers": [{"question_index": 0, "answer": "", "unsolvable": True,
                      "key_steps": ["statement is false for n=3"]}],
         "skipped": False, "parse_ok": True},
    ]
    p = judge_prompt(generation, solutions)
    assert "DECLARED UNSOLVABLE" in p


def test_judge_prompt_includes_solver_bias_profiles():
    """The judge should be told each solver's known error profile so it does
    not treat the three votes as equally reliable."""
    generation = {
        "board": "AQA",
        "topic": "Integration",
        "output_parsed": {"questions": [{"text": "Find x", "marks": 2, "commandWord": "Find",
                                         "difficulty": "foundation", "markScheme": []}]},
    }
    solutions = [
        {"solver_model": "openrouter/x-ai/grok-4.3", "solver_short": "grok43",
         "answers": [], "skipped": False, "parse_ok": True},
        {"solver_model": "openrouter/google/gemini-3.5-flash", "solver_short": "gemini-35-flash",
         "answers": [], "skipped": False, "parse_ok": True},
    ]
    p = judge_prompt(generation, solutions)
    assert "not yet characterised" in p  # grok43 profile
    assert "over-confirm" in p  # gemini-35-flash profile


def test_judge_prompt_agreement_keys_match_solvers():
    """The agreement template must name each solver that actually ran,
    not a hardcoded model."""
    generation = {
        "board": "AQA",
        "topic": "Integration",
        "output_parsed": {"questions": [{"text": "Find x", "marks": 2, "commandWord": "Find",
                                         "difficulty": "foundation", "markScheme": []}]},
    }
    solutions = [
        {"solver_model": "openrouter/deepseek/deepseek-v4-pro", "solver_short": "deepseek-pro",
         "answers": [], "skipped": False, "parse_ok": True},
        {"solver_model": "openrouter/openai/gpt-5.4", "solver_short": "gpt54",
         "answers": [], "skipped": False, "parse_ok": True},
    ]
    p = judge_prompt(generation, solutions)
    assert "deepseek-pro_agrees" in p
    assert "gpt54_agrees" in p
    assert "sonnet46_agrees" not in p


def test_specialist_prompt_injects_custom_guards():
    p = specialist_system_prompt("pure", "Edexcel", guards="ZZGUARDZZ")
    assert "ZZGUARDZZ" in p
    assert STRAND_GUARDS["pure"] not in p


def test_specialist_prompt_defaults_to_strand_guards():
    p = specialist_system_prompt("pure", "Edexcel")
    assert STRAND_GUARDS["pure"] in p


def test_prompt_variants_registry_has_baseline_and_no_guards():
    assert PROMPT_VARIANTS["baseline"] is specialist_system_prompt
    assert PROMPT_VARIANTS["no_guards"] is specialist_no_guards
    assert PROMPT_VARIANTS["work_backwards"] is specialist_work_backwards


def test_no_guards_variant_drops_strand_guard_text():
    p = PROMPT_VARIANTS["no_guards"]("pure", "Edexcel")
    assert STRAND_GUARDS["pure"] not in p
    assert "(no strand-specific design guards)" in p


def test_protocol_defaults_to_none_leaves_baseline_unchanged():
    """The protocol injection point must be inert by default: a plain call and an
    explicit protocol=None call both equal the baseline byte-for-byte."""
    assert specialist_system_prompt("pure", "Edexcel", protocol=None) == specialist_system_prompt("pure", "Edexcel")


def test_work_backwards_variant_injects_protocol_and_keeps_guards():
    """work_backwards adds the answer-first protocol while holding STRAND_GUARDS
    constant, so the protocol is the only changed variable vs baseline."""
    p = PROMPT_VARIANTS["work_backwards"]("pure", "Edexcel")
    assert "Answer first" in p              # the protocol's defining instruction
    assert "Reverse engineer" in p
    assert STRAND_GUARDS["pure"] in p       # guards unchanged
    # the only difference from baseline is the injected protocol block
    assert "Answer first" not in specialist_system_prompt("pure", "Edexcel")


def test_work_backwards_differs_from_baseline_only_by_protocol():
    """The A/B must isolate one variable: work_backwards equals baseline with the
    protocol block spliced in and NOTHING else changed. Removing the protocol
    text from the variant must reproduce the baseline byte-for-byte."""
    for strand in ("pure", "mechanics", "statistics"):
        base = specialist_system_prompt(strand, "Edexcel")
        variant = PROMPT_VARIANTS["work_backwards"](strand, "Edexcel")
        assert variant.replace(f"\n{WORK_BACKWARDS_PROTOCOL}\n", "") == base


def test_oracle_defaults_to_run_python_block():
    """Default (oracle=None) keeps the run_python tool paragraph, so the
    baseline prompt is unchanged by the new parameter."""
    p = specialist_system_prompt("pure", "Edexcel")
    assert RUN_PYTHON_ORACLE in p
    assert "You have a run_python tool" in p


def test_oracle_none_equals_plain_call_byte_for_byte():
    assert specialist_system_prompt("pure", "Edexcel", oracle=None) == specialist_system_prompt("pure", "Edexcel")


def test_native_prompt_swaps_tool_paragraph_keeps_guards_and_schema():
    """The native arm replaces ONLY the run_python tool-mechanics paragraph;
    strand guards and the JSON schema stay intact."""
    p = specialist_native_system_prompt("pure", "Edexcel")
    assert "You have a run_python tool" not in p        # mechanics paragraph swapped out
    assert "built-in Python code-execution environment" in p
    assert STRAND_GUARDS["pure"] in p                   # guards unchanged
    assert '"subtopics"' in p and '"markScheme"' in p   # schema present
    assert "final output must contain ONLY the JSON object" in p  # anti-prose emphasis


def test_native_differs_from_baseline_only_by_oracle_block():
    """Strong isolation guarantee: native == baseline with exactly the oracle
    paragraph swapped, and nothing else changed, for every strand."""
    for strand in ("pure", "mechanics", "statistics"):
        base = specialist_system_prompt(strand, "Edexcel")
        native = specialist_native_system_prompt(strand, "Edexcel")
        assert native.replace(NATIVE_CODE_EXEC_ORACLE, RUN_PYTHON_ORACLE) == base


def test_native_work_backwards_layers_protocol_on_native_oracle():
    """The native work-backwards prompt carries BOTH the native code-execution
    oracle and the work-backwards protocol, while keeping guards and schema."""
    p = specialist_native_work_backwards("pure", "Edexcel")
    assert NATIVE_CODE_EXEC_ORACLE in p                 # native sandbox oracle, not run_python
    assert "You have a run_python tool" not in p
    assert "Answer first" in p                          # work-backwards protocol present
    assert "Reverse engineer" in p
    assert STRAND_GUARDS["pure"] in p                   # guards unchanged
    assert '"subtopics"' in p and '"markScheme"' in p   # schema present


def test_native_work_backwards_protocol_uses_sandbox_not_run_python():
    """The native protocol must not advertise the harness run_python tool — its
    step-2 tool reference is the built-in code-execution environment."""
    assert "run_python" not in WORK_BACKWARDS_PROTOCOL_NATIVE
    assert "built-in code-execution environment" in WORK_BACKWARDS_PROTOCOL_NATIVE
    # the local protocol DOES name run_python; the two differ only on that tool ref
    assert "run_python" in WORK_BACKWARDS_PROTOCOL


def test_native_work_backwards_differs_from_native_baseline_only_by_protocol():
    """The native A/B isolates one variable: native work_backwards equals the
    native baseline with the protocol block spliced in and nothing else changed."""
    for strand in ("pure", "mechanics", "statistics"):
        base = specialist_native_system_prompt(strand, "Edexcel")
        variant = specialist_native_work_backwards(strand, "Edexcel")
        assert variant.replace(f"\n{WORK_BACKWARDS_PROTOCOL_NATIVE}\n", "") == base


def test_native_prompt_variants_registry():
    """The native registry maps the shared variant names to their native
    builders, so generate_one can select by the same name as the local A/B."""
    assert NATIVE_PROMPT_VARIANTS["baseline"] is specialist_native_system_prompt
    assert NATIVE_PROMPT_VARIANTS["work_backwards"] is specialist_native_work_backwards
    # no_guards has no native form on purpose (kept out so it is rejected up front)
    assert "no_guards" not in NATIVE_PROMPT_VARIANTS
