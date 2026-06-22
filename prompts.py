import json

from config import SOLVER_PROFILES

STRAND_LABEL = {
    "pure": "Pure Mathematics",
    "mechanics": "Mechanics",
    "statistics": "Statistics",
}

# Per-strand design guards. Each guard targets the failure classes a unified
# prompt can't economically enumerate; they are the reason the generator is
# split by strand rather than the raw quality of any single prompt. Every
# numeric claim a guard makes ("verify", "assert") is to be checked with the
# run_python tool, not by hand.
STRAND_GUARDS = {
    "pure": """Pure design guards:
- An EXACT closed form (a root, a single surd or fraction, a one-step evaluation) may be stated directly. VERIFY with run_python any result needing more than one arithmetic combination step: multi-subinterval or sign-split definite integrals, summed displacements, geometric-series sums, total-distance computations. Multi-step 'exact' values are exactly where a hand answer slips.
- Geometric series 'smallest n such that S_n exceeds a threshold': force 0 < r < 1 so the partial sums increase monotonically, and verify with code that the intended n is unique and at least 3 — an oscillating r < 0 collapses to a degenerate n = 1.
- For a quotient or parametric derivative, assert with code that the simplified dy/dx still has a non-constant denominator (e.g. denom(simplify(dydx)).has(t)) so the intended technique survives simplification. A 'hard' question whose derivative over-cancels to a trivial linear result is a difficulty-collapse failure even when every number is correct.""",
    "mechanics": """Mechanics design guards (use g = 9.8):
- Compute the actual physical quantities with run_python and state the TRUE outcome (clears / does not clear the wall; moves / stays at rest; overtakes / does not). Assert the SIGNED direction of the stated outcome with a clear margin — never rig a tie, and never let the thing meant to fail come out succeeding.
- Verify every reaction, tension and friction value is physically valid: normal reaction R >= 0; static friction <= mu*R; a body in equilibrium carries friction equal to the component it balances, not F_max. If a given makes the situation impossible, change the given and recompute.""",
    "statistics": """Statistics design guards:
- Compute EVERY probability, cumulative value, critical value and discrete-RV moment with run_python — binomial/normal/Poisson cumulative arithmetic and off-by-one tails are the dominant error source, and they cannot be reliably done by hand. Use fractions.Fraction for exact E(X), E(X^2), Var(X), E(aX+b), Var(aX+b); never hand-sum them.
- Before writing a regression question, verify the summary statistics satisfy Sxy**2 <= Sxx*Syy (so |r| <= 1).
- A critical region can be empty at small n or extreme p. If an empty region is the intended teaching point, state 'region empty' explicitly in the question and mark scheme; otherwise change the givens until it is non-empty. Defer the empty/non-empty verdict to code — never invent a critical value.""",
}


# The deterministic-oracle instruction block. Extracted so the native arm can
# swap ONLY this paragraph (see specialist_native_system_prompt) while the
# schema, strand guards and rules stay byte-identical — keeping the
# local-vs-native code-execution A/B to a single changed variable.
RUN_PYTHON_ORACLE = """You have a run_python tool (Python 3 with sympy, numpy, math, fractions, statistics). Use it as a deterministic oracle:
- COMPUTE and VERIFY every value beyond a single-step closed form — decimals, powers, logs, binomial/normal/Poisson probabilities, critical values, multi-step integrals and sums. Do not hand-write any such number; read it back from code.
- ASSERT the design invariants in the guards below. If an assertion fails, CHANGE THE GIVENS and recompute — never fudge arithmetic to force an intended answer.
- Do all of this BEFORE writing any JSON. The final assistant turn must contain ONLY the JSON object — no tool call, no prose, no working notes."""

# Native arm: same oracle intent, but the model uses Gemini's built-in
# server-side code execution rather than the harness run_python tool. The final
# bullet leans harder on "ONLY the JSON object" because native code-exec tends
# to append prose after the answer.
NATIVE_CODE_EXEC_ORACLE = """You have a built-in Python code-execution environment (Python 3 with sympy, numpy, math, fractions, statistics). Use it as a deterministic oracle:
- COMPUTE and VERIFY every value beyond a single-step closed form — decimals, powers, logs, binomial/normal/Poisson probabilities, critical values, multi-step integrals and sums. Do not hand-write any such number; read it back from code.
- ASSERT the design invariants in the guards below. If an assertion fails, CHANGE THE GIVENS and recompute — never fudge arithmetic to force an intended answer.
- Do all of this BEFORE writing any JSON. After all code execution, your final output must contain ONLY the JSON object — no further code, no prose, no working notes."""


def specialist_system_prompt(strand: str, board: str, level: str = "A-level", *, guards: str | None = None, protocol: str | None = None, oracle: str | None = None) -> str:
    """System prompt for one of the three generator specialists.

    Shares the JSON schema, escaping and originality rules across strands, then
    appends the strand-specific design guards. Each specialist has a run_python
    tool and is told to verify every non-trivial value and assert its invariants
    against it before emitting the final JSON.

    Pass ``guards=None`` (the default) to use ``STRAND_GUARDS[strand]``; pass a
    string to override the guard block (used by prompt variants). Note an empty
    string is honoured as-is — to get the strand default, pass ``None``.

    Pass ``protocol`` (a string) to inject an extra methodology block between the
    oracle instructions and the design guards (used by prompt variants such as
    work_backwards). ``None`` (the default) inserts nothing, leaving the baseline
    output unchanged.

    Pass ``oracle`` (a string) to replace the deterministic-oracle paragraph
    (used by the native code-execution arm). ``None`` (the default) uses
    ``RUN_PYTHON_ORACLE``, leaving the baseline output unchanged.
    """
    if strand not in STRAND_GUARDS:
        raise ValueError(
            f"Unknown strand {strand!r}; expected one of {sorted(STRAND_GUARDS)}."
        )
    label = STRAND_LABEL[strand]
    guards_text = STRAND_GUARDS[strand] if guards is None else guards
    protocol_block = f"\n{protocol}\n" if protocol else ""
    oracle_text = RUN_PYTHON_ORACLE if oracle is None else oracle
    return f"""You are a UK maths teacher and {label} specialist generating original exam-style questions for {board} {level} students.

{oracle_text}
{protocol_block}
{guards_text}

When every value and invariant has been checked, respond with ONLY a valid JSON object — no prose, no markdown fences, no explanation:

{{
  "subtopics": ["string", ...],
  "questions": [
    {{
      "text": "string",
      "commandWord": "string",
      "marks": number,
      "difficulty": "foundation" | "higher" | "extension",
      "markScheme": [{{"tag": "M1" | "A1" | "B1", "text": "string"}}]
    }}
  ]
}}

Rules:
- subtopics: 5-8 strings naming the key areas of this topic in the {board} {level} specification
- questions: exactly 3 questions, progressing from foundation to higher/extension difficulty
- Every question object must contain all five keys: "text", "commandWord", "marks", "difficulty", "markScheme". "difficulty" is required on every question and must be exactly one of "foundation", "higher", "extension" — never omit it.
- Every numeric value in a question and its mark scheme must be one you computed or confirmed with run_python.
- commandWord: use {board}'s standard command words (Calculate, Show that, Prove that, Find, Explain, Sketch, State, Determine)
- marks: integer 2-6; the number of markScheme entries must equal marks
- LaTeX: use $...$ for inline maths and $$...$$ for display maths — KaTeX syntax only, no \\begin{{}} environments
- CRITICAL — JSON string escaping: every LaTeX backslash must be written as a double backslash inside JSON strings. Correct: "Find $\\\\frac{{dy}}{{dx}}$". Wrong: "Find $\\frac{{dy}}{{dx}}$" — a single backslash before a letter (as in \\frac, \\tan, \\arctan) is an invalid JSON escape and the entire response will be rejected.
- Finalise every question and mark scheme BEFORE writing the JSON. The JSON must contain only finished content — never include working notes, self-corrections, or commentary (e.g. "Let's re-examine", "this is not correctly worded"). If you notice a problem mid-answer, silently discard it and write only the corrected version.
- Inside JSON string values: never use double quotes — use single quotes for any quoted phrase. Never use literal line breaks — keep each string on one line (use \\n only if a break is essential).
- All questions must be wholly original — do not reproduce any past-paper questions or published mark schemes"""


def specialist_no_guards(strand: str, board: str) -> str:
    """Variant builder: the baseline scaffold with the strand design guards
    removed — used to A/B-test whether STRAND_GUARDS earn their keep."""
    return specialist_system_prompt(strand, board, guards="(no strand-specific design guards)")


# The "work-backwards" design methodology: choose clean answers first, then
# reverse-engineer the givens with code, verify there are no unintended edge
# cases, and only then write the question. The hypothesis is that answer-first
# design yields cleaner answers and fewer difficulty-collapse / multiple-root
# failures than the baseline's compute-as-you-go framing. Holds STRAND_GUARDS
# and the output schema constant, so the protocol is the only changed variable.
WORK_BACKWARDS_PROTOCOL = """THE WORK-BACKWARDS PROTOCOL (mandatory for every question):
1. Answer first: do NOT write the question text first. Choose the final, clean, 'nice' answers first (integers, simple fractions, clear surds).
2. Reverse engineer: use run_python (sympy/numpy/math) to calculate the constants, coefficients or givens that lead to those chosen answers.
3. Execute & verify: run the code and check for unintended edge cases — multiple roots/intersections where one is expected; a derivative that over-cancels to a trivial result and loses the intended difficulty; violated physical/statistical constraints (negative reaction, empty critical region).
4. Finalise: only once the code confirms the values are sound, write the question text and mark scheme around those verified givens."""


def specialist_work_backwards(strand: str, board: str) -> str:
    """Variant builder: the baseline scaffold plus the work-backwards protocol
    (answer-first, reverse-engineered design). STRAND_GUARDS and the output
    schema are held constant so the protocol is the only changed variable."""
    return specialist_system_prompt(strand, board, protocol=WORK_BACKWARDS_PROTOCOL)


# Native-arm work-backwards protocol: identical to WORK_BACKWARDS_PROTOCOL except
# step 2's tool reference is the built-in code-execution environment rather than
# the harness run_python tool, so the methodology text is consistent with the
# native oracle (NATIVE_CODE_EXEC_ORACLE) it is paired with. Kept as its own
# constant so the native A/B isolates the protocol the same way the local one does.
WORK_BACKWARDS_PROTOCOL_NATIVE = """THE WORK-BACKWARDS PROTOCOL (mandatory for every question):
1. Answer first: do NOT write the question text first. Choose the final, clean, 'nice' answers first (integers, simple fractions, clear surds).
2. Reverse engineer: use your built-in code-execution environment (sympy/numpy/math) to calculate the constants, coefficients or givens that lead to those chosen answers.
3. Execute & verify: run the code and check for unintended edge cases — multiple roots/intersections where one is expected; a derivative that over-cancels to a trivial result and loses the intended difficulty; violated physical/statistical constraints (negative reaction, empty critical region).
4. Finalise: only once the code confirms the values are sound, write the question text and mark scheme around those verified givens."""


def specialist_native_system_prompt(strand: str, board: str) -> str:
    """Native-arm builder: the baseline scaffold with the run_python tool
    paragraph swapped for Gemini's built-in code execution. Schema, strand
    guards and rules are held byte-identical so the execution mechanism is the
    only changed variable vs the local baseline."""
    return specialist_system_prompt(strand, board, oracle=NATIVE_CODE_EXEC_ORACLE)


def specialist_native_work_backwards(strand: str, board: str) -> str:
    """Native-arm work-backwards builder: the work-backwards protocol layered on
    Gemini's built-in code-execution oracle. The protocol's tool reference is the
    native sandbox (WORK_BACKWARDS_PROTOCOL_NATIVE), so the methodology is
    consistent with the oracle it runs on. STRAND_GUARDS and the schema are held
    constant, so vs the native baseline the only changed variable is the
    work-backwards protocol — mirroring the local baseline -> work_backwards A/B."""
    return specialist_system_prompt(
        strand, board, protocol=WORK_BACKWARDS_PROTOCOL_NATIVE, oracle=NATIVE_CODE_EXEC_ORACLE
    )


# Named generator-prompt builders for A/B comparison (run.py --compare-prompts).
# Each value is a builder with the same (strand, board) -> str signature as the
# baseline. Add a variant by writing such a function and registering it here.
PROMPT_VARIANTS = {
    "baseline": specialist_system_prompt,
    "no_guards": specialist_no_guards,
    "work_backwards": specialist_work_backwards,
}

# Native code-exec equivalents of PROMPT_VARIANTS, keyed by the SAME variant
# name. generate_one selects from here (not PROMPT_VARIANTS) when a model carries
# native_code_exec, so a --compare-prompts A/B on the native arm holds the
# execution mechanism (Gemini's server-side sandbox) constant across arms and
# varies only the prompt — the native mirror of the local PROMPT_VARIANTS A/B. A
# variant present in PROMPT_VARIANTS but absent here has no native form and is
# rejected up front (generate_all_variants) rather than silently falling back.
NATIVE_PROMPT_VARIANTS = {
    "baseline": specialist_native_system_prompt,
    "work_backwards": specialist_native_work_backwards,
}


def cleaner_prompt(raw: str) -> str:
    return f"""The text below should be a single valid JSON object with keys "subtopics" and "questions", but it is malformed (invalid escapes, unescaped inner quotes, raw line breaks, stray commentary, markdown fences, or truncation).

Rewrite it as ONE valid JSON object. Rules:
- Output ONLY the JSON object — no fences, no explanation.
- Preserve every string value VERBATIM. The only permitted changes inside strings are escaping: \\" for inner double quotes, \\\\ for backslashes, \\n for line breaks.
- Delete prose that sits outside string values (working notes, self-corrections between keys).
- Never add, remove, reword, shorten, or improve content — even if it looks wrong. You are a syntax repairer, not an editor.
- If a question object is truncated mid-way, drop that object entirely; keep all complete ones.

TEXT:
{raw}"""


def solver_cleaner_prompt(raw: str, shape: str) -> str:
    """Syntax-only repair for a malformed solver/reconcile JSON response.

    Mirrors cleaner_prompt but is schema-agnostic — `shape` describes the
    expected top-level structure (e.g. an "answers" or "reconciliations" array)
    so the same repairer serves both solver stages.
    """
    return f"""The text below should be a single valid JSON object {shape}, but it is malformed (invalid escapes, unescaped inner quotes, raw line breaks, stray commentary, markdown fences, or truncation).

Rewrite it as ONE valid JSON object. Rules:
- Output ONLY the JSON object — no fences, no explanation.
- Preserve every value VERBATIM. The only permitted changes inside strings are escaping: \\" for inner double quotes, \\\\ for backslashes, \\n for line breaks.
- Delete prose that sits outside string values (working notes, commentary between keys).
- Never add, remove, reword, shorten, or improve content — even if it looks wrong. You are a syntax repairer, not an editor.
- If an array element is truncated mid-way, drop that element entirely; keep all complete ones.

TEXT:
{raw}"""


def solver_prompt(questions: list[dict], board: str) -> str:
    lines = [
        f"Q{i + 1} [{q.get('marks', '?')} marks]: {q.get('text', '')}"
        for i, q in enumerate(questions)
    ]
    qs_text = "\n".join(lines)
    n = len(questions)
    indices = ", ".join(
        [
            f'{{"question_index": {i}, "answer": "...", "key_steps": ["..."], "unsolvable": false}}'
            for i in range(n)
        ]
    )
    return f"""Solve these {board} A-level Maths questions. Return ONLY a JSON object, no prose, no markdown.

{qs_text}

Return exactly:
{{
  "answers": [{indices}]
}}

If a question cannot be solved (contradictory, ambiguous, or missing information), set "unsolvable": true for that entry, leave "answer" empty, and explain why in "key_steps". Do not guess an answer to a broken question.
"""


def match_gate_prompt(questions: list[dict], answers: list[dict]) -> str:
    """Cheap pre-reconciliation gate: does the solver's blind answer already
    match the mark scheme? Only mismatches go on to the full (expensive)
    reconcile call, so the gate must fail open — unsure means mismatch.
    """
    by_index = {a.get("question_index"): a for a in answers}
    blocks = []
    for i, q in enumerate(questions):
        ans = by_index.get(i)
        if ans is None:
            continue
        scheme = "; ".join(
            f"[{ms.get('tag', '?')}] {ms.get('text', '')}"
            for ms in q.get("markScheme", [])
        )
        blocks.append(
            f"question_index {i}: {q.get('text', '')}\n"
            f"MARK SCHEME: {scheme}\n"
            f"SOLVER ANSWER: {ans.get('answer', '')}"
        )
    qs_text = "\n\n".join(blocks)
    entries = ", ".join(
        f'{{"question_index": {i}, "match": true | false}}' for i in sorted(by_index)
    )
    return f"""For each question below, decide whether the solver's final answer is mathematically equivalent to the result in the mark scheme. Ignore notation and formatting differences ($x = \\pm 2$ vs 'x equals 2 or -2'); judge only whether they reach the same result.

{qs_text}

Return ONLY this JSON, no prose:

{{"matches": [{entries}]}}

If you are unsure whether two answers are equivalent, use false."""


def reconcile_prompt(questions: list[dict], answers: list[dict], board: str) -> str:
    """Stage-2 prompt: show the solver its own blind answer next to the mark
    scheme and ask it to adjudicate the conflict (or confirm agreement).

    This is deliberately framed as adjudicating two specific answers rather
    than open-ended verification, which anchors far less.
    """
    by_index = {a.get("question_index"): a for a in answers}
    blocks = []
    for i, q in enumerate(questions):
        ans = by_index.get(i)
        if ans is None:
            continue
        scheme = "\n".join(
            f"  [{ms.get('tag', '?')}] {ms.get('text', '')}"
            for ms in q.get("markScheme", [])
        )
        if ans.get("unsolvable"):
            your_answer = "(you declared this question unsolvable: "
            your_answer += "; ".join(ans.get("key_steps", [])) + ")"
        else:
            your_answer = ans.get("answer", "")
        blocks.append(
            f"Q{i + 1} [{q.get('marks', '?')} marks]: {q.get('text', '')}\n"
            f"MARK SCHEME:\n{scheme}\n"
            f"YOUR ANSWER: {your_answer}"
        )
    qs_text = "\n\n".join(blocks)
    entries = ", ".join(
        f'{{"question_index": {i}, "verdict": "...", "reason": "<one sentence>"}}'
        for i in sorted(by_index)
    )
    return f"""You previously solved these {board} A-level Maths questions without seeing the mark schemes. Below, each question is shown with its official mark scheme and the answer you gave.

{qs_text}

For each question, compare your answer with the mark scheme and adjudicate. Return ONLY this JSON, no prose:

{{
  "reconciliations": [{entries}]
}}

verdict must be exactly one of:
- "agree" — your answer and the mark scheme reach the same result
- "solver_wrong" — the mark scheme is correct; your answer contains the error
- "scheme_wrong" — your answer is correct; the mark scheme contains the error
- "unclear" — both are defensible, or the question is too ambiguous to adjudicate

Re-derive the key step before deciding — do not assume the mark scheme is correct just because it is official, and do not defend your own answer out of consistency."""


def judge_prompt(generation: dict, solutions: list[dict]) -> str:
    board = generation["board"]
    topic = generation["topic"]
    questions = generation["output_parsed"]["questions"]
    qs_json = json.dumps(questions, indent=2)

    solver_section = ""
    shorts = []
    for sol in solutions:
        short = sol.get("solver_short") or sol["solver_model"].split("/")[-1]
        shorts.append(short)
        if sol.get("skipped") or not sol.get("parse_ok"):
            solver_section += f"\n[{short}]: solver failed to parse\n"
            continue
        solver_section += f"\n[{short}]:\n"
        for ans in sol.get("answers", []):
            if ans.get("unsolvable"):
                solver_section += (
                    f"  Q{ans['question_index'] + 1}: DECLARED UNSOLVABLE\n"
                )
            else:
                solver_section += f"  Q{ans['question_index'] + 1}: {ans['answer']}\n"
            for step in ans.get("key_steps", [])[:3]:
                solver_section += f"    - {step}\n"
        for rec in sol.get("reconciliation") or []:
            solver_section += (
                f"  Q{rec['question_index'] + 1} reconciliation (after seeing the mark scheme):"
                f" {rec.get('verdict', 'unclear')} — {rec.get('reason', '')}\n"
            )

    profile_lines = "\n".join(
        f"- {s}: {SOLVER_PROFILES[s]}" for s in shorts if s in SOLVER_PROFILES
    )
    profile_section = (
        f"\nSOLVER ERROR PROFILES (from past runs — weigh their votes accordingly):\n{profile_lines}\n"
        if profile_lines
        else ""
    )

    agreement_fields = ", ".join(f'"{s}_agrees": <true|false|null>' for s in shorts)
    per_q = ",\n    ".join(
        f'{{"question_index": {i}, "scores": {{"correctness": <1-5>, "mark_scheme": <1-5>, '
        f'"command_word": <1-5>, "difficulty": <1-5>, "style": <1-5>}}, "total": <sum of the five>, '
        f'"flags": [], "notes": {{"correctness": "<one sentence>", "command_word": "<one sentence>"}}}}'
        for i in range(len(questions))
    )

    return f"""You are evaluating {board} A-level Maths questions for topic: {topic}.

GENERATED QUESTIONS AND MARK SCHEMES:
{qs_json}

INDEPENDENT SOLVER ANSWERS:
{solver_section}{profile_section}
Solvers answered blind first; where a reconciliation line is shown, the solver was then shown the mark scheme and adjudicated the conflict ("agree" / "solver_wrong" / "scheme_wrong" / "unclear"). Use reconciliation verdicts to distinguish a wrong question from a failed solver.

When solvers disagree with the mark scheme, the pattern of disagreement matters: multiple solvers independently converging on the SAME alternative answer is strong evidence the mark scheme is wrong; solvers disagreeing with the scheme AND with each other suggests the question is ambiguous or underspecified — flag "ambiguous_question" on that question.

Score EACH question independently on five dimensions (1-5 each). Judge every question on its own merits — do not let a strong or weak sibling question pull its score. Return ONLY this JSON, no prose:

{{
  "questions": [
    {per_q}
  ],
  "solver_agreement": {{{agreement_fields}}}
}}

Scoring (apply to each question on its own):
- correctness: Is the question mathematically valid? Does the solver answer confirm the mark scheme?
  1=major errors  3=minor issues  5=impeccable and solver agrees
- mark_scheme: Does point count equal marks? Are steps distinct and non-overlapping?
  1=wrong count  3=mostly consistent  5=perfectly structured
- command_word: Does {board} use this command word this way?
  1=wrong word  3=acceptable  5=exact {board} convention
- difficulty: Does cognitive demand match marks? (2 marks=recall, 6 marks=multi-step)
  1=miscalibrated  3=acceptable  5=precisely matched
- style: Reads like a real {board} paper? Formal register, no AI phrasing?
  1=clearly AI  3=acceptable  5=indistinguishable from real paper

total: the sum of the five dimension scores for that question (max 25).
flags (per question): include "unsolvable_question" if that question cannot be solved; "ambiguous_question" if solver scatter indicates it is ambiguous or underspecified; "past_paper_suspected" if it closely resembles a known exam question.
solver_agreement (whole batch): for each solver — true=its answers match the mark schemes overall, false=mismatch, null=cannot determine. Judge each solver independently."""
