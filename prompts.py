import json

STRAND_LABEL = {
    "pure": "Pure Mathematics",
    "mechanics": "Mechanics",
    "statistics": "Statistics",
}

# Per-strand design guards. Each guard targets the failure classes a unified
# prompt can't economically enumerate. They are split across the chain: the
# drafter reads them as DESIGN INTENT (keep the question non-degenerate), the
# optimiser reads them as WHAT TO VERIFY in the sandbox before finalising values.
STRAND_GUARDS = {
    "pure": """Pure design guards:
- Multi-step 'exact' values (multi-subinterval or sign-split definite integrals, summed displacements, geometric-series sums, total-distance computations) are exactly where a hand answer slips — they must be computed, not guessed.
- Geometric series 'smallest n such that S_n exceeds a threshold': force 0 < r < 1 so the partial sums increase monotonically, and make sure the intended n is unique and at least 3 — an oscillating r < 0 collapses to a degenerate n = 1.
- For a quotient or parametric derivative, the simplified dy/dx must still have a non-constant denominator so the intended technique survives simplification. A 'hard' question whose derivative over-cancels to a trivial linear result is a difficulty-collapse failure even when every number is correct.""",
    "mechanics": """Mechanics design guards (use g = 9.8):
- State the TRUE physical outcome (clears / does not clear the wall; moves / stays at rest; overtakes / does not), with a clear signed margin — never rig a tie, and never let the thing meant to fail come out succeeding.
- Every reaction, tension and friction value must be physically valid: normal reaction R >= 0; static friction <= mu*R; a body in equilibrium carries friction equal to the component it balances, not F_max. If a given makes the situation impossible, the given must change.""",
    "statistics": """Statistics design guards:
- Binomial/normal/Poisson cumulative arithmetic and off-by-one tails are the dominant error source and cannot be reliably done by hand — every probability, cumulative value, critical value and discrete-RV moment must be computed exactly (use fractions for exact E(X), E(X^2), Var(X), E(aX+b), Var(aX+b)).
- A regression question's summary statistics must satisfy Sxy**2 <= Sxx*Syy (so |r| <= 1).
- A critical region can be empty at small n or extreme p. If an empty region is the intended teaching point, state 'region empty' explicitly; otherwise the givens must change until it is non-empty. The empty/non-empty verdict is a computed fact, never invented.""",
}


# --- Stage 1: Drafter --------------------------------------------------------

def drafter_system_prompt(strand: str, board: str, n: int, level: str = "A-level") -> str:
    """System prompt for the drafter: brainstorm n original question scenarios.

    The drafter writes the QUESTION TEXT only — no mark scheme, and the numbers
    need not be neat (the optimiser solves it and tweaks the constants). It is
    strand-aware so the design guards keep the scenario non-degenerate, but it
    runs no code.
    """
    if strand not in STRAND_GUARDS:
        raise ValueError(
            f"Unknown strand {strand!r}; expected one of {sorted(STRAND_GUARDS)}."
        )
    label = STRAND_LABEL[strand]
    ladder = (
        "a single well-judged question"
        if n == 1
        else f"exactly {n} questions, progressing from foundation to higher/extension difficulty"
    )
    return f"""You are a creative UK maths teacher and {label} specialist drafting original exam-style questions for {board} {level} students.

Brainstorm {ladder}. Focus on interesting, non-trivial scenarios that test genuine understanding — you may combine sub-areas of the topic. Do NOT worry about the exact numbers being perfectly neat; a later stage solves each question and tweaks the constants. Do NOT solve the questions and do NOT write a mark scheme.

Keep these design intentions in mind so a scenario doesn't collapse to something trivial once solved:
{STRAND_GUARDS[strand]}

Respond with ONLY a valid JSON object — no prose, no markdown fences:

{{
  "drafts": ["question text", ...]
}}

Rules:
- "drafts": {ladder}, each a self-contained question statement (the scenario, the givens, and what is asked).
- Use $...$ for inline maths and $$...$$ for display maths — KaTeX syntax only, no \\begin{{}} environments.
- CRITICAL — JSON string escaping: every LaTeX backslash must be a double backslash inside JSON strings. Correct: "Find $\\\\frac{{dy}}{{dx}}$". A single backslash before a letter is an invalid JSON escape and the whole response is rejected.
- Inside JSON strings: never use double quotes (use single quotes for any quoted phrase); keep each string on one line.
- All questions must be wholly original — do not reproduce any past-paper questions."""


# --- Stage 2: Maths optimiser (Google native sandbox) ------------------------

def optimiser_prompt(drafts: list[str], strand: str, board: str) -> str:
    """User prompt for the optimiser: solve each draft, neaten the constants,
    build the mark scheme, and emit every computed value latex-ready.

    Runs on Google's server-side code-execution sandbox — the deterministic
    oracle for the whole chain. The `values` map is the anti-hallucination
    payload the typesetter must reuse verbatim.
    """
    guards = STRAND_GUARDS.get(strand, "")
    blocks = "\n".join(f"DRAFT {i}: {d}" for i, d in enumerate(drafts))
    return f"""You are an expert {board} A-level Maths problem solver and Python programmer finalising {len(drafts)} draft question(s).

You have a built-in Python code-execution environment (Python 3 with sympy, numpy, math, fractions, statistics). Use it as a deterministic oracle:
- SOLVE each draft fully in code. If the answers or intermediate steps are 'ugly', tweak the question's constants to make the numbers neat, and re-solve until they are. Verify EVERY value beyond a one-step closed form by reading it back from code — never hand-write a number.
- VERIFY the design invariants below; if one fails, change the givens and recompute rather than fudging the answer.
- Determine the distinct logical steps for the mark scheme (one mark per distinct step; marks must equal the number of mark-scheme entries).

{guards}

DRAFTS TO FINALISE:
{blocks}

After all code execution, respond with ONLY a valid JSON object — no further code, no prose, no markdown fences:

{{
  "questions": [
    {{
      "text": "finalised question text, with the neatened constants",
      "commandWord": "string",
      "marks": number,
      "difficulty": "foundation" | "higher" | "extension",
      "markScheme": [{{"tag": "M1" | "A1" | "B1", "text": "string"}}],
      "values": {{"label": "latex-ready string", ...}}
    }}
  ]
}}

Rules:
- One question object per draft, in the same order.
- "values": every computed quantity the question and mark scheme rely on (givens, intermediate results, final answers), each as a ready-to-typeset LaTeX string (e.g. "dydx": "\\\\frac{{3}}{{2x+1}}", "area": "\\\\frac{{7}}{{3}}"). This is the source of truth the typesetter will use verbatim, so it must be complete and exact.
- commandWord: a standard {board} command word (Calculate, Show that, Prove that, Find, Determine, Sketch, State, Explain).
- marks: integer 2-6; the number of markScheme entries must equal marks.
- LaTeX: $...$ inline, $$...$$ display — KaTeX only, no \\begin{{}} environments.
- CRITICAL — JSON string escaping: every LaTeX backslash is a double backslash inside JSON strings. A single backslash before a letter is an invalid JSON escape and the whole response is rejected.
- Inside JSON strings: never use double quotes (single quotes for quoted phrases); one line per string.
- Questions must stay wholly original — do not converge on a known past-paper question."""


# --- Stage 3: Typesetter (OpenRouter, no sandbox) ----------------------------

def typesetter_prompt(questions: list[dict], board: str) -> str:
    """User prompt for the typesetter: render each finalised question + mark
    scheme as HTML, using the optimiser's exact `values` so it can't invent
    numbers.
    """
    payload = json.dumps(questions, indent=2)
    return f"""You are an expert LaTeX typesetter preparing {board} A-level Maths questions for display.

You are given finalised questions, each with a "values" map of verified, computed quantities (LaTeX strings). Your job is purely presentational: lay out the question and its mark scheme as clean HTML.

CRITICAL: every number and expression you render must come from the question's "text", "markScheme", or "values" — use the values VERBATIM. Do NOT recompute, simplify, or invent any mathematics. If a value is needed, it is in "values".

FINALISED QUESTIONS:
{payload}

Respond with ONLY a valid JSON object — no prose, no markdown fences:

{{
  "questions": [
    {{
      "question_html": "HTML for the question statement",
      "mark_scheme_html": "HTML for the mark scheme"
    }}
  ]
}}

Rules:
- One object per input question, in the same order.
- HTML only (e.g. <p>, <ol>, <li>, <strong>); put maths in KaTeX delimiters $...$ (inline) and $$...$$ (display) so it renders in the browser.
- mark_scheme_html should show each step with its mark tag (M1/A1/B1) — an <ol> or a tagged list is fine.
- CRITICAL — JSON string escaping: every LaTeX backslash is a double backslash inside JSON strings. Inside JSON strings never use double quotes (use single quotes in any HTML attribute); keep each string on one line."""


# --- Stage 4: Judge panel ----------------------------------------------------

def maths_judge_prompt(questions: list[dict], board: str) -> str:
    """User prompt for the maths judge (Google native sandbox): re-derive each
    question against its mark scheme and return a hard correct/incorrect verdict.
    """
    payload = json.dumps(questions, indent=2)
    n = len(questions)
    entries = ", ".join(
        f'{{"question_index": {i}, "maths_correct": true | false, "note": "<one sentence>"}}'
        for i in range(n)
    )
    return f"""You are a senior {board} A-level Maths examiner checking the mathematics of {n} finalised question(s).

You have a built-in Python code-execution environment (sympy, numpy, math, fractions, statistics). Use it as a deterministic oracle — do NOT do the arithmetic in your head. For each question, independently re-derive the answer in code and check it against the stated mark scheme and "values".

QUESTIONS:
{payload}

Respond with ONLY this JSON, no prose:

{{"questions": [{entries}]}}

A question is maths_correct=true only if it is well-posed, solvable, and its mark scheme / final values match your code-verified result. Set false if the answer is wrong, the question is contradictory or unsolvable, or the mark count doesn't fit the work. The "note" justifies the verdict in one sentence."""


def style_judge_prompt(questions: list[dict], board: str) -> str:
    """User prompt for the style/suitability judge (Opus): score command word,
    difficulty calibration and exam-paper style. No correctness dimension — that
    is the maths judge's gate.
    """
    payload = json.dumps(questions, indent=2)
    per_q = ",\n    ".join(
        f'{{"question_index": {i}, "scores": {{"command_word": <1-5>, "difficulty": <1-5>, '
        f'"style": <1-5>}}, "suitability_total": <sum of the three>, "flags": [], '
        f'"notes": {{"style": "<one sentence>"}}}}'
        for i in range(len(questions))
    )
    return f"""You are a senior {board} A-level Maths examiner judging the SUITABILITY of {len(questions)} question(s) for a real exam paper. Assume the mathematics has already been verified — judge only presentation and pitch.

QUESTIONS:
{payload}

Score EACH question independently on three dimensions (1-5 each). Judge every question on its own merits. Return ONLY this JSON, no prose:

{{
  "questions": [
    {per_q}
  ]
}}

Scoring (apply to each question on its own):
- command_word: Does {board} use this command word this way?
  1=wrong word  3=acceptable  5=exact {board} convention
- difficulty: Does cognitive demand match the marks? (2 marks=recall, 6 marks=multi-step)
  1=miscalibrated  3=acceptable  5=precisely matched
- style: Reads like a real {board} paper? Formal register, no AI phrasing?
  1=clearly AI  3=acceptable  5=indistinguishable from a real paper

suitability_total: the sum of the three dimension scores (max 15).
flags (per question): include "past_paper_suspected" if it closely resembles a known exam question; "ambiguous_question" if the wording is underspecified."""


# --- Shared JSON repair ------------------------------------------------------

def cleaner_prompt_for(raw: str, shape: str) -> str:
    """Syntax-only repair prompt for a malformed stage JSON response.

    `shape` describes the expected top-level structure (e.g. 'with a top-level
    "questions" array') so the same repairer serves every stage. Pair it with a
    stage-specific shape string via functools.partial when calling clean_json.
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
