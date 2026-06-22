"""Draft -> Refine A/B: model-authored LaTeX vs host-rendered SymPy.

One drafter (Haiku) brainstorms N draft questions; each draft is then finalised
by TWO refiner arms (both Haiku, both given the run_python oracle so the only
difference is *who authors the LaTeX*):

  string : the classic arm — the refiner writes LaTeX into a JSON string itself
           ({question_latex, markscheme_latex, ...}).
  sympy  : the refiner emits a SymPy *program* (TOPIC/DIFFICULTY/QUESTION/
           MARKSCHEME segment lists); sympy_render.render_from_sympy appends a
           host epilogue that calls sympy.latex() — the model never writes LaTeX
           or JSON escapes, and answer values are rendered from the same objects
           the program asserts on.

Opus judges every produced question (correctness + quality, 1-10). The report
compares the two arms on pass rate, avg scores, refiner token spend, and the
failure modes that motivate the sympy arm (parse failures, escape bugs).

Run (from evals/):
    uv run python draft_refine_sympy.py                 # N=3, both arms
    uv run python draft_refine_sympy.py --n 6
    uv run python draft_refine_sympy.py --arms sympy    # one arm only
"""
import argparse
import json
import os
import re
import time
from datetime import datetime, timezone

import litellm
from dotenv import load_dotenv

import config
from parse_utils import parse_json_robust
from sandbox import RUN_PYTHON_TOOL, format_tool_result, run_python
from sympy_render import render_from_sympy

DRAFTER_MODEL = {"id": "openrouter/anthropic/claude-haiku-4-5", "short": "haiku-drafter"}
REFINER_MODEL = {"id": "openrouter/anthropic/claude-haiku-4-5", "short": "haiku-refiner"}
JUDGE_MODEL = config.JUDGE_MODEL  # Opus

REFINER_REASONING = {"effort": "low"}
DRAFTER_REASONING = {"effort": "low"}

PASS_CORRECTNESS = 8  # a question "passes" at >=8/10 correctness and >=6/10 quality
PASS_QUALITY = 6


# --------------------------------------------------------------------------- #
# LLM plumbing (sync; mirrors config.request_kwargs / response_cost routing)
# --------------------------------------------------------------------------- #
class Spend:
    def __init__(self):
        self.prompt = 0
        self.completion = 0
        self.cost = 0.0
        self.calls = 0

    def add(self, model, resp):
        self.calls += 1
        usage = getattr(resp, "usage", None)
        if usage:
            self.prompt += getattr(usage, "prompt_tokens", 0) or 0
            self.completion += getattr(usage, "completion_tokens", 0) or 0
        self.cost += config.response_cost(model, resp)

    def as_dict(self):
        return {
            "prompt_tokens": self.prompt,
            "completion_tokens": self.completion,
            "total_tokens": self.prompt + self.completion,
            "cost_usd": round(self.cost, 6),
            "calls": self.calls,
        }


def complete(model, messages, reasoning, spend, *, tools=None):
    kwargs = dict(
        model=model["id"],
        messages=messages,
        **config.request_kwargs(model, reasoning),
    )
    if tools:
        kwargs["tools"] = tools
    resp = litellm.completion(**kwargs)
    spend.add(model, resp)
    return resp.choices[0].message


# --------------------------------------------------------------------------- #
# Stage 1: drafter
# --------------------------------------------------------------------------- #
DRAFTER_SYSTEM = """You are a creative A-Level Mathematics examiner. Brainstorm highly original, non-trivial exam questions, each combining a different pair of syllabus topics (e.g. calculus+trigonometry, mechanics+vectors, probability+series). Do not worry about neat numbers yet. Do NOT solve them or write mark schemes.

CONSTRAINTS: original inventions only — never reproduce or paraphrase a real AQA/Edexcel/OCR past-paper question. Target Edexcel A-Level Maths.

Output ONLY a JSON object in a ```json block: {"drafts": ["full text of question 1", "full text of question 2", ...]}. Each array element is one self-contained draft question as a plain string. No markdown headers, no numbering, no solutions."""


def draft_questions(n, spend):
    user = f"Produce exactly {n} DISTINCT draft questions as the JSON object described."
    msg = complete(
        DRAFTER_MODEL,
        [{"role": "system", "content": DRAFTER_SYSTEM}, {"role": "user", "content": user}],
        DRAFTER_REASONING,
        spend,
    )
    parsed, err = parse_json_robust(msg.content or "")
    drafts = (parsed or {}).get("drafts") if isinstance(parsed, dict) else None
    if not drafts:
        raise SystemExit(f"drafter produced no parseable drafts (err={err}): {(msg.content or '')[:300]!r}")
    return [str(d).strip() for d in drafts[:n] if str(d).strip()]


# --------------------------------------------------------------------------- #
# Stage 2a: STRING refiner (model authors the LaTeX)
# --------------------------------------------------------------------------- #
STRING_SYSTEM = r"""You are an expert A-Level Mathematics examiner and Python programmer finalising a draft exam question into a flawless question + mark scheme.

Use the run_python tool as a DETERMINISTIC ORACLE: solve the draft, compute every non-trivial value, and assert your design invariants. If answers are ugly (nasty decimals, un-factorable quadratics, no closed form), tweak the givens and recompute until the numbers are neat (integers, simple fractions, clean surds, multiples of pi). A part with no closed-form answer must be reworked until it has one.

When done, output ONLY a JSON object in a ```json block:
{
  "topic": "...",
  "difficulty": "higher" | "extension",
  "question_latex": "the finalised question in LaTeX",
  "markscheme_latex": "the step-by-step solution in LaTeX",
  "verification_summary": "what you tweaked and the verified final answers"
}
CRITICAL JSON/LaTeX escaping: every LaTeX backslash must be a DOUBLE backslash inside the JSON string, e.g. "Find $\\frac{dy}{dx}$"."""


def refine_string(draft, spend):
    messages = [
        {"role": "system", "content": STRING_SYSTEM},
        {"role": "user", "content": f"DRAFT QUESTION:\n{draft}"},
    ]
    content = ""
    for i in range(config.CODE_LOOP_MAX_ITERS):
        tools = None if i == config.CODE_LOOP_MAX_ITERS - 1 else [RUN_PYTHON_TOOL]
        msg = complete(REFINER_MODEL, messages, REFINER_REASONING, spend, tools=tools)
        if getattr(msg, "tool_calls", None):
            messages.append(msg.model_dump())
            for tc in msg.tool_calls:
                try:
                    code = json.loads(tc.function.arguments).get("code", "")
                except (json.JSONDecodeError, AttributeError):
                    code = ""
                out = format_tool_result(run_python(code))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
            continue
        content = msg.content or ""
        break
    payload, err = parse_json_robust(content)
    if payload is None:
        snippet = (content or "<blank>")[:400].replace("\n", "\\n")
        return None, f"json parse failed: {err} | final_content[:400]={snippet!r}"
    return payload, None


# --------------------------------------------------------------------------- #
# Stage 2b: SYMPY refiner (model emits a program; host renders the LaTeX)
# --------------------------------------------------------------------------- #
SYMPY_SYSTEM = r"""You are an expert A-Level Mathematics examiner and Python programmer. Finalise the draft into a flawless question + mark scheme by writing a SINGLE Python program using SymPy. You do NOT write any LaTeX or JSON — a host renderer turns your SymPy objects into LaTeX.

Your program MUST define these module-level names:
  TOPIC         : str
  DIFFICULTY    : "higher" or "extension"
  QUESTION      : list of segments for the question
  MARKSCHEME    : list of segments for the worked solution
  VERIFICATION  : str — what you tweaked and the verified final answers

A *segment* is one of:
  ("text", "literal prose")        # plain words, NO math, NO LaTeX
  ("math", <sympy expression>)     # rendered inline as $...$ by the host
  ("display", <sympy expression>)  # rendered as a display block $$...$$
A bare "string" is shorthand for ("text", "string").

RULES:
1. Build every mathematical object as a SymPy expression (sympy as sp). Never put math inside a "text" segment and never type LaTeX yourself.
2. EVERY numeric answer the question asks for must appear as a ("math", expr) segment whose expr is the value your program COMPUTED — never a hand-typed number. This is enforced: the rendered answer is whatever SymPy computed.
3. Verify with SymPy and ASSERT your invariants (e.g. assert sp.simplify(area) == expected). If a result is ugly or has no closed form, tweak the givens in code and recompute until neat. A failing assert is correct behaviour for a broken question — fix the givens.
4. Do NOT print anything or call sympy.latex yourself; the host appends the rendering + printing.

Output ONLY one ```python code block — the program. No prose before or after."""


def _extract_python(text):
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def refine_sympy(draft, spend):
    messages = [
        {"role": "system", "content": SYMPY_SYSTEM},
        {"role": "user", "content": f"DRAFT QUESTION:\n{draft}"},
    ]
    last_err = None
    for attempt in range(2):  # one retry, feeding the sandbox error back
        msg = complete(REFINER_MODEL, messages, REFINER_REASONING, spend)
        program = _extract_python(msg.content or "")
        payload, err = render_from_sympy(program)
        if payload is not None:
            return payload, None
        last_err = err
        messages.append({"role": "assistant", "content": msg.content or ""})
        messages.append({
            "role": "user",
            "content": f"Your program failed when executed:\n{err}\nFix it and output the corrected program as one ```python block.",
        })
    return None, f"sympy program failed: {last_err}"


# --------------------------------------------------------------------------- #
# Stage 3: judge (Opus)
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = """You are a harsh, specific A-Level Mathematics examiner judging a generated question. Independently check the mathematics. Score two axes 1-10:
- correctness: math flawless? mark scheme correctly & completely solves the stated question? answers are exact closed forms (penalise no-closed-form "find the value" parts and any answer that contradicts the working)?
- quality: original, non-trivial, appropriately difficult for Edexcel A-level, reads like a real exam (not AI-generated), command words appropriate.
Output ONLY a JSON object: {"correctness": <1-10>, "quality": <1-10>, "comments": "<specific justification>"}"""


def judge_question(payload, spend):
    user = "Question (JSON):\n" + json.dumps(
        {k: payload.get(k) for k in ("topic", "difficulty", "question_latex", "markscheme_latex")},
        indent=2,
    )
    msg = complete(
        JUDGE_MODEL,
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        config.JUDGE_REASONING,
        spend,
    )
    parsed, _ = parse_json_robust(msg.content or "")
    if not isinstance(parsed, dict):
        return {"correctness": 0, "quality": 0, "comments": "judge parse failure"}
    return parsed


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
ARMS = {"string": refine_string, "sympy": refine_sympy}


def run_arm(name, drafts, spend):
    refine = ARMS[name]
    items = []
    for i, draft in enumerate(drafts):
        print(f"  [{name}] refining draft {i + 1}/{len(drafts)} ...", flush=True)
        try:
            payload, err = refine(draft, spend)
        except Exception as e:
            payload, err = None, f"{type(e).__name__}: {e}"
        items.append({"draft_index": i, "draft": draft, "payload": payload, "error": err})
        time.sleep(0.3)
    return items


def summarise(name, items, refiner_spend, judge_spend):
    produced = [it for it in items if it["payload"]]
    failures = [it for it in items if not it["payload"]]
    scored = [it for it in produced if "judge" in it]
    corr = [it["judge"].get("correctness", 0) for it in scored]
    qual = [it["judge"].get("quality", 0) for it in scored]
    passes = sum(
        1 for it in scored
        if it["judge"].get("correctness", 0) >= PASS_CORRECTNESS
        and it["judge"].get("quality", 0) >= PASS_QUALITY
    )
    n = len(items)
    return {
        "arm": name,
        "n_drafts": n,
        "produced": len(produced),
        "refiner_failures": len(failures),
        "avg_correctness": round(sum(corr) / len(corr), 2) if corr else None,
        "avg_quality": round(sum(qual) / len(qual), 2) if qual else None,
        "passes": passes,
        "pass_rate": round(passes / n, 2) if n else None,
        "refiner_spend": refiner_spend.as_dict(),
        "judge_spend": judge_spend.as_dict(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="number of draft questions")
    ap.add_argument("--arms", default="string,sympy", help="comma list: string,sympy")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    load_dotenv()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set (evals/.env)")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    run_id = args.run_id or datetime.now(timezone.utc).strftime("dr-sympy-%Y%m%d-%H%M%S")

    draft_spend = Spend()
    print(f"=== Draft-Refine SymPy A/B (run {run_id}) ===")
    print(f"Drafting {args.n} questions (Haiku) ...", flush=True)
    drafts = draft_questions(args.n, draft_spend)
    print(f"  got {len(drafts)} drafts")

    summaries = {}
    details = {}
    for name in arms:
        print(f"\n--- arm: {name} ---")
        refiner_spend = Spend()
        judge_spend = Spend()
        items = run_arm(name, drafts, refiner_spend)
        for it in items:
            if it["payload"]:
                print(f"  [{name}] judging draft {it['draft_index'] + 1} ...", flush=True)
                it["judge"] = judge_question(it["payload"], judge_spend)
        summaries[name] = summarise(name, items, refiner_spend, judge_spend)
        details[name] = items

    report = {
        "run_id": run_id,
        "n": args.n,
        "drafter_spend": draft_spend.as_dict(),
        "drafts": drafts,
        "summaries": summaries,
        "details": details,
    }
    out = config.results_path(run_id, "draft_refine_sympy", "json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Console comparison table.
    print("\n=== A/B SUMMARY ===")
    cols = ["arm", "produced", "refiner_failures", "avg_correctness", "avg_quality",
            "passes", "pass_rate"]
    print(" | ".join(c.ljust(16) for c in cols))
    for name in arms:
        s = summaries[name]
        print(" | ".join(str(s.get(c)).ljust(16) for c in cols))
    print("\nRefiner spend (the metric that matters for tokens):")
    for name in arms:
        rs = summaries[name]["refiner_spend"]
        print(f"  {name:7s}  completion_tokens={rs['completion_tokens']:6d}  "
              f"total_tokens={rs['total_tokens']:6d}  cost=${rs['cost_usd']:.4f}  "
              f"calls={rs['calls']}")
    print(f"\nFull report: {out}")


if __name__ == "__main__":
    main()
