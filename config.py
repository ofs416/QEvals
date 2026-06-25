import os

import litellm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# The ranked generator candidates — the DRAFTER stage. Each drafts question
# scenarios only (no mark scheme, rough numbers); the maths optimiser owns the
# arithmetic, so a generator no longer runs code and carries no native_code_exec.
# Drafting is the cheap, creative half of the chain, so the candidates are
# cheap models (Gemini Flash Lite + Chinese flash tiers). A model may carry its
# own "reasoning" override; absent that, GENERATOR_REASONING applies
# ("reasoning": None opts a model out of the param entirely).
MODELS = [
    {"id": "gemini/gemini-3.1-flash-lite", "short": "gemini-lite"},
    {"id": "openrouter/qwen/qwen3.6-flash", "short": "qwen-flash"},
    {"id": "openrouter/deepseek/deepseek-v4-flash", "short": "deepseek"},
]

# --- Fixed helper stages (shared across every candidate, not ranked) ---------

# Maths optimiser: solves each draft, neatens the constants, builds the mark
# scheme, and emits every computed value latex-ready. Runs direct on Google AI
# Studio with the server-side code-execution sandbox (native_code_exec) — the
# deterministic oracle for the whole chain. Needs GEMINI_API_KEY.
OPTIMISER_MODEL = {
    "id": "gemini/gemini-3.5-flash",
    "short": "gemini-opt",
    "native_code_exec": True,
}

# Typesetter: turns the optimiser's finalized maths + values into HTML for the
# question and mark scheme. No sandbox — it only formats, reusing the optimiser's
# exact values so it can't hallucinate numbers. A cheap formatter is plenty.
TYPESETTER_MODEL = {"id": "openrouter/z-ai/glm-4.7-flash", "short": "typeset"}

# Maths judge: re-verifies each finalized question against its mark scheme using
# Google's sandbox, returning a hard maths-correct verdict per question.
MATHS_JUDGE_MODEL = {
    "id": "gemini/gemini-3.5-flash",
    "short": "gemini-judge",
    "native_code_exec": True,
}

# Repairs malformed JSON from any stage (syntax only, never content). Cheap and
# shared across all stages so it doesn't bias the comparison.
CLEANER_MODEL = {
    "id": "openrouter/anthropic/claude-haiku-4-5",
    "short": "haiku-cleaner",
}

# Cross-provider backup for the Haiku cleaner. The primary route (Anthropic/Haiku
# via OpenRouter) intermittently drops connections instantly (Timeout with
# time-taken 0s) or returns blank finish_reason 'error' responses that exhaust
# all retries — a longer backoff doesn't help an instant-fail, so the fix is to
# fall through to a capable cheap model on a *different* provider rather than give
# up. gpt-5.4-nano is OpenAI, a different upstream than Anthropic, and competent
# at structured-JSON tasks.
CLEANER_FALLBACK_MODEL = {
    "id": "openrouter/openai/gpt-5.4-nano",
    "short": "gpt-nano-cleaner",
}

# Style/suitability judge: confirms command word, difficulty calibration and
# exam-paper style. The maths is the maths judge's job, so this judge scores no
# correctness dimension. Claude Opus.
STYLE_JUDGE_MODEL = {"id": "openrouter/anthropic/claude-opus-4-6", "short": "opus46"}

# How many questions the drafter produces per (topic, board), carried through
# every downstream stage. A knob, not a constant, so 1-vs-3 can be A/B'd for
# quality (run.py --questions-per-topic overrides it). Nothing downstream
# hardcodes a count — stages iterate over however many questions exist.
QUESTIONS_PER_TOPIC = 3

TOPICS = [
    "Differentiation",
    "Integration",
    "Trigonometry",
    "Sequences & Series",
    "Exponentials & Logarithms",
    "Vectors",
    "Proof",
    "Algebra & Functions",
    "Statistical Distributions",
    "Hypothesis Testing",
    "Probability",
    "Kinematics",
    "Forces & Newton's Laws",
    "Moments",
]

# Edexcel only: board style differences don't change the model ranking enough
# to justify doubling the run cost.
BOARDS = ["Edexcel"]

# A topic is routed to one of three strands (Pure / Mechanics / Statistics) on
# its ASSESSED OBJECTIVE, not the technique it happens to use — Vectors is Pure
# tooling Mechanics borrows, an E(X) integral is Statistics even though it
# integrates. The strand selects the design guards (see prompts.STRAND_GUARDS)
# fed to the drafter (design intent) and optimiser (what to verify/preserve).
# Every TOPICS entry must appear here; strand_for_topic raises on a gap so a new
# topic can't silently fall through to the wrong strand.
STRANDS = ("pure", "mechanics", "statistics")

TOPIC_STRANDS = {
    "Differentiation": "pure",
    "Integration": "pure",
    "Trigonometry": "pure",
    "Sequences & Series": "pure",
    "Exponentials & Logarithms": "pure",
    "Vectors": "pure",
    "Proof": "pure",
    "Algebra & Functions": "pure",
    "Statistical Distributions": "statistics",
    "Hypothesis Testing": "statistics",
    "Probability": "statistics",
    "Kinematics": "mechanics",
    "Forces & Newton's Laws": "mechanics",
    "Moments": "mechanics",
}

# A question passes the panel only if the maths judge confirms it correct AND
# its Opus suitability total (command_word + difficulty + style, 3-15) meets
# this. The maths gate is hard pass/fail; this is the quality gate.
SUITABILITY_THRESHOLD = 11


# Request body addition that makes OpenRouter return the actual billed cost
# in the response's usage object (usage.cost).
USAGE_INCLUDE = {"usage": {"include": True}}

# Drafter chain-of-thought budget. The drafter only brainstorms scenarios (no
# maths), so a low cap keeps the cost comparison fair across candidates. "effort"
# is the portable knob OpenRouter normalises across families; per-model
# "reasoning" overrides (or None to opt out) live on the MODELS entries.
GENERATOR_REASONING = {"effort": "low"}

# The optimiser does the heavy maths in Google's sandbox — give it room to
# reason about which constants make the numbers neat.
OPTIMISER_REASONING = {"effort": "medium"}

# The typesetter only formats given values into HTML; no reasoning needed.
TYPESETTER_REASONING = None

# Both judges run at high effort — scoring quality gates every downstream metric
# and judge cost is shared eval overhead, excluded from per-candidate Cost/pass Q.
JUDGE_REASONING = {"effort": "high"}


# --- Routing: OpenRouter vs direct provider --------------------------------
# litellm dispatches on the id prefix: `openrouter/...` goes through OpenRouter,
# `gemini/...` calls Google AI Studio directly (needs GEMINI_API_KEY). The two
# routes differ in how requests are built and how cost is recovered, so the
# request/cost plumbing keys off route_of() rather than hard-coding OpenRouter.
# Adding a model on a new route is just a matter of its id prefix; OpenRouter
# stays fully wired for every other (and future) candidate.


def route_of(model: dict) -> str:
    """Return how this model's requests are dispatched: 'openrouter' or 'gemini'.

    Derived from the litellm id prefix (litellm itself dispatches on it), with an
    optional explicit ``model['route']`` override. Anything not recognised as a
    direct provider defaults to 'openrouter'.
    """
    if "route" in model:
        return model["route"]
    if model["id"].startswith("gemini/"):
        return "gemini"
    return "openrouter"


def _reasoning_effort(reasoning) -> str | None:
    """Map a reasoning config to litellm's portable ``reasoning_effort`` for
    direct providers. Returns None to omit the param (a non-reasoning model, or a
    config that doesn't express an effort level — e.g. a hard token budget, which
    no direct-route model currently uses)."""
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        return reasoning["effort"]
    return None


def request_kwargs(model: dict, reasoning) -> dict:
    """Per-route request kwargs (extra_body + reasoning param) for a completion.

    OpenRouter takes USAGE_INCLUDE (so it reports the billed cost), an optional
    provider routing block, and the portable ``reasoning`` knob — all in
    extra_body. The direct Gemini route takes none of those: cost is computed
    locally from litellm's price table, and chain-of-thought is capped via the
    top-level ``reasoning_effort`` param litellm maps to Gemini's thinkingConfig.
    """
    if route_of(model) == "gemini":
        effort = _reasoning_effort(reasoning)
        return {"reasoning_effort": effort} if effort is not None else {}

    extra_body = dict(USAGE_INCLUDE)
    if reasoning is not None:
        extra_body["reasoning"] = reasoning
    if model.get("provider"):
        extra_body["provider"] = model["provider"]
    return {"extra_body": extra_body}


def response_cost(model: dict, response) -> float:
    """Cost of this request in USD, recovered per route.

    OpenRouter reports the actual billed cost in ``usage.cost`` (enabled by
    USAGE_INCLUDE). Direct providers don't, so the cost is computed from
    litellm's local price table. Either way, a missing/unusable cost yields a
    loud 0.0 rather than a silent wrong number.
    """
    if route_of(model) != "openrouter":
        try:
            cost = litellm.completion_cost(
                completion_response=response, model=model["id"]
            )
        except Exception as e:
            print(
                f"[cost] litellm.completion_cost failed for {model['short']}: {e} -- recording $0"
            )
            return 0.0
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            return float(cost)
        print(f"[cost] no completion_cost for {model['short']} -- recording $0")
        return 0.0

    cost = getattr(response.usage, "cost", None)
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        return float(cost)
    print(f"[cost] no OpenRouter cost in response for {model['short']} -- recording $0")
    return 0.0


def results_path(run_id: str, stage: str, ext: str = "jsonl") -> str:
    """Per-run output path, foldered by output type.

    Each stage writes into its own subdirectory (results/<stage>/<run_id>.<ext>)
    so the run artefacts are grouped by kind rather than interleaved in one flat
    directory. Report writers pass stage="html"/"markdown" to split the two
    rendered formats into their own folders.
    """
    stage_dir = os.path.join(RESULTS_DIR, stage)
    os.makedirs(stage_dir, exist_ok=True)
    return os.path.join(stage_dir, f"{run_id}.{ext}")


def strand_for_topic(topic: str) -> str:
    """Route a topic to its generator specialist (pure/mechanics/statistics).

    Raises on an unmapped topic rather than guessing — a new topic must be
    assigned a strand in TOPIC_STRANDS explicitly.
    """
    try:
        return TOPIC_STRANDS[topic]
    except KeyError:
        raise ValueError(
            f"No strand mapping for topic {topic!r}. Add it to TOPIC_STRANDS "
            f"(valid strands: {STRANDS})."
        )


def model_by_short(short: str) -> dict:
    for m in MODELS:
        if m["short"] == short:
            return m
    raise ValueError(
        f"Unknown model short name: {short!r}. Valid: {[m['short'] for m in MODELS]}"
    )
