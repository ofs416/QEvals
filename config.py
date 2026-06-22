import os
import shlex

import litellm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# A generator may carry its own "reasoning" override; absent that, the global
# GENERATOR_REASONING default applies. "reasoning": None opts a model out
# entirely (for non-reasoning models whose endpoints reject the param).
MODELS = [
    {"id": "openrouter/openai/gpt-5.4-nano", "short": "gpt-nano"},
    # Gemini goes direct to Google AI Studio (needs GEMINI_API_KEY) — the chosen
    # production provider. Going direct also makes the old OpenRouter
    # google-ai-studio provider pin (added because the default route returned
    # blank finish_reason 'error' responses under load) obsolete: we ARE
    # google-ai-studio now. Cost is computed from litellm's price table.
    {"id": "gemini/gemini-2.5-flash", "short": "gemini-flash"},
    {"id": "gemini/gemini-3.1-flash-lite", "short": "gemini-lite"},
    # Experiment arm (2026-06): identical model, but uses Gemini's server-side
    # code_execution sandbox instead of the harness-local run_python loop.
    # A/B against gemini-lite for the native-code-exec quality/price experiment
    # (docs/superpowers/specs/2026-06-16-gemini-native-codeexec-experiment-design.md).
    {
        "id": "gemini/gemini-3.1-flash-lite",
        "short": "gemini-lite-native",
        "native_code_exec": True,
    },
    {"id": "openrouter/mistralai/mistral-small-2603", "short": "mistral-small4"},
    {"id": "openrouter/qwen/qwen3.6-flash", "short": "qwen-flash"},
    {"id": "openrouter/qwen/qwen3.7-plus", "short": "qwen-plus"},
    {"id": "openrouter/deepseek/deepseek-v4-flash", "short": "deepseek"},
    {"id": "openrouter/z-ai/glm-4.7-flash", "short": "GLMflash"},
    {"id": "openrouter/xiaomi/mimo-v2.5", "short": "mimo"},
    # kimi thinks by default and inverts the effort knob (probe: effort=low
    # produced MORE reasoning tokens than default, and a reasoning max_tokens
    # budget is only loosely respected). {"enabled": False} is the only control
    # that works — cuts ~6k tokens/generation to a few hundred. Note this is
    # different from "reasoning": None, which would omit the param and leave
    # kimi's default thinking on.
    {
        "id": "openrouter/moonshotai/kimi-k2.5",
        "short": "kimi25",
        "reasoning": {"max_tokens": 4096},
    },
    {"id": "openrouter/x-ai/grok-4.3", "short": "grok43"},
    {
        "id": "openrouter/minimax/minimax-m3",
        "short": "minimax",
        "reasoning": {"max_tokens": 4096},
    },
    # Solver-tier models added as generator candidates (2026-06): tests whether
    # solving-grade strength buys enough generation quality (correctness +
    # mark_scheme) to justify the higher cost against the cheap tier on
    # Cost/pass Q. grok-4.3 already appears above and in SOLVER_MODELS.
    {"id": "openrouter/openai/gpt-5.5", "short": "gpt55"},
    {"id": "gemini/gemini-3.5-flash", "short": "gemini-35-flash"},
]

# Four solvers from different families: agreement between independent solvers
# (vs the mark scheme) distinguishes "question is wrong" from "solver failed".
# A solver may carry its own "reasoning" override (see SOLVER_REASONING); absent
# that, the global default applies.
SOLVER_MODELS = [
    # Replaced claude-opus-4.8 (2026-06): ~40% of solver spend for a vote the
    # judge was told to discount (over-reject profile), and the Anthropic
    # perspective is already represented by the judge. grok-4.3 adds a family
    # not present anywhere else in the pipeline at ~6x lower cost; probe run
    # (_probe_grok.py) caught both known scheme_wrong cases without
    # false-flagging the control.
    {"id": "openrouter/x-ai/grok-4.3", "short": "grok43"},
    {"id": "openrouter/openai/gpt-5.5", "short": "gpt55"},
    {"id": "gemini/gemini-3.5-flash", "short": "gemini-35-flash"},
    # Fourth family (2026-06): replaced minimax-m3, then opus-4.6, both of which
    # hit the OpenRouter instant-connection-failure / blank finish_reason 'error'
    # pattern under load (opus especially, being the heaviest model with the
    # tightest capacity — and it duplicated JUDGE_MODEL, so its vote wasn't
    # independent). qwen3.7-plus is a solving-grade reasoning model from a family
    # (Alibaba) present nowhere else in the pipeline, reliable on OpenRouter, and
    # far cheaper than opus.
    # {"id": "openrouter/qwen/qwen3.7-plus", "short": "qwen-plus"},
    # Temporarily disabled (2026-06): the deepseek-v4-pro endpoint on
    # OpenRouter was returning instant connection failures and blank
    # completions (finish_reason 'error'), exhausting retries on every call.
    # Re-enable once the provider settles. When re-enabled, note it ignores
    # "effort" (reasons regardless), so the global effort cap can't bound it —
    # the hard reasoning max_tokens budget below is the only knob that keeps it
    # from blowing the solve max_tokens on chain-of-thought.
    # {
    #     "id": "openrouter/deepseek/deepseek-v4-pro",
    #     "short": "deepseek-pro",
    #     "reasoning": {"max_tokens": 2048},
    # },
]

# Known error profile per solver, fed to the judge so it does not weigh the
# votes equally. Derived from agreement analysis of past runs; update when the
# solver line-up or observed behaviour changes.
SOLVER_PROFILES = {
    "grok43": "newly added; error profile not yet characterised — treat its vote with mild caution",
    "gpt55": "balanced error profile",
    "gemini-35-flash": "credulous; known to over-confirm (may agree with a flawed mark scheme)",
    "qwen-plus": "newly added; error profile not yet characterised — treat its vote with mild caution",
    "deepseek-pro": "reasoning model, newly added; error profile not yet characterised — treat its vote with mild caution",
}

# Repairs malformed JSON from any generator OR solver (syntax only, never
# content). Cheap and shared across all models so it doesn't bias the comparison.
CLEANER_MODEL = {
    "id": "openrouter/anthropic/claude-haiku-4-5",
    "short": "haiku-cleaner",
}

# Cross-provider backup for the Haiku cleaner/gate calls. The primary route
# (Anthropic/Haiku via OpenRouter) intermittently drops connections instantly
# (Timeout with time-taken 0s) or returns blank finish_reason 'error' responses
# that exhaust all retries — a longer backoff doesn't help an instant-fail, so
# the fix is to fall through to a capable cheap model on a *different* provider
# rather than give up (cleaner) or fail open (gate). gpt-5.4-nano is OpenAI, a
# different upstream than Anthropic, and competent at structured-JSON tasks.
CLEANER_FALLBACK_MODEL = {
    "id": "openrouter/openai/gpt-5.4-nano",
    "short": "gpt-nano-cleaner",
}

JUDGE_MODEL = {"id": "openrouter/anthropic/claude-opus-4-6", "short": "opus46"}

# Cheap equivalence check that gates the reconciliation pass: the full
# (expensive) solver reconcile call only fires for answers the gate says don't
# match the mark scheme. Shared across all solvers so it doesn't bias the
# comparison; instructed to fail open (unsure -> mismatch -> reconcile anyway).
GATE_MODEL = {"id": "openrouter/anthropic/claude-haiku-4-5", "short": "haiku-gate"}

# Same cross-provider backup idea for the gate (see CLEANER_FALLBACK_MODEL): on a
# primary-route outage the gate tries this before falling open, so a transient
# Haiku blip doesn't push every answer into the expensive reconcile call.
GATE_FALLBACK_MODEL = {
    "id": "openrouter/openai/gpt-5.4-nano",
    "short": "gpt-nano-gate",
}

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

# The generator is split into three strand specialists (Pure / Mechanics /
# Statistics), each carrying its own design guards (see prompts.STRAND_GUARDS).
# A topic is routed to one specialist on its ASSESSED OBJECTIVE, not the
# technique it happens to use — Vectors is Pure tooling Mechanics borrows, an
# E(X) integral is Statistics even though it integrates. Every TOPICS entry must
# appear here; strand_for_topic raises on a gap so a new topic can't silently
# fall through to the wrong specialist.
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

# A generation passes when the judge total meets this. Shared by the judge
# stage (scheme_wrong consensus hard-fail) and the report.
PASS_THRESHOLD = 18


# Request body addition that makes OpenRouter return the actual billed cost
# in the response's usage object (usage.cost).
USAGE_INCLUDE = {"usage": {"include": True}}

# Solver chain-of-thought budget. "effort" is the portable knob OpenRouter
# normalises across the Anthropic / OpenAI / Gemini families; solvers that ignore
# it (e.g. deepseek-v4-pro) override with a hard token budget on their own dict.
# Set to "high" for stronger verification — but note high effort means a long
# reasoning trace, which previously truncated the JSON answer at the old 8192
# max_tokens (gemini-3.5-flash burned ~8188 thinking tokens and emitted nothing
# parseable). The solve/reconcile calls now run at 16384 max_tokens to give that
# trace headroom; keep them in sync if you raise this further.
SOLVER_REASONING = {"effort": "medium"}

# Same idea for the generators under test: cap chain-of-thought so reasoning
# models (minimax-m3 was burning >20k thinking tokens per generation) don't
# dominate the cost comparison. Applied uniformly so no candidate gets an
# unfair thinking budget; per-model "reasoning" overrides (or None to opt
# out) live on the MODELS entries.
GENERATOR_REASONING = {"effort": "medium"}

# The judge scores every question on five dimensions in one batched call, so it
# runs at "high" — stronger reasoning here is worth the spend (judge cost is
# shared eval overhead, excluded from the per-candidate Cost/pass Q) and the
# scoring quality gates every downstream metric. Kept above medium deliberately,
# unlike the generator/solver knobs.
JUDGE_REASONING = {"effort": "high"}

# --- Code-in-loop generation (the run_python tool) -------------------------
# Each specialist gets a Python tool during generation so it computes and
# verifies every non-trivial value (binomial/normal/decimal/multi-step) and
# asserts its design invariants against a deterministic oracle instead of
# guessing arithmetic. This is NOT in-generator self-verification (an LLM
# reasoning about its own correctness, which past runs found harmful) — a Python
# interpreter is a deterministic oracle, the correct thing to put in the loop.

# Command used to execute generator-emitted Python. Defaults to
# `uv run --with sympy python -` so sympy/numpy are importable regardless of
# which interpreter runs the harness (uv caches the resolved env after the first
# call). Override with EVAL_SANDBOX_CMD (a shell-quoted command) to point at a
# prepared interpreter, e.g. "python" when sympy is already installed.
SANDBOX_CMD = shlex.split(
    os.environ.get("EVAL_SANDBOX_CMD", "uv run --with sympy python -")
)

# Max generator<->python rounds per generation before tools are removed and a
# final answer is forced. Bounds cost and stops a runaway tool loop.
CODE_LOOP_MAX_ITERS = 8

# Per-call wall-clock cap (s) and stdout/stderr cap (chars) for sandboxed code.
SANDBOX_TIMEOUT = 30
SANDBOX_OUTPUT_LIMIT = 4000


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
