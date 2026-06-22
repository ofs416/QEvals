# evals/CLAUDE.md

LLM evaluation pipeline for comparing question-generation models. See the root [`CLAUDE.md`](../CLAUDE.md) for project-wide context.

## Purpose

Benchmarks candidate LLMs on their ability to generate well-formed AQA/Edexcel A-level Maths questions. Outputs a scored HTML/Markdown report used to decide which model(s) to use in the Pergrad production pipeline.

## Running

```
cd evals
python run.py                          # full pipeline (all models, all topics)
python run.py --models gemini-flash,gpt-nano   # subset of models
python run.py --skip-generate          # load existing generations, re-solve + re-judge
python run.py --skip-solve             # load existing generations + solutions, re-judge
python run.py --run-id my-run-id       # use a fixed run ID instead of a timestamp
python run.py --models gpt55 --compare-prompts baseline,no_guards  # A/B two generator prompts on one model
```

Requires API keys in `evals/.env`: `OPENROUTER_API_KEY` for OpenRouter-routed models, and `GEMINI_API_KEY` for the direct-Gemini models (see **Routing** below). `litellm` dispatches each call on its model-id prefix.

**Prompt A/B (`--compare-prompts`):** holds one generator model fixed (`--models` must resolve to exactly one) and runs every named variant from `prompts.PROMPT_VARIANTS` across all topics/boards. Generations are tagged with `prompt_variant`, and the report groups by variant instead of model — the summary table, topic matrix, and viewer tabs show one entry per prompt. Add a variant by writing a `(strand, board) -> str` builder and registering it in `PROMPT_VARIANTS` (baseline = the current `specialist_system_prompt`; `no_guards` drops the strand design guards as a worked example).

## Pipeline Stages

| Stage | Script | Output file |
|-------|--------|-------------|
| 1. Generate | `generate.py` | `results/generations/<run_id>.jsonl` |
| 2. Solve | `solve.py` | `results/solutions/<run_id>.jsonl` |
| 3. Judge | `judge.py` | `results/judgements/<run_id>.jsonl` |
| 4. Report | `report.py` | `results/markdown/<run_id>.md`, `results/html/<run_id>.html` |

Each stage can be skipped with `--skip-<stage>` and loads its input from the previous JSONL.

### Generate

Generation is split into **three strand specialists** — Pure, Mechanics, Statistics — instead of one unified prompt. Each `(topic, board)` is routed to a specialist on its *assessed objective* (`config.strand_for_topic`, backed by `TOPIC_STRANDS`): Vectors is Pure (tooling Mechanics borrows), an `E(X)` integral is Statistics even though it integrates. `TOPIC_STRANDS` must cover every `TOPICS` entry — `strand_for_topic` raises on a gap rather than guessing. The split's payoff is not raw quality but carrying **strand-specific design guards** (`prompts.STRAND_GUARDS`) a unified prompt can't economically enumerate: the Pure geometric-series monotonicity / difficulty-collapse guards, the Mechanics physical-validity / signed-outcome guards, the Statistics exact-moment / regression / empty-tail guards.

The harness still **ranks the candidate `MODELS`** — every model now generates through the same specialist split, so the prompts are a shared baseline rather than a per-model variable. The specialist prompt (`prompts.specialist_system_prompt`) enforces JSON-only output with the same strict schema as before: `{ subtopics, questions[] }`, 4 questions foundation→extension.

**Code-in-loop:** each specialist is given a `run_python` tool (`sandbox.RUN_PYTHON_TOOL`) and told to use it as a deterministic oracle — compute and verify every value beyond a one-step closed form (decimals, powers, logs, binomial/normal/Poisson probabilities, multi-step integrals, discrete-RV moments) and assert its design invariants, changing the givens and recomputing on a failed assertion rather than fudging arithmetic. This is **not** the in-generator self-verification past runs found harmful (an LLM reasoning about its own correctness); a Python interpreter is a deterministic oracle, the right thing to put in the loop. `generate_one` runs a tool-calling loop: the model alternates `run_python` calls (executed via `sandbox.run_python`, off-thread) and tool results until it emits the final JSON, capped at `config.CODE_LOOP_MAX_ITERS` rounds — on the last round the tools are withdrawn to force an answer. Tokens and per-route cost (see **Routing**) accumulate across every round. Each record carries `strand`, `code_loop_iters`, and `python_runs`.

`sandbox.run_python` shells out to `config.SANDBOX_CMD` (default `uv run --with sympy python -`, so sympy/numpy are importable regardless of which interpreter runs the harness; override with `EVAL_SANDBOX_CMD`) with a `SANDBOX_TIMEOUT` wall-clock cap and `SANDBOX_OUTPUT_LIMIT` truncation. It is a plain subprocess, **not a security boundary** — the executed code is model-generated; point `SANDBOX_CMD` at a throwaway interpreter/container if isolation matters.

Generator chain-of-thought is capped via `config.GENERATOR_REASONING` (global `effort: low`), applied uniformly so no candidate gets an unfair thinking budget. Per-model overrides live on the `MODELS` entries; `"reasoning": None` opts a non-reasoning model out of the param entirely.

JSON is parsed robustly via `parse_utils.parse_json_robust` — it tries vanilla `json.loads`, then backslash repair, then `ast.literal_eval`. If all three fail, the output is sent to the **cleaner** (`cleaner.clean_json`, Claude Haiku) which repairs syntax without changing content. The cleaner is shared by the generation **and** solver stages and has a cross-provider fallback: when the primary Haiku route is down (instant connection-timeouts or blank `finish_reason: 'error'` responses that exhaust retries) or returns unparseable text, it falls through to `config.CLEANER_FALLBACK_MODEL` (gpt-5.4-nano, a different upstream) before giving up. `json_parse_ok_raw` records whether cleaning was needed.

### Solve

Each solver model runs two passes per generation:

1. **Blind solve** — solvers attempt the questions without seeing the mark scheme. A solver may declare a question `unsolvable` instead of guessing at a broken one.
2. **Reconciliation** — gated by a cheap equivalence check (`config.GATE_MODEL`, Claude Haiku): answers the gate confirms as matching the mark scheme get an automatic `agree` verdict; only mismatches and declared-unsolvable answers go to the full solver reconcile call. On a primary-route outage the gate tries `config.GATE_FALLBACK_MODEL` (gpt-5.4-nano, a different upstream) before failing open (unsure/unparseable → mismatch → reconcile anyway), so a transient Haiku blip doesn't push every answer into the expensive reconcile call. Both the blind solve and reconcile responses go through `cleaner.clean_json` on a parse failure (same cleaner + fallback as generation); recovered records set `cleaned: true`, and unrecoverable ones keep `solver_raw` / `reconcile_raw`. In the reconcile call the solver sees its own blind answer next to the mark scheme and adjudicates the conflict: `agree`, `solver_wrong`, `scheme_wrong`, or `unclear`. Framing it as adjudicating two specific answers anchors far less than open-ended verification, and the blind pass stays intact as the defence against self-consistent wrong schemes.

Both results live on the same solution record (`answers` + `reconciliation`, with separate `solver_*`/`reconcile_*` token and cost fields). When a solve or reconcile response fails to parse, the raw text is kept on the record (`solver_raw` / `reconcile_raw`) for post-run diagnosis. The judge receives blind answers, reconciliation verdicts, and each solver's known bias profile (`config.SOLVER_PROFILES`) so it can weigh the votes when distinguishing "question is wrong" from "solver failed". Solver chain-of-thought is capped via `config.SOLVER_REASONING`.

### Judge

Claude Opus scores **each question** on 5 dimensions (1–5 each, max 25) in a single per-batch call that returns a `questions[]` array:
- **correctness** — mathematical validity + solver agreement
- **mark_scheme** — point count matches marks, steps are distinct
- **command_word** — correct board-specific usage
- **difficulty** — cognitive demand matches mark count
- **style** — reads like a real exam paper, not AI-generated

A question passes at ≥18/25 (`config.PASS_THRESHOLD`). `flags` and `notes` are per question; `solver_agreement` stays batch-level. A parse/judge failure yields `questions: []` plus a batch-level failure flag (`json_parse_failure`, `judge_parse_failure`, `api_error`). Per-question flags: `unsolvable_question`, `ambiguous_question` (solvers disagree with the scheme and each other), `past_paper_suspected`, `scheme_wrong_consensus`.

**Hard rule (not judge discretion):** when ≥2 independent solvers reconcile the *same* question as `scheme_wrong`, `judge.apply_scheme_wrong_consensus` caps **that question's** correctness at 2, caps its total below the pass threshold, and adds the `scheme_wrong_consensus` flag to it — a consensus-confirmed wrong mark scheme can never let that question pass. Other questions in the batch are unaffected.

The judge is instructed to read the *pattern* of solver disagreement: multiple solvers converging on the same alternative answer implicates the mark scheme; solvers scattering implicates the question (→ `ambiguous_question`).

### Report

Produces a Markdown table (tabulate, github format) and a self-contained HTML viewer. The HTML viewer uses the same Playfair Display / DM Sans / KaTeX stack as the frontend, with per-model tabs and sortable score order. Cost breakdown (generation / solving / judging) is included.

All model-level metrics are computed over **individual questions**, not whole batches:

- **Pass rate / Avg score** — pass rate = questions scoring ≥18 / questions actually parsed (a quality signal; parse/judge failures contribute no questions and are excluded from the denominator). Avg score is the mean per-question /25.
- **Cost/pass Q** — **generation** cost only (including spend on batches that produced nothing usable) divided by the number of **individually passing** questions; the real price of a usable question, and the metric that carries the failure penalty (a flaky model's wasted spend inflates it). Solver and judge costs are excluded on purpose — both are shared eval-harness overhead, not the candidate's production cost, and solving (the dominant, model-varying term) would otherwise make this metric measure the verification ensemble rather than the generator. "—" when a model has no passing questions.
- **Per-topic score matrix** — topic × model mean per-question judge score (across boards), sorted hardest topic first, in both the Markdown report and the HTML summary tab. Used to spot topics where only stronger models hold up.

The HTML viewer groups questions under their topic batch (header shows an "X/N passed" summary) and renders each question's own /25 score badge, flags, and notes.

## Key Files

| File | Role |
|------|------|
| `config.py` | `MODELS`, `SOLVER_MODELS`, `CLEANER_MODEL`, `CLEANER_FALLBACK_MODEL`, `GATE_MODEL`, `GATE_FALLBACK_MODEL`, `JUDGE_MODEL`, `TOPICS`, `STRANDS`/`TOPIC_STRANDS`/`strand_for_topic`, `BOARDS`, `SANDBOX_CMD`/`CODE_LOOP_MAX_ITERS`, cost helpers |
| `prompts.py` | `specialist_system_prompt` + `STRAND_GUARDS`, `cleaner_prompt`, `solver_cleaner_prompt`, `solver_prompt`, `judge_prompt` |
| `sandbox.py` | `run_python` (subprocess code execution) + `RUN_PYTHON_TOOL` schema + `format_tool_result`, for the code-in-loop generator |
| `llm_utils.py` | `acompletion_with_retry` — litellm wrapper with exponential backoff (4 attempts; 5s base for rate limits, 1s for others). `retry_on_blank=True` also retries blank-content responses (provider `finish_reason: 'error'` arrives as a 200 with empty content, never an exception) — used by all stages except generation, which falls back to `reasoning_content` |
| `cleaner.py` | `clean_json` — syntax-only JSON repair with cross-provider fallback (`CLEANER_MODEL` → `CLEANER_FALLBACK_MODEL`); shared by the generation and solver stages |
| `parse_utils.py` | `parse_json_robust`, `extract_first_json`, `repair_backslashes`, `extract_fenced_json`, `valid_generation` |
| `generate.py` | Stage 1 async generation loop |
| `solve.py` | Stage 2 async solving loop |
| `judge.py` | Stage 3 async judging loop |
| `report.py` | Stage 4 report generation (Markdown + HTML) |
| `run.py` | CLI entrypoint — orchestrates all stages |
| `_probe_reasoning.py` | One-off probe script for testing reasoning model behaviour (not part of the main pipeline) |

## Routing (OpenRouter vs direct Gemini)

litellm dispatches each call on its model-id prefix, and `config.route_of` mirrors that into the two things that differ per route:

- **`openrouter/<provider>/<model>`** — routed through OpenRouter (needs `OPENROUTER_API_KEY`). Requests carry `USAGE_INCLUDE` (so OpenRouter returns the billed cost in `usage.cost`), an optional `provider` routing block, and the portable `reasoning` knob — all in `extra_body`. `response_cost` reads `usage.cost`.
- **`gemini/<model>`** — direct to Google AI Studio (needs `GEMINI_API_KEY`). The Gemini candidates run here (the chosen production provider). No `extra_body`: chain-of-thought is capped via the top-level `reasoning_effort` param litellm maps to Gemini's `thinkingConfig`, and `response_cost` computes cost from `litellm.completion_cost` (litellm's price table) since direct providers don't report a billed cost.

`config.request_kwargs(model, reasoning)` builds the per-route request kwargs and is the single place every stage (generate/solve/judge/cleaner/gate) constructs them — OpenRouter stays fully wired for every non-Gemini and future candidate. Add a model on a route by its id prefix, or force one with an explicit `"route"` key on the model dict.

**Native code execution (experiment):** a generator model entry may carry `native_code_exec: True` (e.g. `gemini-lite-native`). `generate_one` then swaps the harness-local `run_python` tool loop for a single call with Gemini's server-side `code_execution` tool (`tools=[{"code_execution": {}}]`). The native arm selects its system prompt from `prompts.NATIVE_PROMPT_VARIANTS` by `prompt_variant` (default `baseline` = `specialist_native_system_prompt`, the baseline prompt with only the oracle paragraph swapped); the local `prompt_builder` is ignored for native models since it carries no code-exec form. Run the mechanism A/B with `python run.py --models gemini-lite,gemini-lite-native`; the two report rows compare quality and Cost/pass Q.

**Native prompt A/B:** to compare prompts *within* the native sandbox (mechanism held constant), run `python run.py --models gemini-lite-native --compare-prompts baseline,work_backwards`. `NATIVE_PROMPT_VARIANTS` mirrors `PROMPT_VARIANTS` by variant name (`baseline` → `specialist_native_system_prompt`, `work_backwards` → `specialist_native_work_backwards`, which layers `WORK_BACKWARDS_PROTOCOL_NATIVE` on the native oracle). A variant present in `PROMPT_VARIANTS` but absent from `NATIVE_PROMPT_VARIANTS` (e.g. `no_guards`) has no native form and is rejected up front by `generate_all_variants` rather than silently falling back to the native baseline. Add a native variant by writing its `specialist_native_*` builder and registering it under the matching name.

**Cost caveat:** `response_cost` prices the direct route from `litellm.completion_cost` (token-based), which may not capture any separate Google charge for code execution, so the native arm's price can be slightly undercounted.

## Models

Defined in `config.py`. IDs are either OpenRouter paths (`openrouter/<provider>/<model>`) or direct-Gemini paths (`gemini/<model>`); see **Routing** above.

- **Generator models** (`MODELS`) — the candidates being compared
- **Solver models** (`SOLVER_MODELS`) — four independent families for answer verification. Chain-of-thought is capped per `config.SOLVER_REASONING` (global `effort: low`), with per-solver overrides for models that ignore `effort` (e.g. `deepseek-pro` uses a hard reasoning-token budget)
- **Cleaner model** (`CLEANER_MODEL`) — Claude Haiku; syntax-only JSON repair, shared across all generators **and solvers**, with a cross-provider backup (`CLEANER_FALLBACK_MODEL`, gpt-5.4-nano) for primary-route outages
- **Gate model** (`GATE_MODEL`) — Claude Haiku; cheap reconciliation equivalence check, with the same cross-provider backup (`GATE_FALLBACK_MODEL`) before failing open
- **Judge model** (`JUDGE_MODEL`) — Claude Opus; scores all generations

## Results

`evals/results/` — gitignored. Each run produces 5 files, one per output type, foldered by kind: `generations/<run_id>.jsonl`, `solutions/<run_id>.jsonl`, `judgements/<run_id>.jsonl`, `markdown/<run_id>.md`, `html/<run_id>.html`. Open the HTML report in a browser; the Markdown table is useful for quick comparison in a terminal or PR diff.
