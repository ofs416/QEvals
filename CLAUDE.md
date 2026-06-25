# QEvals/CLAUDE.md

LLM evaluation pipeline for comparing question-generation models. This is the root of the `QEvals` repo.

## Purpose

Benchmarks candidate LLMs on their ability to draft well-formed Edexcel A-level Maths questions. Outputs a scored HTML/Markdown report used to decide which model(s) to use in the Pergrad production pipeline.

The pipeline follows the **drafter → maths optimiser → typesetter → judge-panel** archetype (the four-role design sketched in `gem_evals/`). The candidate **generators are only drafters** — the maths correctness and rendering are isolated into fixed downstream stages, so what's ranked is the creative-drafting skill. The repo also carries earlier prototypes and a separate exam-rendering subproject — see **Repository Layout** at the end.

## Running

Run from the repo root:

```
python run.py                              # full pipeline (all candidate models, all topics)
python run.py --models gpt55,grok43        # subset of generator candidates
python run.py --questions-per-topic 1      # draft 1 question/topic instead of the default 3 (A/B the count)
python run.py --skip-generate              # load existing drafts, re-optimise + re-typeset + re-judge
python run.py --skip-optimise              # load drafts + optimisations, re-typeset + re-judge
python run.py --skip-typeset               # also load typesettings
python run.py --run-id my-run-id           # use a fixed run ID instead of a timestamp
```

Requires API keys in `.env` (gitignored): `OPENROUTER_API_KEY` for the OpenRouter-routed stages (drafter, typesetter, style judge), and `GEMINI_API_KEY` for the direct-Gemini sandbox stages (optimiser, maths judge — see **Routing** below). `litellm` dispatches each call on its model-id prefix.

**Questions-per-topic (`--questions-per-topic N`, default `config.QUESTIONS_PER_TOPIC`):** the one knob the chain is built to A/B. The drafter emits `N` drafts and every downstream stage iterates over however many questions exist — nothing hardcodes a count — so a `N=1` run and a `N=3` run are directly comparable on quality.

## Pipeline Stages

| Stage | Script | Model(s) | Output file |
|-------|--------|----------|-------------|
| 1. Draft | `generate.py` | candidate `MODELS` (OpenRouter) | `results/generations/<run_id>.jsonl` |
| 2. Optimise | `optimise.py` | `OPTIMISER_MODEL` (Gemini, native sandbox) | `results/optimisations/<run_id>.jsonl` |
| 3. Typeset | `typeset.py` | `TYPESETTER_MODEL` (OpenRouter) | `results/typesettings/<run_id>.jsonl` |
| 4. Judge | `judge.py` | `MATHS_JUDGE_MODEL` + `STYLE_JUDGE_MODEL` | `results/judgements/<run_id>.jsonl` |
| 5. Report | `report.py` | — | `results/markdown/<run_id>.md`, `results/html/<run_id>.html` |

Each stage can be skipped with `--skip-<stage>` and loads its input from the previous JSONL. Records are keyed by `generation_id` throughout; a downstream stage carries `skipped: true` (and produces nothing) when its input failed to parse.

### Draft (the ranked stage)

The drafter is the candidate being compared. It is strand-aware — each `(topic, board)` is routed to Pure / Mechanics / Statistics on its *assessed objective* (`config.strand_for_topic`, backed by `TOPIC_STRANDS`): Vectors is Pure (tooling Mechanics borrows), an `E(X)` integral is Statistics even though it integrates. `TOPIC_STRANDS` must cover every `TOPICS` entry — `strand_for_topic` raises on a gap. The strand selects the **design guards** (`prompts.STRAND_GUARDS`) the drafter reads as *design intent* (keep the scenario non-degenerate: geometric-series monotonicity, difficulty-collapse, physical validity, regression shape, empty critical regions).

`prompts.drafter_system_prompt(strand, board, n)` asks for `n` original question scenarios with rough numbers and **no mark scheme** — the drafter runs no code; the optimiser owns the maths. Output is JSON-only: `{ "drafts": [str, ...] }`. A single completion (no tool loop). Chain-of-thought is capped via `config.GENERATOR_REASONING` (global `effort: low`), applied uniformly so no candidate gets an unfair thinking budget; per-model `"reasoning"` overrides live on the `MODELS` entries (`None` opts a non-reasoning model out of the param).

### Optimise (the deterministic oracle)

`OPTIMISER_MODEL` (Gemini, direct to Google AI Studio with `native_code_exec`) solves each draft in Google's **server-side code-execution sandbox** (`tools=[{"code_execution": {}}]`), tweaks the constants until the numbers are neat, builds the mark scheme, and emits a `values` map of verified latex-ready quantities. The record's `questions[]` carry `{ text, commandWord, marks, difficulty, markScheme, values }`. The `values` map is the **anti-hallucination payload** the typesetter must reuse verbatim. `prompts.optimiser_prompt` feeds the strand guards as the "what to verify/preserve" list. Reasoning is capped via `config.OPTIMISER_REASONING`.

### Typeset

`TYPESETTER_MODEL` (OpenRouter, no sandbox) renders each finalised question and mark scheme as HTML (`{ question_html, mark_scheme_html }`), told to use the optimiser's exact `values` and never recompute. The per-question `marks`/`difficulty`/`commandWord` are carried over from the optimiser record (the typesetter returns only HTML; `typeset._merge` pairs them by index). HTML uses KaTeX `$...$`/`$$...$$` delimiters so the report renders it in the browser.

### Judge (two gates, both must pass)

A finalised question is checked by two independent judges, run concurrently per batch:

- **`MATHS_JUDGE_MODEL`** (Gemini, native sandbox) re-derives the maths in code and returns a hard per-question `maths_correct` verdict + `note` (`prompts.maths_judge_prompt`).
- **`STYLE_JUDGE_MODEL`** (Claude Opus) scores `command_word`, `difficulty`, `style` (1–5 each → `suitability_total` /15) + `flags`/`notes` (`prompts.style_judge_prompt`). It scores **no** correctness dimension — that is the maths judge's gate.

A question `passed` only if `maths_correct AND suitability_total >= config.SUITABILITY_THRESHOLD`. A judge call that fails to parse fails its gate (conservative: an unverified question does not pass) and adds a batch flag (`maths_judge_parse_failure`, `style_judge_parse_failure`). A skipped/failed optimiser input yields `questions: []` + `optimiser_failure`; an exception yields `api_error`. Per-question style flags: `past_paper_suspected`, `ambiguous_question`.

### Report

Produces a Markdown table (tabulate, github format) and a self-contained HTML viewer (Playfair Display / DM Sans / KaTeX stack, per-model tabs, sortable). The viewer embeds the typesetter's `question_html` + `mark_scheme_html` directly and shows each question's PASS/FAIL, a maths ✓/✗ badge, and its suitability /15. Cost breakdown has four lines: generation / optimise / typeset / judging.

All model-level metrics are computed over **individual questions**, not whole batches:

- **Pass rate / Maths OK / Avg /15** — pass rate = `passed` questions / questions judged (failed batches contribute no questions and are excluded — a quality signal). Maths OK = `maths_correct` rate. Avg /15 is the mean per-question suitability.
- **Cost/pass Q** — **generation (drafter)** cost only (including spend on batches that produced nothing usable) divided by the number of **individually passing** questions. The optimiser / typesetter / judge are fixed shared eval-harness overhead, not the candidate's production cost, so they're excluded — exactly the rationale that excluded solver/judge cost before. "—" when a model has no passing questions.
- **Per-topic suitability matrix** — topic × model mean suitability (across boards), sorted hardest topic first, in both the Markdown report and the HTML summary tab.

## Key Files

| File | Role |
|------|------|
| `config.py` | `MODELS`, `OPTIMISER_MODEL`, `TYPESETTER_MODEL`, `MATHS_JUDGE_MODEL`, `STYLE_JUDGE_MODEL`, `CLEANER_MODEL`/`CLEANER_FALLBACK_MODEL`, `QUESTIONS_PER_TOPIC`, `SUITABILITY_THRESHOLD`, `TOPICS`, `STRANDS`/`TOPIC_STRANDS`/`strand_for_topic`, `BOARDS`, routing + cost helpers |
| `prompts.py` | `drafter_system_prompt` + `STRAND_GUARDS`, `optimiser_prompt`, `typesetter_prompt`, `maths_judge_prompt`, `style_judge_prompt`, `cleaner_prompt_for` |
| `generate.py` | Stage 1 async drafting loop |
| `optimise.py` | Stage 2 async optimise loop (Gemini native sandbox) |
| `typeset.py` | Stage 3 async typeset loop (+ `_merge`) |
| `judge.py` | Stage 4 async two-gate panel |
| `report.py` | Stage 5 report generation (Markdown + HTML) |
| `run.py` | CLI entrypoint — orchestrates all stages |
| `llm_utils.py` | `acompletion_with_retry` — litellm wrapper with exponential backoff (4 attempts; 5s base for rate limits, 1s for others). `retry_on_blank=True` also retries blank-content responses (provider `finish_reason: 'error'` arrives as a 200 with empty content, never an exception) |
| `cleaner.py` | `clean_json` — syntax-only JSON repair with cross-provider fallback (`CLEANER_MODEL` → `CLEANER_FALLBACK_MODEL`); shared by every stage via `cleaner_prompt_for(shape)` |
| `parse_utils.py` | `parse_json_robust`, `extract_first_json`, `repair_backslashes`, `extract_fenced_json`, `valid_generation`, `valid_drafts` |

## Routing (OpenRouter vs direct Gemini)

litellm dispatches each call on its model-id prefix, and `config.route_of` mirrors that into the two things that differ per route:

- **`openrouter/<provider>/<model>`** — routed through OpenRouter (needs `OPENROUTER_API_KEY`). Used by the drafter candidates, the typesetter, the style judge, and the cleaner. Requests carry `USAGE_INCLUDE` (so OpenRouter returns the billed cost in `usage.cost`), an optional `provider` routing block, and the portable `reasoning` knob — all in `extra_body`. `response_cost` reads `usage.cost`.
- **`gemini/<model>`** — direct to Google AI Studio (needs `GEMINI_API_KEY`). Used by the optimiser and maths judge, which need the **server-side code-execution sandbox** (`native_code_exec: True` → `tools=[{"code_execution": {}}]`). No `extra_body`: chain-of-thought is capped via the top-level `reasoning_effort` param litellm maps to Gemini's `thinkingConfig`, and `response_cost` computes cost from `litellm.completion_cost` since direct providers don't report a billed cost.

`config.request_kwargs(model, reasoning)` builds the per-route request kwargs and is the single place every stage constructs them. Add a model on a route by its id prefix, or force one with an explicit `"route"` key on the model dict.

**Cost caveat:** `response_cost` prices the direct route from `litellm.completion_cost` (token-based), which may not capture any separate Google charge for code execution, so the optimiser / maths-judge spend can be slightly undercounted. (These are shared overhead, excluded from Cost/pass Q.)

## Models

Defined in `config.py`. IDs are either OpenRouter paths (`openrouter/<provider>/<model>`) or direct-Gemini paths (`gemini/<model>`); see **Routing** above.

- **Generator candidates** (`MODELS`) — the OpenRouter drafters being ranked
- **Optimiser** (`OPTIMISER_MODEL`) — Gemini with native sandbox; the deterministic maths oracle (fixed, shared)
- **Typesetter** (`TYPESETTER_MODEL`) — OpenRouter; HTML rendering, no sandbox (fixed, shared)
- **Maths judge** (`MATHS_JUDGE_MODEL`) — Gemini with native sandbox; hard maths-correct gate
- **Style judge** (`STYLE_JUDGE_MODEL`) — Claude Opus; suitability gate
- **Cleaner** (`CLEANER_MODEL`) — Claude Haiku; syntax-only JSON repair shared across stages, with a cross-provider backup (`CLEANER_FALLBACK_MODEL`, gpt-5.4-nano) for primary-route outages

## Results

`results/` — gitignored. Each run produces five JSONL/rendered files, foldered by kind: `generations/`, `optimisations/`, `typesettings/`, `judgements/`, plus `markdown/<run_id>.md` and `html/<run_id>.html`. Open the HTML report in a browser; the Markdown table is useful for quick comparison in a terminal or PR diff.

## Repository Layout

Beyond the `run.py` pipeline and its `tests/`, the repo carries one auxiliary directory. It is not imported by the main pipeline.

**`gem_evals/`** — the prompt sets that inspired this architecture: `prompts.txt` (drafter / optimiser / typesetter / reviewer) and `prompts_backwards.txt` (the optimiser work-backwards variant). The work-backwards optimiser prompt is a natural future A/B (hold the chain fixed, swap `optimiser_prompt`).
