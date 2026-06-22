#!/usr/bin/env python3
import argparse, asyncio, os
from datetime import datetime, timezone
import litellm
litellm.suppress_debug_info = True

from config import MODELS, SOLVER_MODELS, TOPICS, BOARDS, RESULTS_DIR, results_path, model_by_short
from generate import generate_all, generate_all_variants, save_jsonl, load_jsonl
from solve import solve_all
from judge import judge_all
from report import generate_report
from prompts import PROMPT_VARIANTS


def parse_args():
    p = argparse.ArgumentParser(description="Run the LLM eval pipeline")
    p.add_argument("--run-id", default=None, help="Run identifier (default: timestamp)")
    p.add_argument("--skip-generate", action="store_true", help="Skip generation, load existing JSONL")
    p.add_argument("--skip-solve", action="store_true", help="Skip solving, load existing JSONL")
    p.add_argument("--skip-judge", action="store_true", help="Skip judging, load existing JSONL")
    p.add_argument("--models", default=None, help="Comma-separated short names to test, e.g. haiku,gpt4o-mini")
    p.add_argument("--prompt", default="baseline",
                   help="PROMPT_VARIANTS name to use for a normal model-ranking run (default: baseline)")
    p.add_argument("--compare-prompts", default=None,
                   help="Comma-separated PROMPT_VARIANTS names to A/B on one fixed --models model")
    return p.parse_args()


def resolve_prompt(prompt_arg):
    """Resolve --prompt to its builder for a normal run. Errors on an unknown name."""
    if prompt_arg not in PROMPT_VARIANTS:
        raise ValueError(
            f"Unknown prompt variant {prompt_arg!r}; available: {sorted(PROMPT_VARIANTS)}"
        )
    return PROMPT_VARIANTS[prompt_arg]


def resolve_prompt_comparison(models_arg, compare_arg):
    """Validate args for a prompt A/B run. Returns (model_dict, [variant_name, ...]).

    Raises ValueError on misuse: fewer than two variants, an unknown variant
    name, or not exactly one fixed model.
    """
    names = [s.strip() for s in compare_arg.split(",") if s.strip()]
    if len(names) < 2:
        raise ValueError("--compare-prompts needs at least two variant names")
    unknown = [n for n in names if n not in PROMPT_VARIANTS]
    if unknown:
        raise ValueError(
            f"Unknown prompt variant(s) {unknown}; available: {sorted(PROMPT_VARIANTS)}"
        )
    shorts = [s.strip() for s in (models_arg or "").split(",") if s.strip()]
    if not shorts:
        raise ValueError(
            "--compare-prompts requires exactly one --models (the fixed generator model); got none"
        )
    if len(shorts) != 1:
        raise ValueError(
            "--compare-prompts requires exactly one --models (the fixed generator model); "
            f"got {len(shorts)}: {shorts}"
        )
    return model_by_short(shorts[0]), names


async def main():
    from dotenv import load_dotenv
    load_dotenv()

    args = parse_args()
    if args.compare_prompts and args.prompt != "baseline":
        raise SystemExit("--prompt and --compare-prompts are mutually exclusive: "
                         "--compare-prompts already names the variants to run.")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

    gen_path    = results_path(run_id, "generations")
    sol_path    = results_path(run_id, "solutions")
    jdg_path    = results_path(run_id, "judgements")
    md_path     = results_path(run_id, "markdown", ext="md")
    html_path   = results_path(run_id, "html", ext="html")

    models = None
    if args.models:
        models = [model_by_short(s.strip()) for s in args.models.split(",")]

    # Stage 1: Generate
    if args.skip_generate:
        print(f"[generate] Loading {gen_path}")
        generations = load_jsonl(gen_path)
    elif args.compare_prompts:
        model, variant_names = resolve_prompt_comparison(args.models, args.compare_prompts)
        print(f"[generate] Comparing {len(variant_names)} prompts x {len(TOPICS)} topics "
              f"x {len(BOARDS)} board(s) on {model['short']} ...")
        generations = await generate_all_variants(run_id, model, variant_names)
        save_jsonl(generations, gen_path)
        ok = sum(1 for g in generations if g["json_parse_ok"])
        print(f"[generate] Done -- {ok}/{len(generations)} parsed OK -> {gen_path}")
    else:
        prompt_builder = resolve_prompt(args.prompt)
        label = "" if args.prompt == "baseline" else f" with prompt '{args.prompt}'"
        print(f"[generate] Running {len(models or MODELS)} models x {len(TOPICS)} topics x {len(BOARDS)} board(s){label} ...")
        generations = await generate_all(run_id, models=models, prompt_builder=prompt_builder)
        save_jsonl(generations, gen_path)
        ok = sum(1 for g in generations if g["json_parse_ok"])
        print(f"[generate] Done -- {ok}/{len(generations)} parsed OK -> {gen_path}")

    # Stage 2: Solve
    if args.skip_solve:
        print(f"[solve] Loading {sol_path}")
        solutions = load_jsonl(sol_path)
    else:
        print(f"[solve] Running {len(SOLVER_MODELS)} solver(s) x {len(generations)} generations ...")
        solutions = await solve_all(generations)
        save_jsonl(solutions, sol_path)
        skipped = sum(1 for s in solutions if s.get("skipped"))
        ok = sum(1 for s in solutions if s.get("parse_ok"))
        print(f"[solve] Done -- {ok}/{len(solutions)} parsed OK ({skipped} skipped) -> {sol_path}")

    # Stage 3: Judge
    if args.skip_judge:
        print(f"[judge] Loading {jdg_path}")
        judgements = load_jsonl(jdg_path)
    else:
        print(f"[judge] Judging {len(generations)} generations ...")
        judgements = await judge_all(generations, solutions)
        save_jsonl(judgements, jdg_path)
        print(f"[judge] Done -> {jdg_path}")

    # Stage 4: Report
    generate_report(run_id, generations, solutions, judgements, md_path, html_path)
    print(f"[report] {md_path}")
    print(f"[report] {html_path}")


if __name__ == "__main__":
    asyncio.run(main())
