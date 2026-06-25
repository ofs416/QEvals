#!/usr/bin/env python3
import argparse, asyncio
from datetime import datetime, timezone
import litellm
litellm.suppress_debug_info = True

from config import MODELS, TOPICS, BOARDS, QUESTIONS_PER_TOPIC, results_path, model_by_short
from generate import generate_all, save_jsonl, load_jsonl
from optimise import optimise_all
from typeset import typeset_all
from judge import judge_all
from report import generate_report


def parse_args():
    p = argparse.ArgumentParser(description="Run the LLM eval pipeline")
    p.add_argument("--run-id", default=None, help="Run identifier (default: timestamp)")
    p.add_argument("--skip-generate", action="store_true", help="Skip drafting, load existing JSONL")
    p.add_argument("--skip-optimise", action="store_true", help="Skip optimising, load existing JSONL")
    p.add_argument("--skip-typeset", action="store_true", help="Skip typesetting, load existing JSONL")
    p.add_argument("--skip-judge", action="store_true", help="Skip judging, load existing JSONL")
    p.add_argument("--models", default=None, help="Comma-separated short names to test, e.g. gpt55,grok43")
    p.add_argument("--questions-per-topic", type=int, default=QUESTIONS_PER_TOPIC,
                   help=f"Drafts per (topic, board) (default: {QUESTIONS_PER_TOPIC})")
    return p.parse_args()


async def main():
    from dotenv import load_dotenv
    load_dotenv()

    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

    gen_path = results_path(run_id, "generations")
    opt_path = results_path(run_id, "optimisations")
    ts_path  = results_path(run_id, "typesettings")
    jdg_path = results_path(run_id, "judgements")
    md_path  = results_path(run_id, "markdown", ext="md")
    html_path = results_path(run_id, "html", ext="html")

    models = None
    if args.models:
        models = [model_by_short(s.strip()) for s in args.models.split(",")]

    # Stage 1: Draft
    if args.skip_generate:
        print(f"[generate] Loading {gen_path}")
        generations = load_jsonl(gen_path)
    else:
        n = args.questions_per_topic
        print(f"[generate] Drafting {len(models or MODELS)} models x {len(TOPICS)} topics x {len(BOARDS)} board(s), {n} q/topic ...")
        generations = await generate_all(run_id, models=models, n=n)
        save_jsonl(generations, gen_path)
        ok = sum(1 for g in generations if g["json_parse_ok"])
        print(f"[generate] Done -- {ok}/{len(generations)} parsed OK -> {gen_path}")

    # Stage 2: Optimise
    if args.skip_optimise:
        print(f"[optimise] Loading {opt_path}")
        optimisations = load_jsonl(opt_path)
    else:
        print(f"[optimise] Optimising {len(generations)} batches ...")
        optimisations = await optimise_all(generations)
        save_jsonl(optimisations, opt_path)
        ok = sum(1 for o in optimisations if o.get("parse_ok"))
        print(f"[optimise] Done -- {ok}/{len(optimisations)} parsed OK -> {opt_path}")

    # Stage 3: Typeset
    if args.skip_typeset:
        print(f"[typeset] Loading {ts_path}")
        typesettings = load_jsonl(ts_path)
    else:
        print(f"[typeset] Typesetting {len(optimisations)} batches ...")
        typesettings = await typeset_all(optimisations)
        save_jsonl(typesettings, ts_path)
        ok = sum(1 for t in typesettings if t.get("parse_ok"))
        print(f"[typeset] Done -- {ok}/{len(typesettings)} parsed OK -> {ts_path}")

    # Stage 4: Judge
    if args.skip_judge:
        print(f"[judge] Loading {jdg_path}")
        judgements = load_jsonl(jdg_path)
    else:
        print(f"[judge] Judging {len(optimisations)} batches ...")
        judgements = await judge_all(optimisations)
        save_jsonl(judgements, jdg_path)
        print(f"[judge] Done -> {jdg_path}")

    # Stage 5: Report
    generate_report(run_id, generations, optimisations, typesettings, judgements, md_path, html_path)
    print(f"[report] {md_path}")
    print(f"[report] {html_path}")


if __name__ == "__main__":
    asyncio.run(main())
