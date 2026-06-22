"""One-off probe: can x-ai/grok-4.3 act as a solver? Runs the real
solve_one + reconcile_one path on the two generations whose mark schemes
all four production solvers flagged as scheme_wrong, plus one clean
control generation. Not part of the main pipeline.
"""
import asyncio, json
from dotenv import load_dotenv
import litellm
litellm.suppress_debug_info = True

from generate import load_jsonl
from solve import solve_one, reconcile_one

GROK = {"id": "openrouter/x-ai/grok-4.3", "short": "grok43"}

# (generation_id, question_index all solvers called scheme_wrong; None = control)
TARGETS = [
    ("777d2784-5661-44da-a153-9a486b51c002", 1),   # Integration: answer should be (2sqrt2+2)/15
    ("2300aad9-f91f-4fe3-91d2-c54d6f43d542", 0),   # Sequences: answer should be d=1
    ("06cf1180-22fa-43b7-824b-e2c385a52bcc", None),  # control: passed, only opus48 dissented
]


async def main():
    load_dotenv()
    gens = {g["generation_id"]: g for g in load_jsonl("results/generations_gemlite_qwenplus.jsonl")}

    for gid, flagged_qi in TARGETS:
        gen = gens[gid]
        print("=" * 70)
        print(f"{gen['topic']} ({gid[:8]})  flagged question: {flagged_qi}")
        try:
            solution = await solve_one(gen, GROK)
        except Exception as e:
            print(f"  solve failed: {type(e).__name__}: {e}")
            continue
        print(f"  blind solve: parse_ok={solution['parse_ok']} "
              f"latency={solution['latency_ms'] / 1000:.0f}s "
              f"cost=${solution['solver_cost_usd']:.4f} "
              f"out_tokens={solution['solver_output_tokens']}")
        for a in solution.get("answers", []):
            print(f"    q{a.get('question_index')}: unsolvable={a.get('unsolvable', False)} "
                  f"answer={str(a.get('final_answer') or a.get('answer'))[:160]}")
        if not solution["parse_ok"]:
            print("  RAW:", (solution.get("solver_raw") or "")[:500])
            continue
        try:
            rec = await reconcile_one(gen, solution, GROK)
        except Exception as e:
            print(f"  reconcile failed: {type(e).__name__}: {e}")
            continue
        print(f"  reconcile: ok={rec['reconcile_ok']} cost=${rec['reconcile_cost_usd']:.4f}")
        for r in rec.get("reconciliation", []):
            mark = " <-- flagged by all 4 prod solvers" if r.get("question_index") == flagged_qi else ""
            print(f"    q{r.get('question_index')}: {r.get('verdict')}{mark}")
            print(f"      {str(r.get('reason'))[:250]}")


if __name__ == "__main__":
    asyncio.run(main())
