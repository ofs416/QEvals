#!/usr/bin/env python3
"""Manual feasibility probe / hard gate for the native code-exec experiment.

Run from evals/:  python _probe_native.py

Gate: only proceed to the full A/B if (1) the lite model actually executed code
server-side (SHA-256 ground truth), (2) the native generation path parses to a
valid generation, and (3) response_cost returns a non-zero price.
"""
import asyncio
import hashlib

from dotenv import load_dotenv

load_dotenv()

import litellm

litellm.suppress_debug_info = True

from config import model_by_short, request_kwargs, response_cost
from generate import generate_one
from parse_utils import valid_generation

LITE_ID = "gemini/gemini-3.1-flash-lite"


async def ground_truth() -> bool:
    """(1) Prove server-side execution: a SHA-256 of a random nonce can't be
    recalled, so a matching digest means the model really ran the code."""
    nonce = "native-probe-" + hashlib.sha256(str(id(object())).encode()).hexdigest()[:12]
    truth = hashlib.sha256(nonce.encode()).hexdigest()
    prompt = (
        f"Run Python to compute hashlib.sha256({nonce!r}.encode()).hexdigest() "
        "and reply with ONLY the 64-char hex digest."
    )
    r = await litellm.acompletion(
        model=LITE_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        timeout=90,
        tools=[{"code_execution": {}}],
    )
    out = (r.choices[0].message.content or "").strip().lower()
    digest = "".join(c for c in out if c in "0123456789abcdef")[:64]
    ok = digest == truth
    print(f"[1] server-side execution: match={ok} (got {digest[:16]}...)")
    return ok


async def full_path() -> bool:
    """(2)+(3): the harness native path parses and prices."""
    model = model_by_short("gemini-lite-native")
    rec = await generate_one(model, "Integration", "Edexcel", "probe")
    parsed = rec["json_parse_ok"] and valid_generation(rec["output_parsed"])
    priced = rec["cost_usd"] > 0
    print(f"[2] native generation parses: {parsed} (json_parse_ok={rec['json_parse_ok']})")
    print(f"[3] non-zero price: {priced} (cost_usd={rec['cost_usd']})")
    if not parsed:
        print("    raw output (first 500 chars):", (rec["output_raw"] or "")[:500])
    return parsed and priced


async def main():
    ge = await ground_truth()
    fp = await full_path()
    print("\nGATE:", "PASS — proceed to full run" if (ge and fp) else "FAIL — do not run the A/B")


if __name__ == "__main__":
    asyncio.run(main())
