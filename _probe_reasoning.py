import asyncio
import os
import sys

import litellm
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv()

MODEL = sys.argv[1] if len(sys.argv) > 1 else "openrouter/deepseek/deepseek-v4-pro"

QUESTION = "Solve: a geometric series has first term 250 and ratio 0.8. Find the smallest n with S_n > 245. Reply with just the number."


async def probe(label, extra):
    body = {"usage": {"include": True}, **extra}
    try:
        r = await litellm.acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": QUESTION}],
            max_tokens=8192,
            timeout=120,
            extra_body=body,
        )
        u = r.usage
        details = getattr(u, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", None) if details else None
        print(f"{label:<28} completion={u.completion_tokens:>5}  reasoning={reasoning}  "
              f"finish={r.choices[0].finish_reason}  answer={str(r.choices[0].message.content)[:40]!r}")
    except Exception as e:
        print(f"{label:<28} FAIL {type(e).__name__}: {str(e)[:120]}")


async def main():
    print(f"model: {MODEL}")
    await probe("default", {})
    await probe("effort=low", {"reasoning": {"effort": "low"}})
    await probe("reasoning max_tokens=1024", {"reasoning": {"max_tokens": 1024}})
    await probe("reasoning disabled", {"reasoning": {"enabled": False}})


asyncio.run(main())
