"""Sandboxed Python execution for the code-in-loop generation stage.

Generator specialists call the ``run_python`` tool to compute and verify every
non-trivial value, assert their design invariants, and change givens when an
assertion fires — turning each generator into a deterministic-oracle-backed
author rather than one guessing arithmetic (binomial/normal cumulative
off-by-ones, multi-step integrals, discrete-RV moments).

Execution is a plain subprocess with a wall-clock timeout reading the code from
stdin. This is a local research harness, not a security boundary: the code is
model-generated and runs with the harness's own privileges. Point SANDBOX_CMD
at a throwaway interpreter/container if that matters for your environment.
"""
import subprocess

from config import SANDBOX_CMD, SANDBOX_OUTPUT_LIMIT, SANDBOX_TIMEOUT

# OpenAI-style function schema advertised to the generator. Kept here next to
# the executor so the tool contract and its implementation stay together.
RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Execute Python 3 and return its stdout/stderr. sympy, numpy, math, "
            "fractions and statistics are importable. Use it as a deterministic "
            "oracle: COMPUTE and VERIFY every value beyond a one-step closed form "
            "and ASSERT your design invariants before writing the final question "
            "JSON. print() anything you need to read back."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source to execute. Must print() its results.",
                }
            },
            "required": ["code"],
        },
    },
}


def _truncate(s: str | None, limit: int) -> str:
    s = s or ""
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n...[truncated {len(s) - limit} chars]"


def run_python(
    code: str,
    *,
    cmd: list[str] | None = None,
    timeout: int | None = None,
    output_limit: int | None = None,
) -> dict:
    """Run ``code`` in a subprocess, returning a structured result.

    Never raises for a misbehaving program: timeouts, non-zero exits and a
    missing interpreter all come back as a dict the loop can hand straight to
    the model so it can react.
    """
    cmd = cmd or SANDBOX_CMD
    timeout = SANDBOX_TIMEOUT if timeout is None else timeout
    output_limit = SANDBOX_OUTPUT_LIMIT if output_limit is None else output_limit
    try:
        proc = subprocess.run(
            cmd,
            input=code,
            capture_output=True,
            text=True,
            # Force UTF-8 for stdin/stdout: model-generated code is full of math
            # unicode (pi, sqrt, integral signs, ...) and the locale default is
            # cp1252 on Windows, which raises UnicodeEncodeError on the input.
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "timed_out": True,
            "returncode": None,
            "stdout": "",
            "stderr": f"timed out after {timeout}s",
        }
    except FileNotFoundError as e:
        return {
            "ok": False,
            "timed_out": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"sandbox interpreter not found ({cmd[0]!r}): {e}",
        }
    return {
        "ok": proc.returncode == 0,
        "timed_out": False,
        "returncode": proc.returncode,
        "stdout": _truncate(proc.stdout, output_limit),
        "stderr": _truncate(proc.stderr, output_limit),
    }


def format_tool_result(result: dict) -> str:
    """Render a run_python result as the string content of a tool message."""
    if result.get("timed_out"):
        return f"ERROR: {result.get('stderr', 'timed out')}"
    parts = []
    if result.get("stdout"):
        parts.append(result["stdout"].rstrip())
    if result.get("stderr"):
        parts.append("STDERR:\n" + result["stderr"].rstrip())
    if not parts:
        parts.append("(no output)")
    if not result.get("ok"):
        parts.append(f"[exit code {result.get('returncode')}]")
    return "\n".join(parts)
