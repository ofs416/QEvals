"""Host-side LaTeX rendering for the draft-refine *sympy-program* arm.

The refiner does not emit LaTeX strings. It emits a Python program that builds
SymPy objects and lays out the question / mark scheme as a list of *segments* —
each segment either literal prose ``("text", "...")`` or a verified SymPy object
``("math", expr)`` / ``("display", expr)``. This module appends a fixed host
epilogue (which the model never authors) that calls ``sympy.latex()`` on those
objects and prints the assembled strings as JSON.

Two payoffs over a model that writes LaTeX into a JSON string itself:
  * No JSON/LaTeX backslash escaping ever enters the token stream — the model
    emits ``sin(2*t)``, the host renders ``\\sin{\\left(2 t \\right)}``.
  * The rendered LaTeX is produced from the *same objects* the program computed
    and asserted on, so a question's stated answer cannot drift from the value
    SymPy actually verified (the failure that produced a wrong slope distance in
    the string arm). Numeric answers must be ``("math", expr)`` segments, so the
    model literally cannot hand-narrate a number the program didn't compute.

``render_from_sympy`` runs the whole thing once in the sandbox: the model's own
``assert`` invariants execute in the same pass, so a broken question fails here
(non-zero exit) rather than rendering a wrong one.
"""
import json

from sandbox import run_python

_START = "===RENDER_JSON_START==="
_END = "===RENDER_JSON_END==="

# Appended verbatim after the model's program. The model defines TOPIC,
# DIFFICULTY, QUESTION, MARKSCHEME, VERIFICATION; this turns the segment lists
# into LaTeX. It is host-owned: the model never writes a latex() call.
_EPILOGUE = f'''

# ====================== HOST EPILOGUE (model never authors this) ============
import json as _json
import sympy as _sp


def _seg_latex(_seg):
    # Lenient: a bare string is prose; otherwise (kind, value[, ...]).
    if isinstance(_seg, str):
        return _seg
    _kind = _seg[0]
    _val = _seg[1]
    if _kind == "text":
        return str(_val)
    if _kind in ("math", "display"):
        _expr = _val if isinstance(_val, _sp.Basic) else _sp.sympify(_val)
        _tex = _sp.latex(_expr)
        return ("$$" + _tex + "$$") if _kind == "display" else ("$" + _tex + "$")
    raise ValueError("unknown segment kind: %r" % (_kind,))


def _render(_segs, _name):
    if not isinstance(_segs, (list, tuple)) or not _segs:
        raise ValueError("%s must be a non-empty list of segments" % _name)
    return "".join(_seg_latex(_s) for _s in _segs)


_payload = {{
    "topic": str(TOPIC),
    "difficulty": str(DIFFICULTY),
    "question_latex": _render(QUESTION, "QUESTION"),
    "markscheme_latex": _render(MARKSCHEME, "MARKSCHEME"),
    "verification_summary": str(globals().get("VERIFICATION", "")),
}}
print("{_START}")
print(_json.dumps(_payload))
print("{_END}")
'''


def render_from_sympy(program_src: str, **run_kwargs) -> tuple[dict | None, str | None]:
    """Execute a refiner SymPy program and render its LaTeX host-side.

    ``program_src`` must define module-level ``TOPIC``, ``DIFFICULTY``,
    ``QUESTION`` and ``MARKSCHEME`` (and optionally ``VERIFICATION``). Returns
    ``(payload, None)`` on success or ``(None, error)`` if the program raised,
    timed out, or didn't emit the sentinel-wrapped JSON — the error string is the
    sandbox stderr (the model's failing assert, a NameError for a missing
    variable, etc.) so a retry can feed it straight back to the model.
    """
    full = program_src + _EPILOGUE
    result = run_python(full, **run_kwargs)
    stdout = result.get("stdout", "") or ""
    if _START not in stdout or _END not in stdout:
        stderr = (result.get("stderr") or "").strip()
        if result.get("timed_out"):
            stderr = stderr or "timed out"
        return None, stderr or "program produced no render output"
    blob = stdout.split(_START, 1)[1].split(_END, 1)[0].strip()
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError as e:
        return None, f"render JSON decode failed: {e}"
    return payload, None
