import ast
import re
import warnings


def extract_first_json(text: str | None) -> str | None:
    """Return the first balanced {...} object from text, or None if absent.

    Tracks both single- and double-quoted string boundaries so that braces
    inside string values don't throw off the depth counter.
    """
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    quote_char: str | None = None  # current string delimiter, or None
    i = start
    while i < len(text):
        ch = text[i]
        if quote_char:
            if ch == "\\":
                i += 2  # skip escaped character
                continue
            if ch == quote_char:
                quote_char = None
        else:
            if ch in ('"', "'"):
                quote_char = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1
    return None


def repair_backslashes(raw: str) -> str:
    """Double single backslashes that aren't valid JSON escape sequences.

    The negative lookbehind prevents re-processing the second backslash of an
    already-correct '\\\\x' pair, which is the bug in a simple lookahead-only regex.
    """
    return re.sub(r"(?<!\\)\\(?![\"\\\/bfnrtu])", r"\\\\", raw)


def extract_fenced_json(text: str) -> str | None:
    """Return the contents of the first ```json (or bare ```) fenced block.

    Preferred over brace counting when a fence is present: the brace tracker's
    escape skipping can desync on invalid escapes (e.g. a lone backslash in
    LaTeX like \\implies) and run off the end of otherwise-complete output.
    """
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    return m.group(1) if m else None


def valid_generation(parsed: object) -> bool:
    """True if parsed output has the shape downstream stages depend on.

    Guards against false-positive parses where chain-of-thought contains a
    small JSON fragment (e.g. a single mark-scheme item) that parses fine
    but is not a generation.
    """
    return (
        isinstance(parsed, dict)
        and isinstance(parsed.get("questions"), list)
        and len(parsed["questions"]) > 0
    )


def valid_drafts(parsed: object) -> bool:
    """True if parsed drafter output has a non-empty 'drafts' list of strings."""
    return (
        isinstance(parsed, dict)
        and isinstance(parsed.get("drafts"), list)
        and len(parsed["drafts"]) > 0
        and all(isinstance(d, str) for d in parsed["drafts"])
    )


def parse_json_robust(text: str | None) -> tuple[dict | None, str | None]:
    """Extract and parse the first JSON object from text.

    Returns (parsed_dict, None) on success, or (None, error_message) on failure.
    Extraction prefers a markdown-fenced block, falling back to the first
    balanced {...}. Repair passes in order:
      1. Vanilla json.loads
      2. Fix invalid LaTeX backslash escapes, retry json.loads
      3. ast.literal_eval — handles Python-style single-quoted dicts from
         reasoning models whose chain-of-thought uses Python dict syntax
    """
    import json

    if not text:
        return None, "no JSON object found"

    candidates = []
    fenced = extract_fenced_json(text)
    if fenced:
        candidates.append(fenced)
    braced = extract_first_json(text)
    if braced and braced != fenced:
        candidates.append(braced)
    if not candidates:
        return None, "no JSON object found"

    final_err: str | None = None
    for raw in candidates:
        # strict=False permits raw control characters (literal newlines/tabs)
        # inside string values, which LLMs emit routinely in multi-line text.
        try:
            return json.loads(raw, strict=False), None
        except json.JSONDecodeError:
            pass

        try:
            return json.loads(repair_backslashes(raw), strict=False), None
        except json.JSONDecodeError as exc:
            final_err = final_err or str(exc)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                obj = ast.literal_eval(raw)
            if isinstance(obj, dict):
                return obj, None
        except (ValueError, SyntaxError):
            pass

    return None, final_err or "no JSON object found"
