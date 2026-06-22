"""Unit tests for sympy_render.render_from_sympy (no API calls).

Run:  uv run --with sympy python test_sympy_render.py
"""
from sympy_render import render_from_sympy


def test_renders_objects_to_latex():
    src = '''
import sympy as sp
t = sp.symbols('t')
TOPIC = "Kinematics"
DIFFICULTY = "higher"
v = 1 - sp.cos(2*t)
s_at_pi = sp.pi
QUESTION = [("text", "Find the velocity "), ("math", v), ("text", ".")]
MARKSCHEME = [("text", "s = "), ("math", s_at_pi)]
VERIFICATION = "ok"
'''
    payload, err = render_from_sympy(src)
    assert err is None, err
    # LaTeX comes out of sympy.latex(), not from the model — verbose form proves it.
    assert payload["question_latex"] == r"Find the velocity $1 - \cos{\left(2 t \right)}$."
    assert payload["markscheme_latex"] == r"s = $\pi$"
    assert payload["topic"] == "Kinematics"
    assert payload["difficulty"] == "higher"


def test_answer_value_comes_from_computed_object():
    # The anti-drift property: the displayed answer is rendered from the SAME
    # object the program computed, so it can't disagree with the computation.
    src = '''
import sympy as sp
TOPIC = "Vectors"; DIFFICULTY = "extension"
t = (8 - 2*sp.sqrt(3))/5
x = 6*t
dist = sp.simplify(x / sp.cos(sp.rad(30)))
QUESTION = [("text", "Find the distance.")]
MARKSCHEME = [("text", "distance = "), ("math", dist), ("text", " m")]
'''
    payload, err = render_from_sympy(src)
    assert err is None, err
    # (32*sqrt(3) - 24)/5, the correct value — rendered, not narrated.
    assert r"\sqrt{3}" in payload["markscheme_latex"]
    assert "32" in payload["markscheme_latex"]
    assert payload["verification_summary"] == ""  # optional var defaults empty


def test_display_math_and_bare_string_segment():
    src = '''
import sympy as sp
x = sp.symbols('x')
TOPIC = "Integration"; DIFFICULTY = "higher"
QUESTION = ["A bare string is treated as prose. ", ("display", sp.Integral(x**2, x))]
MARKSCHEME = [("math", sp.Rational(1, 3))]
VERIFICATION = "v"
'''
    payload, err = render_from_sympy(src)
    assert err is None, err
    assert payload["question_latex"].startswith("A bare string is treated as prose. $$")
    assert payload["question_latex"].endswith("$$")
    assert payload["markscheme_latex"] == r"$\frac{1}{3}$"


def test_failing_assert_surfaces_as_error():
    # A broken question must fail here (non-zero exit), not render silently.
    src = '''
import sympy as sp
TOPIC = "Proof"; DIFFICULTY = "higher"
answer = sp.Integer(2) + 2
assert answer == 5, "design invariant violated: expected neat 5"
QUESTION = [("text", "x")]
MARKSCHEME = [("text", "y")]
'''
    payload, err = render_from_sympy(src)
    assert payload is None
    assert "design invariant violated" in err


def test_missing_required_variable_surfaces_as_error():
    src = '''
TOPIC = "X"; DIFFICULTY = "higher"
QUESTION = [("text", "q")]
# MARKSCHEME deliberately undefined
'''
    payload, err = render_from_sympy(src)
    assert payload is None
    assert "MARKSCHEME" in err  # NameError names it


def test_bad_segment_kind_surfaces_as_error():
    src = '''
TOPIC = "X"; DIFFICULTY = "higher"
QUESTION = [("texxt", "typo kind")]
MARKSCHEME = [("text", "y")]
'''
    payload, err = render_from_sympy(src)
    assert payload is None
    assert "unknown segment kind" in err


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
