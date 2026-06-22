import sys

from sandbox import RUN_PYTHON_TOOL, format_tool_result, run_python

# Run against the test interpreter (no sympy needed for these checks) so the
# suite stays fast and offline, independent of the configured SANDBOX_CMD.
PY = [sys.executable]


def test_run_python_captures_stdout():
    r = run_python("print(6*7)", cmd=PY)
    assert r["ok"] is True
    assert "42" in r["stdout"]


def test_run_python_reports_error_without_raising():
    r = run_python("raise ValueError('boom')", cmd=PY)
    assert r["ok"] is False
    assert r["timed_out"] is False
    assert "boom" in r["stderr"]


def test_run_python_times_out():
    r = run_python("import time; time.sleep(5)", cmd=PY, timeout=1)
    assert r["timed_out"] is True
    assert r["ok"] is False


def test_run_python_truncates_long_output():
    r = run_python("print('x' * 10000)", cmd=PY, output_limit=100)
    assert "truncated" in r["stdout"]
    assert len(r["stdout"]) < 200


def test_run_python_missing_interpreter():
    r = run_python("print(1)", cmd=["definitely-not-a-real-interpreter-xyz"])
    assert r["ok"] is False
    assert "not found" in r["stderr"]


def test_format_tool_result_includes_stdout():
    out = format_tool_result(
        {"ok": True, "timed_out": False, "returncode": 0, "stdout": "42", "stderr": ""}
    )
    assert "42" in out


def test_format_tool_result_flags_failure():
    out = format_tool_result(
        {"ok": False, "timed_out": False, "returncode": 1, "stdout": "", "stderr": "Traceback"}
    )
    assert "exit code 1" in out
    assert "Traceback" in out


def test_run_python_tool_schema_shape():
    assert RUN_PYTHON_TOOL["type"] == "function"
    assert RUN_PYTHON_TOOL["function"]["name"] == "run_python"
    assert "code" in RUN_PYTHON_TOOL["function"]["parameters"]["properties"]
