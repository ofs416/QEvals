import pytest
from report import (
    assemble_viewer_data,
    build_combined_report,
    compute_model_stats,
    compute_topic_matrix,
    format_markdown_table,
    format_topic_table,
)


def _q(passed=True, maths=True, suit=13, flags=None, notes=None):
    return {
        "question_index": 0,
        "maths_correct": maths,
        "maths_note": "",
        "scores": {"command_word": 4, "difficulty": 4, "style": 5},
        "suitability_total": suit,
        "passed": passed,
        "flags": flags or [],
        "notes": notes or {},
    }


def _gen(short, gid, cost, topic="Algebra", board="Edexcel", parse_ok=True, questions=None):
    return {
        "model_short": short,
        "generation_id": gid,
        "topic": topic,
        "board": board,
        "json_parse_ok": parse_ok,
        "cost_usd": cost,
        "latency_ms": 1000,
        "_judgement": {"questions": questions if questions is not None else [_q()], "flags": []},
    }


# --- model stats -------------------------------------------------------------

def test_stats_avg_suitability_and_pass_rate():
    rows = [_gen("m", "g1", 0.01, questions=[_q(passed=True, suit=13)]),
            _gen("m", "g2", 0.01, questions=[_q(passed=False, suit=9)])]
    stats = compute_model_stats(rows)
    assert stats["m"]["avg_suit"] == 11.0
    assert stats["m"]["pass_rate"] == 0.5


def test_stats_maths_rate():
    rows = [_gen("m", "g1", 0.01, questions=[_q(maths=True)]),
            _gen("m", "g2", 0.01, questions=[_q(maths=False, passed=False)])]
    stats = compute_model_stats(rows)
    assert stats["m"]["maths_rate"] == 0.5


def test_cost_per_pass_uses_generation_cost_only():
    rows = [_gen("m", "g1", 0.01, questions=[_q(passed=True), {**_q(passed=True), "question_index": 1}]),
            _gen("m", "g2", 0.01, questions=[])]  # failed batch, still costs
    stats = compute_model_stats(rows)
    # 0.02 total gen cost / 2 passing questions
    assert stats["m"]["cost_per_pass_usd"] == pytest.approx(0.01)


def test_cost_per_pass_is_none_when_nothing_passes():
    rows = [_gen("m", "g1", 0.01, questions=[_q(passed=False)])]
    stats = compute_model_stats(rows)
    assert stats["m"]["cost_per_pass_usd"] is None
    assert "—" in format_markdown_table(stats)


def test_pass_rate_excludes_failed_batches():
    good = _gen("m", "g1", 0.01, questions=[_q(passed=True)])
    failed = _gen("m", "g2", 0.01, parse_ok=False, questions=[])
    stats = compute_model_stats([good, failed])
    assert stats["m"]["pass_rate"] == 1.0


def test_json_fail_rate():
    rows = [_gen("m", "g1", 0.01, parse_ok=False, questions=[]),
            _gen("m", "g2", 0.01, questions=[_q()])]
    stats = compute_model_stats(rows)
    assert stats["m"]["json_fail_pct"] == 50.0


def test_markdown_table_has_new_columns():
    rows = [_gen("m", "g1", 0.01, questions=[_q(passed=True, maths=True, suit=13)])]
    md = format_markdown_table(compute_model_stats(rows))
    assert "Maths OK" in md
    assert "Avg /15" in md
    assert "Cost/pass Q" in md


# --- topic matrix ------------------------------------------------------------

def test_topic_matrix_averages_across_boards():
    rows = [_gen("m", "g1", 0.01, topic="Vectors", board="AQA", questions=[_q(suit=14)]),
            _gen("m", "g2", 0.01, topic="Vectors", board="Edexcel", questions=[_q(suit=10)])]
    matrix = compute_topic_matrix(rows)
    assert matrix["Vectors"]["m"] == 12.0


def test_topic_matrix_omits_model_with_only_failed_batch():
    failed = _gen("bad", "g1", 0.01, topic="Vectors", parse_ok=False, questions=[])
    assert compute_topic_matrix([failed]) == {}


def test_topic_table_hardest_first():
    rows = [_gen("m", "g1", 0.01, topic="Algebra", questions=[_q(suit=14)]),
            _gen("m", "g2", 0.01, topic="Vectors", questions=[_q(suit=8)])]
    md = format_topic_table(compute_topic_matrix(rows), ["m"])
    assert md.index("Vectors") < md.index("Algebra")


# --- viewer ------------------------------------------------------------------

def _vgen(short, gid, topic="Algebra"):
    return {"generation_id": gid, "model_short": short, "topic": topic, "board": "Edexcel",
            "json_parse_ok": True, "cost_usd": 0.01, "latency_ms": 1000}


def _vts(gid, n=1):
    return {"generation_id": gid, "questions": [
        {"question_html": f"<p>Q{i} html</p>", "mark_scheme_html": "<ol><li>M1</li></ol>",
         "marks": 2, "difficulty": "foundation", "commandWord": "Find"} for i in range(n)]}


def _vjdg(gid, passed=True, maths=True, suit=13, n=1):
    return {"generation_id": gid, "flags": [], "questions": [
        {"question_index": i, "maths_correct": maths, "maths_note": "",
         "scores": {}, "suitability_total": suit, "passed": passed, "flags": [], "notes": {}}
        for i in range(n)]}


def test_viewer_pairs_html_with_verdict():
    data = assemble_viewer_data([_vgen("m", "g1")], [_vts("g1")], [_vjdg("g1", passed=True)])
    batch = data["m"][0]
    assert batch["passed"] == 1
    assert batch["maths_ok"] == 1
    q = batch["questions"][0]
    assert q["question_html"] == "<p>Q0 html</p>"
    assert q["mark_scheme_html"] == "<ol><li>M1</li></ol>"
    assert q["passed"] is True


def test_viewer_groups_by_model():
    data = assemble_viewer_data(
        [_vgen("a", "g1"), _vgen("b", "g2")],
        [_vts("g1"), _vts("g2")],
        [_vjdg("g1"), _vjdg("g2")],
    )
    assert set(data) == {"a", "b"}


def test_viewer_missing_typeset_still_shows_verdict():
    # judge scored it but typeset produced nothing — verdict shows, html blank
    data = assemble_viewer_data([_vgen("m", "g1")], [], [_vjdg("g1", passed=True)])
    q = data["m"][0]["questions"][0]
    assert q["question_html"] == ""
    assert q["passed"] is True


# --- combined report ---------------------------------------------------------

def test_combined_report_embeds_html_and_costs():
    gens = [_vgen("kimi", "g1")]
    opts = [{"generation_id": "g1", "opt_cost_usd": 0.05}]
    tss = [{**_vts("g1"), "ts_cost_usd": 0.02}]
    jdgs = [{**_vjdg("g1"), "maths_judge_cost_usd": 0.03, "style_judge_cost_usd": 0.04}]
    html = build_combined_report("MYRUN", gens, opts, tss, jdgs)
    assert "MYRUN" in html
    assert "kimi" in html
    assert "Q0 html" in html          # typeset HTML embedded
    assert "Optimise" in html         # four-line cost breakdown
    assert "Typeset" in html


def test_combined_report_has_katex():
    html = build_combined_report("R", [_vgen("m", "g1")], [], [_vts("g1")], [_vjdg("g1")])
    assert "katex" in html.lower()
