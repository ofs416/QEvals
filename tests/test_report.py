import pytest
from report import compute_model_stats, format_markdown_table, PASS_THRESHOLD
from report import assemble_viewer_data, build_combined_report
from report import compute_topic_matrix, format_topic_table


def _gen(short, score, cost, parse_ok=True, flags=None, n_questions=1):
    return {
        "model_short": short,
        "json_parse_ok": parse_ok,
        "cost_usd": cost,
        "latency_ms": 1000,
        "_judgement": {
            "questions": [
                {"question_index": i, "total": score, "scores": {}, "flags": []}
                for i in range(n_questions)
            ],
            "flags": flags or [],
        },
    }


def test_compute_stats_avg_score():
    rows = [_gen("haiku", 20, 0.006), _gen("haiku", 22, 0.007)]
    stats = compute_model_stats(rows)
    assert stats["haiku"]["avg_score"] == 21.0


def test_compute_stats_pass_rate():
    rows = [_gen("haiku", 20, 0.006), _gen("haiku", 16, 0.007)]
    stats = compute_model_stats(rows)
    assert stats["haiku"]["pass_rate"] == 0.5


def test_compute_stats_score_per_dollar():
    rows = [_gen("haiku", 20, 0.01)]
    stats = compute_model_stats(rows)
    assert stats["haiku"]["score_per_dollar"] == pytest.approx(2000.0)


def test_compute_stats_total_cost_per_model():
    rows = [_gen("haiku", 20, 0.006), _gen("haiku", 22, 0.007), _gen("qwen", 19, 0.001)]
    stats = compute_model_stats(rows)
    assert stats["haiku"]["total_cost_usd"] == pytest.approx(0.013)
    assert stats["qwen"]["total_cost_usd"] == pytest.approx(0.001)


def test_compute_stats_json_fail_rate():
    rows = [_gen("haiku", 0, 0.005, parse_ok=False), _gen("haiku", 20, 0.005)]
    stats = compute_model_stats(rows)
    assert stats["haiku"]["json_fail_pct"] == 50.0


def test_raw_json_fail_counts_cleaned_rows():
    # row rescued by the cleaner: final parse OK, raw parse failed
    rescued = _gen("haiku", 20, 0.005)
    rescued["json_parse_ok_raw"] = False
    rows = [rescued, _gen("haiku", 22, 0.005)]
    stats = compute_model_stats(rows)
    assert stats["haiku"]["raw_json_fail_pct"] == 50.0
    assert stats["haiku"]["json_fail_pct"] == 0.0


def test_raw_json_fail_falls_back_for_old_records():
    # pre-cleaner records have no json_parse_ok_raw field
    rows = [_gen("haiku", 0, 0.005, parse_ok=False), _gen("haiku", 20, 0.005)]
    stats = compute_model_stats(rows)
    assert stats["haiku"]["raw_json_fail_pct"] == 50.0


def test_pass_threshold_is_18():
    assert PASS_THRESHOLD == 18


# ── cost per passing question ────────────────────────────────────────────────

def _costed_gen(short, gid, score, topic="Algebra", board="AQA",
                n_questions=2, cost=0.01, judge_cost=0.005):
    row = _gen(short, score, cost, n_questions=n_questions)
    row["generation_id"] = gid
    row["topic"] = topic
    row["board"] = board
    row["output_parsed"] = {"questions": [{"text": f"Q{i}"} for i in range(n_questions)]}
    row["_judgement"]["judge_cost_usd"] = judge_cost
    return row


def test_cost_per_pass_uses_generation_cost_only():
    # Generation cost includes the failed batch (g2) but excludes solver/judge
    # overhead entirely. gen total 0.01+0.01 = 0.02 over g1's 2 passing questions.
    rows = [_costed_gen("m", "g1", 20), _costed_gen("m", "g2", 10)]
    stats = compute_model_stats(rows)
    assert stats["m"]["cost_per_pass_usd"] == pytest.approx(0.01)


def test_cost_per_pass_ignores_solver_and_judge_costs():
    # A huge judge cost on the record must not inflate the generator's price —
    # only generation cost_usd counts.
    rows = [_costed_gen("m", "g1", 20, n_questions=2, cost=0.01, judge_cost=5.0)]
    stats = compute_model_stats(rows)
    assert stats["m"]["cost_per_pass_usd"] == pytest.approx(0.005)


def test_cost_per_pass_is_none_when_nothing_passes():
    rows = [_costed_gen("m", "g1", 10)]
    stats = compute_model_stats(rows)
    assert stats["m"]["cost_per_pass_usd"] is None
    assert "—" in format_markdown_table(stats)


def test_cost_per_pass_in_markdown_table():
    rows = [_costed_gen("m", "g1", 20)]
    stats = compute_model_stats(rows)
    md = format_markdown_table(stats)
    assert "Cost/pass Q" in md
    # generation cost 0.01 over 2 passing questions
    assert "$0.00500" in md


def test_pass_rate_excludes_failed_generations():
    # g1: 2 questions both pass; g2: parse failure (no questions). Pass rate is
    # quality-only: 2/2 = 1.0, the failure does not enter the denominator.
    good = _costed_gen("m", "g1", 20, n_questions=2)
    failed = _gen("m", 0, 0.01, parse_ok=False, n_questions=0)
    failed["generation_id"] = "g2"
    stats = compute_model_stats([good, failed])
    assert stats["m"]["pass_rate"] == 1.0


def test_cost_per_pass_includes_failed_generation_cost():
    # g1 passes (2 q, $0.01); g2 fails producing nothing but still cost $0.03.
    # Cost/pass Q = (0.01 + 0.03) / 2 = 0.02 — failures penalise the price.
    good = _costed_gen("m", "g1", 20, n_questions=2, cost=0.01)
    failed = _gen("m", 0, 0.03, parse_ok=False, n_questions=0)
    failed["generation_id"] = "g2"
    stats = compute_model_stats([good, failed])
    assert stats["m"]["cost_per_pass_usd"] == pytest.approx(0.02)


# ── per-topic matrix ─────────────────────────────────────────────────────────

def test_topic_matrix_averages_across_boards():
    rows = [_costed_gen("m", "g1", 20, topic="Vectors", board="AQA"),
            _costed_gen("m", "g2", 10, topic="Vectors", board="Edexcel"),
            _costed_gen("m", "g3", 22, topic="Algebra", board="AQA")]
    matrix = compute_topic_matrix(rows)
    assert matrix["Vectors"]["m"] == 15.0
    assert matrix["Algebra"]["m"] == 22.0


def test_topic_matrix_omits_model_with_only_failed_generations():
    # A model whose only generation in a topic failed contributes no questions,
    # so it is omitted from that topic's cell; a topic with no data drops out.
    failed = _gen("bad", 0, 0.01, parse_ok=False, n_questions=0)
    failed["generation_id"] = "g1"
    failed["topic"] = "Vectors"
    assert compute_topic_matrix([failed]) == {}


def test_topic_matrix_keeps_passing_model_drops_failed_in_same_topic():
    good = _costed_gen("good", "g1", 20, topic="Vectors", n_questions=2)
    failed = _gen("bad", 0, 0.01, parse_ok=False, n_questions=0)
    failed["generation_id"] = "g2"
    failed["topic"] = "Vectors"
    matrix = compute_topic_matrix([good, failed])
    assert matrix["Vectors"]["good"] == 20.0
    assert "bad" not in matrix["Vectors"]


def test_topic_table_lists_hardest_topic_first():
    rows = [_costed_gen("m", "g1", 22, topic="Algebra"),
            _costed_gen("m", "g2", 10, topic="Vectors")]
    matrix = compute_topic_matrix(rows)
    md = format_topic_table(matrix, ["m"])
    assert md.index("Vectors") < md.index("Algebra")


def test_topic_table_shows_dash_for_missing_model():
    matrix = {"Algebra": {"m1": 20.0}}
    md = format_topic_table(matrix, ["m1", "m2"])
    assert "—" in md


def test_markdown_table_contains_model_names():
    stats = {
        "haiku": {"avg_score": 20.5, "pass_rate": 0.85, "avg_cost_usd": 0.006,
                  "total_cost_usd": 0.168, "score_per_dollar": 3416,
                  "raw_json_fail_pct": 0.0, "json_fail_pct": 0.0,
                  "p50_latency_ms": 900},
    }
    md = format_markdown_table(stats)
    assert "haiku" in md
    assert "20.5" in md
    assert "85%" in md



# ── Fixtures ─────────────────────────────────────────────────────────────────

def _viewer_gen(short, topic, gid, board="AQA", questions=None):
    return {
        "generation_id": gid,
        "model_short": short,
        "topic": topic,
        "board": board,
        "output_parsed": {
            "questions": questions or [
                {
                    "text": "Find $x$ where $x^2 = 4$.",
                    "marks": 2,
                    "commandWord": "Find",
                    "difficulty": "foundation",
                    "markScheme": [
                        {"tag": "M1", "text": "Sets up $x^2 = 4$."},
                        {"tag": "A1", "text": "$x = \\pm 2$."},
                    ],
                }
            ]
        },
        "json_parse_ok": True,
        "cost_usd": 0.01,
        "latency_ms": 1000,
    }


def _viewer_jdg(gid, score, flags=None, notes=None):
    return {
        "generation_id": gid,
        "questions": [
            {"question_index": 0,
             "scores": {"correctness": 4, "mark_scheme": 4,
                        "command_word": 4, "difficulty": 4, "style": 4},
             "total": score,
             "flags": flags or [],
             "notes": notes or {}},
        ],
        "flags": [],
    }


# ── assemble_viewer_data ──────────────────────────────────────────────────────

def test_assemble_viewer_data_groups_by_model():
    gens = [_viewer_gen("gpt-mini", "Algebra", "g1"),
            _viewer_gen("kimi",     "Vectors", "g2")]
    jdgs = [_viewer_jdg("g1", 12), _viewer_jdg("g2", 22)]
    data = assemble_viewer_data(gens, jdgs)
    assert "gpt-mini" in data
    assert "kimi" in data
    assert len(data["gpt-mini"]) == 1
    assert len(data["kimi"]) == 1


def test_assemble_viewer_data_sorts_batches_ascending():
    gens = [_viewer_gen("m", "T1", "g1"), _viewer_gen("m", "T2", "g2"), _viewer_gen("m", "T3", "g3")]
    jdgs = [_viewer_jdg("g1", 20), _viewer_jdg("g2", 10), _viewer_jdg("g3", 15)]
    data = assemble_viewer_data(gens, jdgs)
    scores = [b["avg_score"] for b in data["m"]]
    assert scores == [10, 15, 20]


def test_assemble_viewer_data_per_question_score_and_pass():
    gens = [_viewer_gen("m", "T", "g1")]
    jdgs = [_viewer_jdg("g1", 22)]
    data = assemble_viewer_data(gens, jdgs)
    batch = data["m"][0]
    assert batch["passed"] == 1
    assert batch["n_questions"] == 1
    q = batch["questions"][0]
    assert q["score"] == 22
    assert q["passed"] is True


def test_assemble_viewer_data_per_question_flags_and_notes():
    gens = [_viewer_gen("m", "T", "g1")]
    jdgs = [_viewer_jdg("g1", 8, flags=["unsolvable_question"],
                         notes={"correctness": "Mark scheme is wrong."})]
    data = assemble_viewer_data(gens, jdgs)
    q = data["m"][0]["questions"][0]
    assert "unsolvable_question" in q["flags"]
    assert q["notes"]["correctness"] == "Mark scheme is wrong."
    assert q["passed"] is False


def test_assemble_viewer_data_includes_questions():
    gens = [_viewer_gen("m", "T", "g1")]
    jdgs = [_viewer_jdg("g1", 18)]
    data = assemble_viewer_data(gens, jdgs)
    assert len(data["m"][0]["questions"]) == 1
    assert data["m"][0]["questions"][0]["text"] == "Find $x$ where $x^2 = 4$."


def test_assemble_viewer_data_missing_judgement_gives_zero_score():
    gens = [_viewer_gen("m", "T", "g1")]
    data = assemble_viewer_data(gens, [])  # no judgements
    q = data["m"][0]["questions"][0]
    assert q["score"] == 0
    assert q["passed"] is False


def test_assemble_viewer_data_failed_generation_has_no_questions():
    gen = _viewer_gen("m", "T", "g1")
    gen["output_parsed"] = None
    data = assemble_viewer_data([gen], [{"generation_id": "g1", "questions": [],
                                         "flags": ["json_parse_failure"]}])
    batch = data["m"][0]
    assert batch["questions"] == []
    assert batch["n_questions"] == 0
    assert "json_parse_failure" in batch["batch_flags"]


def test_assemble_viewer_data_multi_question_mixed_pass_fail():
    gen = _viewer_gen("m", "T", "g1", questions=[
        {"text": "Q1", "marks": 2, "markScheme": []},
        {"text": "Q2", "marks": 3, "markScheme": []},
    ])
    jdg = {
        "generation_id": "g1",
        "questions": [
            {"question_index": 0, "scores": {}, "total": 22, "flags": [], "notes": {}},
            {"question_index": 1, "scores": {}, "total": 10, "flags": [], "notes": {}},
        ],
        "flags": [],
    }
    data = assemble_viewer_data([gen], [jdg])
    batch = data["m"][0]
    assert batch["n_questions"] == 2
    assert batch["passed"] == 1            # only q0 (22) passes
    assert batch["avg_score"] == 16.0      # mean(22, 10)
    assert batch["questions"][0]["passed"] is True
    assert batch["questions"][1]["passed"] is False


# ── build_combined_report ────────────────────────────────────────────────────

def test_viewer_html_contains_model_name():
    gens = [_viewer_gen("kimi", "Algebra", "g1")]
    jdgs = [_viewer_jdg("g1", 22)]
    html = build_combined_report("TESTRUN", gens, jdgs)
    assert "kimi" in html


def test_viewer_html_contains_run_id():
    gens = [_viewer_gen("kimi", "Algebra", "g1")]
    jdgs = [_viewer_jdg("g1", 22)]
    html = build_combined_report("MYRUN", gens, jdgs)
    assert "MYRUN" in html


def test_viewer_html_contains_embedded_json():
    gens = [_viewer_gen("kimi", "Algebra", "g1")]
    jdgs = [_viewer_jdg("g1", 22)]
    html = build_combined_report("R", gens, jdgs)
    assert "const DATA" in html


def test_viewer_html_contains_question_text():
    gens = [_viewer_gen("kimi", "Algebra", "g1")]
    jdgs = [_viewer_jdg("g1", 22)]
    html = build_combined_report("R", gens, jdgs)
    assert "Find $x$ where $x^2 = 4$." in html


def test_viewer_html_contains_flag():
    gens = [_viewer_gen("m", "T", "g1")]
    jdgs = [_viewer_jdg("g1", 8, flags=["unsolvable_question"])]
    html = build_combined_report("R", gens, jdgs)
    assert "unsolvable_question" in html


def test_viewer_html_contains_correctness_note():
    gens = [_viewer_gen("m", "T", "g1")]
    jdgs = [_viewer_jdg("g1", 8, notes={"correctness": "The mark scheme is incorrect."})]
    html = build_combined_report("R", gens, jdgs)
    assert "The mark scheme is incorrect." in html


def test_viewer_html_shows_per_question_score():
    gens = [_viewer_gen("m", "T", "g1")]
    jdgs = [_viewer_jdg("g1", 22)]
    html = build_combined_report("R", gens, jdgs)
    # the per-question score (22) must be embedded in the viewer data
    assert '"score": 22' in html or '"score":22' in html


def test_viewer_html_renders_per_question_score_badge():
    gens = [_viewer_gen("m", "T", "g1")]
    jdgs = [_viewer_jdg("g1", 22)]
    html = build_combined_report("R", gens, jdgs)
    # the JS must render each question's own /25 score badge
    assert "q.score" in html


def test_viewer_html_shows_passed_summary_markup():
    gens = [_viewer_gen("m", "T", "g1")]
    jdgs = [_viewer_jdg("g1", 22)]
    html = build_combined_report("R", gens, jdgs)
    assert "passed" in html  # batch header renders "X/N passed"


def test_viewer_html_contains_katex_cdn():
    gens = [_viewer_gen("m", "T", "g1")]
    jdgs = [_viewer_jdg("g1", 18)]
    html = build_combined_report("R", gens, jdgs)
    assert "katex" in html.lower()


def test_solve_cost_includes_reconciliation_cost():
    gens = [_viewer_gen("m", "T", "g1")]
    jdgs = [_viewer_jdg("g1", 20)]
    sols = [{"generation_id": "g1", "solver_cost_usd": 0.01, "reconcile_cost_usd": 0.02}]
    html = build_combined_report("R", gens, jdgs, solutions=sols)
    assert "$0.0300" in html  # solving line = blind solve + reconciliation


def test_solve_cost_tolerates_records_without_reconcile_field():
    # pre-reconciliation runs have no reconcile_cost_usd
    gens = [_viewer_gen("m", "T", "g1")]
    jdgs = [_viewer_jdg("g1", 20)]
    sols = [{"generation_id": "g1", "solver_cost_usd": 0.01}]
    html = build_combined_report("R", gens, jdgs, solutions=sols)
    assert "$0.0100" in html


def test_combined_report_contains_summary_table():
    gens = [_viewer_gen("kimi", "Algebra", "g1")]
    jdgs = [_viewer_jdg("g1", 22)]
    html = build_combined_report("R", gens, jdgs)
    assert "SUMMARY_HTML" in html
    assert "stats-table" in html


def test_combined_report_contains_topic_matrix():
    gens = [_viewer_gen("kimi", "Algebra", "g1"),
            _viewer_gen("kimi", "Vectors", "g2")]
    jdgs = [_viewer_jdg("g1", 22), _viewer_jdg("g2", 10)]
    html = build_combined_report("R", gens, jdgs)
    assert "Per-topic scores" in html
    assert "Vectors" in html


# ── prompt_variant grouping ───────────────────────────────────────────────────

def test_stats_group_by_prompt_variant():
    a = _gen("gpt", 20, 0.01); a["prompt_variant"] = "baseline"
    b = _gen("gpt", 10, 0.01); b["prompt_variant"] = "no_guards"
    stats = compute_model_stats([a, b])
    assert set(stats) == {"baseline", "no_guards"}
    assert stats["baseline"]["avg_score"] == 20.0
    assert stats["no_guards"]["avg_score"] == 10.0


def test_stats_fall_back_to_model_short_without_variant():
    stats = compute_model_stats([_gen("gpt", 20, 0.01)])
    assert set(stats) == {"gpt"}


def test_topic_matrix_groups_by_prompt_variant():
    a = _costed_gen("gpt", "g1", 20, topic="Algebra"); a["prompt_variant"] = "baseline"
    b = _costed_gen("gpt", "g2", 10, topic="Algebra"); b["prompt_variant"] = "no_guards"
    matrix = compute_topic_matrix([a, b])
    assert matrix["Algebra"]["baseline"] == 20.0
    assert matrix["Algebra"]["no_guards"] == 10.0


def test_viewer_data_groups_by_prompt_variant():
    g1 = _viewer_gen("gpt", "T", "g1"); g1["prompt_variant"] = "baseline"
    g2 = _viewer_gen("gpt", "T", "g2"); g2["prompt_variant"] = "no_guards"
    data = assemble_viewer_data([g1, g2], [_viewer_jdg("g1", 20), _viewer_jdg("g2", 10)])
    assert set(data) == {"baseline", "no_guards"}
    assert data["baseline"][0]["avg_score"] == 20.0
    assert data["no_guards"][0]["avg_score"] == 10.0
