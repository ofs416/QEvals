import html as _html
import json
import statistics
from collections import defaultdict
from tabulate import tabulate
from jinja2 import Template

from config import SUITABILITY_THRESHOLD


def _stage_cost(records: list[dict], *fields: str) -> float:
    """Sum the named cost fields across a stage's records (missing -> 0)."""
    return sum(sum(r.get(f, 0) for f in fields) for r in records)


def _q_entries(judgement: dict) -> list[dict]:
    """Per-question judge entries carried by a judgement, [] for a failed batch."""
    return judgement.get("questions") or []


def join_records(generations: list[dict], judgements: list[dict]) -> list[dict]:
    jmap = {j["generation_id"]: j for j in judgements}
    rows = []
    for gen in generations:
        row = dict(gen)
        row["_judgement"] = jmap.get(gen["generation_id"], {"questions": [], "flags": ["no_judgement"]})
        rows.append(row)
    return rows


def assemble_viewer_data(
    generations: list[dict], typesettings: list[dict], judgements: list[dict]
) -> dict:
    """Per-model batches for the HTML viewer. Each question pairs the typesetter's
    HTML with its panel verdict, keyed by the judge's question_index (the scoring
    source of truth); HTML is pulled from the matching typeset entry by index."""
    tmap = {t["generation_id"]: t for t in typesettings}
    jmap = {j["generation_id"]: j for j in judgements}
    by_model: dict[str, list] = defaultdict(list)

    for gen in generations:
        gid = gen["generation_id"]
        jdg = jmap.get(gid, {})
        ts_questions = (tmap.get(gid, {}) or {}).get("questions", [])

        merged = []
        for qj in _q_entries(jdg):
            i = qj.get("question_index", len(merged))
            tq = ts_questions[i] if isinstance(i, int) and i < len(ts_questions) else {}
            merged.append({
                "question_html": tq.get("question_html", ""),
                "mark_scheme_html": tq.get("mark_scheme_html", ""),
                "commandWord": tq.get("commandWord", ""),
                "marks": tq.get("marks"),
                "difficulty": tq.get("difficulty", ""),
                "maths_correct": bool(qj.get("maths_correct")),
                "suitability_total": qj.get("suitability_total", 0),
                "passed": bool(qj.get("passed")),
                "flags": qj.get("flags", []),
                "notes": qj.get("notes", {}),
                "maths_note": qj.get("maths_note", ""),
            })

        totals = [m["suitability_total"] for m in merged]
        by_model[gen["model_short"]].append({
            "topic": gen.get("topic", ""),
            "board": gen.get("board", ""),
            "passed": sum(1 for m in merged if m["passed"]),
            "maths_ok": sum(1 for m in merged if m["maths_correct"]),
            "n_questions": len(merged),
            "avg_score": round(statistics.mean(totals), 1) if totals else 0,
            "batch_flags": jdg.get("flags", []),
            "questions": merged,
        })

    return {m: sorted(batches, key=lambda b: b["avg_score"]) for m, batches in by_model.items()}


def compute_model_stats(rows: list[dict]) -> dict:
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model_short"]].append(row)

    stats = {}
    for short, model_rows in by_model.items():
        q_entries = [q for r in model_rows for q in _q_entries(r["_judgement"])]
        costs = [r["cost_usd"] for r in model_rows]
        latencies = sorted(r["latency_ms"] for r in model_rows)
        json_fails = sum(1 for r in model_rows if not r["json_parse_ok"])
        raw_fails = sum(1 for r in model_rows if not r.get("json_parse_ok_raw", r["json_parse_ok"]))

        n_questions = len(q_entries)
        passed = sum(1 for q in q_entries if q.get("passed"))
        maths_ok = sum(1 for q in q_entries if q.get("maths_correct"))
        suit_totals = [q.get("suitability_total", 0) for q in q_entries]
        avg_suit = statistics.mean(suit_totals) if suit_totals else 0
        avg_cost = statistics.mean(costs)
        # Quality only: failed drafts contribute no questions, so they don't
        # enter these denominators. Their cost is penalised via cost_per_pass.
        pass_rate = passed / n_questions if n_questions else 0
        maths_rate = maths_ok / n_questions if n_questions else 0
        p50 = latencies[len(latencies) // 2]

        # Drafter (generation) cost only — optimiser/typeset/judge are fixed
        # shared overhead, not the candidate's production cost. Numerator is the
        # FULL drafter spend (including failed batches), so this is the all-in
        # price of a usable question.
        cost_per_pass = sum(costs) / passed if passed else None

        stats[short] = {
            "avg_suit": round(avg_suit, 1),
            "pass_rate": pass_rate,
            "maths_rate": maths_rate,
            "avg_cost_usd": avg_cost,
            "total_cost_usd": sum(costs),
            "cost_per_pass_usd": cost_per_pass,
            "raw_json_fail_pct": round(100 * raw_fails / len(model_rows), 1),
            "json_fail_pct": round(100 * json_fails / len(model_rows), 1),
            "p50_latency_ms": p50,
        }
    return stats


def _fmt_cost_per_pass(s: dict) -> str:
    v = s.get("cost_per_pass_usd")
    return f"${v:.5f}" if v is not None else "—"


_HEADERS = ["Model", "Avg /15", "Pass rate", "Maths OK", "Avg cost/gen", "Cost/pass Q",
            "Total cost", "Raw JSON fail%", "JSON fail%", "p50 latency"]


def format_markdown_table(stats: dict) -> str:
    rows = sorted(stats.items(), key=lambda x: -x[1]["avg_suit"])
    table = [
        [
            short,
            f"{s['avg_suit']}/15",
            f"{s['pass_rate']*100:.0f}%",
            f"{s['maths_rate']*100:.0f}%",
            f"${s['avg_cost_usd']:.5f}",
            _fmt_cost_per_pass(s),
            f"${s['total_cost_usd']:.4f}",
            f"{s['raw_json_fail_pct']:.1f}%",
            f"{s['json_fail_pct']:.1f}%",
            f"{s['p50_latency_ms']}ms",
        ]
        for short, s in rows
    ]
    return tabulate(table, headers=_HEADERS, tablefmt="github")


def compute_topic_matrix(rows: list[dict]) -> dict:
    """{topic: {model: avg per-question suitability}} across boards. A model with
    only failed batches in a topic contributes no questions and is omitted."""
    cells: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        suits = [q.get("suitability_total", 0) for q in _q_entries(r["_judgement"])]
        cells[r.get("topic", "")][r["model_short"]].extend(suits)
    matrix = {
        topic: {m: round(statistics.mean(v), 1) for m, v in models.items() if v}
        for topic, models in cells.items()
    }
    return {topic: cell for topic, cell in matrix.items() if cell}


def _topics_hardest_first(topic_matrix: dict) -> list[str]:
    return sorted(topic_matrix, key=lambda t: statistics.mean(topic_matrix[t].values()))


def format_topic_table(topic_matrix: dict, model_order: list[str]) -> str:
    table = []
    for topic in _topics_hardest_first(topic_matrix):
        row = [topic] + [
            f"{topic_matrix[topic][m]}" if m in topic_matrix[topic] else "—"
            for m in model_order
        ]
        table.append(row)
    return tabulate(table, headers=["Topic"] + model_order, tablefmt="github")


def _build_topic_matrix_html(topic_matrix: dict, model_order: list[str]) -> str:
    if not topic_matrix:
        return ""
    header_cells = "<th>Topic</th>" + "".join(f"<th>{_html.escape(m)}</th>" for m in model_order)
    body_rows = ""
    for topic in _topics_hardest_first(topic_matrix):
        cells = ""
        for m in model_order:
            v = topic_matrix[topic].get(m)
            if v is None:
                cells += "<td>—</td>"
            else:
                color = "#f59e0b" if v >= SUITABILITY_THRESHOLD else "#dc2626"
                cells += f'<td style="font-weight:600;color:{color}">{v}</td>'
        body_rows += f'<tr><td style="font-weight:500;color:#1c1917">{_html.escape(topic)}</td>{cells}</tr>'
    return (
        f'<h3 class="cost-heading">Per-topic suitability (avg /15, hardest first)</h3>'
        f'<div class="table-scroll"><table class="stats-table">'
        f'<thead><tr>{header_cells}</tr></thead><tbody>{body_rows}</tbody>'
        f'</table></div>'
    )


def _build_summary_html(stats, n_gens, costs: dict, topic_matrix=None) -> str:
    rows_sorted = sorted(stats.items(), key=lambda x: -x[1]["avg_suit"])
    total_cost = sum(costs.values())

    header_cells = "".join(f"<th>{h}</th>" for h in _HEADERS)
    body_rows = ""
    for short, s in rows_sorted:
        color = "#f59e0b" if s["avg_suit"] >= SUITABILITY_THRESHOLD else "#dc2626"
        body_rows += (
            f'<tr>'
            f'<td style="font-weight:500;color:#1c1917">{_html.escape(short)}</td>'
            f'<td style="font-weight:700;color:{color}">{s["avg_suit"]}/15</td>'
            f'<td>{s["pass_rate"]*100:.0f}%</td>'
            f'<td>{s["maths_rate"]*100:.0f}%</td>'
            f'<td>${s["avg_cost_usd"]:.5f}</td>'
            f'<td>{_fmt_cost_per_pass(s)}</td>'
            f'<td>${s["total_cost_usd"]:.4f}</td>'
            f'<td>{s["raw_json_fail_pct"]:.1f}%</td>'
            f'<td>{s["json_fail_pct"]:.1f}%</td>'
            f'<td>{s["p50_latency_ms"]}ms</td>'
            f'</tr>'
        )

    model_order = [short for short, _ in rows_sorted]
    topic_html = _build_topic_matrix_html(topic_matrix or {}, model_order)

    return (
        f'<div class="summary">'
        f'<p class="summary-meta">{n_gens} generations &nbsp;&middot;&nbsp;'
        f' Total cost: <strong>${total_cost:.2f}</strong></p>'
        f'<div class="table-scroll"><table class="stats-table">'
        f'<thead><tr>{header_cells}</tr></thead><tbody>{body_rows}</tbody>'
        f'</table></div>'
        f'{topic_html}'
        f'<h3 class="cost-heading">Cost breakdown</h3>'
        f'<table class="cost-table"><tbody>'
        f'<tr><td>Generation</td><td>${costs["generation"]:.4f}</td></tr>'
        f'<tr><td>Optimise</td><td>${costs["optimise"]:.4f}</td></tr>'
        f'<tr><td>Typeset</td><td>${costs["typeset"]:.4f}</td></tr>'
        f'<tr><td>Judging</td><td>${costs["judge"]:.4f}</td></tr>'
        f'</tbody></table>'
        f'</div>'
    )


COMBINED_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Eval Report — {{ run_id }}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=DM+Sans:opsz,wght@9..40,400;9..40,500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
          onload="initViewer()"></script>
  <style>
    :root {
      --cream: #FAF6E4;
      --forest-800: #007f8c;
      --stone-900: #1c1917;
      --gray-600: #4b5563;
      --gray-100: #f3f4f6;
      --amber-500: #f59e0b;
      --red-600: #dc2626;
      --green-700: #047857;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'DM Sans', sans-serif; background: var(--cream); color: var(--stone-900); min-height: 100vh; }
    #header {
      position: sticky; top: 0; z-index: 10;
      background: var(--cream); border-bottom: 1px solid #e5e7eb;
      padding: 0.75rem 2rem;
    }
    .header-inner {
      max-width: 900px; margin: 0 auto;
      display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;
    }
    .run-label {
      font-family: 'Playfair Display', serif; font-weight: 700;
      font-size: 1.1rem; color: var(--stone-900);
    }
    .tab-btns { display: flex; gap: 0.375rem; flex-wrap: wrap; flex: 1; }
    .tab-btn {
      font-family: 'DM Sans', sans-serif; font-size: 0.8125rem; font-weight: 500;
      padding: 0.3rem 0.75rem; border-radius: 0.375rem; border: none;
      cursor: pointer; background: var(--gray-100); color: var(--gray-600);
      transition: background 0.1s, color 0.1s;
    }
    .tab-btn.active { background: var(--amber-500); color: var(--stone-900); }
    .tab-btn.summary-btn { border: 1px solid #d1d5db; background: white; }
    .tab-btn.summary-btn.active { background: var(--forest-800); color: white; border-color: var(--forest-800); }
    .sort-btn {
      font-family: 'DM Sans', sans-serif; font-size: 0.8125rem; font-weight: 500;
      padding: 0.3rem 0.75rem; border-radius: 0.375rem;
      border: 1px solid #d1d5db; background: white;
      cursor: pointer; color: var(--gray-600); white-space: nowrap;
    }
    #content { max-width: 900px; margin: 0 auto; padding: 2rem; }

    /* Summary */
    .summary-meta { font-size: 0.9375rem; color: var(--gray-600); margin-bottom: 1.5rem; }
    .table-scroll { overflow-x: auto; margin-bottom: 2rem; }
    .stats-table { border-collapse: collapse; width: 100%; font-size: 0.875rem; white-space: nowrap; }
    .stats-table th {
      background: var(--gray-100); color: var(--gray-600);
      font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em;
      padding: 0.5rem 0.75rem; text-align: left; border-bottom: 2px solid #e5e7eb;
    }
    .stats-table td { padding: 0.5rem 0.75rem; border-bottom: 1px solid #f3f4f6; color: var(--gray-600); }
    .stats-table tbody tr:hover { background: white; }
    .cost-heading {
      font-family: 'Playfair Display', serif; font-weight: 700; font-size: 1rem;
      margin-bottom: 0.75rem; color: var(--stone-900);
    }
    .cost-table { border-collapse: collapse; }
    .cost-table td { padding: 0.3rem 1.5rem 0.3rem 0; color: var(--gray-600); font-size: 0.875rem; }
    .cost-table td:first-child { color: var(--stone-900); font-weight: 500; }

    /* Viewer */
    .batch { margin-bottom: 3rem; }
    .batch-divider { border: none; border-top: 2px solid #d1d5db; margin-bottom: 1rem; }
    .batch-header { display: flex; align-items: baseline; gap: 0.75rem; margin-bottom: 0.5rem; flex-wrap: wrap; }
    .batch-score { font-family: 'Playfair Display', serif; font-weight: 700; font-size: 1.1rem; }
    .batch-score.pass { color: var(--amber-500); }
    .batch-score.fail { color: var(--red-600); }
    .q-score-badge { font-weight: 700; font-size: 0.8125rem; }
    .q-score-badge.pass { color: var(--amber-500); }
    .q-score-badge.fail { color: var(--red-600); }
    .maths-badge { font-size: 0.6875rem; font-weight: 600; padding: 0.125rem 0.4rem; border-radius: 0.25rem; }
    .maths-ok { background: #d1fae5; color: var(--green-700); }
    .maths-bad { background: #fee2e2; color: #991b1b; }
    .batch-meta { font-size: 0.875rem; color: var(--gray-600); }
    .flag-badge {
      display: inline-block; font-size: 0.6875rem; font-weight: 500;
      padding: 0.125rem 0.5rem; border-radius: 9999px;
      text-transform: uppercase; letter-spacing: 0.05em;
    }
    .flag-unsolvable { background: #fee2e2; color: #991b1b; }
    .flag-ambiguous { background: #ede9fe; color: #5b21b6; }
    .flag-past-paper { background: #fef3c7; color: #92400e; }
    .batch-note, .q-note {
      font-size: 0.8125rem; color: var(--gray-600);
      font-style: italic; margin-bottom: 1rem; line-height: 1.5;
    }
    .question { margin-bottom: 1.5rem; }
    .question-divider { border: none; border-top: 1px solid #e5e7eb; margin: 1.25rem 0; }
    .question-meta { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; flex-wrap: wrap; }
    .q-num { font-family: 'Playfair Display', serif; font-weight: 700; font-size: 0.9375rem; }
    .marks-badge {
      font-size: 0.6875rem; font-weight: 500; padding: 0.125rem 0.4rem;
      background: var(--gray-100); color: var(--gray-600); border-radius: 0.25rem;
    }
    .difficulty-chip {
      font-size: 0.6875rem; font-weight: 500; padding: 0.125rem 0.4rem;
      border-radius: 0.25rem; text-transform: capitalize;
    }
    .difficulty-foundation { background: #d1fae5; color: #065f46; }
    .difficulty-higher { background: #dbeafe; color: #1e40af; }
    .difficulty-extension { background: #ede9fe; color: #4c1d95; }
    .cmd-word { font-size: 0.6875rem; color: var(--forest-800); font-style: italic; }
    .question-text { font-size: 1rem; line-height: 1.6; color: var(--stone-900); margin-bottom: 0.875rem; }
    .mark-scheme { border-left: 3px solid #e5e7eb; padding-left: 1rem; font-size: 0.9rem; line-height: 1.5; color: var(--gray-600); }
    .mark-scheme ol, .mark-scheme ul { padding-left: 1.25rem; }
    .empty { text-align: center; padding: 4rem 2rem; color: var(--gray-600); font-style: italic; }
    .katex { font-size: 1em !important; }
  </style>
</head>
<body>

<div id="header">
  <div class="header-inner">
    <span class="run-label">{{ run_id }}</span>
    <div class="tab-btns" id="tab-btns"></div>
    <button class="sort-btn" id="sort-btn" onclick="toggleSort()">&#8593; Score</button>
  </div>
</div>

<div id="content"></div>

<script>
const DATA = {{ data_json }};
const MODELS = {{ models_json }};
const SUMMARY_HTML = {{ summary_html_json }};
const SUIT_THRESHOLD = {{ suit_threshold }};

let currentView = '__summary__';
let sortAsc = true;

function initViewer() {
  renderTabBtns();
  renderContent();
}

function renderTabBtns() {
  const c = document.getElementById('tab-btns');
  const isSummary = currentView === '__summary__';
  document.getElementById('sort-btn').style.display = isSummary ? 'none' : '';

  let html = `<button class="tab-btn summary-btn${isSummary ? ' active' : ''}" onclick="selectView('__summary__')">Summary</button>`;
  html += MODELS.map(m =>
    `<button class="tab-btn${m === currentView ? ' active' : ''}" onclick="selectView('${m}')">${m}</button>`
  ).join('');
  c.innerHTML = html;
}

function selectView(v) {
  currentView = v;
  renderTabBtns();
  renderContent();
}

function toggleSort() {
  sortAsc = !sortAsc;
  document.getElementById('sort-btn').textContent = (sortAsc ? '\\u2191' : '\\u2193') + ' Score';
  renderContent();
}

function diffClass(d) {
  return {foundation: 'difficulty-foundation', higher: 'difficulty-higher', extension: 'difficulty-extension'}[d] || '';
}

function flagHtml(flag) {
  if (flag === 'ambiguous_question') return '<span class="flag-badge flag-ambiguous">ambiguous</span>';
  if (flag === 'past_paper_suspected') return '<span class="flag-badge flag-past-paper">past paper</span>';
  return '<span class="flag-badge flag-unsolvable">' + flag + '</span>';
}

function e(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function renderContent() {
  if (currentView === '__summary__') {
    document.getElementById('content').innerHTML = SUMMARY_HTML;
    return;
  }

  const batches = (DATA[currentView] || []).slice().sort((a, b) =>
    sortAsc ? a.avg_score - b.avg_score : b.avg_score - a.avg_score
  );

  if (!batches.length) {
    document.getElementById('content').innerHTML = '<p class="empty">No data for this model.</p>';
    return;
  }

  const html = batches.map(function(batch) {
    const flagsHtml = (batch.batch_flags || []).map(flagHtml).join(' ');
    const batchScoreClass = batch.n_questions && batch.passed === batch.n_questions
      ? 'pass' : (batch.passed === 0 ? 'fail' : '');

    const questionsHtml = (batch.questions || []).map(function(q, qi) {
      const qPassClass = q.passed ? 'pass' : 'fail';
      const mathsBadge = q.maths_correct
        ? '<span class="maths-badge maths-ok">&#10003; maths</span>'
        : '<span class="maths-badge maths-bad">&#10007; maths</span>';
      const qFlags = (q.flags || []).map(flagHtml).join(' ');
      const styleNote = (q.notes && q.notes.style)
        ? '<p class="q-note">' + e(q.notes.style) + '</p>' : '';
      const mathsNote = (!q.maths_correct && q.maths_note)
        ? '<p class="q-note">' + e(q.maths_note) + '</p>' : '';

      return (qi > 0 ? '<hr class="question-divider">' : '') +
        '<div class="question">' +
        '<div class="question-meta">' +
          '<span class="q-num">Q' + (qi + 1) + '</span>' +
          '<span class="q-score-badge ' + qPassClass + '">' + (q.passed ? 'PASS' : 'FAIL') + '</span>' +
          mathsBadge +
          '<span class="marks-badge">' + e(q.suitability_total) + '/15 suit</span>' +
          (q.marks != null ? '<span class="marks-badge">' + e(q.marks) + 'm</span>' : '') +
          '<span class="difficulty-chip ' + diffClass(q.difficulty) + '">' + e(q.difficulty || '') + '</span>' +
          '<span class="cmd-word">' + e(q.commandWord || '') + '</span>' +
          qFlags +
        '</div>' +
        '<div class="question-text">' + (q.question_html || '<em>(no typeset output)</em>') + '</div>' +
        mathsNote + styleNote +
        '<div class="mark-scheme">' + (q.mark_scheme_html || '') + '</div>' +
        '</div>';
    }).join('');

    return '<div class="batch">' +
      '<hr class="batch-divider">' +
      '<div class="batch-header">' +
        '<span class="batch-score ' + batchScoreClass + '">' + batch.passed + '/' + batch.n_questions + ' passed</span>' +
        '<span class="batch-meta">' + e(batch.topic) + ' &middot; ' + e(batch.board) + ' &middot; ' + batch.maths_ok + ' maths ok</span>' +
        flagsHtml +
      '</div>' +
      questionsHtml +
      '</div>';
  }).join('');

  document.getElementById('content').innerHTML = html;

  renderMathInElement(document.getElementById('content'), {
    delimiters: [
      {left: '$$', right: '$$', display: true},
      {left: '$',  right: '$',  display: false},
    ],
    throwOnError: false,
  });
}
</script>
</body>
</html>"""


def build_combined_report(
    run_id: str,
    generations: list[dict],
    optimisations: list[dict],
    typesettings: list[dict],
    judgements: list[dict],
    stats: dict | None = None,
) -> str:
    rows = join_records(generations, judgements)
    if stats is None:
        stats = compute_model_stats(rows)
    topic_matrix = compute_topic_matrix(rows)

    data = assemble_viewer_data(generations, typesettings, judgements)
    models_sorted = sorted(
        data.keys(),
        key=lambda m: -statistics.mean(b["avg_score"] for b in data[m]) if data[m] else 0,
    )

    costs = {
        "generation": _stage_cost(generations, "cost_usd"),
        "optimise": _stage_cost(optimisations, "opt_cost_usd"),
        "typeset": _stage_cost(typesettings, "ts_cost_usd"),
        "judge": _stage_cost(judgements, "maths_judge_cost_usd", "style_judge_cost_usd"),
    }
    summary_html = _build_summary_html(stats, len(generations), costs, topic_matrix)

    return Template(COMBINED_TEMPLATE).render(
        run_id=_html.escape(run_id),
        suit_threshold=SUITABILITY_THRESHOLD,
        data_json=json.dumps(data, ensure_ascii=False),
        models_json=json.dumps(models_sorted),
        summary_html_json=json.dumps(summary_html),
    )


def generate_report(
    run_id: str,
    generations: list[dict],
    optimisations: list[dict],
    typesettings: list[dict],
    judgements: list[dict],
    md_path: str,
    html_path: str,
) -> None:
    rows = join_records(generations, judgements)
    stats = compute_model_stats(rows)
    topic_matrix = compute_topic_matrix(rows)
    model_order = [short for short, _ in sorted(stats.items(), key=lambda x: -x[1]["avg_suit"])]

    costs = {
        "generation": _stage_cost(generations, "cost_usd"),
        "optimise": _stage_cost(optimisations, "opt_cost_usd"),
        "typeset": _stage_cost(typesettings, "ts_cost_usd"),
        "judge": _stage_cost(judgements, "maths_judge_cost_usd", "style_judge_cost_usd"),
    }

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Eval Report — {run_id}\n\n")
        f.write(format_markdown_table(stats))
        f.write("\n\n## Per-topic suitability (avg /15 across boards, hardest first)\n\n")
        f.write(format_topic_table(topic_matrix, model_order))
        f.write(
            f"\n\n**Total cost:** generation ${costs['generation']:.4f} | "
            f"optimise ${costs['optimise']:.4f} | typeset ${costs['typeset']:.4f} | "
            f"judging ${costs['judge']:.4f}\n"
        )

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_combined_report(run_id, generations, optimisations, typesettings, judgements, stats=stats))
