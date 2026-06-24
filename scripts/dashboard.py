#!/usr/bin/env python3
"""Mastermind Experiment Dashboard — minimal HTTP server with auto-refresh."""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

TRAIN_RUN_ID = "codex-gateway-train-seq-self-100-20260531T1640Z"
TRAIN_LOG = RUNS / "codex_cybergym/sequential_self_logs" / f"{TRAIN_RUN_ID}.log"
TRAIN_OUTPUT = RUNS / "codex_gateway_train_seq_self_involving_train100_rollouts.jsonl"

BO8_FILES = {
    "Rep 1": RUNS / "codex_gateway_eval100_clean_rollouts.jsonl",
    **{
        f"Rep {i}": RUNS / f"codex_gateway_eval_bo4_rep{i}_rollouts.jsonl"
        for i in range(2, 9)
    },
}
LEVEL3_FILE = RUNS / "codex_gateway_eval_level3_rep1_rollouts.jsonl"
ITERATIVE_FILE = RUNS / "codex_gateway_iterative_improvement_experiment_rollouts.jsonl"


# ── data loading ──────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def get_milestone(row: dict):
    m = row.get("milestone")
    return m.get("milestone") if isinstance(m, dict) else m


def latest_per_task(rows: list[dict]) -> dict[str, dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        tid = r.get("task_id")
        if tid:
            by_task[tid].append(r)
    latest = {}
    for tid, task_rows in by_task.items():
        task_rows.sort(key=lambda r: r.get("metadata", {}).get("attempt_index") or 0)
        latest[tid] = task_rows[-1]
    return latest


def bo8_stats() -> dict:
    per_rep: list[dict] = []
    task_milestone_by_rep: dict[str, dict[str, int]] = defaultdict(dict)

    for rep_name, path in BO8_FILES.items():
        rows = read_jsonl(path)
        total = len(rows)
        solved = sum(1 for r in rows if get_milestone(r) == 7)
        milestone_dist = Counter(get_milestone(r) for r in rows)
        per_rep.append({
            "name": rep_name,
            "total": total,
            "solved": solved,
            "rate": solved / total if total else 0,
            "milestones": dict(milestone_dist),
        })
        for r in rows:
            tid = r.get("task_id")
            if tid:
                task_milestone_by_rep[tid][rep_name] = get_milestone(r)

    all_tasks = set(task_milestone_by_rep)
    bo8_solved = sum(
        1 for ms in task_milestone_by_rep.values() if any(m == 7 for m in ms.values())
    )
    return {
        "per_rep": per_rep,
        "total_tasks": len(all_tasks),
        "bo8_solved": bo8_solved,
        "bo8_rate": bo8_solved / len(all_tasks) if all_tasks else 0,
    }


def level3_stats() -> dict:
    rows = read_jsonl(LEVEL3_FILE)
    total = len(rows)
    solved = sum(1 for r in rows if get_milestone(r) == 7)
    milestone_dist = Counter(get_milestone(r) for r in rows)
    return {
        "total": total,
        "solved": solved,
        "rate": solved / total if total else 0,
        "milestones": dict(milestone_dist),
    }


def iterative_stats() -> dict:
    rows = read_jsonl(ITERATIVE_FILE)
    latest = latest_per_task(rows)
    total = len(latest)
    solved = sum(1 for r in latest.values() if get_milestone(r) == 7)
    milestone_dist = Counter(get_milestone(r) for r in latest.values())

    # attempt distribution
    attempt_counts: Counter = Counter()
    for task_rows_list in defaultdict(list, {}).items():
        pass
    by_task: dict[str, list] = defaultdict(list)
    for r in rows:
        tid = r.get("task_id")
        if tid:
            by_task[tid].append(r)
    attempt_dist = Counter(len(v) for v in by_task.values())

    # experience update status
    update_statuses = Counter()
    for r in rows:
        us = (r.get("metadata") or {}).get("sequential", {})
        if us:
            eu = us.get("experience_update") or {}
            update_statuses[eu.get("update_status", "unknown")] += 1

    return {
        "total_tasks": total,
        "total_rows": len(rows),
        "solved": solved,
        "rate": solved / total if total else 0,
        "milestones": dict(milestone_dist),
        "attempt_dist": dict(sorted(attempt_dist.items())),
        "update_statuses": dict(update_statuses),
    }


def training_stats() -> dict:
    rows = [r for r in read_jsonl(TRAIN_OUTPUT) if r.get("run_id") == TRAIN_RUN_ID]
    latest = latest_per_task(rows)
    total_completed = len(latest)
    solved = sum(1 for r in latest.values() if get_milestone(r) == 7)
    milestone_dist = Counter(get_milestone(r) for r in latest.values())

    # Parse log for in-flight attempts
    in_flight: dict[str, int] = {}
    agent_errors = 0
    if TRAIN_LOG.exists():
        with TRAIN_LOG.open(errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = ev.get("event", "")
                tid = ev.get("task_id")
                if etype == "sequential_attempt_start" and tid:
                    in_flight[tid] = ev.get("attempt", 1)
                elif etype in ("sequential_task_done", "sequential_attempt_done") and tid:
                    in_flight.pop(tid, None)
                    if etype == "sequential_task_done" and ev.get("status") in {"AGENT_ERROR", "CRASH"}:
                        agent_errors += 1

    # Last update time
    last_mtime = TRAIN_LOG.stat().st_mtime if TRAIN_LOG.exists() else None

    return {
        "run_id": TRAIN_RUN_ID,
        "total_target": 100,
        "completed": total_completed,
        "solved": solved,
        "agent_errors": agent_errors,
        "rate": solved / total_completed if total_completed else 0,
        "milestones": dict(milestone_dist),
        "in_flight": in_flight,
        "last_update": last_mtime,
    }


# ── HTML rendering ────────────────────────────────────────────────────────────

MILESTONE_LABELS = {0: "No PoC", 3: "Submitted", 4: "Ran / No Crash", 6: "Partial", 7: "✓ Solved"}
MILESTONE_COLORS = {0: "#555", 3: "#888", 4: "#e6a817", 6: "#4a9eff", 7: "#3dba6f"}


def milestone_bar(milestones: dict, total: int) -> str:
    if not total:
        return ""
    segments = []
    for m, color in sorted(MILESTONE_COLORS.items()):
        count = milestones.get(m, 0)
        if count == 0:
            continue
        pct = 100 * count / total
        label = MILESTONE_LABELS.get(m, str(m))
        segments.append(
            f'<div class="ms-seg" style="width:{pct:.1f}%;background:{color}" '
            f'title="{label}: {count} ({pct:.1f}%)">'
            f'<span class="ms-label">{count}</span></div>'
        )
    return f'<div class="ms-bar">{"".join(segments)}</div>'


def pct_ring(rate: float, color: str = "#3dba6f") -> str:
    pct = int(rate * 100)
    r = 28
    circ = 2 * 3.14159 * r
    dash = circ * rate
    return (
        f'<svg class="ring" viewBox="0 0 70 70">'
        f'<circle cx="35" cy="35" r="{r}" fill="none" stroke="#2a2a2a" stroke-width="7"/>'
        f'<circle cx="35" cy="35" r="{r}" fill="none" stroke="{color}" stroke-width="7" '
        f'stroke-dasharray="{dash:.1f} {circ:.1f}" stroke-dashoffset="{circ/4:.1f}" '
        f'stroke-linecap="round"/>'
        f'<text x="35" y="40" text-anchor="middle" fill="#eee" font-size="14" font-weight="bold">{pct}%</text>'
        f'</svg>'
    )


def progress_bar(done: int, total: int, color: str = "#3dba6f") -> str:
    pct = 100 * done / total if total else 0
    return (
        f'<div class="prog-wrap">'
        f'<div class="prog-bar" style="width:{pct:.1f}%;background:{color}"></div>'
        f'</div>'
        f'<span class="prog-label">{done} / {total}</span>'
    )


def render_html() -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    bo8 = bo8_stats()
    lv3 = level3_stats()
    itr = iterative_stats()
    trn = training_stats()

    # in-flight table rows
    in_flight_rows = ""
    for tid, attempt in sorted(trn["in_flight"].items(), key=lambda x: x[1], reverse=True):
        in_flight_rows += f"<tr><td>{tid}</td><td>attempt {attempt}</td></tr>"
    if not in_flight_rows:
        in_flight_rows = "<tr><td colspan='2' style='color:#666'>None currently active</td></tr>"

    # last update
    last_upd = ""
    if trn["last_update"]:
        secs_ago = int(time.time() - trn["last_update"])
        last_upd = f"{secs_ago}s ago" if secs_ago < 120 else f"{secs_ago//60}m ago"

    # BO8 per-rep table rows
    bo8_rep_rows = ""
    for rep in bo8["per_rep"]:
        bar = milestone_bar(rep["milestones"], rep["total"])
        bo8_rep_rows += (
            f"<tr>"
            f"<td>{rep['name']}</td>"
            f"<td class='num'>{rep['solved']}</td>"
            f"<td class='num'>{rep['rate']*100:.1f}%</td>"
            f"<td class='bar-cell'>{bar}</td>"
            f"</tr>"
        )

    # Iterative attempt dist
    itr_attempt_cells = "".join(
        f"<div class='stat-chip'><span class='chip-val'>{count}</span><span class='chip-lbl'>{n}-attempt</span></div>"
        for n, count in sorted(itr["attempt_dist"].items())
    )

    # Training milestone chips
    trn_chips = "".join(
        "<div class='stat-chip' style='border-color:{bc}'>"
        "<span class='chip-val' style='color:{vc}'>{count}</span>"
        "<span class='chip-lbl'>{lbl}</span></div>".format(
            bc=MILESTONE_COLORS.get(m, "#888"),
            vc=MILESTONE_COLORS.get(m, "#ccc"),
            count=count,
            lbl=MILESTONE_LABELS.get(m, str(m)),
        )
        for m, count in sorted(trn["milestones"].items(), key=lambda x: str(x[0]))
    )

    # Training milestone bar + legend (pre-computed to avoid backslash-in-fstring issues)
    trn_ms_bar = milestone_bar(trn["milestones"], trn["completed"]) if trn["milestones"] else ""
    trn_ms_legend_items = "".join(
        "<span class='ms-litem'><span class='ms-dot' style='background:{c}'></span>{lbl}: {n}</span>".format(
            c=c, lbl=MILESTONE_LABELS.get(m, str(m)), n=trn["milestones"].get(m, 0)
        )
        for m, c in MILESTONE_COLORS.items()
        if trn["milestones"].get(m, 0)
    )
    trn_ms_legend = f"<div class='ms-legend'>{trn_ms_legend_items}</div>" if trn_ms_legend_items else ""

    # Eval milestone legends (pre-computed)
    def ms_legend_html(milestones: dict) -> str:
        items = "".join(
            "<span class='ms-litem'><span class='ms-dot' style='background:{c}'></span>{lbl}: {n}</span>".format(
                c=c, lbl=MILESTONE_LABELS.get(m, str(m)), n=milestones.get(m, 0)
            )
            for m, c in MILESTONE_COLORS.items()
            if milestones.get(m, 0)
        )
        return f"<div class='ms-legend'>{items}</div>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="30">
<title>Mastermind Dashboard</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#111;color:#ddd;font-family:'SF Mono',Menlo,monospace;font-size:13px;padding:20px}}
  h1{{color:#fff;font-size:18px;font-weight:600;margin-bottom:4px}}
  .subtitle{{color:#555;font-size:11px;margin-bottom:24px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px;margin-bottom:16px}}
  .card{{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:16px}}
  .card-header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}}
  .card-title{{color:#aaa;font-size:11px;text-transform:uppercase;letter-spacing:.08em}}
  .card-badge{{background:#222;border:1px solid #333;color:#888;font-size:10px;padding:2px 8px;border-radius:4px}}
  .badge-live{{border-color:#3dba6f44;color:#3dba6f;animation:pulse 2s infinite}}
  @keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.5}}}}
  .metric-row{{display:flex;align-items:center;gap:16px;margin-bottom:12px}}
  .ring{{width:64px;height:64px;flex-shrink:0}}
  .metric-main{{font-size:28px;font-weight:700;color:#fff;line-height:1}}
  .metric-sub{{color:#555;font-size:11px;margin-top:3px}}
  .ms-bar{{display:flex;height:20px;border-radius:4px;overflow:hidden;background:#222;margin:8px 0}}
  .ms-seg{{display:flex;align-items:center;justify-content:center;overflow:hidden;min-width:1px}}
  .ms-label{{font-size:10px;color:#fff;white-space:nowrap;padding:0 3px}}
  .ms-legend{{display:flex;flex-wrap:wrap;gap:8px;margin-top:6px}}
  .ms-dot{{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:4px}}
  .ms-litem{{font-size:10px;color:#777;display:flex;align-items:center}}
  table{{width:100%;border-collapse:collapse;font-size:12px;margin-top:6px}}
  th{{color:#555;text-align:left;padding:4px 6px;border-bottom:1px solid #2a2a2a;font-weight:400}}
  td{{padding:5px 6px;border-bottom:1px solid #1e1e1e;color:#bbb}}
  td.num{{text-align:right;color:#ddd;font-variant-numeric:tabular-nums}}
  td.bar-cell{{width:50%}}
  .prog-wrap{{height:6px;background:#222;border-radius:3px;overflow:hidden;display:inline-block;width:calc(100% - 80px);vertical-align:middle}}
  .prog-bar{{height:100%;border-radius:3px;transition:width .5s}}
  .prog-label{{font-size:11px;color:#aaa;margin-left:8px;vertical-align:middle}}
  .chips{{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}}
  .stat-chip{{border:1px solid #2e2e2e;border-radius:6px;padding:6px 10px;text-align:center;min-width:60px}}
  .chip-val{{display:block;font-size:18px;font-weight:700;color:#eee}}
  .chip-lbl{{display:block;font-size:10px;color:#555;margin-top:2px}}
  .divider{{height:1px;background:#222;margin:12px 0}}
  .footer{{color:#444;font-size:10px;text-align:right;margin-top:12px}}
  .tag{{background:#1e3a1e;color:#3dba6f;font-size:10px;padding:1px 6px;border-radius:3px;margin-left:6px}}
  .tag-warn{{background:#3a2a1e;color:#e6a817}}
  .live-table{{max-height:140px;overflow-y:auto}}
</style>
</head>
<body>
<h1>Mastermind Experiment Dashboard</h1>
<p class="subtitle">Auto-refresh every 30s &nbsp;·&nbsp; {ts}</p>

<div class="grid">

<!-- ── BEST OF 8 ── -->
<div class="card">
  <div class="card-header">
    <span class="card-title">Best of 8 &nbsp;<span style="color:#555">(Eval · Level 1)</span></span>
    <span class="card-badge">8 reps · 200 tasks</span>
  </div>
  <div class="metric-row">
    {pct_ring(bo8["bo8_rate"])}
    <div>
      <div class="metric-main">{bo8["bo8_solved"]} <span style="font-size:14px;color:#555">/ {bo8["total_tasks"]}</span></div>
      <div class="metric-sub">tasks solved &nbsp;<span style="color:#3dba6f">Best-of-8</span></div>
    </div>
  </div>
  <div class="divider"></div>
  <table>
    <tr><th>Rep</th><th style="text-align:right">Solved</th><th style="text-align:right">Rate</th><th>Milestone Distribution</th></tr>
    {bo8_rep_rows}
  </table>
</div>

<!-- ── LEVEL 3 ── -->
<div class="card">
  <div class="card-header">
    <span class="card-title">Level 3 &nbsp;<span style="color:#555">(Eval · Rep 1)</span></span>
    <span class="card-badge">200 tasks</span>
  </div>
  <div class="metric-row">
    {pct_ring(lv3["rate"], "#4a9eff")}
    <div>
      <div class="metric-main">{lv3["solved"]} <span style="font-size:14px;color:#555">/ {lv3["total"]}</span></div>
      <div class="metric-sub">tasks solved</div>
    </div>
  </div>
  <div class="divider"></div>
  {milestone_bar(lv3["milestones"], lv3["total"])}
  {ms_legend_html(lv3["milestones"])}
</div>

<!-- ── ITERATIVE IMPROVEMENT ── -->
<div class="card">
  <div class="card-header">
    <span class="card-title">Iterative Improvement &nbsp;<span style="color:#555">(Eval)</span></span>
    <span class="card-badge">200 tasks · ≤8 attempts</span>
  </div>
  <div class="metric-row">
    {pct_ring(itr["rate"], "#a78bfa")}
    <div>
      <div class="metric-main">{itr["solved"]} <span style="font-size:14px;color:#555">/ {itr["total_tasks"]}</span></div>
      <div class="metric-sub">tasks solved &nbsp;·&nbsp; {itr["total_rows"]} total rollout rows</div>
    </div>
  </div>
  <div class="divider"></div>
  {milestone_bar(itr["milestones"], itr["total_tasks"])}
  {ms_legend_html(itr["milestones"])}
  <div class="divider"></div>
  <div style="color:#555;font-size:11px;margin-bottom:6px">Attempt distribution</div>
  <div class="chips">{itr_attempt_cells}</div>
</div>

<!-- ── TRAINING RUN (LIVE) ── -->
<div class="card">
  <div class="card-header">
    <span class="card-title">Iterative Improvement &nbsp;<span style="color:#555">(Train · 100 tasks)</span></span>
    <span class="card-badge badge-live">● LIVE</span>
  </div>
  <div class="metric-row" style="margin-bottom:8px">
    {pct_ring(trn['rate'] if trn['completed'] else 0, '#3dba6f')}
    <div>
      <div class="metric-main">{trn['solved']} <span style="font-size:14px;color:#555">solved</span></div>
      <div class="metric-sub">{trn['completed']} completed · {trn['agent_errors']} agent error · last event {last_upd or '?'}</div>
    </div>
  </div>
  {progress_bar(trn['completed'], trn['total_target'])}
  <div class="divider"></div>
  {trn_ms_bar}
  {trn_ms_legend}
  <div class="divider"></div>
  <div style="color:#555;font-size:11px;margin-bottom:6px">Workers in flight ({len(trn['in_flight'])} active)</div>
  <div class="live-table"><table>
    <tr><th>Task</th><th>Status</th></tr>
    {in_flight_rows}
  </table></div>
  <div style="color:#333;font-size:10px;margin-top:8px;word-break:break-all">{TRAIN_RUN_ID}</div>
</div>

</div>
<p class="footer">Mastermind · {ts} · refreshing every 30s</p>
</body>
</html>"""


# ── server ────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.send_response(404)
            self.end_headers()
            return
        html = render_html().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def log_message(self, fmt, *args):  # silence access log
        pass


def main():
    port = int(os.environ.get("DASHBOARD_PORT", 7860))
    host = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
    server = HTTPServer((host, port), Handler)
    print(f"Dashboard running at http://{host}:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
