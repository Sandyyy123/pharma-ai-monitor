"""
Build HTML report combining synthesis clusters + judging scores.
"""

import json, sys
import pandas as pd
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent))
from config import DRUG

SYNTHESIS = Path(__file__).parent / "data" / "synthesis.json"
JUDGED = Path(__file__).parent / "data" / "judged_answers.csv"
OUT = Path(__file__).parent / "output" / "report.html"

VERDICT_COLOR = {
    "owned": "#22c55e", "partial": "#f59e0b", "not owned": "#ef4444",
    "aligned": "#22c55e", "missing qualifier": "#f59e0b",
    "unsupported": "#ef4444", "contradicted": "#dc2626",
}


def verdict_badge(v):
    c = VERDICT_COLOR.get(v, "#94a3b8")
    return f'<span style="background:{c};color:white;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600">{v}</span>'


def build_report():
    with open(SYNTHESIS) as f:
        clusters = json.load(f)

    has_judging = JUDGED.exists()
    df = pd.read_csv(JUDGED) if has_judging else None

    total = sum(c["count"] for c in clusters)
    today = date.today().strftime("%B %d, %Y")

    # --- Global judging stats ---
    global_stats = ""
    if has_judging:
        pct_aligned = round(100 * (df["pi_verdict"] == "aligned").sum() / len(df), 1)
        pct_hedged = round(100 * (df["hedge_count"] > 0).sum() / len(df), 1)
        pct_owned = round(100 * (df["verdict"] == "owned").sum() / len(df), 1)
        pct_contradicted = round(100 * (df["pi_verdict"] == "contradicted").sum() / len(df), 1)
        global_stats = f"""
        <div class="stat"><div class="num">{pct_owned}%</div><div class="lbl">Answers - Brand Owned</div></div>
        <div class="stat" style="border-color:#22c55e"><div class="num" style="color:#16a34a">{pct_aligned}%</div><div class="lbl">PI Aligned (claim-level)</div></div>
        <div class="stat" style="border-color:#f59e0b"><div class="num" style="color:#d97706">{pct_hedged}%</div><div class="lbl">Contain Hedges / Qualifiers</div></div>
        <div class="stat" style="border-color:#ef4444"><div class="num" style="color:#dc2626">{pct_contradicted}%</div><div class="lbl">Contradicted PI</div></div>"""

    # --- Cluster cards ---
    cards = ""
    for i, c in enumerate(clusters):
        bar_w = c["share_pct"]
        model_pills = "".join(
            f'<span style="background:#e0e7ff;color:#3730a3;padding:2px 8px;border-radius:12px;font-size:11px;margin-right:4px">{m}: {n}</span>'
            for m, n in c["model_breakdown"].items()
        )
        evidence_items = "".join(
            f'<li style="margin-bottom:6px;color:#475569;font-size:13px">"{e[:200]}{"..." if len(e)>200 else ""}"</li>'
            for e in c["evidence"]
        )

        # judging stats per cluster
        judge_html = ""
        if has_judging:
            q = c["dominant_question"]
            cdf = df[df["question"] == q]
            if len(cdf) > 0:
                cl_aligned = round(100 * (cdf["pi_verdict"] == "aligned").sum() / len(cdf), 0)
                cl_hedged = round(100 * (cdf["hedge_count"] > 0).sum() / len(cdf), 0)
                cl_owned = round(100 * (cdf["verdict"] == "owned").sum() / len(cdf), 0)
                judge_html = f"""
                <div style="display:flex;gap:16px;margin:12px 0;flex-wrap:wrap">
                  <div style="background:#f0fdf4;border-radius:8px;padding:8px 14px;text-align:center">
                    <div style="font-size:20px;font-weight:700;color:#16a34a">{cl_owned:.0f}%</div>
                    <div style="font-size:11px;color:#64748b">Brand Owned</div>
                  </div>
                  <div style="background:#fefce8;border-radius:8px;padding:8px 14px;text-align:center">
                    <div style="font-size:20px;font-weight:700;color:#d97706">{cl_hedged:.0f}%</div>
                    <div style="font-size:11px;color:#64748b">Hedged Claims</div>
                  </div>
                  <div style="background:#f0fdf4;border-radius:8px;padding:8px 14px;text-align:center">
                    <div style="font-size:20px;font-weight:700;color:#16a34a">{cl_aligned:.0f}%</div>
                    <div style="font-size:11px;color:#64748b">PI Aligned</div>
                  </div>
                </div>"""

        cards += f"""
        <div style="background:white;border-radius:12px;padding:24px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,0.08);border-left:5px solid #2563eb">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
            <div>
              <span style="font-size:12px;color:#64748b;font-weight:600">KEY MESSAGE {i+1}</span>
              <h3 style="margin:4px 0 0;font-size:18px;color:#1e3a5f">{c['label']}</h3>
            </div>
            <div style="text-align:right">
              <div style="font-size:28px;font-weight:700;color:#2563eb">{c['share_pct']}%</div>
              <div style="font-size:12px;color:#64748b">{c['count']} of {total} answers</div>
            </div>
          </div>
          <div style="background:#f1f5f9;border-radius:6px;height:8px;margin-bottom:12px">
            <div style="background:#2563eb;height:8px;border-radius:6px;width:{bar_w}%"></div>
          </div>
          <div style="margin-bottom:8px">
            <span style="font-size:12px;color:#64748b;font-weight:600">DOMINANT QUESTION: </span>
            <span style="font-size:13px;color:#334155">{c['dominant_question']}</span>
          </div>
          <div style="margin-bottom:8px">{model_pills}</div>
          {judge_html}
          <div>
            <div style="font-size:12px;font-weight:600;color:#64748b;margin-bottom:6px">REPRESENTATIVE EVIDENCE</div>
            <ul style="margin:0;padding-left:18px">{evidence_items}</ul>
          </div>
        </div>"""

    # --- Judging detail table ---
    table_html = ""
    if has_judging:
        sample = df.sample(min(20, len(df)), random_state=42)
        rows = ""
        for _, r in sample.iterrows():
            rows += f"""
            <tr>
              <td style="font-size:12px;max-width:200px">{r['question'][:80]}...</td>
              <td style="font-size:11px;max-width:250px;color:#475569">{str(r['answer'])[:120]}...</td>
              <td style="text-align:center">{r['model'].split('/')[1]}</td>
              <td style="text-align:center;color:#7c3aed;font-size:12px">{str(r['hedges'])[:60] if pd.notna(r['hedges']) and r['hedges'] else '-'}</td>
              <td style="text-align:center">{verdict_badge(r['verdict'])}</td>
              <td style="text-align:center">{verdict_badge(r['pi_verdict'])}</td>
            </tr>"""

        table_html = f"""
        <h2>Sample Answer-Level Judging (20 random rows)</h2>
        <div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse;background:white;border-radius:10px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.08);font-size:13px">
          <thead>
            <tr style="background:#1e3a5f;color:white">
              <th style="padding:10px 8px;text-align:left">Question</th>
              <th style="padding:10px 8px;text-align:left">AI Answer</th>
              <th style="padding:10px 8px">Model</th>
              <th style="padding:10px 8px">Hedges Found</th>
              <th style="padding:10px 8px">Brand Verdict</th>
              <th style="padding:10px 8px">PI Verdict</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{DRUG} AI Monitor Report</title>
<style>
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; margin:0; background:#f8fafc; color:#1e293b; }}
  .header {{ background:linear-gradient(135deg,#1e3a5f,#2563eb); color:white; padding:32px 40px; }}
  .header h1 {{ margin:0 0 6px; font-size:26px; }}
  .header p {{ margin:0; opacity:.85; font-size:14px; }}
  .stats {{ display:flex; gap:16px; padding:24px 40px; flex-wrap:wrap; }}
  .stat {{ background:white; border-radius:10px; padding:16px 24px; border-left:4px solid #2563eb; box-shadow:0 1px 3px rgba(0,0,0,.1); min-width:150px; }}
  .stat .num {{ font-size:28px; font-weight:700; color:#1e3a5f; }}
  .stat .lbl {{ font-size:12px; color:#64748b; margin-top:4px; }}
  .body {{ padding:0 40px 40px; }}
  h2 {{ font-size:18px; color:#1e3a5f; border-bottom:2px solid #e2e8f0; padding-bottom:8px; margin:24px 0 16px; }}
  td {{ padding:10px 8px; border-bottom:1px solid #e2e8f0; vertical-align:top; }}
  tr:last-child td {{ border-bottom:none; }}
</style>
</head>
<body>
<div class="header">
  <h1>{DRUG} - AI Monitor Report</h1>
  <p>What AI models are telling patients and HCPs about {DRUG} | {total} answers analyzed | {today}</p>
</div>
<div class="stats">
  <div class="stat"><div class="num">{total}</div><div class="lbl">AI Answers Analyzed</div></div>
  <div class="stat"><div class="num">{len(clusters)}</div><div class="lbl">Key Message Themes</div></div>
  {global_stats}
</div>
<div class="body">
  <h2>Key Messages - What AI is Teaching Users About {DRUG}</h2>
  {cards}
  {table_html}
</div>
</body>
</html>"""

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        f.write(html)
    print(f"Report saved to {OUT}")
    return str(OUT)


if __name__ == "__main__":
    build_report()
