"""
Compare NLI method vs LLM-as-judge method side by side.
Output: output/method_comparison.html
"""

import sys, shutil
import pandas as pd
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent))
from config import DRUG

NLI_CSV = Path(__file__).parent / "data" / "judged_answers.csv"
LLM_CSV = Path(__file__).parent / "data" / "judged_llm.csv"
OUT = Path(__file__).parent / "output" / "method_comparison.html"

VERDICT_COLOR = {
    "owned": "#22c55e", "partial": "#f59e0b", "not owned": "#ef4444", "not_owned": "#ef4444",
    "aligned": "#22c55e", "missing qualifier": "#f59e0b", "missing_qualifier": "#f59e0b",
    "unsupported": "#ef4444", "contradicted": "#dc2626", "overstated": "#f97316",
    "error": "#94a3b8",
}

def badge(v):
    c = VERDICT_COLOR.get(str(v).lower(), "#94a3b8")
    return f'<span style="background:{c};color:white;padding:2px 8px;border-radius:10px;font-size:11px">{v}</span>'

def stat_box(label, val, color="#2563eb"):
    return f'<div style="background:white;border-radius:8px;padding:14px 20px;border-left:4px solid {color};box-shadow:0 1px 3px rgba(0,0,0,.1)"><div style="font-size:24px;font-weight:700;color:{color}">{val}</div><div style="font-size:12px;color:#64748b;margin-top:2px">{label}</div></div>'

def compare():
    nli = pd.read_csv(NLI_CSV)
    llm = pd.read_csv(LLM_CSV)
    n = min(len(nli), len(llm))
    nli = nli.iloc[:n]
    llm = llm.iloc[:n]

    today = date.today().strftime("%B %d, %Y")

    # --- Stats ---
    def pct(series, val):
        return f"{round(100*(series==val).sum()/len(series),1)}%"

    nli_owned = pct(nli['verdict'], 'owned')
    nli_aligned = pct(nli['pi_verdict'], 'aligned')
    nli_unsup = pct(nli['pi_verdict'], 'unsupported')
    nli_hedged = f"{round(100*(nli['hedge_count']>0).sum()/len(nli),1)}%"

    llm_owned = pct(llm['brand_verdict_llm'], 'owned')
    llm_aligned = pct(llm['pi_verdict_llm'], 'aligned')
    llm_unsup = pct(llm['pi_verdict_llm'], 'unsupported')
    llm_sharp = f"{round(llm['sharpness_score'].mean(),1)}/10"

    # --- Sample rows ---
    sample = nli.sample(min(15, n), random_state=42).index
    rows_html = ""
    for idx in sample:
        nr = nli.iloc[idx]
        lr = llm.iloc[idx]
        rows_html += f"""
        <tr>
          <td style="font-size:11px;max-width:160px;color:#475569">{str(nr['question'])[:70]}...</td>
          <td style="font-size:11px;max-width:180px">{str(nr['answer'])[:100]}...</td>
          <td style="text-align:center">{badge(nr['verdict'])}<br><small style="color:#7c3aed">{str(nr.get('hedges',''))[:40] or '-'}</small></td>
          <td style="text-align:center">{badge(nr['pi_verdict'])}</td>
          <td style="text-align:center">{badge(lr['brand_verdict_llm'])}<br><small style="color:#64748b;font-size:10px">{str(lr.get('brand_verdict_reason',''))[:50]}</small></td>
          <td style="text-align:center">{badge(lr['pi_verdict_llm'])}<br><small style="color:#64748b;font-size:10px">{str(lr.get('pi_verdict_reason',''))[:50]}</small></td>
          <td style="text-align:center;font-weight:700;color:#2563eb">{lr.get('sharpness_score','?')}/10</td>
          <td style="font-size:10px;color:#7c3aed;max-width:150px">{str(lr.get('hedges_llm',''))[:60] or '-'}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{DRUG} - Method Comparison</title>
<style>
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; margin:0; background:#f8fafc; color:#1e293b; }}
  .header {{ background:linear-gradient(135deg,#1e3a5f,#2563eb); color:white; padding:28px 36px; }}
  .header h1 {{ margin:0 0 4px; font-size:22px; }}
  .header p {{ margin:0; opacity:.85; font-size:13px; }}
  .section {{ padding:20px 36px; }}
  h2 {{ font-size:17px; color:#1e3a5f; border-bottom:2px solid #e2e8f0; padding-bottom:6px; margin:20px 0 14px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; margin-bottom:24px; }}
  .method-box {{ background:white; border-radius:12px; padding:20px; box-shadow:0 1px 4px rgba(0,0,0,.08); }}
  .method-box h3 {{ margin:0 0 14px; font-size:15px; color:#1e3a5f; border-bottom:1px solid #e2e8f0; padding-bottom:8px; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:8px; }}
  table {{ width:100%; border-collapse:collapse; background:white; border-radius:10px; overflow:hidden; box-shadow:0 1px 4px rgba(0,0,0,.08); font-size:12px; }}
  th {{ background:#1e3a5f; color:white; padding:9px 7px; text-align:left; font-size:11px; }}
  td {{ padding:9px 7px; border-bottom:1px solid #f1f5f9; vertical-align:top; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:hover {{ background:#f8fafc; }}
</style>
</head>
<body>
<div class="header">
  <h1>{DRUG} - Judging Method Comparison</h1>
  <p>NLI-based scoring vs LLM-as-judge | {n} answers | {today}</p>
</div>

<div class="section">
  <h2>Method 1 vs Method 2 - Key Stats</h2>
  <div class="grid">
    <div class="method-box">
      <h3>Method 1 - NLI + Hedge Detector (fast, no API cost)</h3>
      <div class="stats">
        {stat_box("Brand Owned", nli_owned)}
        {stat_box("PI Aligned", nli_aligned, "#22c55e")}
        {stat_box("PI Unsupported", nli_unsup, "#ef4444")}
        {stat_box("Answers with Hedges", nli_hedged, "#f59e0b")}
      </div>
      <p style="font-size:13px;color:#64748b;margin:10px 0 0">Strengths: fast, cheap, catches hedges reliably.<br>Weakness: NLI model sees full answer vs one PI sentence - context mismatch causes over-flagging as unsupported.</p>
    </div>
    <div class="method-box">
      <h3>Method 2 - LLM-as-Judge (GPT-4o-mini, structured rubric)</h3>
      <div class="stats">
        {stat_box("Brand Owned", llm_owned)}
        {stat_box("PI Aligned", llm_aligned, "#22c55e")}
        {stat_box("PI Unsupported", llm_unsup, "#ef4444")}
        {stat_box("Avg Sharpness Score", llm_sharp, "#7c3aed")}
      </div>
      <p style="font-size:13px;color:#64748b;margin:10px 0 0">Strengths: extracts individual claims, gives reasoning, scores sharpness.<br>Weakness: costs API credits per answer, reasoning can vary.</p>
    </div>
  </div>

  <h2>Side-by-Side Sample (15 rows)</h2>
  <div style="overflow-x:auto">
  <table>
    <thead>
      <tr>
        <th>Question</th>
        <th>AI Answer</th>
        <th>NLI Brand Verdict + Hedges</th>
        <th>NLI PI Verdict</th>
        <th>LLM Brand Verdict + Reason</th>
        <th>LLM PI Verdict + Reason</th>
        <th>Sharpness</th>
        <th>LLM Hedges Found</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
  </div>
</div>
</body>
</html>"""

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        f.write(html)
    win_path = "/mnt/c/Users/grove/Downloads/pharma_method_comparison.html"
    shutil.copy(str(OUT), win_path)
    print(f"Comparison saved and copied to {win_path}")


if __name__ == "__main__":
    compare()
