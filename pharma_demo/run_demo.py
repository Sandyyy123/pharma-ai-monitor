#!/usr/bin/env python3
"""
Pharma AI Scoring Demo - Before vs After
Fetches real Wegovy PI from openFDA, runs baseline + improved scorer on 10 synthetic rows.
Outputs HTML report to /mnt/c/Users/grove/Downloads/pharma_scoring_demo.html
"""

import json
import sys
import requests
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from demo_data import ROWS
from baseline_scorer import score_baseline
from improved_scorer import score_improved

OUT_PATH = "/mnt/c/Users/grove/Downloads/pharma_scoring_demo.html"


def fetch_wegovy_pi_claims() -> list:
    print("Fetching Wegovy PI from openFDA...")
    url = "https://api.fda.gov/drug/label.json?search=openfda.brand_name:Wegovy&limit=1"
    r = requests.get(url, timeout=15)
    data = r.json()
    result = data["results"][0]

    sections = [
        result.get("indications_and_usage", [""])[0],
        result.get("warnings_and_cautions", [""])[0],
        result.get("adverse_reactions", [""])[0],
        result.get("dosage_and_administration", [""])[0],
        result.get("contraindications", [""])[0],
    ]

    claims = []
    for section in sections:
        if not section:
            continue
        sentences = [s.strip() for s in section.replace("\n", " ").split(".") if len(s.strip()) > 40]
        claims.extend(sentences[:8])

    print(f"  Extracted {len(claims)} PI claim sentences.")
    return claims


def run_all(pi_claims: list) -> list:
    results = []
    for i, row in enumerate(ROWS):
        print(f"  Scoring row {i+1}/10: {row['question'][:60]}...")
        b = score_baseline(row["ai_answer"], row["brand_message"])
        imp = score_improved(row["ai_answer"], row["brand_message"], pi_claims)
        results.append({
            "id": row["id"],
            "question": row["question"],
            "ai_answer": row["ai_answer"],
            "expected_issues": ", ".join(row["expected_issues"]) if row["expected_issues"] else "None (clean answer)",
            # baseline
            "base_sim": b["similarity"],
            "base_verdict": b["verdict"],
            "base_pi": b["pi_alignment"],
            # improved
            "adj_sim": imp["adjusted_similarity"],
            "hedges": ", ".join(imp["hedges_found"]) if imp["hedges_found"] else "-",
            "wrong_form": ", ".join(imp["wrong_formulations"]) if imp["wrong_formulations"] else "-",
            "imp_verdict": imp["verdict"],
            "pi_verdict": imp["pi_verdict"],
            "matched_pi": imp["matched_pi_claim"] or "-",
        })
    return results


def build_html(results: list) -> str:
    verdict_color = {
        "owned": "#22c55e",
        "partial": "#f59e0b",
        "not owned": "#ef4444",
        "aligned": "#22c55e",
        "missing qualifier": "#f59e0b",
        "unsupported": "#ef4444",
        "contradicted": "#dc2626",
        "N/A": "#94a3b8",
    }

    rows_html = ""
    for r in results:
        bc = verdict_color.get(r["base_verdict"], "#94a3b8")
        ic = verdict_color.get(r["imp_verdict"], "#94a3b8")
        pc = verdict_color.get(r["pi_verdict"], "#94a3b8")
        changed = r["base_verdict"] != r["imp_verdict"]
        row_bg = "#fef9c3" if changed else "white"

        rows_html += f"""
        <tr style="background:{row_bg}">
          <td style="font-weight:600;color:#1e3a5f">{r['id']}</td>
          <td style="font-size:13px">{r['question']}</td>
          <td style="font-size:12px;color:#475569">{r['ai_answer'][:120]}...</td>
          <td style="font-size:12px;color:#dc2626">{r['expected_issues']}</td>
          <td style="text-align:center">{r['base_sim']}</td>
          <td style="text-align:center"><span style="background:{bc};color:white;padding:2px 8px;border-radius:4px;font-size:12px">{r['base_verdict']}</span></td>
          <td style="text-align:center">{r['base_pi']}</td>
          <td style="font-size:12px;color:#7c3aed">{r['hedges']}</td>
          <td style="font-size:12px;color:#b45309">{r['wrong_form']}</td>
          <td style="text-align:center">{r['adj_sim']}</td>
          <td style="text-align:center"><span style="background:{ic};color:white;padding:2px 8px;border-radius:4px;font-size:12px">{r['imp_verdict']}</span></td>
          <td style="text-align:center"><span style="background:{pc};color:white;padding:2px 8px;border-radius:4px;font-size:12px">{r['pi_verdict']}</span></td>
          <td style="font-size:11px;color:#475569">{r['matched_pi']}</td>
        </tr>"""

    changed_count = sum(1 for r in results if r["base_verdict"] != r["imp_verdict"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Pharma AI Scoring - Before vs After</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #f8fafc; color: #1e293b; }}
  .header {{ background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%); color: white; padding: 32px 40px; }}
  .header h1 {{ margin: 0 0 8px; font-size: 26px; }}
  .header p {{ margin: 0; opacity: 0.85; font-size: 15px; }}
  .stats {{ display: flex; gap: 20px; padding: 24px 40px; }}
  .stat {{ background: white; border-radius: 10px; padding: 16px 24px; border-left: 4px solid #2563eb; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .stat .num {{ font-size: 28px; font-weight: 700; color: #1e3a5f; }}
  .stat .label {{ font-size: 13px; color: #64748b; margin-top: 4px; }}
  .section {{ padding: 0 40px 40px; }}
  .section h2 {{ font-size: 18px; color: #1e3a5f; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.08); font-size: 13px; }}
  th {{ background: #1e3a5f; color: white; padding: 10px 8px; text-align: left; font-size: 12px; }}
  td {{ padding: 10px 8px; border-bottom: 1px solid #e2e8f0; vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }}
  .leg {{ display: flex; align-items: center; gap: 6px; font-size: 13px; }}
  .dot {{ width: 14px; height: 14px; border-radius: 3px; }}
  .changed-note {{ background: #fef9c3; border: 1px solid #fde047; border-radius: 6px; padding: 10px 16px; margin-bottom: 16px; font-size: 13px; color: #713f12; }}
</style>
</head>
<body>
<div class="header">
  <h1>Pharma AI Scoring - Before vs After</h1>
  <p>Wegovy PI sourced from openFDA. 10 synthetic rows modeled on real client examples.</p>
</div>

<div class="stats">
  <div class="stat"><div class="num">10</div><div class="label">AI Answers Scored</div></div>
  <div class="stat"><div class="num">{changed_count}</div><div class="label">Verdicts Changed After Fix</div></div>
  <div class="stat"><div class="num">3</div><div class="label">Fix Layers Applied</div></div>
</div>

<div class="section">
  <h2>Fix Layers Applied</h2>
  <div class="legend">
    <div class="leg"><div class="dot" style="background:#7c3aed"></div>Layer 1: Hedge/qualifier detector</div>
    <div class="leg"><div class="dot" style="background:#b45309"></div>Layer 2: Formulation/drug name validator</div>
    <div class="leg"><div class="dot" style="background:#2563eb"></div>Layer 3: Claim-level NLI (aligned / missing qualifier / unsupported / contradicted)</div>
  </div>
  <div class="changed-note">Yellow rows = verdict changed between baseline and improved scorer.</div>
  <h2>Scoring Results - Before vs After</h2>
  <div style="overflow-x:auto">
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Question</th>
        <th>AI Answer</th>
        <th>Expected Issues</th>
        <th>Base Sim</th>
        <th>Base Verdict</th>
        <th>Base PI</th>
        <th>Hedges Found</th>
        <th>Wrong Formulation</th>
        <th>Adj Sim</th>
        <th>New Verdict</th>
        <th>PI Verdict</th>
        <th>Matched PI Claim</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
  </div>
</div>
</body>
</html>"""


def main():
    print("=== Pharma AI Scoring Demo ===")
    pi_claims = fetch_wegovy_pi_claims()
    print("Running scorers on 10 rows...")
    results = run_all(pi_claims)
    print("Building HTML report...")
    html = build_html(results)
    with open(OUT_PATH, "w") as f:
        f.write(html)
    print(f"Done. Report saved to {OUT_PATH}")
    return results


if __name__ == "__main__":
    main()
