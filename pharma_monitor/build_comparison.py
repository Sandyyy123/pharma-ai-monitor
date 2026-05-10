"""
Build pharma method comparison report in RAG dashboard style.
Dark theme, Chart.js, tab navigation, progress bars.
"""

import json, shutil
import pandas as pd
from pathlib import Path
from datetime import date

BASE = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "output" / "method_comparison.html"
WIN = "pharma_method_comparison.html"

def pct(series, val):
    return round(100 * (series == val).sum() / len(series), 1)

def load_all():
    client = pd.read_csv(BASE / "judged_client_method.csv")
    nli    = pd.read_csv(BASE / "judged_answers.csv")
    llm    = pd.read_csv(BASE / "judged_llm.csv")
    sent   = pd.read_csv(BASE / "judged_sentence_nli.csv")
    zs     = pd.read_csv(BASE / "judged_zeroshot.csv")
    ent    = pd.read_csv(BASE / "judged_entity.csv")
    return client, nli, llm, sent, zs, ent

def bar(label, val, max_val=100, color="#00d4aa"):
    w = round(val / max_val * 100)
    return f'''<div class="br">
      <div class="bl">{label}</div>
      <div class="bbg"><div class="bf" style="width:{w}%;background:{color}"></div></div>
      <div class="bv">{val}%</div>
    </div>'''

def badge(v):
    v = str(v).lower()
    if v in ("owned","aligned","clean","specific"):
        return f'<span class="badge bg">{v}</span>'
    if v in ("partial","partially aligned","flagged","missing_qualifier","missing qualifier"):
        return f'<span class="badge br2">{v}</span>'
    if v in ("not owned","not aligned","unsupported","contradicted","vague"):
        return f'<span class="badge bred">{v}</span>'
    return f'<span class="badge bgr">{v}</span>'

def build():
    client, nli, llm, sent, zs, ent = load_all()
    n = len(client)
    today = date.today().strftime("%B %d, %Y")

    # --- Compute all stats ---
    # M0 client
    c_owned     = pct(client["brand_verdict"], "owned")
    c_partial   = pct(client["brand_verdict"], "partial")
    c_pi_aligned= pct(client["pi_verdict_client"], "aligned")
    c_pi_part   = pct(client["pi_verdict_client"], "partially aligned")
    c_issues    = round(100 - c_owned, 1)

    # M1 NLI
    n_owned     = pct(nli["verdict"], "owned")
    n_hedged    = round(100*(nli["hedge_count"]>0).sum()/n, 1)
    n_pi_align  = pct(nli["pi_verdict"], "aligned")
    n_wrong_form= round(100*nli["wrong_formulation"].notna().sum()/n, 1)

    # M2 LLM
    l_owned     = pct(llm["brand_verdict_llm"], "owned")
    l_pi_align  = pct(llm["pi_verdict_llm"], "aligned")
    l_unsup     = pct(llm["pi_verdict_llm"], "unsupported")
    l_sharp     = round(llm["sharpness_score"].mean(), 1)
    l_hedged    = round(100*llm["hedges_llm"].notna().sum()/n, 1)

    # M3 Sentence NLI
    s_owned     = pct(sent["final_verdict"], "owned")
    s_contra    = round(100*(sent["n_contradicted"]>0).sum()/n, 1)
    s_avg_align = round(sent["n_aligned"].mean(), 2)
    s_avg_unsup = round(sent["n_unsupported"].mean(), 2)

    # M4 Zero-shot
    top_labels  = zs["top_label"].value_counts()
    top1_label  = top_labels.index[0]
    top1_pct    = round(100*top_labels.iloc[0]/n, 1)
    top2_label  = top_labels.index[1]
    top2_pct    = round(100*top_labels.iloc[1]/n, 1)
    pct_vague   = pct(zs["specificity"], "vague")
    pct_specific= pct(zs["specificity"], "specific")

    # M5 Entity
    e_clean     = pct(ent["entity_verdict"], "clean")
    e_flagged   = pct(ent["entity_verdict"], "flagged")
    e_wrong_drug= round(100*ent["wrong_drug"].notna().sum()/n, 1)
    e_approval  = round(100*ent["approval_correct"].sum()/n, 1)

    # --- Chart data ---
    chart_problems = {
        "Client (M0)": c_issues,
        "NLI+Hedge (M1)": n_hedged,
        "LLM Judge (M2)": l_hedged,
        "Sent-NLI (M3)": s_contra,
        "Zero-shot (M4)": pct_vague,
        "Entity (M5)": e_flagged,
    }
    chart_owned = {
        "Client (M0)": c_owned,
        "NLI+Hedge (M1)": n_owned,
        "LLM Judge (M2)": l_owned,
        "Sent-NLI (M3)": s_owned,
    }

    # --- Sample rows ---
    sample_idx = list(client.sample(15, random_state=42).index)
    rows_html = ""
    for idx in sample_idx:
        c = client.iloc[idx]; nl = nli.iloc[idx]; lm = llm.iloc[idx]
        sn = sent.iloc[idx]; zrow = zs.iloc[idx]; en = ent.iloc[idx]
        hedges_found = str(nl.get('hedges','')) if pd.notna(nl.get('hedges','')) and nl.get('hedges','') else '-'
        wrong_drug = str(en['wrong_drug'])[:20] if pd.notna(en['wrong_drug']) and en['wrong_drug'] else '-'
        rows_html += f"""<tr>
          <td>{str(c['model']).split('/')[1]}</td>
          <td style="max-width:140px;font-size:0.72rem">{str(c['question'])[:60]}...</td>
          <td style="max-width:150px;font-size:0.72rem">{str(c['answer'])[:80]}...</td>
          <td>{badge(c['brand_verdict'])}<br>{badge(c['pi_verdict_client'])}</td>
          <td>{badge(nl['verdict'])}<br><span style="color:#7c3aed;font-size:0.68rem">{hedges_found[:25]}</span></td>
          <td>{badge(lm['brand_verdict_llm'])}<br>{badge(lm['pi_verdict_llm'])}<br><span style="color:#00d4aa;font-size:0.68rem">{lm.get('sharpness_score','?')}/10</span></td>
          <td>{badge(sn['final_verdict'])}<br><span style="font-size:0.68rem;color:#3fb950">✓{sn['n_aligned']} ✗{sn['n_contradicted']}</span></td>
          <td><span style="font-size:0.72rem">{str(zrow['top_label'])[:18]}</span><br>{badge(zrow['specificity'])}</td>
          <td>{badge(en['entity_verdict'])}<br><span style="color:#f85149;font-size:0.68rem">{wrong_drug}</span></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Pharma AI Monitor - Method Comparison</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0d1117;color:#c9d1d9;line-height:1.6}}
.hdr{{background:linear-gradient(135deg,#161b22,#0d1117);border-bottom:1px solid #30363d;padding:20px 32px}}
.hdr h1{{color:#00d4aa;font-size:1.8rem}}.hdr .sub{{color:#8b949e;font-size:0.85rem;margin-top:4px}}
.stats{{display:flex;gap:10px;padding:12px 32px;background:#161b22;border-bottom:1px solid #30363d;flex-wrap:wrap}}
.st{{text-align:center;min-width:80px}}.st .v{{font-size:1.4rem;font-weight:bold;color:#00d4aa}}.st .l{{font-size:0.7rem;color:#8b949e}}
.nav{{display:flex;background:#161b22;border-bottom:2px solid #30363d;padding:0 16px;overflow-x:auto}}
.nt{{padding:10px 16px;cursor:pointer;color:#8b949e;font-size:0.82rem;border-bottom:2px solid transparent;margin-bottom:-2px;white-space:nowrap;transition:.2s}}
.nt:hover{{color:#e6edf3}}.nt.act{{color:#00d4aa;border-bottom-color:#00d4aa}}
.cnt{{padding:20px 32px;max-width:1500px;margin:0 auto}}
.pnl{{display:none}}.pnl.act{{display:block}}
.cd{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:14px}}
.cd h3{{color:#e6edf3;margin-bottom:10px;font-size:1.05rem}}
.cd h4{{color:#00d4aa;margin:12px 0 6px;font-size:0.95rem}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.g3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}}
@media(max-width:900px){{.g2,.g3{{grid-template-columns:1fr}}}}
.br{{display:flex;align-items:center;gap:6px;margin:3px 0}}
.bl{{width:160px;text-align:right;font-size:0.78rem;color:#8b949e}}
.bbg{{flex:1;height:20px;background:#21262d;border-radius:4px;overflow:hidden}}
.bf{{height:100%;background:linear-gradient(90deg,#00d4aa,#00b894);border-radius:4px;transition:width .6s}}
.bv{{width:50px;font-size:0.78rem;color:#00d4aa}}
table{{width:100%;border-collapse:collapse;font-size:0.78rem}}
th{{background:#161b22;color:#8b949e;padding:8px 6px;text-align:left}}
td{{padding:7px 6px;border-bottom:1px solid #21262d;vertical-align:top}}
tr:hover td{{background:#161b2288}}
.badge{{display:inline-block;padding:1px 7px;border-radius:10px;font-size:0.68rem;font-weight:bold}}
.bg{{background:#0d2818;color:#3fb950;border:1px solid #3fb950}}
.br2{{background:#2d2200;color:#d29922;border:1px solid #d29922}}
.bgr{{background:#21262d;color:#8b949e;border:1px solid #484f58}}
.bred{{background:#2d0000;color:#f85149;border:1px solid #f85149}}
.insight{{background:#0d2818;border:1px solid #3fb950;border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:0.85rem;color:#3fb950}}
.warning{{background:#2d2200;border:1px solid #d29922;border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:0.85rem;color:#d29922}}
.good{{color:#3fb950}}.ok{{color:#d29922}}.bad{{color:#f85149}}
</style>
</head>
<body>

<div class="hdr">
  <h1>Pharma AI Monitor - 6-Method Comparison</h1>
  <div class="sub">Wegovy brand scoring | {n} AI answers | GPT-4o-mini + Grok-3-mini + Claude Haiku | {today}</div>
</div>

<div class="stats">
  <div class="st"><div class="v">{n}</div><div class="l">Answers Scored</div></div>
  <div class="st"><div class="v">6</div><div class="l">Methods Tested</div></div>
  <div class="st"><div class="v">3</div><div class="l">AI Models</div></div>
  <div class="st"><div class="v" style="color:#f85149">{e_wrong_drug}%</div><div class="l">Wrong Drug Refs</div></div>
  <div class="st"><div class="v" style="color:#d29922">{n_hedged}%</div><div class="l">Answers Hedged</div></div>
  <div class="st"><div class="v" style="color:#f85149">{pct_vague}%</div><div class="l">Vague/Generic</div></div>
  <div class="st"><div class="v" style="color:#3fb950">{c_owned}%</div><div class="l">Client Says Owned</div></div>
  <div class="st"><div class="v" style="color:#3fb950">{l_sharp}/10</div><div class="l">Avg Sharpness</div></div>
</div>

<div class="nav">
  <div class="nt act" onclick="sp('overview')">Overview</div>
  <div class="nt" onclick="sp('m0')">M0: Client Baseline</div>
  <div class="nt" onclick="sp('m1')">M1: NLI + Hedge</div>
  <div class="nt" onclick="sp('m2')">M2: LLM Judge</div>
  <div class="nt" onclick="sp('m3')">M3: Sentence NLI</div>
  <div class="nt" onclick="sp('m4')">M4: Zero-shot</div>
  <div class="nt" onclick="sp('m5')">M5: Entity</div>
  <div class="nt" onclick="sp('table')">All Rows</div>
</div>

<div class="cnt">

<!-- OVERVIEW -->
<div class="pnl act" id="pnl-overview">
  <div class="warning">Client method reports {c_owned}% brand owned and {c_pi_aligned}% PI aligned. Our 5 additional methods reveal: {e_wrong_drug}% wrong drug references, {n_hedged}% hedged answers, {pct_vague}% vague/generic content - all missed by the client baseline.</div>
  <div class="g2">
    <div class="cd">
      <h3>Problems Detected per Method</h3>
      <canvas id="chart-problems" height="220"></canvas>
    </div>
    <div class="cd">
      <h3>Brand Owned % per Method</h3>
      <canvas id="chart-owned" height="220"></canvas>
    </div>
  </div>
  <div class="cd">
    <h3>Method Summary</h3>
    <table>
      <thead><tr><th>Method</th><th>Approach</th><th>Brand Owned</th><th>PI Aligned</th><th>Key Finding</th><th>Cost</th></tr></thead>
      <tbody>
        <tr><td><span class="good">M0 Client</span></td><td>Cosine + simple LLM prompt</td><td class="good">{c_owned}%</td><td class="good">{c_pi_aligned}%</td><td class="bad">Misses hedges, wrong drugs, vague answers</td><td>Low</td></tr>
        <tr><td><span class="good">M1 NLI+Hedge</span></td><td>Sentence embeddings + hedge regex + NLI</td><td class="ok">{n_owned}%</td><td class="bad">{n_pi_align}%</td><td class="ok">{n_hedged}% answers hedged, {n_wrong_form}% wrong formulation</td><td>None (local)</td></tr>
        <tr><td><span class="good">M2 LLM Judge</span></td><td>GPT extracts claims before judging</td><td class="ok">{l_owned}%</td><td class="ok">{l_pi_align}%</td><td class="ok">Avg sharpness {l_sharp}/10, gives reasoning per answer</td><td>Medium (API)</td></tr>
        <tr><td><span class="good">M3 Sentence NLI</span></td><td>Per-sentence NLI, aggregated verdict</td><td class="bad">{s_owned}%</td><td>-</td><td class="bad">{s_contra}% have contradicted sentences</td><td>None (local)</td></tr>
        <tr><td><span class="good">M4 Zero-shot</span></td><td>BART claim type classification</td><td>-</td><td>-</td><td class="bad">{pct_vague}% answers vague/generic</td><td>None (local)</td></tr>
        <tr><td><span class="good">M5 Entity</span></td><td>Regex: drug names, numbers, approval</td><td>-</td><td>-</td><td class="bad">{e_wrong_drug}% mention wrong drug</td><td>Zero</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- M0 CLIENT -->
<div class="pnl" id="pnl-m0">
  <div class="cd"><h3>M0: Client Baseline - Cosine Similarity + Simple LLM Judge</h3>
    <p style="color:#8b949e;font-size:0.85rem;margin-bottom:12px">His current system. Full answer vs brand message (cosine). Full answer to GPT: "aligned/partially/not aligned?" No claim extraction, no hedge detection.</p>
    {bar("Brand Owned", c_owned)}
    {bar("Brand Partial", c_partial, color="#d29922")}
    {bar("PI Aligned", c_pi_aligned)}
    {bar("PI Partially Aligned", c_pi_part, color="#d29922")}
    <div class="warning" style="margin-top:12px">This method over-reports ownership. It cannot distinguish "Wegovy reduces weight by 15-17%" from "Wegovy may help some patients lose weight eventually."</div>
  </div>
</div>

<!-- M1 NLI -->
<div class="pnl" id="pnl-m1">
  <div class="cd"><h3>M1: NLI + Hedge Detector</h3>
    <p style="color:#8b949e;font-size:0.85rem;margin-bottom:12px">Adds hedge word regex (may/could/investigational) before scoring. Uses NLI (cross-encoder/nli-deberta-v3-small) for PI alignment. No API cost.</p>
    {bar("Brand Owned", n_owned)}
    {bar("Answers Containing Hedges", n_hedged, color="#d29922")}
    {bar("Wrong Formulation Detected", n_wrong_form, color="#f85149")}
    {bar("PI Aligned", n_pi_align)}
    <div class="insight" style="margin-top:12px">Catches {n_hedged}% hedged answers the client system misses. Identifies {n_wrong_form}% wrong drug formulations (Ozempic/Rybelsus/Wegovy Pill).</div>
  </div>
</div>

<!-- M2 LLM -->
<div class="pnl" id="pnl-m2">
  <div class="cd"><h3>M2: LLM Claim Extractor (GPT-4o-mini)</h3>
    <p style="color:#8b949e;font-size:0.85rem;margin-bottom:12px">GPT extracts individual claims from each answer before judging. Returns structured JSON: claims list, hedges, wrong drug refs, PI verdict with reason, sharpness score 0-10.</p>
    {bar("Brand Owned", l_owned)}
    {bar("PI Aligned", l_pi_align)}
    {bar("PI Unsupported", l_unsup, color="#f85149")}
    {bar("Hedges Detected by LLM", l_hedged, color="#d29922")}
    <div class="insight" style="margin-top:12px">Average sharpness score: {l_sharp}/10. Provides human-readable reasoning for every verdict. Most explainable method.</div>
  </div>
</div>

<!-- M3 SENTENCE NLI -->
<div class="pnl" id="pnl-m3">
  <div class="cd"><h3>M3: Sentence-Level NLI</h3>
    <p style="color:#8b949e;font-size:0.85rem;margin-bottom:12px">Splits each answer into sentences. Judges each sentence separately against the closest PI claim. Most granular NLI approach.</p>
    {bar("Final Verdict: Owned", s_owned)}
    {bar("Answers with Contradicted Sentences", s_contra, color="#f85149")}
    <div class="br"><div class="bl">Avg Aligned Sentences</div><div class="bbg"><div class="bf" style="width:{min(s_avg_align*50,100)}%"></div></div><div class="bv">{s_avg_align}</div></div>
    <div class="br"><div class="bl">Avg Unsupported Sentences</div><div class="bbg"><div class="bf" style="width:{min(s_avg_unsup*20,100)}%;background:#f85149"></div></div><div class="bv">{s_avg_unsup}</div></div>
    <div class="warning" style="margin-top:12px">Most strict method - 0% owned. NLI model is very precise: a 3-sentence answer rarely fully entails a single PI sentence. Better for catching contradictions than ownership.</div>
  </div>
</div>

<!-- M4 ZEROSHOT -->
<div class="pnl" id="pnl-m4">
  <div class="cd"><h3>M4: Zero-shot Claim Classifier (BART)</h3>
    <p style="color:#8b949e;font-size:0.85rem;margin-bottom:12px">Facebook BART classifies each answer by claim type (efficacy/safety/comparative/hedge/dosing/approval) and specificity (specific vs vague).</p>
    {bar("Vague/Generic Answers", pct_vague, color="#f85149")}
    {bar("Specific/Evidence-backed", pct_specific)}
    {bar(top1_label[:25], top1_pct, color="#7c3aed")}
    {bar(top2_label[:25], top2_pct, color="#7c3aed")}
    <div class="insight" style="margin-top:12px">{pct_vague}% of answers classified as vague/generic - this is the synthesis problem the client described. Most common claim type: {top1_label} ({top1_pct}%).</div>
  </div>
</div>

<!-- M5 ENTITY -->
<div class="pnl" id="pnl-m5">
  <div class="cd"><h3>M5: Named Entity + Context Validator</h3>
    <p style="color:#8b949e;font-size:0.85rem;margin-bottom:12px">Pure regex - no ML. Checks drug names, percentages, approval status, competitor mentions. Fast and deterministic. Zero API cost.</p>
    {bar("Clean Answers", e_clean)}
    {bar("Flagged Answers", e_flagged, color="#f85149")}
    {bar("Wrong Drug Mentioned", e_wrong_drug, color="#f85149")}
    {bar("Correct Approval Mention", e_approval, color="#d29922")}
    <div class="insight" style="margin-top:12px">{e_wrong_drug}% of answers mention Ozempic, Zepbound, or Rybelsus instead of Wegovy. The client method detects 0 of these. Run this check first as a free pre-filter.</div>
  </div>
</div>

<!-- ALL ROWS TABLE -->
<div class="pnl" id="pnl-table">
  <div class="cd">
    <h3>15 Sample Answers - All 6 Methods Side by Side</h3>
    <div style="overflow-x:auto">
    <table>
      <thead><tr>
        <th>Model</th><th>Question</th><th>AI Answer</th>
        <th>M0<br>Brand/PI</th>
        <th>M1 NLI<br>Verdict/Hedges</th>
        <th>M2 LLM<br>Brand/PI/Sharp</th>
        <th>M3 Sent<br>Verdict/Counts</th>
        <th>M4 Zero<br>Type/Spec</th>
        <th>M5 Entity<br>Verdict/Drug</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
  </div>
</div>

</div><!-- /cnt -->

<script>
function sp(id) {{
  document.querySelectorAll('.pnl').forEach(p => p.classList.remove('act'));
  document.querySelectorAll('.nt').forEach(t => t.classList.remove('act'));
  document.getElementById('pnl-' + id).classList.add('act');
  event.target.classList.add('act');
}}

// Charts
const chartOpts = {{
  responsive:true,
  plugins:{{legend:{{display:false}}}},
  scales:{{
    x:{{ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}},
    y:{{ticks:{{color:'#8b949e',callback:v=>v+'%'}},grid:{{color:'#21262d'}},max:100}}
  }}
}};

new Chart(document.getElementById('chart-problems'), {{
  type:'bar',
  data:{{
    labels:{json.dumps(list({"Client (M0)": c_issues, "NLI+Hedge (M1)": n_hedged, "LLM Judge (M2)": l_hedged, "Sent-NLI (M3)": s_contra, "Zero-shot (M4)": pct_vague, "Entity (M5)": e_flagged}.keys()))},
    datasets:[{{
      label:'Problems Detected %',
      data:{list({"Client (M0)": c_issues, "NLI+Hedge (M1)": n_hedged, "LLM Judge (M2)": l_hedged, "Sent-NLI (M3)": s_contra, "Zero-shot (M4)": pct_vague, "Entity (M5)": e_flagged}.values())},
      backgroundColor:['#484f58','#00d4aa','#7c3aed','#059669','#d97706','#ef4444'],
      borderRadius:4
    }}]
  }},
  options:{{...chartOpts, plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:c=>c.raw+'% flagged'}}}}}}}}
}});

new Chart(document.getElementById('chart-owned'), {{
  type:'bar',
  data:{{
    labels:{json.dumps(list({"Client (M0)": c_owned, "NLI+Hedge (M1)": n_owned, "LLM Judge (M2)": l_owned, "Sent-NLI (M3)": s_owned}.keys()))},
    datasets:[{{
      label:'Brand Owned %',
      data:{list({"Client (M0)": c_owned, "NLI+Hedge (M1)": n_owned, "LLM Judge (M2)": l_owned, "Sent-NLI (M3)": s_owned}.values())},
      backgroundColor:['#484f58','#00d4aa','#7c3aed','#059669'],
      borderRadius:4
    }}]
  }},
  options:chartOpts
}});
</script>
</body>
</html>"""

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        f.write(html)
    shutil.copy(str(OUT), WIN)
    print(f"Saved to {WIN}")

if __name__ == "__main__":
    build()
