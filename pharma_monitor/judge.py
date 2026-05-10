"""
Step 2b: Judge each answer against Wegovy PI.
For every answer: detect hedges, check formulation, run NLI claim-level scoring.
Input:  data/answers.csv
Output: data/judged_answers.csv
"""

import re, sys, csv, requests
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util, CrossEncoder

sys.path.insert(0, str(Path(__file__).parent))
from config import OPENROUTER_BASE

ANSWERS_CSV = Path(__file__).parent / "data" / "answers.csv"
OUT_CSV = Path(__file__).parent / "data" / "judged_answers.csv"
PI_CACHE = Path(__file__).parent / "data" / "wegovy_pi.txt"

HEDGE_PATTERNS = [
    r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bsome experts?\b",
    r"\bsome patients?\b", r"\bsome studies\b", r"\binvestigational\b",
    r"\bnot yet\b", r"\beventually\b", r"\bpotentially\b",
    r"\bsuggests?\b", r"\bbelieve[sd]?\b", r"\bvary\b", r"\buncertain\b",
]
WEGOVY_TERMS = ["wegovy", "semaglutide 2.4", "2.4 mg"]
WRONG_FORM = ["rybelsus", "oral semaglutide", "wegovy pill", "ozempic"]

_embed = None
_nli = None


def get_embed():
    global _embed
    if _embed is None:
        print("Loading embedding model...")
        _embed = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed


def get_nli():
    global _nli
    if _nli is None:
        print("Loading NLI model...")
        _nli = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _nli


def fetch_pi() -> list:
    if PI_CACHE.exists():
        text = PI_CACHE.read_text()
    else:
        print("Fetching Wegovy PI from openFDA...")
        url = "https://api.fda.gov/drug/label.json?search=openfda.brand_name:Wegovy&limit=1"
        r = requests.get(url, timeout=15)
        result = r.json()["results"][0]
        sections = [
            result.get("indications_and_usage", [""])[0],
            result.get("warnings_and_cautions", [""])[0],
            result.get("adverse_reactions", [""])[0],
            result.get("dosage_and_administration", [""])[0],
        ]
        text = " ".join(sections)
        PI_CACHE.write_text(text)

    claims = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 40]
    return claims[:40]


def detect_hedges(text: str) -> list:
    return [p.replace(r"\b", "").strip() for p in HEDGE_PATTERNS if re.search(p, text.lower())]


def check_formulation(text: str) -> dict:
    lower = text.lower()
    return {
        "correct": any(t in lower for t in WEGOVY_TERMS),
        "wrong": [t for t in WRONG_FORM if t in lower],
    }


def judge_answer(answer: str, pi_claims: list, pi_embs) -> dict:
    embed = get_embed()
    nli = get_nli()

    hedges = detect_hedges(answer)
    form = check_formulation(answer)

    emb_ans = embed.encode(answer, convert_to_tensor=True)
    sims = util.cos_sim(emb_ans, pi_embs)[0]
    best_idx = int(sims.argmax())
    best_pi = pi_claims[best_idx]
    best_sim = float(sims[best_idx])

    nli_scores = nli.predict([[best_pi, answer]])
    labels = ["contradiction", "entailment", "neutral"]
    top_label = labels[int(nli_scores[0].argmax())]

    if top_label == "entailment" and hedges:
        pi_verdict = "missing qualifier"
    elif top_label == "entailment":
        pi_verdict = "aligned"
    elif top_label == "contradiction":
        pi_verdict = "contradicted"
    else:
        pi_verdict = "unsupported"

    hedge_penalty = 0.15 * len(hedges) if hedges else 0
    form_penalty = 0.20 if form["wrong"] else (0.10 if not form["correct"] else 0)
    adj_sim = round(max(0, best_sim - hedge_penalty - form_penalty), 3)

    if adj_sim >= 0.45:
        verdict = "owned"
    elif adj_sim >= 0.30:
        verdict = "partial"
    else:
        verdict = "not owned"

    return {
        "hedges": "; ".join(hedges) if hedges else "",
        "hedge_count": len(hedges),
        "wrong_formulation": "; ".join(form["wrong"]) if form["wrong"] else "",
        "correct_drug": form["correct"],
        "base_sim": round(best_sim, 3),
        "adj_sim": adj_sim,
        "verdict": verdict,
        "pi_verdict": pi_verdict,
        "matched_pi_claim": best_pi[:150],
    }


def judge_all():
    df = pd.read_csv(ANSWERS_CSV)
    print(f"Judging {len(df)} answers...")

    pi_claims = fetch_pi()
    embed = get_embed()
    print("Embedding PI claims...")
    pi_embs = embed.encode(pi_claims, convert_to_tensor=True)

    results = []
    for i, row in df.iterrows():
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(df)}]")
        j = judge_answer(row["answer"], pi_claims, pi_embs)
        results.append({**row.to_dict(), **j})

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Judged answers saved to {OUT_CSV}")
    return out_df


if __name__ == "__main__":
    judge_all()
