"""
LLM-as-judge scoring method.
For each answer: GPT-4o extracts claims, matches to PI, returns structured verdict.
Input:  data/answers.csv + data/wegovy_pi.txt
Output: data/judged_llm.csv
"""

import os, sys, json, time, requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))
from config import OPENROUTER_BASE

ANSWERS_CSV = Path(__file__).parent / "data" / "answers.csv"
PI_CACHE = Path(__file__).parent / "data" / "wegovy_pi.txt"
OUT_CSV = Path(__file__).parent / "data" / "judged_llm.csv"

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
JUDGE_MODEL = "openai/gpt-4o-mini"


def get_pi_summary() -> str:
    if PI_CACHE.exists():
        text = PI_CACHE.read_text()
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 40]
        return ". ".join(sentences[:15])
    return ""


JUDGE_PROMPT = """You are a pharmaceutical compliance expert. Your job is to judge whether an AI-generated answer about a drug is accurate compared to the official Prescribing Information (PI).

OFFICIAL PI SUMMARY:
{pi}

AI ANSWER TO JUDGE:
"{answer}"

Analyze this answer and return a JSON object with exactly these fields:
{{
  "claims": ["list each factual claim in the answer as a separate string"],
  "hedges": ["list any hedge words found: may, might, could, some experts, investigational, etc."],
  "wrong_drug_references": ["list any references to wrong drugs or wrong formulations"],
  "pi_verdict": "one of: aligned / missing_qualifier / unsupported / overstated / contradicted",
  "pi_verdict_reason": "one sentence explaining why",
  "brand_verdict": "one of: owned / partial / not_owned",
  "brand_verdict_reason": "one sentence explaining why",
  "sharpness_score": 0-10 (10 = very precise and evidence-backed, 1 = vague and generic)
}}

Return only valid JSON. No explanation outside the JSON."""


def call_llm_judge(answer: str, pi_summary: str) -> dict:
    prompt = JUDGE_PROMPT.format(pi=pi_summary, answer=answer)
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    r = requests.post(f"{OPENROUTER_BASE}/chat/completions", json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def judge_all_llm():
    if not API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    df = pd.read_csv(ANSWERS_CSV)
    pi_summary = get_pi_summary()
    print(f"Judging {len(df)} answers with LLM-as-judge ({JUDGE_MODEL})...")

    results = []
    for i, row in df.iterrows():
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(df)}]")
        try:
            verdict = call_llm_judge(row["answer"], pi_summary)
            time.sleep(0.3)
        except Exception as e:
            verdict = {
                "claims": [], "hedges": [], "wrong_drug_references": [],
                "pi_verdict": "error", "pi_verdict_reason": str(e),
                "brand_verdict": "error", "brand_verdict_reason": "",
                "sharpness_score": 0,
            }
        results.append({
            **row.to_dict(),
            "claims": "; ".join(verdict.get("claims", [])),
            "hedges_llm": "; ".join(verdict.get("hedges", [])),
            "wrong_drug_refs": "; ".join(verdict.get("wrong_drug_references", [])),
            "pi_verdict_llm": verdict.get("pi_verdict", ""),
            "pi_verdict_reason": verdict.get("pi_verdict_reason", ""),
            "brand_verdict_llm": verdict.get("brand_verdict", ""),
            "brand_verdict_reason": verdict.get("brand_verdict_reason", ""),
            "sharpness_score": verdict.get("sharpness_score", 0),
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"LLM judging saved to {OUT_CSV}")
    return out_df


if __name__ == "__main__":
    judge_all_llm()
