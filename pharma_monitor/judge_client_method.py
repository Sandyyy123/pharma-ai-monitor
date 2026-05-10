"""
Replicates the client's current method:
- Full answer vs brand message: cosine similarity → owned/partial/not owned
- Full answer vs PI: simple LLM judge → aligned/partially aligned/not aligned
No claim extraction. No hedge detection. Exactly what he has today.
Input:  data/answers.csv
Output: data/judged_client_method.csv
"""

import os, sys, json, time, requests
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))
from config import BRAND_MESSAGES, OPENROUTER_BASE

ANSWERS_CSV = Path(__file__).parent / "data" / "answers.csv"
PI_CACHE = Path(__file__).parent / "data" / "wegovy_pi.txt"
OUT_CSV = Path(__file__).parent / "data" / "judged_client_method.csv"

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
JUDGE_MODEL = "openai/gpt-4o-mini"
SIM_THRESHOLD_OWNED = 0.50
SIM_THRESHOLD_PARTIAL = 0.35

_embed = None

SIMPLE_JUDGE_PROMPT = """You are a pharmaceutical compliance reviewer.

OFFICIAL PRESCRIBING INFORMATION (summary):
{pi}

AI ANSWER:
"{answer}"

Does this answer align with the Prescribing Information?
Reply with ONLY one of these labels:
- aligned
- partially aligned
- not aligned

Label:"""


def get_embed():
    global _embed
    if _embed is None:
        _embed = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed


def get_pi_summary() -> str:
    if PI_CACHE.exists():
        text = PI_CACHE.read_text()
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 40]
        return ". ".join(sentences[:10])
    return ""


def score_brand(answer: str, brand_messages: list) -> dict:
    embed = get_embed()
    emb_ans = embed.encode(answer, convert_to_tensor=True)
    best_sim = 0.0
    best_msg = ""
    for msg in brand_messages:
        emb_msg = embed.encode(msg, convert_to_tensor=True)
        sim = float(util.cos_sim(emb_ans, emb_msg))
        if sim > best_sim:
            best_sim = sim
            best_msg = msg

    if best_sim >= SIM_THRESHOLD_OWNED:
        verdict = "owned"
    elif best_sim >= SIM_THRESHOLD_PARTIAL:
        verdict = "partial"
    else:
        verdict = "not owned"

    return {"brand_sim": round(best_sim, 3), "brand_verdict": verdict, "matched_msg": best_msg[:80]}


def score_pi_llm(answer: str, pi_summary: str) -> str:
    prompt = SIMPLE_JUDGE_PROMPT.format(pi=pi_summary, answer=answer)
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0,
    }
    r = requests.post(f"{OPENROUTER_BASE}/chat/completions", json=payload, headers=headers, timeout=20)
    r.raise_for_status()
    label = r.json()["choices"][0]["message"]["content"].strip().lower()
    if "partially" in label:
        return "partially aligned"
    if "not" in label:
        return "not aligned"
    return "aligned"


def judge_all():
    if not API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    df = pd.read_csv(ANSWERS_CSV)
    pi_summary = get_pi_summary()
    print(f"Replicating client method on {len(df)} answers...")

    results = []
    for i, row in df.iterrows():
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(df)}]")
        brand = score_brand(row["answer"], BRAND_MESSAGES)
        try:
            pi_verdict = score_pi_llm(row["answer"], pi_summary)
            time.sleep(0.2)
        except Exception as e:
            pi_verdict = f"error: {e}"
        results.append({**row.to_dict(), **brand, "pi_verdict_client": pi_verdict})

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Client method results saved to {OUT_CSV}")
    return out_df


if __name__ == "__main__":
    judge_all()
