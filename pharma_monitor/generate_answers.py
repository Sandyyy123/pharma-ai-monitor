"""
Step 1: Generate AI answers for all questions across all models and runs.
Saves to data/answers.csv
"""

import os, sys, csv, time, requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))
from config import DRUG, QUESTIONS, MODELS, RUNS_PER_QUESTION, OPENROUTER_BASE

OUT = Path(__file__).parent / "data" / "answers.csv"
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

SYSTEM = f"You are a medical information assistant. Answer questions about {DRUG} accurately and concisely in 2-4 sentences."


def call_model(model: str, question: str) -> str:
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        "max_tokens": 300,
        "temperature": 0.7,
    }
    r = requests.post(f"{OPENROUTER_BASE}/chat/completions", json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def generate(dry_run=False):
    if not API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set in .env")
        sys.exit(1)

    rows = []
    total = len(QUESTIONS) * len(MODELS) * RUNS_PER_QUESTION
    count = 0

    for q in QUESTIONS:
        for model in MODELS:
            for run in range(1, RUNS_PER_QUESTION + 1):
                count += 1
                print(f"[{count}/{total}] {model.split('/')[1]} | run {run} | {q[:50]}...")
                if dry_run:
                    answer = f"[DRY RUN] Answer from {model} run {run} for: {q}"
                else:
                    try:
                        answer = call_model(model, q)
                        time.sleep(0.3)
                    except Exception as e:
                        answer = f"ERROR: {e}"
                rows.append({
                    "question": q,
                    "model": model,
                    "run": run,
                    "answer": answer,
                    "drug": DRUG,
                })

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["drug", "question", "model", "run", "answer"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} answers to {OUT}")
    return rows


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    generate(dry_run=dry)
