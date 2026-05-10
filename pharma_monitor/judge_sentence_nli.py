"""
Sentence-level NLI scoring for pharma monitor.
Splits each answer into sentences, matches each to the closest PI claim,
runs NLI scoring, aggregates to answer-level verdict.

Input:  data/answers.csv
PI:     data/wegovy_pi.txt
Output: data/judged_sentence_nli.csv
"""

import re
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util, CrossEncoder

# ----- Paths ----------------------------------------------------------------
BASE = Path(__file__).parent
ANSWERS_CSV = BASE / "data" / "answers.csv"
PI_TXT      = BASE / "data" / "wegovy_pi.txt"
OUT_CSV     = BASE / "data" / "judged_sentence_nli.csv"

# ----- Hedge patterns -------------------------------------------------------
HEDGE_WORDS = re.compile(
    r"\bmay\b|\bmight\b|\bcould\b|\bsome experts?\b|\binvestigational\b"
    r"|\bnot yet\b|\beventually\b",
    re.IGNORECASE,
)

# ----- NLI label mapping (DeBERTa order: contradiction, entailment, neutral) -
LABEL_ORDER = ["contradiction", "entailment", "neutral"]


# ---------------------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    """Split on period (keeping other punctuation), keep sentences > 20 chars."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) > 20:
            sentences.append(s)
    return sentences


def chunk_pi(pi_text: str, max_chars: int = 400) -> list[str]:
    """Split PI text into ~sentence-level chunks for embedding."""
    raw = re.split(r"(?<=[.!?])\s+", pi_text.strip())
    chunks = []
    buf = ""
    for s in raw:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) < max_chars:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    return chunks


def verdict_for_sentence(scores: list[float], has_hedge: bool) -> str:
    """
    scores: [contradiction_score, entailment_score, neutral_score]
    Returns: aligned | missing_qualifier | contradicted | unsupported
    """
    label = LABEL_ORDER[int(np.argmax(scores))]
    if label == "entailment":
        return "missing_qualifier" if has_hedge else "aligned"
    elif label == "contradiction":
        return "contradicted"
    else:
        return "unsupported"


def aggregate_verdicts(sent_verdicts: list[str]) -> str:
    """
    Final answer-level verdict:
      - any contradicted  -> 'contradicted'
      - >50% aligned      -> 'owned'
      - else              -> 'partial'
    """
    if not sent_verdicts:
        return "partial"
    n = len(sent_verdicts)
    if "contradicted" in sent_verdicts:
        return "contradicted"
    n_aligned = sum(1 for v in sent_verdicts if v == "aligned")
    if n_aligned / n > 0.5:
        return "owned"
    return "partial"


# ---------------------------------------------------------------------------
def main():
    # 1. Load PI and PI chunks
    if not PI_TXT.exists():
        print("ERROR: wegovy_pi.txt not found. Fetch with:")
        print('  curl "https://api.fda.gov/drug/label.json?search=openfda.brand_name:Wegovy&limit=1"')
        sys.exit(1)

    pi_text = PI_TXT.read_text(encoding="utf-8")
    pi_chunks = chunk_pi(pi_text)
    print(f"PI loaded: {len(pi_chunks)} chunks")

    # 2. Load answers
    df = pd.read_csv(ANSWERS_CSV)
    print(f"Answers loaded: {len(df)} rows")

    # 3. Load models
    print("Loading SentenceTransformer (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading CrossEncoder (cross-encoder/nli-deberta-v3-small)...")
    nli = CrossEncoder("cross-encoder/nli-deberta-v3-small")

    # 4. Pre-embed PI chunks
    print("Embedding PI chunks...")
    pi_embeddings = embedder.encode(pi_chunks, convert_to_tensor=True, show_progress_bar=True)

    # 5. Process each answer
    results = []

    for idx, row in df.iterrows():
        answer_text = str(row["answer"])
        sentences = split_sentences(answer_text)

        if not sentences:
            # No scoreable sentences - treat as partial
            results.append({
                **row.to_dict(),
                "sentence_verdicts": json.dumps([]),
                "n_aligned": 0,
                "n_missing_qualifier": 0,
                "n_unsupported": 0,
                "n_contradicted": 0,
                "final_verdict": "partial",
                "best_evidence": "",
            })
            continue

        # Embed answer sentences
        sent_embeddings = embedder.encode(sentences, convert_to_tensor=True)

        sent_verdicts = []
        best_sent = ""
        best_score = -1.0

        for i, (sent, sent_emb) in enumerate(zip(sentences, sent_embeddings)):
            # Match to closest PI chunk by cosine similarity
            cos_scores = util.cos_sim(sent_emb.unsqueeze(0), pi_embeddings)[0]
            best_pi_idx = int(cos_scores.argmax())
            best_cos = float(cos_scores[best_pi_idx])
            best_pi_claim = pi_chunks[best_pi_idx]

            # Run NLI
            nli_input = [[best_pi_claim, sent]]
            nli_scores = nli.predict(nli_input)[0]  # shape [3]

            has_hedge = bool(HEDGE_WORDS.search(sent))
            verdict = verdict_for_sentence(nli_scores.tolist(), has_hedge)
            sent_verdicts.append(verdict)

            # Track best_evidence: sentence with highest entailment score
            ent_score = float(nli_scores[1])  # index 1 = entailment
            if ent_score > best_score:
                best_score = ent_score
                best_sent = sent

        # Aggregate
        n_aligned   = sent_verdicts.count("aligned")
        n_mq        = sent_verdicts.count("missing_qualifier")
        n_unsup     = sent_verdicts.count("unsupported")
        n_contra    = sent_verdicts.count("contradicted")
        final       = aggregate_verdicts(sent_verdicts)

        results.append({
            **row.to_dict(),
            "sentence_verdicts":    json.dumps(sent_verdicts),
            "n_aligned":            n_aligned,
            "n_missing_qualifier":  n_mq,
            "n_unsupported":        n_unsup,
            "n_contradicted":       n_contra,
            "final_verdict":        final,
            "best_evidence":        best_sent,
        })

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} answers...")

    # 6. Save output
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV} ({len(out_df)} rows)")

    # 7. Summary stats
    n_total    = len(out_df)
    n_owned    = (out_df["final_verdict"] == "owned").sum()
    n_partial  = (out_df["final_verdict"] == "partial").sum()
    n_contra   = (out_df["final_verdict"] == "contradicted").sum()
    avg_aligned = out_df["n_aligned"].mean()

    print("\n=== Summary Stats ===")
    print(f"  % owned:        {100 * n_owned / n_total:.1f}%  ({n_owned}/{n_total})")
    print(f"  % partial:      {100 * n_partial / n_total:.1f}%  ({n_partial}/{n_total})")
    print(f"  % contradicted: {100 * n_contra / n_total:.1f}%  ({n_contra}/{n_total})")
    print(f"  avg n_aligned per answer: {avg_aligned:.2f}")

    # Verify row count
    if len(out_df) == 135:
        print("\nRow count check: PASSED (135 rows)")
    else:
        print(f"\nRow count check: FAILED (expected 135, got {len(out_df)})")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
