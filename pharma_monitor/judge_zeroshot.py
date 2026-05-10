"""
Zero-shot classification scoring for pharma AI monitor.

Classifies each answer into claim types using facebook/bart-large-mnli,
then scores specificity (specific/vague) with a second classification pass.
"""

import csv
import json
from collections import Counter
from pathlib import Path

from transformers import pipeline

# ── Paths ──────────────────────────────────────────────────────────────────
INPUT_CSV  = Path(__file__).parent / "data" / "answers.csv"
OUTPUT_CSV = Path(__file__).parent / "data" / "judged_zeroshot.csv"

# ── Labels ─────────────────────────────────────────────────────────────────
CLAIM_LABELS = [
    "efficacy claim",
    "safety claim",
    "comparative claim",
    "hedge or speculation",
    "dosing information",
    "approval status",
]

SPECIFICITY_LABELS = ["specific and evidence-backed", "vague and generic"]

SCORE_THRESHOLD = 0.15
TOP_N = 2


def load_classifier():
    print("Loading facebook/bart-large-mnli (downloads ~1.6 GB on first use)...")
    clf = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
    )
    print("Model loaded.\n")
    return clf


def classify_answer(clf, text: str) -> dict:
    """Run both classification passes on a single answer text."""
    # Pass 1: multi-label claim type
    result1 = clf(text, candidate_labels=CLAIM_LABELS, multi_label=True)
    label_scores = dict(zip(result1["labels"], result1["scores"]))

    # Sort by score descending, pick top 2 above threshold
    sorted_labels = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
    above_threshold = [(l, s) for l, s in sorted_labels if s > SCORE_THRESHOLD]

    top_label  = above_threshold[0][0] if len(above_threshold) > 0 else sorted_labels[0][0]
    top_score  = above_threshold[0][1] if len(above_threshold) > 0 else sorted_labels[0][1]
    second_label = above_threshold[1][0] if len(above_threshold) > 1 else ""
    second_score = above_threshold[1][1] if len(above_threshold) > 1 else 0.0

    # Pass 2: specificity (binary)
    result2 = clf(text, candidate_labels=SPECIFICITY_LABELS, multi_label=False)
    specificity_raw = result2["labels"][0]  # highest-scoring label
    specificity = "specific" if "specific" in specificity_raw else "vague"

    return {
        "top_label":    top_label,
        "top_score":    round(top_score, 4),
        "second_label": second_label,
        "second_score": round(second_score, 4),
        "specificity":  specificity,
        # All 6 label scores as individual columns
        **{f"score_{lbl.replace(' ', '_')}": round(label_scores[lbl], 4) for lbl in CLAIM_LABELS},
    }


def main():
    clf = load_classifier()

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Processing {len(rows)} answers...\n")

    output_rows = []
    for i, row in enumerate(rows, 1):
        if i % 10 == 0 or i == 1:
            print(f"  [{i}/{len(rows)}] drug={row['drug']}  model={row['model']}")
        scores = classify_answer(clf, row["answer"])
        output_rows.append({**row, **scores})

    # ── Write output CSV ────────────────────────────────────────────────────
    fieldnames = list(rows[0].keys()) + [
        "top_label", "top_score", "second_label", "second_score", "specificity",
    ] + [f"score_{lbl.replace(' ', '_')}" for lbl in CLAIM_LABELS]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nOutput written to {OUTPUT_CSV}  ({len(output_rows)} rows)\n")

    # ── Summary ─────────────────────────────────────────────────────────────
    top_labels   = [r["top_label"] for r in output_rows]
    specificities = [r["specificity"] for r in output_rows]

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nMost common top labels:")
    for label, count in Counter(top_labels).most_common():
        pct = count / len(output_rows) * 100
        print(f"  {label:<30s}  {count:>3d}  ({pct:.1f}%)")

    n_specific = specificities.count("specific")
    n_vague    = specificities.count("vague")
    print(f"\nSpecificity:")
    print(f"  specific : {n_specific:>3d}  ({n_specific / len(output_rows) * 100:.1f}%)")
    print(f"  vague    : {n_vague:>3d}  ({n_vague    / len(output_rows) * 100:.1f}%)")

    print("\nAverage score per label:")
    for lbl in CLAIM_LABELS:
        col = f"score_{lbl.replace(' ', '_')}"
        avg = sum(float(r[col]) for r in output_rows) / len(output_rows)
        print(f"  {lbl:<30s}  {avg:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
