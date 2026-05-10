"""
Named entity + context validation scoring for Wegovy PI answers.

PI Summary:
  - Wegovy = semaglutide 2.4mg injection
  - Approved weight loss: 15-17% body weight
  - Approved for cardiovascular risk reduction
  - Once-weekly subcutaneous injection

Input:  data/answers.csv
Output: data/judged_entity.csv
"""

import re
import sys
import csv
import pandas as pd
from pathlib import Path

ANSWERS_CSV = Path(__file__).parent / "data" / "answers.csv"
OUT_CSV     = Path(__file__).parent / "data" / "judged_entity.csv"

# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

CORRECT_DRUG_PATTERNS = [
    re.compile(r"\bwegovy\b", re.IGNORECASE),
    re.compile(r"\bsemaglutide\s+2\.4\b", re.IGNORECASE),
    re.compile(r"\b2\.4\s*mg\b", re.IGNORECASE),
]

# Wrong formulations / wrong-route drugs (keyed by label -> pattern)
WRONG_DRUG_PATTERNS = {
    "rybelsus":         re.compile(r"\brybelsus\b", re.IGNORECASE),
    "ozempic":          re.compile(r"\bozempic\b", re.IGNORECASE),
    "zepbound":         re.compile(r"\bzepbound\b", re.IGNORECASE),
    "oral semaglutide": re.compile(r"\boral\s+semaglutide\b", re.IGNORECASE),
    "wegovy pill":      re.compile(r"\bwegovy\s+pill\b", re.IGNORECASE),
}

COMPETITOR_PATTERNS = {
    "tirzepatide": re.compile(r"\btirzepatide\b", re.IGNORECASE),
    "zepbound":    re.compile(r"\bzepbound\b", re.IGNORECASE),
    "mounjaro":    re.compile(r"\bmounjaro\b", re.IGNORECASE),
}

# Percentage extraction: handles "15%", "15 %", "15-17%", "15–17%"
PCT_RANGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*%"
)
PCT_SINGLE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")

# Correct Wegovy efficacy range per PI (with a small buffer for rounding)
PCT_LOW  = 14.0
PCT_HIGH = 18.0

HEDGE_PATTERNS = [
    re.compile(r"\bmay\b",           re.IGNORECASE),
    re.compile(r"\bmight\b",         re.IGNORECASE),
    re.compile(r"\bcould\b",         re.IGNORECASE),
    re.compile(r"\bsome\b",          re.IGNORECASE),
    re.compile(r"\bpossibly\b",      re.IGNORECASE),
    re.compile(r"\bpotentially\b",   re.IGNORECASE),
    re.compile(r"\binvestigational\b", re.IGNORECASE),
    re.compile(r"\bnot\s+yet\b",     re.IGNORECASE),
    re.compile(r"\beventually\b",    re.IGNORECASE),
]

# Approval phrases
APPROVAL_CORRECT_PATTERNS = [
    re.compile(r"\bfda[- ]approved\b",          re.IGNORECASE),
    re.compile(r"\bapproved\s+by\s+the\s+fda\b", re.IGNORECASE),
    re.compile(r"\bfda\s+approval\b",            re.IGNORECASE),
    re.compile(r"\bapproved\s+for\b",            re.IGNORECASE),
]
APPROVAL_FLAG_PATTERNS = [
    re.compile(r"\bnot\s+yet\s+approved\b",  re.IGNORECASE),
    re.compile(r"\bawaiting\s+approval\b",   re.IGNORECASE),
    re.compile(r"\binvestigational\b",       re.IGNORECASE),
    re.compile(r"\bpending\s+approval\b",    re.IGNORECASE),
    re.compile(r"\bnot\s+approved\b",        re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def check_correct_drug(text: str) -> bool:
    return any(p.search(text) for p in CORRECT_DRUG_PATTERNS)


def check_wrong_drugs(text: str) -> list:
    found = [label for label, p in WRONG_DRUG_PATTERNS.items() if p.search(text)]
    return found


def check_competitor(text: str) -> bool:
    return any(p.search(text) for p in COMPETITOR_PATTERNS.values())


def extract_percentages(text: str) -> list:
    """
    Return a deduplicated list of all numeric percentage values found.
    Ranges like '15-17%' expand to [15.0, 17.0].
    """
    found = []
    # Ranges first (so we don't double-count via single-pct pattern)
    range_spans = set()
    for m in PCT_RANGE_RE.finditer(text):
        found.extend([float(m.group(1)), float(m.group(2))])
        range_spans.add(m.span())

    # Singles not already captured inside a range match
    for m in PCT_SINGLE_RE.finditer(text):
        # check if this match overlaps any range match span
        overlaps = any(
            rs[0] <= m.start() < rs[1]
            for rs in range_spans
        )
        if not overlaps:
            found.append(float(m.group(1)))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for v in found:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def check_number_accuracy(pcts: list):
    """
    None  → no percentages found
    True  → at least one pct is within [PCT_LOW, PCT_HIGH]
    False → percentages found but ALL are outside that range
    """
    if not pcts:
        return None
    in_range = [p for p in pcts if PCT_LOW <= p <= PCT_HIGH]
    return len(in_range) > 0


def count_hedges(text: str) -> int:
    total = 0
    for p in HEDGE_PATTERNS:
        total += len(p.findall(text))
    return total


def check_approval(text: str):
    """
    Returns (approval_correct, approval_flag).
    approval_correct: True if any positive approval phrase found, False if flag found, None if neither.
    approval_flag: True if any red-flag phrase found.
    """
    flag = any(p.search(text) for p in APPROVAL_FLAG_PATTERNS)
    correct = any(p.search(text) for p in APPROVAL_CORRECT_PATTERNS)

    if flag:
        approval_correct = False
    elif correct:
        approval_correct = True
    else:
        approval_correct = None

    return approval_correct, flag


def entity_verdict(correct_drug, wrong_drugs, hedge_count, number_accurate, approval_flag) -> str:
    """
    "clean"   : correct_drug=True, no wrong_drug, hedge_count <= 1, number_accurate != False
    "flagged" : any wrong_drug OR approval_flag OR hedge_count >= 3
    "partial" : everything else
    """
    if wrong_drugs or approval_flag or hedge_count >= 3:
        return "flagged"
    if correct_drug and not wrong_drugs and hedge_count <= 1 and number_accurate is not False:
        return "clean"
    return "partial"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def score_row(answer: str) -> dict:
    correct_drug      = check_correct_drug(answer)
    wrong_drugs       = check_wrong_drugs(answer)
    competitor        = check_competitor(answer)
    pcts              = extract_percentages(answer)
    num_accurate      = check_number_accuracy(pcts)
    hedges            = count_hedges(answer)
    approval_correct, approval_flag = check_approval(answer)
    verdict           = entity_verdict(correct_drug, wrong_drugs, hedges, num_accurate, approval_flag)

    return {
        "correct_drug":       correct_drug,
        "wrong_drug":         "|".join(wrong_drugs) if wrong_drugs else "",
        "competitor_mention": competitor,
        "extracted_pcts":     "|".join(str(p) for p in pcts) if pcts else "",
        "number_accurate":    "" if num_accurate is None else num_accurate,
        "hedge_count":        hedges,
        "approval_flag":      approval_flag,
        "approval_correct":   "" if approval_correct is None else approval_correct,
        "entity_verdict":     verdict,
    }


def main():
    if not ANSWERS_CSV.exists():
        print(f"ERROR: Input file not found: {ANSWERS_CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(ANSWERS_CSV)
    print(f"Loaded {len(df)} rows from {ANSWERS_CSV.name}")

    # Score every row
    score_cols = [
        "correct_drug", "wrong_drug", "competitor_mention",
        "extracted_pcts", "number_accurate", "hedge_count",
        "approval_flag", "approval_correct", "entity_verdict",
    ]
    scores = df["answer"].fillna("").apply(score_row).apply(pd.Series)

    out = pd.concat([df, scores], axis=1)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(out)} rows -> {OUT_CSV}")

    # ---------------------------------------------------------------------------
    # Summary stats
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ENTITY SCORING SUMMARY")
    print("=" * 60)

    total = len(out)

    # Verdict distribution
    vc = out["entity_verdict"].value_counts()
    print("\nVerdict distribution:")
    for verdict in ["clean", "flagged", "partial"]:
        n = vc.get(verdict, 0)
        print(f"  {verdict:<10}: {n:>4}  ({100*n/total:.1f}%)")

    # Correct drug
    n_correct = out["correct_drug"].sum()
    print(f"\nCorrect drug mentioned : {n_correct}/{total} ({100*n_correct/total:.1f}%)")

    # Wrong drugs
    n_wrong = (out["wrong_drug"] != "").sum()
    print(f"Wrong drug mentioned   : {n_wrong}/{total} ({100*n_wrong/total:.1f}%)")
    if n_wrong:
        # Flatten and count individual labels
        from collections import Counter
        flat = []
        for v in out.loc[out["wrong_drug"] != "", "wrong_drug"]:
            flat.extend(v.split("|"))
        for label, cnt in Counter(flat).most_common():
            print(f"    {label}: {cnt}")

    # Competitor mentions
    n_comp = out["competitor_mention"].sum()
    print(f"Competitor mentioned   : {n_comp}/{total} ({100*n_comp/total:.1f}%)")

    # Number accuracy
    num_col = out["number_accurate"]
    n_true  = (num_col == True).sum()
    n_false = (num_col == False).sum()
    n_none  = (num_col == "").sum()
    print(f"\nNumber accuracy:")
    print(f"  In-range (True)  : {n_true}")
    print(f"  Out-of-range (F) : {n_false}")
    print(f"  No % found (None): {n_none}")

    # Hedge counts
    print(f"\nHedge word stats:")
    print(f"  Mean   : {out['hedge_count'].mean():.2f}")
    print(f"  Median : {out['hedge_count'].median():.0f}")
    print(f"  Max    : {out['hedge_count'].max()}")
    high_hedge = (out["hedge_count"] >= 3).sum()
    print(f"  Rows with hedge_count >= 3: {high_hedge} ({100*high_hedge/total:.1f}%)")

    # Approval flag
    n_flag = out["approval_flag"].sum()
    n_appr = (out["approval_correct"] == True).sum()
    print(f"\nApproval flag (red flag): {n_flag}/{total} ({100*n_flag/total:.1f}%)")
    print(f"Approval correct mention: {n_appr}/{total} ({100*n_appr/total:.1f}%)")

    # Per-model breakdown
    if "model" in out.columns:
        print("\nVerdict by model:")
        pivot = out.groupby("model")["entity_verdict"].value_counts().unstack(fill_value=0)
        for col in ["clean", "flagged", "partial"]:
            if col not in pivot.columns:
                pivot[col] = 0
        print(pivot[["clean", "flagged", "partial"]].to_string())

    print("\nDone.")
    sys.exit(0)


if __name__ == "__main__":
    main()
