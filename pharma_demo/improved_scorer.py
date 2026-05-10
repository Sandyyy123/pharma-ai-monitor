"""
Improved scorer - adds three layers on top of cosine similarity:
1. Hedge/qualifier detector (before scoring)
2. Formulation/drug name validator
3. Claim-level NLI verdict (aligned / missing qualifier / unsupported / contradicted)
"""

import re
from sentence_transformers import SentenceTransformer, util, CrossEncoder

_embed_model = None
_nli_model = None

HEDGE_PATTERNS = [
    r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bsome experts?\b",
    r"\bsome patients?\b", r"\bsome studies\b", r"\binvestigational\b",
    r"\bnot yet\b", r"\bnot widely available\b", r"\beventually\b",
    r"\bpotentially\b", r"\bsuggests?\b", r"\bbelieve[sd]?\b",
    r"\bvary\b", r"\buncertain\b", r"\bpromising\b",
]

WEGOVY_TERMS = ["wegovy", "semaglutide 2.4", "2.4 mg"]
WRONG_FORMULATION_TERMS = ["rybelsus", "oral semaglutide", "wegovy pill", "glp-1 pill"]
THRESHOLD = 0.45
HEDGE_PENALTY = 0.15
FORMULATION_PENALTY = 0.20


def _get_embed():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _get_nli():
    global _nli_model
    if _nli_model is None:
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _nli_model


def detect_hedges(text: str) -> list:
    found = []
    lower = text.lower()
    for pattern in HEDGE_PATTERNS:
        if re.search(pattern, lower):
            found.append(re.sub(r"\\b", "", pattern).strip())
    return found


def check_formulation(text: str) -> dict:
    lower = text.lower()
    has_correct = any(t in lower for t in WEGOVY_TERMS)
    has_wrong = [t for t in WRONG_FORMULATION_TERMS if t in lower]
    return {"correct_drug_mentioned": has_correct, "wrong_formulations": has_wrong}


def classify_claim(ai_claim: str, pi_claim: str) -> str:
    nli = _get_nli()
    scores = nli.predict([[pi_claim, ai_claim]])
    # scores: [contradiction, entailment, neutral] for deberta-v3-small
    labels = ["contradiction", "entailment", "neutral"]
    top = labels[int(scores[0].argmax())]

    hedges = detect_hedges(ai_claim)
    if top == "entailment" and hedges:
        return "missing qualifier"
    if top == "entailment":
        return "aligned"
    if top == "contradiction":
        return "contradicted"
    return "unsupported"


def score_improved(ai_answer: str, brand_message: str, pi_claims: list = None) -> dict:
    embed = _get_embed()

    # Step 1 - base similarity
    emb_answer = embed.encode(ai_answer, convert_to_tensor=True)
    emb_msg = embed.encode(brand_message, convert_to_tensor=True)
    sim = float(util.cos_sim(emb_answer, emb_msg))

    # Step 2 - hedge detection
    hedges = detect_hedges(ai_answer)
    hedge_penalty = HEDGE_PENALTY if hedges else 0.0

    # Step 3 - formulation check
    form = check_formulation(ai_answer)
    formulation_penalty = FORMULATION_PENALTY if form["wrong_formulations"] else 0.0
    if not form["correct_drug_mentioned"] and not form["wrong_formulations"]:
        formulation_penalty = HEDGE_PENALTY  # ambiguous drug reference

    adjusted_sim = max(0.0, sim - hedge_penalty - formulation_penalty)

    # Step 4 - verdict
    if adjusted_sim >= THRESHOLD:
        verdict = "owned"
    elif adjusted_sim >= 0.30:
        verdict = "partial"
    else:
        verdict = "not owned"

    # Step 5 - claim-level PI alignment (if PI claims provided)
    pi_verdict = "N/A"
    best_pi_claim = None
    if pi_claims:
        emb_pi = embed.encode(pi_claims, convert_to_tensor=True)
        cosines = util.cos_sim(emb_answer, emb_pi)[0]
        best_idx = int(cosines.argmax())
        best_pi_claim = pi_claims[best_idx][:120] + "..."
        pi_verdict = classify_claim(ai_answer, pi_claims[best_idx])

    return {
        "base_similarity": round(sim, 3),
        "hedges_found": hedges,
        "wrong_formulations": form["wrong_formulations"],
        "correct_drug_mentioned": form["correct_drug_mentioned"],
        "adjusted_similarity": round(adjusted_sim, 3),
        "verdict": verdict,
        "pi_verdict": pi_verdict,
        "matched_pi_claim": best_pi_claim,
    }
