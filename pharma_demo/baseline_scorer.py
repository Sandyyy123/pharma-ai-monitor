"""
Baseline scorer - simulates the current broken system.
Simple cosine similarity between answer and brand message.
No hedge detection, no formulation check, no claim-level logic.
"""

from sentence_transformers import SentenceTransformer, util

MODEL = SentenceTransformer("all-MiniLM-L6-v2")
THRESHOLD = 0.45


def score_baseline(ai_answer: str, brand_message: str) -> dict:
    emb_answer = MODEL.encode(ai_answer, convert_to_tensor=True)
    emb_msg = MODEL.encode(brand_message, convert_to_tensor=True)
    sim = float(util.cos_sim(emb_answer, emb_msg))

    if sim >= THRESHOLD:
        verdict = "owned"
    elif sim >= 0.30:
        verdict = "partial"
    else:
        verdict = "not owned"

    return {
        "similarity": round(sim, 3),
        "verdict": verdict,
        "pi_alignment": "aligned" if sim >= THRESHOLD else "unclear",
    }
