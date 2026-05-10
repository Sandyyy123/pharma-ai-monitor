"""
Step 2: Embed all answers, cluster, extract key messages with evidence + volume.
Input:  data/answers.csv
Output: data/synthesis.json
"""

import json, sys
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLUSTERS

ANSWERS_CSV = Path(__file__).parent / "data" / "answers.csv"
OUT_JSON = Path(__file__).parent / "data" / "synthesis.json"

EMBED_MODEL = "all-MiniLM-L6-v2"


def extract_sentences(text: str) -> list:
    return [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 30]


def synthesize():
    df = pd.read_csv(ANSWERS_CSV)
    print(f"Loaded {len(df)} answers.")

    print("Embedding answers...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(df["answer"].tolist(), show_progress_bar=True)

    print(f"Clustering into {N_CLUSTERS} key message groups...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(embeddings)

    results = []
    for cluster_id in range(N_CLUSTERS):
        mask = df["cluster"] == cluster_id
        cluster_df = df[mask]
        cluster_embs = embeddings[mask.values]
        centroid = km.cluster_centers_[cluster_id]

        # find 3 answers closest to centroid = best representatives
        sims = cosine_similarity([centroid], cluster_embs)[0]
        top_idx = sims.argsort()[::-1][:3]
        evidence = cluster_df.iloc[top_idx]["answer"].tolist()

        # dominant question for this cluster
        top_question = cluster_df["question"].value_counts().index[0]

        # model breakdown
        model_counts = cluster_df["model"].apply(lambda x: x.split("/")[1]).value_counts().to_dict()

        # generate cluster label from most common words in evidence
        combined = " ".join(evidence).lower()
        label = generate_label(combined, cluster_id)

        results.append({
            "cluster_id": cluster_id,
            "label": label,
            "count": int(mask.sum()),
            "share_pct": round(100 * mask.sum() / len(df), 1),
            "dominant_question": top_question,
            "model_breakdown": model_counts,
            "evidence": evidence,
        })

    results.sort(key=lambda x: x["count"], reverse=True)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Synthesis saved to {OUT_JSON}")
    return results


def generate_label(text: str, cluster_id: int) -> str:
    keywords = [
        ("weight loss", "Weight loss efficacy"),
        ("side effect", "Side effects and tolerability"),
        ("cardiovascular", "Cardiovascular benefits"),
        ("dose", "Dosing and administration"),
        ("compare", "Comparative positioning"),
        ("long-term", "Long-term safety"),
        ("insurance", "Access and cost"),
        ("oral", "Oral formulation interest"),
        ("diabetes", "Diabetes and comorbidities"),
        ("clinical trial", "Clinical evidence"),
    ]
    for kw, label in keywords:
        if kw in text:
            return label
    return f"Key message theme {cluster_id + 1}"


if __name__ == "__main__":
    synthesize()
