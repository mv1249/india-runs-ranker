"""
Generates the three missing .npy artifacts (career_embeddings, skill_embeddings,
jd_vectors) from candidates.jsonl using the SBERT model.

features.parquet must already exist (it's produced by feature_extractor.py).
This script reads candidates in the same order as features.parquet and writes:
  career_embeddings.npy  — (N, D)
  skill_embeddings.npy   — (N, D)
  jd_vectors.npy         — (2, D)  [career query, skill query]
  candidate_ids.npy      — (N,)    candidate_id strings, parallel to embeddings

Usage:
    python embed_only.py \
        --candidates "../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" \
        --out_dir .
"""

import argparse
import json
import math
import ssl
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

JD_QUERY = (
    "Senior AI engineer building production ranking retrieval matching systems. "
    "Embedding-based retrieval sentence transformers vector databases FAISS Qdrant "
    "hybrid search BM25. Evaluation frameworks NDCG MRR MAP offline online correlation. "
    "Learning-to-rank recsys NLP transformers fine-tuning LLM integration. "
    "Shipped real systems to production users at product companies not research only. "
    "Strong Python code quality. A/B testing recruiter engagement metrics."
)

SKILL_QUERY = (
    "embeddings vector search FAISS Qdrant Pinecone sentence-transformers BGE E5 "
    "information retrieval BM25 hybrid search learning-to-rank NDCG MRR MAP "
    "NLP transformers fine-tuning LoRA recommendation systems XGBoost LightGBM "
    "Python MLOps MLflow feature engineering"
)


def build_career_text(c: dict) -> str:
    parts = []
    for job in c["career_history"]:
        parts.append(
            f"{job['title']} at {job['company']} ({job['industry']}): {job['description']}"
        )
    summary = c["profile"].get("summary", "")
    headline = c["profile"].get("headline", "")
    return f"{headline}. {summary} " + " ".join(parts)


def build_skill_text(c: dict) -> str:
    tokens = []
    for s in c["skills"]:
        name = s["name"]
        prof = s["proficiency"]
        endorse = s.get("endorsements", 0)
        dur = s.get("duration_months", 0)
        trust = (
            {"beginner": 0.2, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}[prof]
            * math.log1p(endorse)
            * math.log1p(dur)
        )
        repeats = max(1, min(3, int(trust / 2)))
        tokens.extend([name] * repeats)
    return " ".join(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    print(f"Loading candidates from {args.candidates}...")
    candidates = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates):,} candidates.")

    feat_df = pd.read_parquet(out_dir / "features.parquet")
    # Build an order map so embeddings align with features.parquet row order
    feat_order = {cid: i for i, cid in enumerate(feat_df["candidate_id"])}
    candidates.sort(key=lambda c: feat_order.get(c["candidate_id"], 999999))

    # Auto-select best device: MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    print(f"Loading SBERT model: {args.model}")
    model = SentenceTransformer(args.model, device=device)

    print("Building texts...")
    career_texts = [build_career_text(c) for c in tqdm(candidates, desc="career text")]
    skill_texts  = [build_skill_text(c)  for c in tqdm(candidates, desc="skill text")]
    cids = [c["candidate_id"] for c in candidates]

    print("Encoding career texts (largest batch)...")
    career_vecs = model.encode(
        career_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print("Encoding skill texts...")
    skill_vecs = model.encode(
        skill_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    jd_career_vec = model.encode([JD_QUERY], normalize_embeddings=True, convert_to_numpy=True)
    jd_skill_vec  = model.encode([SKILL_QUERY], normalize_embeddings=True, convert_to_numpy=True)

    np.save(out_dir / "career_embeddings.npy", career_vecs)
    np.save(out_dir / "skill_embeddings.npy",  skill_vecs)
    np.save(out_dir / "jd_vectors.npy",        np.vstack([jd_career_vec, jd_skill_vec]))
    np.save(out_dir / "candidate_ids.npy",     np.array(cids))

    print(f"Saved career_embeddings.npy  shape={career_vecs.shape}")
    print(f"Saved skill_embeddings.npy   shape={skill_vecs.shape}")
    print(f"Saved jd_vectors.npy         shape=(2, {career_vecs.shape[1]})")
    print(f"Saved candidate_ids.npy      shape=({len(cids)},)")
    print("Done.")


if __name__ == "__main__":
    main()
