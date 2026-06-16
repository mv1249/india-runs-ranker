"""
Offline pre-computation step.
Reads candidates.jsonl, computes:
  1. career_embeddings.npy  — SBERT vectors for career descriptions (100K x 384)
  2. features.parquet       — structured feature matrix (100K rows)

Run once; output is used by rank.py (which must finish in <5 min, CPU-only).

Usage:
    python feature_extractor.py --candidates "../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
"""

import argparse
import json
import math
import os
import re
import ssl
from datetime import date, datetime
from pathlib import Path

# macOS Python 3.11 ships without system CA bundle linked; patch before any network call
ssl._create_default_https_context = ssl._create_unverified_context
import httpx as _httpx
_orig_httpx_client = _httpx.Client
class _NoVerifyHttpxClient(_orig_httpx_client):
    def __init__(self, *a, **kw):
        kw["verify"] = False
        super().__init__(*a, **kw)
_httpx.Client = _NoVerifyHttpxClient

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFERENCE_DATE = date(2026, 6, 16)

CONSULTING_GIANTS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree", "ltimindtree",
    "l&t infotech", "niit technologies", "cyient", "mphasiis",
}

PRODUCT_INDUSTRIES = {
    "ai/ml", "e-commerce", "food delivery", "fintech", "transportation",
    "edtech", "healthtech", "saas", "gaming", "adtech", "media tech",
    "software", "internet", "marketplace", "cloud", "cybersecurity",
}

ML_TITLE_KEYWORDS = {
    "ml engineer", "machine learning", "ai engineer", "nlp engineer",
    "search engineer", "recommendation", "applied ml", "data scientist",
    "research engineer", "ranking engineer", "retrieval", "recsys",
    "information retrieval", "deep learning engineer",
}

TIER_A_SKILLS = {
    "vector search", "embedding", "embeddings", "faiss", "qdrant", "pinecone",
    "weaviate", "milvus", "opensearch", "elasticsearch", "bm25", "hybrid search",
    "information retrieval", "sentence transformers", "sentence-transformers",
    "bge", "e5", "reranking", "re-ranking", "learning to rank", "learning-to-rank",
    "ltr", "ndcg", "mrr", "map", "retrieval evaluation", "dense retrieval",
    "sparse retrieval", "ann search", "approximate nearest neighbor",
    "vector database", "semantic search",
}

TIER_B_SKILLS = {
    "nlp", "transformers", "hugging face", "huggingface", "fine-tuning", "fine tuning",
    "lora", "qlora", "peft", "recommendation systems", "search", "ranking",
    "feature engineering", "xgboost", "lightgbm", "a/b testing", "mlflow",
    "ray", "mlops", "feature store", "online learning", "bandit",
    "knowledge graph", "bert", "gpt", "llm", "rag", "generative ai",
    "text classification", "named entity recognition", "ner",
}

TIER_C_SKILLS = {
    "python", "pytorch", "tensorflow", "scikit-learn", "sklearn", "spark",
    "pyspark", "kafka", "distributed systems", "docker", "kubernetes",
    "sql", "airflow", "dbt", "data engineering", "mlpipeline",
}

NEGATIVE_SKILLS = {
    "cad", "solidworks", "creo", "ansys", "autocad", "matlab",
    "accounting", "tally", "excel", "powerpoint", "sap",
    "yolo", "object detection", "image segmentation", "3d reconstruction",
    "speech recognition", "tts", "text to speech", "asr",
}

INDIA_PREFERRED_CITIES = {
    "pune", "noida", "delhi", "gurgaon", "gurugram", "hyderabad",
    "bangalore", "bengaluru", "mumbai", "chennai", "ncr",
}

# The JD query used to embed — captures intent, not just keywords
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

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_candidates(path: str):
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def build_career_text(candidate: dict) -> str:
    parts = []
    for job in candidate["career_history"]:
        parts.append(
            f"{job['title']} at {job['company']} ({job['industry']}): {job['description']}"
        )
    summary = candidate["profile"].get("summary", "")
    headline = candidate["profile"].get("headline", "")
    return f"{headline}. {summary} " + " ".join(parts)


def build_skill_text(candidate: dict) -> str:
    """Trust-weighted skill text: repeat high-trust skills, omit low-trust."""
    tokens = []
    for s in candidate["skills"]:
        name = s["name"]
        prof = s["proficiency"]
        endorse = s.get("endorsements", 0)
        dur = s.get("duration_months", 0)
        trust = (
            {"beginner": 0.2, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}[prof]
            * math.log1p(endorse)
            * math.log1p(dur)
        )
        # Repeat proportional to trust (capped at 3x) so SBERT attends more to it
        repeats = max(1, min(3, int(trust / 2)))
        tokens.extend([name] * repeats)
    return " ".join(tokens)


def skill_trust_score(candidate: dict) -> float:
    score = 0.0
    for s in candidate["skills"]:
        name_lower = s["name"].lower()
        prof = s["proficiency"]
        endorse = s.get("endorsements", 0)
        dur = s.get("duration_months", 0)

        # Skip skills that are pure negatives
        if any(k in name_lower for k in NEGATIVE_SKILLS):
            continue

        prof_w = {"beginner": 0.2, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}[prof]
        trust = prof_w * math.log1p(endorse) * math.log1p(dur)

        if any(k in name_lower for k in TIER_A_SKILLS):
            score += trust * 3.0
        elif any(k in name_lower for k in TIER_B_SKILLS):
            score += trust * 1.5
        elif any(k in name_lower for k in TIER_C_SKILLS):
            score += trust * 0.5

    return score


def career_fit_score(candidate: dict) -> tuple[float, bool]:
    """Returns (score, is_consulting_only)."""
    history = candidate["career_history"]
    score = 0.0
    consulting_count = 0
    total_jobs = len(history)

    for job in history:
        company_lower = job["company"].lower()
        title_lower = job["title"].lower()
        industry_lower = job["industry"].lower()
        dur = min(job["duration_months"], 48)  # cap at 4 years to avoid inflation

        is_consulting = any(c in company_lower for c in CONSULTING_GIANTS)
        is_product_industry = any(ind in industry_lower for ind in PRODUCT_INDUSTRIES)
        is_ml_title = any(t in title_lower for t in ML_TITLE_KEYWORDS)

        if is_consulting:
            consulting_count += 1

        if is_ml_title:
            score += dur * 2.5
        if is_product_industry and not is_consulting:
            score += 15
        if not is_consulting:
            score += dur * 0.3  # general product experience

    consulting_only = (consulting_count == total_jobs) and total_jobs > 0

    if consulting_only:
        score *= 0.25  # JD explicitly calls this out as disqualifier

    return score, consulting_only


def location_score(candidate: dict) -> float:
    sig = candidate["redrob_signals"]
    loc = candidate["profile"].get("location", "").lower()
    country = candidate["profile"].get("country", "")
    relocate = sig.get("willing_to_relocate", False)

    if country == "India":
        in_pref_city = any(city in loc for city in INDIA_PREFERRED_CITIES)
        if in_pref_city:
            return 1.0
        elif relocate:
            return 0.75
        else:
            return 0.5  # India but non-preferred city
    else:
        return 0.3 if relocate else 0.05


def availability_score(candidate: dict) -> float:
    sig = candidate["redrob_signals"]

    # Recency decay — linear over 6 months
    try:
        last_active = date.fromisoformat(sig["last_active_date"])
    except Exception:
        last_active = date(2020, 1, 1)
    days_inactive = max(0, (REFERENCE_DATE - last_active).days)
    recency = max(0.0, 1.0 - days_inactive / 180.0)

    # Notice period buckets
    notice = sig.get("notice_period_days", 60)
    if notice <= 0:
        notice_score = 1.0
    elif notice <= 30:
        notice_score = 0.95
    elif notice <= 60:
        notice_score = 0.85
    elif notice <= 90:
        notice_score = 0.65
    elif notice <= 120:
        notice_score = 0.40
    elif notice <= 150:
        notice_score = 0.20
    else:
        notice_score = 0.10

    open_flag = float(sig.get("open_to_work_flag", False))
    resp_rate = sig.get("recruiter_response_rate", 0.0)
    interview_rate = sig.get("interview_completion_rate", 0.5)
    offer_rate = sig.get("offer_acceptance_rate", -1)
    offer_score = offer_rate if offer_rate >= 0 else 0.5  # unknown → neutral

    composite = (
        0.30 * open_flag
        + 0.25 * recency
        + 0.20 * resp_rate
        + 0.15 * interview_rate
        + 0.10 * offer_score
    )
    return composite * notice_score


def honeypot_flag(candidate: dict) -> float:
    """Returns 0.0 if honeypot detected, 1.0 otherwise."""
    history = candidate["career_history"]
    claimed_yoe = candidate["profile"].get("years_of_experience", 0)

    # Check 1: impossible timeline (career months >> claimed YoE by >3 years)
    total_months = sum(h.get("duration_months", 0) for h in history)
    if total_months / 12.0 > claimed_yoe + 3.0:
        return 0.0

    # Check 2: many expert/advanced skills with 0 duration (bulk keyword listing)
    zero_dur_experts = sum(
        1 for s in candidate["skills"]
        if s["proficiency"] in ("expert", "advanced") and s.get("duration_months", 0) <= 1
    )
    if zero_dur_experts >= 4:
        return 0.0

    # Check 3: suspiciously high expert count with low endorsements
    expert_skills = [s for s in candidate["skills"] if s["proficiency"] == "expert"]
    if len(expert_skills) >= 7:
        avg_endorse = sum(s.get("endorsements", 0) for s in expert_skills) / len(expert_skills)
        if avg_endorse < 3:
            return 0.0

    return 1.0


def yoe_fit(candidate: dict) -> float:
    """Score how close YoE is to the JD sweet spot of 5-9 years."""
    yoe = candidate["profile"].get("years_of_experience", 0)
    if 5 <= yoe <= 9:
        return 1.0
    elif 4 <= yoe < 5 or 9 < yoe <= 12:
        return 0.8
    elif 3 <= yoe < 4 or 12 < yoe <= 15:
        return 0.5
    else:
        return 0.2


def education_score(candidate: dict) -> float:
    tier_scores = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.25, "unknown": 0.4}
    degrees = candidate.get("education", [])
    if not degrees:
        return 0.3
    best = max(tier_scores.get(d.get("tier", "unknown"), 0.4) for d in degrees)
    return best


def github_score_norm(candidate: dict) -> float:
    g = candidate["redrob_signals"].get("github_activity_score", -1)
    if g < 0:
        return 0.2  # no GitHub linked → neutral, not zero
    return min(g / 100.0, 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out_dir", default=".", help="Output directory for artifacts")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SBERT model name")
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates):,} candidates.")

    # -----------------------------------------------------------------------
    # Structured feature extraction
    # -----------------------------------------------------------------------
    print("Computing structured features...")
    rows = []
    for c in tqdm(candidates, desc="Features"):
        cid = c["candidate_id"]
        s_trust = skill_trust_score(c)
        c_score, consulting_only = career_fit_score(c)
        loc = location_score(c)
        avail = availability_score(c)
        hp = honeypot_flag(c)
        yoe = yoe_fit(c)
        edu = education_score(c)
        gh = github_score_norm(c)

        rows.append({
            "candidate_id": cid,
            "skill_trust_score": s_trust,
            "career_score": c_score,
            "location_score": loc,
            "availability_score": avail,
            "honeypot_flag": hp,
            "yoe_fit": yoe,
            "education_score": edu,
            "github_score": gh,
            "consulting_only": int(consulting_only),
            "years_of_experience": c["profile"].get("years_of_experience", 0),
            "current_title": c["profile"].get("current_title", ""),
            "location": c["profile"].get("location", ""),
            "country": c["profile"].get("country", ""),
            "notice_period_days": c["redrob_signals"].get("notice_period_days", 60),
            "open_to_work_flag": int(c["redrob_signals"].get("open_to_work_flag", False)),
            "recruiter_response_rate": c["redrob_signals"].get("recruiter_response_rate", 0),
            "last_active_date": c["redrob_signals"].get("last_active_date", ""),
        })

    feat_df = pd.DataFrame(rows)
    feat_path = out_dir / "features.parquet"
    feat_df.to_parquet(feat_path, index=False)
    print(f"Saved features → {feat_path}")

    # -----------------------------------------------------------------------
    # SBERT embedding
    # -----------------------------------------------------------------------
    print(f"Loading SBERT model: {args.model}")
    model = SentenceTransformer(args.model)

    print("Building career texts...")
    career_texts = [build_career_text(c) for c in candidates]
    skill_texts = [build_skill_text(c) for c in candidates]

    print("Encoding career texts (this is the slow part)...")
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

    # Encode JD queries
    jd_career_vec = model.encode([JD_QUERY], normalize_embeddings=True, convert_to_numpy=True)
    jd_skill_vec = model.encode([SKILL_QUERY], normalize_embeddings=True, convert_to_numpy=True)

    # Save
    career_path = out_dir / "career_embeddings.npy"
    skill_path = out_dir / "skill_embeddings.npy"
    jd_path = out_dir / "jd_vectors.npy"
    cid_path = out_dir / "candidate_ids.npy"

    np.save(career_path, career_vecs)
    np.save(skill_path, skill_vecs)
    np.save(jd_path, np.vstack([jd_career_vec, jd_skill_vec]))
    np.save(cid_path, np.array([c["candidate_id"] for c in candidates]))

    print(f"Saved career embeddings → {career_path}  shape={career_vecs.shape}")
    print(f"Saved skill embeddings  → {skill_path}   shape={skill_vecs.shape}")
    print(f"Saved JD vectors        → {jd_path}")
    print("Pre-computation complete.")


if __name__ == "__main__":
    main()
