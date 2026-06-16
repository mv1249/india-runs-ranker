"""
Redrob Ranker — HuggingFace Spaces Gradio sandbox.

Accepts a small JSON array of candidates (same schema as candidates.jsonl),
runs the ranking pipeline inline, and returns a ranked CSV.

The JD vectors are embedded from JD_QUERY at startup.
No pre-computed artifacts needed — works self-contained on the input batch.

Deploy to HuggingFace Spaces:
  - Runtime: CPU Basic (free tier)
  - requirements.txt: sentence-transformers, numpy, pandas, gradio

Usage locally:
  pip install gradio sentence-transformers numpy pandas
  python gradio_app.py
"""

import csv
import io
import json
import math
import ssl
from datetime import date

ssl._create_default_https_context = ssl._create_unverified_context

import gradio as gr
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants (mirrored from feature_extractor.py / rank.py)
# ---------------------------------------------------------------------------

REFERENCE_DATE = date(2026, 6, 16)

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

CONSULTING_GIANTS = {
    "tcs","infosys","wipro","accenture","cognizant","capgemini",
    "hcl","tech mahindra","mphasis","hexaware","mindtree","ltimindtree",
}
PRODUCT_INDUSTRIES = {
    "ai/ml","e-commerce","food delivery","fintech","transportation",
    "edtech","healthtech","saas","gaming","adtech","media tech",
    "software","internet","marketplace","cloud","cybersecurity",
}
ML_TITLE_KEYWORDS = {
    "ml engineer","machine learning","ai engineer","nlp engineer",
    "search engineer","recommendation","applied ml","data scientist",
    "research engineer","ranking engineer","retrieval","recsys",
    "information retrieval","deep learning engineer",
}
TIER_A = {"vector search","embedding","embeddings","faiss","qdrant","pinecone",
           "weaviate","milvus","opensearch","elasticsearch","bm25","hybrid search",
           "information retrieval","sentence transformers","bge","e5","reranking",
           "re-ranking","learning to rank","ltr","ndcg","mrr","map","semantic search"}
TIER_B = {"nlp","transformers","hugging face","fine-tuning","lora","qlora","peft",
           "recommendation systems","search","ranking","feature engineering","xgboost",
           "lightgbm","mlops","rag","llm","bert","text classification"}
TIER_C = {"python","pytorch","tensorflow","scikit-learn","spark","kafka",
           "docker","kubernetes","sql","airflow"}
INDIA_CITIES = {"pune","noida","delhi","gurgaon","gurugram","hyderabad",
                "bangalore","bengaluru","mumbai","chennai","ncr"}

W_CAREER_SBERT = 0.35
W_SKILL_SBERT  = 0.20
W_SKILL_TRUST  = 0.20
W_CAREER_FIT   = 0.12
W_AVAILABILITY = 0.08
W_LOCATION     = 0.03
W_YOE          = 0.01
W_EDUCATION    = 0.01

# ---------------------------------------------------------------------------
# Model — loaded once at startup
# ---------------------------------------------------------------------------

print("Loading SBERT model...")
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
JD_CAREER_VEC = MODEL.encode([JD_QUERY], normalize_embeddings=True)[0]
JD_SKILL_VEC  = MODEL.encode([SKILL_QUERY], normalize_embeddings=True)[0]
print("Model ready.")


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------

def build_career_text(c):
    parts = []
    for job in c.get("career_history", []):
        parts.append(f"{job['title']} at {job['company']} ({job['industry']}): {job['description']}")
    return f"{c['profile'].get('headline','')}. {c['profile'].get('summary','')} " + " ".join(parts)


def build_skill_text(c):
    tokens = []
    for s in c.get("skills", []):
        prof_w = {"beginner":0.2,"intermediate":0.5,"advanced":0.8,"expert":1.0}.get(s["proficiency"],0.5)
        trust = prof_w * math.log1p(s.get("endorsements",0)) * math.log1p(s.get("duration_months",0))
        repeats = max(1, min(3, int(trust/2)))
        tokens.extend([s["name"]] * repeats)
    return " ".join(tokens)


def skill_trust(c):
    score = 0.0
    for s in c.get("skills", []):
        n = s["name"].lower()
        prof_w = {"beginner":0.2,"intermediate":0.5,"advanced":0.8,"expert":1.0}.get(s["proficiency"],0.5)
        trust = prof_w * math.log1p(s.get("endorsements",0)) * math.log1p(s.get("duration_months",0))
        if any(k in n for k in TIER_A): score += trust * 3.0
        elif any(k in n for k in TIER_B): score += trust * 1.5
        elif any(k in n for k in TIER_C): score += trust * 0.5
    return score


def career_fit(c):
    score, consulting_count, total = 0.0, 0, len(c.get("career_history", []))
    for job in c.get("career_history", []):
        co = job["company"].lower(); ti = job["title"].lower(); ind = job["industry"].lower()
        dur = min(job.get("duration_months",0), 48)
        is_c = any(x in co for x in CONSULTING_GIANTS)
        if is_c: consulting_count += 1
        if any(t in ti for t in ML_TITLE_KEYWORDS): score += dur * 2.5
        if any(p in ind for p in PRODUCT_INDUSTRIES) and not is_c: score += 15
        if not is_c: score += dur * 0.3
    if total > 0 and consulting_count == total: score *= 0.25
    return score, (total > 0 and consulting_count == total)


def location_sc(c):
    sig = c.get("redrob_signals", {})
    loc = c["profile"].get("location","").lower()
    ctry = c["profile"].get("country","")
    rel = sig.get("willing_to_relocate", False)
    if ctry == "India":
        if any(city in loc for city in INDIA_CITIES): return 1.0
        return 0.75 if rel else 0.5
    return 0.3 if rel else 0.05


def availability_sc(c):
    sig = c.get("redrob_signals", {})
    try: last = date.fromisoformat(sig["last_active_date"])
    except: last = date(2020,1,1)
    days = max(0,(REFERENCE_DATE - last).days)
    recency = max(0.0, 1.0 - days/180.0)
    notice = sig.get("notice_period_days", 60)
    ns = {0:1.0,30:0.95,60:0.85,90:0.65,120:0.4,150:0.2}.get(notice, 0.1 if notice > 150 else 0.5)
    comp = (0.30*float(sig.get("open_to_work_flag",False))
          + 0.25*recency
          + 0.20*sig.get("recruiter_response_rate",0)
          + 0.15*sig.get("interview_completion_rate",0.5)
          + 0.10*max(sig.get("offer_acceptance_rate",-1), 0))
    return comp * ns


def honeypot_check(c):
    h = c.get("career_history", [])
    yoe = c["profile"].get("years_of_experience", 0)
    total_m = sum(x.get("duration_months",0) for x in h)
    if total_m/12 > yoe + 3: return 0.0
    zero_dur = sum(1 for s in c.get("skills",[]) if s["proficiency"] in ("expert","advanced") and s.get("duration_months",0)<=1)
    if zero_dur >= 4: return 0.0
    return 1.0


def yoe_fit(c):
    y = c["profile"].get("years_of_experience", 0)
    if 5 <= y <= 9: return 1.0
    if 4 <= y < 5 or 9 < y <= 12: return 0.8
    if 3 <= y < 4 or 12 < y <= 15: return 0.5
    return 0.2


def edu_score(c):
    tier = {"tier_1":1.0,"tier_2":0.75,"tier_3":0.5,"tier_4":0.25,"unknown":0.4}
    eds = c.get("education", [])
    return max((tier.get(d.get("tier","unknown"),0.4) for d in eds), default=0.3)


def normalize(arr):
    mn, mx = arr.min(), arr.max()
    if mx == mn: return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# ---------------------------------------------------------------------------
# Main ranking function
# ---------------------------------------------------------------------------

def rank_candidates(json_input: str) -> tuple[str, str]:
    """Returns (csv_string, status_message)."""
    try:
        candidates = json.loads(json_input)
        if not isinstance(candidates, list):
            candidates = [candidates]
    except json.JSONDecodeError as e:
        return "", f"JSON parse error: {e}"

    if len(candidates) == 0:
        return "", "No candidates provided."
    if len(candidates) > 200:
        return "", "Please provide 200 or fewer candidates for the sandbox demo."

    # Build texts
    career_texts = [build_career_text(c) for c in candidates]
    skill_texts  = [build_skill_text(c)  for c in candidates]

    # Encode
    career_vecs = MODEL.encode(career_texts, normalize_embeddings=True, show_progress_bar=False)
    skill_vecs  = MODEL.encode(skill_texts,  normalize_embeddings=True, show_progress_bar=False)

    career_sims = career_vecs @ JD_CAREER_VEC
    skill_sims  = skill_vecs  @ JD_SKILL_VEC

    rows = []
    for i, c in enumerate(candidates):
        cf, consulting_only = career_fit(c)
        rows.append({
            "candidate_id":  c["candidate_id"],
            "current_title": c["profile"].get("current_title",""),
            "location":      c["profile"].get("location",""),
            "country":       c["profile"].get("country",""),
            "years_of_experience": c["profile"].get("years_of_experience",0),
            "career_sbert":  float(career_sims[i]),
            "skill_sbert":   float(skill_sims[i]),
            "skill_trust":   skill_trust(c),
            "career_score":  cf,
            "location_score":location_sc(c),
            "availability":  availability_sc(c),
            "honeypot_flag": honeypot_check(c),
            "yoe_fit":       yoe_fit(c),
            "education":     edu_score(c),
            "open_to_work":  int(c.get("redrob_signals",{}).get("open_to_work_flag",False)),
            "notice_days":   c.get("redrob_signals",{}).get("notice_period_days",60),
            "resp_rate":     c.get("redrob_signals",{}).get("recruiter_response_rate",0),
            "consulting_only": int(consulting_only),
        })

    df = pd.DataFrame(rows)
    n_feat = normalize(df["skill_trust"].values)
    n_career = normalize(df["career_score"].values)
    n_csbert = normalize(df["career_sbert"].values)
    n_ssbert = normalize(df["skill_sbert"].values)

    raw = (W_CAREER_SBERT * n_csbert + W_SKILL_SBERT * n_ssbert
         + W_SKILL_TRUST * n_feat + W_CAREER_FIT * n_career
         + W_AVAILABILITY * df["availability"].values
         + W_LOCATION * df["location_score"].values
         + W_YOE * df["yoe_fit"].values
         + W_EDUCATION * df["education"].values) * df["honeypot_flag"].values

    df["final_score"] = raw
    df = df.sort_values(["final_score","candidate_id"], ascending=[False,True]).reset_index(drop=True)

    top_n = min(100, len(df))
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=["candidate_id","rank","score","reasoning"])
    writer.writeheader()
    for rank_pos, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
        sim_str = f"semantic={row['career_sbert']:.2f}"
        sk_str = f"skill_trust={row['skill_trust']:.0f}"
        avail_note = ""
        if row["notice_days"] > 90:
            avail_note = f"; notice={row['notice_days']}d"
        reasoning = (f"{row['current_title']} | {sim_str}, {sk_str} | "
                     f"{'India' if row['country']=='India' else row['country']}"
                     f"{avail_note}")
        writer.writerow({"candidate_id": row["candidate_id"], "rank": rank_pos,
                         "score": round(float(row["final_score"]),6), "reasoning": reasoning})

    return out.getvalue(), f"Ranked {top_n} candidates. Top result: {df.iloc[0]['candidate_id']} — {df.iloc[0]['current_title']}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

SAMPLE = json.dumps([
    {
        "candidate_id": "CAND_0000001",
        "profile": {"current_title":"ML Engineer","years_of_experience":6,
                    "location":"Bangalore, Karnataka","country":"India",
                    "headline":"ML engineer at product startup","summary":"Built recsys at scale"},
        "career_history":[{"title":"ML Engineer","company":"Swiggy","industry":"Food Delivery",
                           "duration_months":36,"description":"Built embedding-based retrieval using FAISS"}],
        "skills":[{"name":"FAISS","proficiency":"expert","endorsements":20,"duration_months":36},
                  {"name":"Embeddings","proficiency":"expert","endorsements":15,"duration_months":30}],
        "education":[{"degree":"B.Tech CS","tier":"tier_1"}],
        "redrob_signals":{"open_to_work_flag":True,"last_active_date":"2026-06-10",
                          "recruiter_response_rate":0.8,"interview_completion_rate":0.9,
                          "offer_acceptance_rate":0.7,"notice_period_days":30,
                          "willing_to_relocate":True,"github_activity_score":40}
    }
], indent=2)

demo = gr.Interface(
    fn=rank_candidates,
    inputs=gr.Textbox(label="Candidate JSON (array of candidate objects)", value=SAMPLE, lines=20),
    outputs=[
        gr.Textbox(label="Ranked CSV output", lines=15),
        gr.Textbox(label="Status"),
    ],
    title="Redrob Candidate Ranker — Sandbox",
    description=(
        "Paste up to 200 candidates in the same JSON schema as `candidates.jsonl`. "
        "Returns a ranked CSV with `candidate_id, rank, score, reasoning`."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
