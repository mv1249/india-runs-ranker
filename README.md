# India Runs — Multi-Signal Cascade Ranker

**Redrob India Runs Data & AI Challenge 2026**  
Rank the top-100 candidates for a Senior AI Engineer role from a pool of 100,000 profiles.

---

## The Problem

Given:
- `candidates.jsonl` — 100,000 candidate profiles (487 MB), each with career history, skills (with proficiency, endorsements, duration), education, location, and behavioral signals
- A **Job Description** — Senior AI Engineer, retrieval / ranking / recsys focus, India-based, product company preferred, 5–9 YoE

Produce:
- A `team_xxx.csv` with exactly 100 rows: `candidate_id, rank, score, reasoning`
- Scores must be non-increasing (validator-enforced)
- Fewer than 10 honeypots in the top-100 (disqualifier if exceeded)
- Reasoning strings must be specific to each candidate — no templated filler

**Evaluation:**
| Metric | Weight |
|---|---|
| NDCG@10 | 50% |
| NDCG@50 | 30% |
| Manual reasoning review | Stage 4 |
| Honeypot rate < 10% in top-100 | Hard disqualifier |

---

## Why Simple Approaches Fail

| Approach | The Problem |
|---|---|
| **Keyword / BM25** | Surfaces HR managers who listed "Pinecone" on their profile |
| **Pure cosine similarity on skills** | Honeypots (coherent skill text, impossible timelines) score high |
| **LLM per candidate** | 100K × 2s = 55+ hours; violates compute constraints |
| **Skills-only ranking** | Misses ML engineers who describe systems in plain language without buzzwords |

---

## Our Approach: Multi-Signal Cascade Ranker

A **deterministic, interpretable, feature-engineering-first** system that combines semantic understanding with trust-weighted structured signals. No per-candidate LLM calls. No vector databases. Fully reproducible.

### Two-Phase Architecture

```
OFFLINE (run once — unconstrained time)
────────────────────────────────────────────────────────────────
candidates.jsonl (100K, 487MB)
        │
        ▼
feature_extractor.py ──► features.parquet       (100K × 18 signals, 2.5MB)
        │
        ▼
SBERT Encoder (all-MiniLM-L6-v2, Apple MPS GPU, batch=256)
        │
        ├──► career_embeddings.npy   (100K × 384, 146MB)
        ├──► skill_embeddings.npy    (100K × 384, 146MB)
        └──► jd_vectors.npy          (2 × 384, career + skill query)
        
        + candidate_ids.npy          (alignment index, 4.6MB)

ONLINE (rank.py — pure numpy, CPU-only, < 5 seconds for 100K)
────────────────────────────────────────────────────────────────
Load artifacts → matmul cosine similarity → weighted scoring
→ × honeypot flag → top-100 sort → submission CSV
```

**Key insight:** Pre-computation happens once, offline. The ranking step is pure numpy matrix multiplication — 100K cosine similarities in under 100ms, no network, no inference API.

---

## The Five Signals

### Signal 1 — Career SBERT Similarity (35%)

**Model:** `sentence-transformers/all-MiniLM-L6-v2` — 22M parameters, 384-dim embeddings, designed for sentence-level semantic similarity.

We embed each candidate's **full career history** as a single passage:
```
{title} at {company} ({industry}): {description}. 
{next_title} at {next_company} ...
```

We also embed a carefully **distilled JD intent query** (not the raw JD text — the *intent*):

> *"Senior AI engineer building production ranking retrieval matching systems. Embedding-based retrieval, vector databases, FAISS, Qdrant, hybrid search, BM25. Learning-to-rank, NDCG, MRR, recsys, NLP. Shipped real systems to production users at product companies."*

Cosine similarity between a candidate's career vector and this JD vector becomes `career_sbert_score`.

**Why this works:**  
A RecSys engineer at Swiggy who writes *"led migration from keyword-search to embedding-based retrieval, offline-online correlation"* scores **0.66** cosine similarity.  
A Project Manager at Wipro whose career descriptions cover project delivery and stakeholder management scores **0.31** — even if their skill list says "Pinecone, FAISS, Embeddings."

---

### Signal 2 — Skill SBERT Similarity (20%)

We construct a **trust-weighted skill text** for each candidate. Skills are repeated in the text proportional to `proficiency × log(1 + endorsements) × log(1 + duration_months)`. A skill with 4 months and 4 endorsements barely appears; a skill with 88 months and 34 endorsements dominates.

This weighted text is then embedded by SBERT and compared against a **skill-focused JD query:**

> *"FAISS Qdrant Pinecone BM25 sentence-transformers NDCG learning-to-rank information retrieval hybrid search NLP transformers fine-tuning LoRA RAG recommendation systems."*

**Why separate from Signal 1:**  
Career descriptions often describe *what was built* in plain language. Skill texts focus on *technology names*. Separating them allows each to contribute independently — a plain-language career description scores high on Signal 1; a strong tool-specific skill set scores high on Signal 2.

---

### Signal 3 — Trust-Weighted Skill Score (20%)

An explicit structured score that kills keyword stuffers independently of any embedding.

**Formula:**
```python
skill_trust = proficiency_weight × log(1 + endorsements) × log(1 + duration_months)
```

**Tier taxonomy:**
| Tier | Weight | Skills |
|---|---|---|
| **A** | 3× | FAISS, Qdrant, Pinecone, BM25, sentence-transformers, NDCG, learning-to-rank, information retrieval, hybrid search, vector databases |
| **B** | 1.5× | NLP, transformers, fine-tuning, LoRA, RAG, LLMs, recommendation systems, XGBoost, PyTorch |
| **C** | 0.5× | Python, SQL, Docker, Kubernetes, general engineering |

**Real contrast:**
| Candidate | Skill | Duration | Endorsements | Trust Score |
|---|---|---|---|---|
| PM @ Wipro | Fine-tuning LLMs (advanced) | 4 months | 4 | **2.1** |
| ML Eng @ Netflix | Elasticsearch (expert) | 96 months | 44 | **183** |

The PM who listed 9 AI skills overnight scores skill trust ≈ 12 total. The genuine ML engineer scores 300+.

---

### Signal 4 — Career Trajectory Score (12%)

Three components that reward the right career shape:

**1. ML title bonus** — Roles matching: ML Engineer, Search Engineer, RecSys Engineer, Applied ML, NLP Engineer, Research Engineer, Data Scientist (with ML focus) → `+duration_months × 2.5` points per role.

**2. Product company premium** — Industries flagged as product-first (AI/ML, E-commerce, Fintech, SaaS, EdTech, Gaming) → +15 points per qualifying role.

**3. Consulting-only penalty** — Candidates whose *entire career* is at Wipro / TCS / Infosys / Accenture / Cognizant → score multiplied by **0.25**. The JD explicitly states preference for people who shipped to real users, not IT services delivery.

---

### Signal 5 — Behavioral Availability (7%)

A perfect-on-paper candidate who hasn't logged in for 6 months is not actually available to hire. This composite captures real reachability:

```python
availability = (
    0.30 × open_to_work_flag
  + 0.25 × recency_score          # linear decay, 0 after 180 days inactive
  + 0.20 × recruiter_response_rate
  + 0.15 × interview_completion_rate
  + 0.10 × offer_acceptance_rate
) × notice_period_multiplier       # 0d→1.0, 30d→0.95, 60d→0.85, 120d→0.40, 180d+→0.10
```

**Effect:**  
A Staff ML Engineer at Paytm/Razorpay with response rate 0.95, active 21 days ago, 60-day notice → availability **0.78** → rises to rank #2.  
A Senior NLP Engineer with response rate 0.16, inactive 8 months → availability **0.13** → drops 30+ ranks despite excellent skill scores.

### Other Signals (combined 6%)

- **Location score (3%):** India-based gets 1.0; clear non-India gets 0.2; ambiguous gets 0.6.
- **GitHub activity (1%):** Normalized public repo/contribution count.
- **Years of experience fit (1%):** Gaussian peaked at 7 YoE (JD target is 5–9).
- **Education tier (1%):** IIT/IISc/top-10 global = 1.0, tier-2 = 0.6, others = 0.3.

---

## Honeypot Defense

Two hard-zero detection rules run before any scoring:

**Rule 1 — Timeline impossibility:**
```python
if total_career_months / 12 > claimed_yoe + 3:
    honeypot_flag = 0   # impossible tenure
```
Catches: "8 years of experience at a company that's been around for 3 years."

**Rule 2 — Bulk expert listing:**
```python
if count(skills where proficiency in ['expert','advanced'] and duration_months <= 1) >= 4:
    honeypot_flag = 0   # skills listed overnight
```
Catches: candidates who added 15 "expert" AI skills to game the system.

**Result:** 35 honeypots identified across 100K candidates. **0 appear in the top-100.** (Hard disqualifier threshold is 10.)

---

## Final Scoring Formula

```python
final_score = (
    0.35 × normalize(career_sbert_cosine)
  + 0.20 × normalize(skill_sbert_cosine)
  + 0.20 × normalize(skill_trust_score)
  + 0.12 × normalize(career_trajectory_score)
  + 0.07 × availability_score
  + 0.03 × location_score
  + 0.01 × github_score_norm
  + 0.01 × yoe_fit
  + 0.01 × education_score
) × honeypot_flag    # 0 for detected honeypots, 1 otherwise
```

All continuous signals are min-max normalized across the full 100K before combination. The honeypot flag is the **only hard zero** — everything else is a gradient.

Tie-breaking: `sort_values(["final_score", "candidate_id"], ascending=[False, True])`.

---

## Results

| Rank | Candidate ID | Title | Career Path | Score |
|---|---|---|---|---|
| 1 | CAND_0045250 | Applied ML Engineer | Rephrase.ai → Paytm | 0.8350 |
| 2 | CAND_0077337 | Staff ML Engineer | Paytm → Razorpay → Glance | 0.8281 |
| 3 | CAND_0071974 | Senior AI Engineer | Netflix → Meta → Mad Street Den | 0.8205 |
| 4 | CAND_0081846 | Lead AI Engineer | Razorpay → Paytm | 0.8196 |
| 5 | CAND_0094056 | NLP Engineer | Rephrase.ai → Adobe | 0.8152 |

**Score range:** 0.6991 – 0.8350 across top-100  
**All India-based** ✅ | **All product-company careers** ✅ | **0 consulting-only** ✅ | **0 honeypots** ✅

**Correctly eliminated:**
- PM @ Wipro listing 9 AI skills → career SBERT 0.31 + consulting penalty → not in top-100
- Honeypot with 8-year claim at a 3-year-old company → hard zero
- Great profile with 8-month inactivity + 0.12 response rate → availability 0.05 → drops 40+ ranks

---

## Repository Structure

```
.
├── feature_extractor.py      # 18-signal structured feature builder
│                             # Outputs: features.parquet
├── embed_only.py             # MPS-accelerated SBERT embedding generation
│                             # Outputs: career_embeddings.npy, skill_embeddings.npy,
│                             #          jd_vectors.npy, candidate_ids.npy
├── rank.py                   # Online ranker — loads artifacts, scores, outputs CSV
├── gradio_app.py             # Self-contained HuggingFace Spaces sandbox
├── requirements.txt          # Reproducible Python environment
├── submission_metadata.yaml  # Approach summary and team details
├── deck.html                 # 11-slide presentation (open in browser, print to PDF)
│
├── features.parquet          # Pre-computed structured features (100K × 18, 2.5MB)
├── candidate_ids.npy         # Candidate ID alignment index (4.6MB)
├── jd_vectors.npy            # JD query embeddings (2 × 384, 3KB)
└── team_xxx.csv              # Final submission — 100 ranked candidates
```

> **Note:** `career_embeddings.npy` and `skill_embeddings.npy` (146MB each) exceed GitHub's file size limit and are excluded from this repo. Regenerate them with `embed_only.py` (see below).

---

## Reproduce from Scratch

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate structured features

```bash
python feature_extractor.py \
  --candidates /path/to/candidates.jsonl \
  --out features.parquet
```

Takes ~3–5 minutes on CPU. Produces `features.parquet` (100K × 18 columns, 2.5MB).

### 3. Generate SBERT embeddings (offline, GPU-accelerated)

```bash
python embed_only.py \
  --candidates /path/to/candidates.jsonl \
  --out_dir .
```

Auto-detects device: **MPS** (Apple Silicon) > CUDA > CPU.  
Runtime: ~18 min on Apple M-series MPS at batch=256.  
Outputs: `career_embeddings.npy`, `skill_embeddings.npy`, `jd_vectors.npy`, `candidate_ids.npy`.

### 4. Run the online ranker

```bash
python rank.py \
  --candidates /path/to/candidates.jsonl \
  --artifacts_dir . \
  --out team_xxx.csv
```

Runs in **< 5 seconds on CPU**. Outputs validated CSV with `candidate_id, rank, score, reasoning`.

### 5. Validate the submission

```bash
python validate_submission.py --submission team_xxx.csv
```

---

## Technical Stack

| Component | Choice | Reason |
|---|---|---|
| Semantic embeddings | `sentence-transformers` · `all-MiniLM-L6-v2` | Best CPU-deployable semantic model at 22M params, 384-dim, 14k tok/sec |
| GPU acceleration | Apple Silicon MPS via PyTorch | Reduced offline embedding time from ~90 min (CPU) to ~18 min |
| Online ranking | Pure `numpy` matmul | `career_vecs @ jd_vec` → 100K cosine similarities in <100ms, zero overhead |
| Feature storage | `pandas` + `pyarrow` parquet | Typed columnar format, fast I/O, <3MB on disk |
| Sandbox | `gradio` on HuggingFace Spaces | Self-contained, no external artifacts, accepts JSON → returns CSV |
| No LLM at inference | Deliberate | Zero latency, zero cost per candidate, deterministic output |

---

## Sandbox

The Gradio sandbox accepts up to 200 candidates as a JSON array and returns a ranked CSV inline — no pre-computed artifacts needed, all feature extraction and scoring runs in-process.

**Link:** [HuggingFace Spaces — redrob-ranker](#) *(deploy `gradio_app.py` to your Space)*

---

## Submission Checklist

- [x] Ranked output CSV (`team_xxx.csv`) — 100 rows, validated, 0 violations
- [x] GitHub repository — clean, reproducible, all code committed
- [x] Presentation deck (`deck.html`) — 11 slides following Redrob template
- [x] Gradio sandbox (`gradio_app.py`) — ready for HuggingFace Spaces deployment
- [x] `submission_metadata.yaml` — approach summary, team details, declarations
- [ ] Rename `team_xxx.csv` to actual team ID before portal submission
- [ ] Fill in GitHub repo + sandbox links in `submission_metadata.yaml`
