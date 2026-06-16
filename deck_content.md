# India Runs AI Challenge — Deck Content
# (Feed this to Gamma or convert to slides)

---

## SLIDE 1 — TITLE
**Making Hiring Smarter**
AI-Powered Candidate Ranking for Senior AI Engineering Roles

India Runs Data & AI Challenge | 2026
Team: [Your Name]

---

## SLIDE 2 — THE PROBLEM
**100,000 candidates. 1 role. 5 minutes.**

- The JD asks for a Senior AI Engineer — retrieval, ranking, recsys, production ML
- A naive keyword search would return HR Managers who listed "Pinecone" on their profile
- The challenge: build a system that understands *fit*, not just word overlap

**Three constraints that ruled out shortcuts:**
1. No per-candidate LLM calls — must run in <5 min on CPU
2. Honeypot candidates with impossible timelines must be caught
3. Reasoning strings must be specific and candidate-grounded (not templated)

---

## SLIDE 3 — THE 4 ARCHETYPES WE HAD TO SOLVE FOR

| Archetype | Example | Challenge |
|---|---|---|
| **The Golden Candidate** | RecSys Eng @ Swiggy, 6 yrs, expert FAISS/SBERT | Uses product language, not buzzwords — naive search misses them |
| **The Keyword Stuffer** | Project Manager @ Wipro, lists 9 AI skills with 4-month duration | Fools cosine similarity, must be penalised |
| **The Behavioral Ghost** | Great profile, last active 8 months ago, 0.12 response rate | Strong on paper, unreachable in practice |
| **The Honeypot** | Claims 8 yrs exp at a 3-year-old company | Temporally impossible — catches pure embedding rankers |

---

## SLIDE 4 — OUR ARCHITECTURE

**Multi-Signal Cascade Ranker**

```
OFFLINE (one-time, unconstrained time):
  candidates.jsonl (100K, 487MB)
       ↓
  Feature Extractor → features.parquet (structured signals)
                    → career_embeddings.npy  (100K × 384)
                    → skill_embeddings.npy   (100K × 384)

ONLINE (ranking step, <5 sec CPU):
  Load artifacts → weighted scoring → top-100 → CSV
```

**Why this split?**
Pre-computation happens once offline. The ranking step is pure numpy matrix multiplication — blazing fast, no network, no LLM calls.

---

## SLIDE 5 — SIGNAL 1: SEMANTIC CAREER FIT (35%)

**Model:** `all-MiniLM-L6-v2` (22M params, 384-dim, CPU-friendly)

**What we embed:**
- Each candidate's full career history as one passage: `{title} at {company} ({industry}): {description}`
- A carefully distilled JD query (not the raw JD text — the *intent*)

**JD Query used:**
> "Senior AI engineer building production ranking retrieval matching systems. Embedding-based retrieval, vector databases, FAISS, Qdrant, hybrid search, BM25. Learning-to-rank, NDCG, MRR, recsys, NLP. Shipped real systems to production users at product companies."

**Why it matters:**
A RecSys engineer at Swiggy who writes *"led migration from keyword-search to embedding-based retrieval, offline-online correlation"* scores 0.66 cosine similarity.
A Project Manager at Wipro whose career descriptions cover project delivery scores 0.31 — even if their skill list says "Pinecone, FAISS, Embeddings."

---

## SLIDE 6 — SIGNAL 2: TRUST-WEIGHTED SKILL MATCH (20% + 20%)

**The core insight:** listing a skill ≠ knowing a skill

**Trust formula:**
```
skill_trust = proficiency_weight × log(1 + endorsements) × log(1 + duration_months)
```

**Tier taxonomy:**
- **Tier A (3× weight):** FAISS, Qdrant, Pinecone, BM25, sentence-transformers, NDCG, learning-to-rank, information retrieval, hybrid search
- **Tier B (1.5×):** NLP, transformers, fine-tuning, LoRA, RAG, LLMs, recommendation systems, XGBoost
- **Tier C (0.5×):** Python, PyTorch, Docker, SQL

**Real contrast:**
| Candidate | Skill | Prof | Duration | Endorsements | Trust |
|---|---|---|---|---|---|
| PM @ Wipro | Fine-tuning LLMs | advanced | 4 months | 4 | **2.1** |
| ML Eng @ Netflix | Elasticsearch | expert | 96 months | 44 | **183** |

---

## SLIDE 7 — SIGNAL 3: CAREER TRAJECTORY (12%)

**Three components:**

1. **ML title bonus** — roles matching: ML Engineer, Search Engineer, RecSys Engineer, Applied ML, NLP Engineer, Research Engineer → +`duration × 2.5` points

2. **Product company premium** — industries like AI/ML, E-commerce, Fintech, SaaS, EdTech → +15 points per role

3. **Consulting penalty** — pure Wipro/TCS/Infosys/Accenture/Cognizant career → score × 0.25 (JD explicitly flags this)

**Why the JD agrees:**
> *"We're looking for people who've shipped systems to real users, not people who've spent years in IT services delivery."*

---

## SLIDE 8 — SIGNAL 4: BEHAVIORAL AVAILABILITY (7%)

**The JD's explicit warning:**
> "A perfect-on-paper candidate who hasn't logged in for 6 months is not actually available."

**Availability composite:**
```
availability = (
    0.30 × open_to_work_flag
  + 0.25 × recency_score         # linear decay over 180 days
  + 0.20 × recruiter_response_rate
  + 0.15 × interview_completion_rate
  + 0.10 × offer_acceptance_rate
) × notice_period_bucket          # 0-day→1.0, 30d→0.95, 120d→0.40, 180d+→0.10
```

**Effect:**
- Staff ML Engineer @ Paytm/Razorpay — resp=0.95, active 21 days ago, notice=60d → availability 0.78 → rises to rank #2
- Senior NLP Engineer — resp=0.16, inactive → availability 0.13 → drops to rank #10 despite excellent skill scores

---

## SLIDE 9 — HONEYPOT DEFENSE

**Catch #1: Timeline impossibility**
```python
total_career_months / 12 > claimed_yoe + 3 → hard zero
```
Catches: "8 years experience at a 3-year-old company" (impossible tenure)

**Catch #2: Bulk keyword listing**
```python
count(expert/advanced skills with duration ≤ 1 month) ≥ 4 → hard zero
```
Catches: candidates who listed 15 "expert" AI skills overnight

**Result:** 35 honeypots identified and hard-zeroed
**Top-100 honeypot rate: 0 / 100** (limit is <10 for disqualification)

---

## SLIDE 10 — FINAL SCORING FORMULA

```
final_score = (
    0.35 × normalize(career_sbert_cosine)
  + 0.20 × normalize(skill_sbert_cosine)
  + 0.20 × normalize(skill_trust_score)
  + 0.12 × normalize(career_trajectory_score)
  + 0.07 × availability_score
  + 0.03 × location_score
  + 0.01 × github_activity_norm
  + 0.01 × yoe_fit
  + 0.01 × education_tier
) × honeypot_flag   ← hard zero for detected honeypots
```

**Key design decisions:**
- SBERT is 55% of the score — semantic understanding dominates
- Skill trust weighting at 20% kills keyword stuffers independently of SBERT
- Availability is additive (not multiplicative) to avoid over-penalising great candidates with normal notice periods
- Honeypot is the only hard zero — everything else is a gradient

---

## SLIDE 11 — TOP-10 RESULTS

| Rank | Candidate | Title | Career Path | SBERT | Skill Trust |
|---|---|---|---|---|---|
| 1 | CAND_0045250 | Applied ML Engineer | Rephrase.ai → Paytm | 0.66 | 357 |
| 2 | CAND_0077337 | Staff ML Engineer | Paytm → Razorpay → Glance | 0.63 | 326 |
| 3 | CAND_0071974 | Senior AI Engineer | **Netflix → Meta** → Mad Street Den | 0.61 | 379 |
| 4 | CAND_0081846 | Lead AI Engineer | Razorpay → Paytm | 0.60 | 361 |
| 5 | CAND_0094056 | NLP Engineer | Rephrase.ai → **Adobe** | 0.69 | 320 |
| 6 | CAND_0088025 | Staff ML Engineer | Product companies | 0.64 | 308 |
| 7 | CAND_0062247 | AI Engineer | Product companies | 0.71 | 253 |
| 8 | CAND_0041610 | RecSys Engineer | Product companies | 0.66 | 267 |
| 9 | CAND_0070398 | ML Engineer | Product companies | 0.70 | 269 |
| 10 | CAND_0033861 | Senior NLP Engineer | Product companies | 0.63 | 349 |

✅ All India-based | ✅ All product-company careers | ✅ 0 consulting-only | ✅ 0 honeypots

---

## SLIDE 12 — WHAT GETS ELIMINATED AND WHY

**The keyword stuffer (PM @ Wipro):**
- Career SBERT: 0.31 — PM/delivery descriptions, not ML
- Skill trust: 12 — "advanced Fine-tuning 4m 4 endorsements" doesn't fool us
- Consulting-only: 0.25× career penalty
- → Not in top-100 ✅

**The honeypot (impossible timeline):**
- career_months/12 > claimed_yoe + 3 → honeypot_flag = 0
- Final score: 0.0 regardless of everything else ✅

**The behavioral ghost (great profile, 0.12 response rate, 8 months inactive):**
- availability_score < 0.05
- Drops from top-20 range to rank 60+ ✅

---

## SLIDE 13 — TECHNICAL STACK & RUNTIME

**Pre-computation (run once offline):**
- `sentence-transformers` — all-MiniLM-L6-v2 (22M params, 384 dims)
- Apple Silicon MPS GPU — ~18 min for 100K × 2 embedding passes
- Output: 2 × 146MB `.npy` files + 2.5MB structured feature parquet

**Ranking (online, <5 seconds CPU):**
- Pure `numpy` matrix multiply: `career_vecs @ jd_vec` → (100K,) in milliseconds
- `pandas` for feature combination and top-100 selection
- Zero network calls, zero LLM inference

**Sandbox:**
- `gradio_app.py` — deployed to HuggingFace Spaces
- Accepts up to 200 candidates as JSON, runs inline, returns ranked CSV

**Reproducibility:**
```bash
python embed_only.py --candidates candidates.jsonl --out_dir ./precomputed
python rank.py --candidates candidates.jsonl --artifacts_dir ./precomputed --out submission.csv
```

---

## SLIDE 14 — WHY THIS BEATS SIMPLER APPROACHES

| Approach | What it gets wrong |
|---|---|
| Keyword / BM25 | Surfaces HR managers with "Pinecone" listed |
| Pure cosine similarity | Honeypots score high (coherent skill text, impossible timeline) |
| LLM per candidate | Too slow (100K × 2s = 55+ hours), violates compute constraint |
| Skills-only ranking | Misses "plain-language" ML engineers who describe systems without buzzwords |

**Our approach wins because:**
1. Career descriptions are judged semantically — buzzwords don't get you in
2. Skill trust weighting means duration × endorsements matter, not just listing
3. Behavioral signals are first-class — availability is part of the score, not a filter
4. Honeypot defense is deterministic and always on

---

## SLIDE 15 — SUMMARY

**What we built:** A deterministic, interpretable, multi-signal cascade ranker

**What makes it work:**
- Semantic understanding of career narratives (not keyword matching)
- Trust-weighted skill scoring that can't be gamed by listing
- Behavioral availability as a scoring signal, not just a pass/fail filter
- Hardcoded honeypot defense

**Results:**
- 100 candidates ranked, CSV validated ✅
- 0 honeypots, 0 consulting-only, 0 keyword stuffers in top-100 ✅
- Every reasoning string grounded in real candidate data ✅
- Ranking step runs in ~5 seconds on CPU ✅

**The insight that guided everything:**
*A recruiter needs to trust the shortlist. That means every rank must be explainable from the candidate's actual history — not from a black box.*
