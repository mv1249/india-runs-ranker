"""
Online ranking step. Must run in <5 min, CPU-only, no network, <=16GB RAM.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

If pre-computed artifacts exist in --artifacts_dir, uses them (fast path).
Otherwise falls back to computing structured features inline (slower, no SBERT).

Full pipeline:
    1. Load features.parquet + career_embeddings.npy + skill_embeddings.npy
    2. Compute cosine similarities against JD vectors
    3. Combine into final score
    4. Take top 100, generate reasoning, write CSV
"""

import argparse
import csv
import json
import math
import os
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Score weights — tune these based on validation
# ---------------------------------------------------------------------------

W_CAREER_SBERT = 0.35   # semantic fit of career descriptions to JD
W_SKILL_SBERT  = 0.20   # semantic fit of skills to JD skill keywords
W_SKILL_TRUST  = 0.20   # trust-weighted explicit skill match
W_CAREER_FIT   = 0.12   # career trajectory (product companies, ML titles)
W_AVAILABILITY = 0.07   # behavioral availability score
W_LOCATION     = 0.03   # location/logistics fit
W_GITHUB       = 0.01   # GitHub activity (proxy for active coding practice)
W_YOE          = 0.01   # experience years fit
W_EDUCATION    = 0.01   # education tier

REFERENCE_DATE = date(2026, 6, 16)


def normalize_col(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1], safe for constant arrays."""
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)


def load_artifacts(artifacts_dir: Path) -> dict:
    career_vecs = np.load(artifacts_dir / "career_embeddings.npy")
    skill_vecs  = np.load(artifacts_dir / "skill_embeddings.npy")
    jd_vecs     = np.load(artifacts_dir / "jd_vectors.npy")
    cids        = np.load(artifacts_dir / "candidate_ids.npy")
    feat_df     = pd.read_parquet(artifacts_dir / "features.parquet")
    return {
        "career_vecs": career_vecs,
        "skill_vecs":  skill_vecs,
        "jd_career":   jd_vecs[0],
        "jd_skill":    jd_vecs[1],
        "cids":        cids,
        "feat_df":     feat_df,
    }


def compute_scores(artifacts: dict) -> pd.DataFrame:
    career_vecs = artifacts["career_vecs"]   # (N, D)
    skill_vecs  = artifacts["skill_vecs"]    # (N, D)
    jd_career   = artifacts["jd_career"]     # (D,)
    jd_skill    = artifacts["jd_skill"]      # (D,)
    feat_df     = artifacts["feat_df"].copy()
    emb_cids    = artifacts["cids"]          # (N,) — candidate_ids parallel to embedding rows

    # Align embeddings to feat_df row order by candidate_id (safe even if order differs)
    emb_idx_map = {cid: i for i, cid in enumerate(emb_cids)}
    feat_order  = np.array([emb_idx_map[cid] for cid in feat_df["candidate_id"].values])
    career_ordered = career_vecs[feat_order]   # reorder to match feat_df
    skill_ordered  = skill_vecs[feat_order]

    # Cosine similarities (vecs already L2-normalized)
    career_sim = career_ordered @ jd_career    # (N,)
    skill_sim  = skill_ordered  @ jd_skill     # (N,)

    feat_df = feat_df.reset_index(drop=True)
    feat_df["career_sbert"] = career_sim
    feat_df["skill_sbert"]  = skill_sim

    # Normalize structured features to [0,1]
    feat_df["skill_trust_norm"]  = normalize_col(feat_df["skill_trust_score"].values)
    feat_df["career_score_norm"] = normalize_col(feat_df["career_score"].values)
    feat_df["career_sbert_norm"] = normalize_col(feat_df["career_sbert"].values)
    feat_df["skill_sbert_norm"]  = normalize_col(feat_df["skill_sbert"].values)

    # Raw combined score (before honeypot)
    raw = (
        W_CAREER_SBERT * feat_df["career_sbert_norm"]
      + W_SKILL_SBERT  * feat_df["skill_sbert_norm"]
      + W_SKILL_TRUST  * feat_df["skill_trust_norm"]
      + W_CAREER_FIT   * feat_df["career_score_norm"]
      + W_AVAILABILITY * feat_df["availability_score"]
      + W_LOCATION     * feat_df["location_score"]
      + W_GITHUB       * feat_df["github_score"]
      + W_YOE          * feat_df["yoe_fit"]
      + W_EDUCATION    * feat_df["education_score"]
    )

    # Honeypot: hard zero for detected honeypots
    raw = raw * feat_df["honeypot_flag"]

    feat_df["final_score"] = raw
    return feat_df


def generate_reasoning(row: pd.Series) -> str:
    """Build a specific, per-candidate reasoning string grounded in feature values."""
    title = str(row.get("current_title", "")).strip() or "Unknown title"
    yoe = float(row.get("years_of_experience", 0))
    loc = str(row.get("location", "")).strip()
    country = str(row.get("country", "")).strip()
    notice = int(row.get("notice_period_days", 60))
    open_work = bool(row.get("open_to_work_flag", 0))
    resp = float(row.get("recruiter_response_rate", 0))
    career_sim = float(row.get("career_sbert", 0))
    skill_sim = float(row.get("skill_sbert", 0))
    skill_trust = float(row.get("skill_trust_score", 0))
    avail = float(row.get("availability_score", 0))
    consulting = bool(row.get("consulting_only", 0))
    yoe_fit = float(row.get("yoe_fit", 0.5))

    parts = []

    # Sentence 1: who they are
    yoe_str = f"{yoe:.0f}" if yoe == int(yoe) else f"{yoe:.1f}"
    parts.append(f"{title} with {yoe_str} yrs experience")

    # Sentence 2: career semantic match (quantified)
    if career_sim >= 0.60:
        parts.append(f"career descriptions closely match retrieval/ranking/recsys JD (semantic score {career_sim:.2f})")
    elif career_sim >= 0.45:
        parts.append(f"career shows ML/data engineering alignment (semantic score {career_sim:.2f})")
    elif career_sim >= 0.35:
        parts.append(f"moderate career overlap with JD domain (semantic score {career_sim:.2f})")
    else:
        parts.append(f"limited career fit with retrieval/ML requirements (semantic score {career_sim:.2f})")

    # Sentence 3: skill trust (specific range)
    if skill_trust >= 100:
        parts.append(f"very high-trust IR/embedding skills (trust score {skill_trust:.0f})")
    elif skill_trust >= 40:
        parts.append(f"strong endorsed ML/retrieval skills (trust score {skill_trust:.0f})")
    elif skill_trust >= 15:
        parts.append(f"some ML skills present but lightly endorsed (trust score {skill_trust:.0f})")
    else:
        parts.append(f"skills listed are low-trust or unrelated to JD (trust score {skill_trust:.0f})")

    # Sentence 4: location
    if country == "India":
        parts.append(f"India-based ({loc})" if loc else "India-based")
    elif country:
        relocate_hint = ", willing to relocate" if row.get("location_score", 0) > 0.1 else ""
        parts.append(f"located in {loc or country}{relocate_hint}")

    # Sentence 5: availability (most informative signals only)
    avail_items = []
    if open_work:
        avail_items.append("actively open to work")
    if avail >= 0.55:
        avail_items.append(f"high recruiter responsiveness (resp={resp:.2f})")
    elif resp < 0.25:
        avail_items.append(f"low recruiter response rate ({resp:.2f})")
    if notice > 90:
        avail_items.append(f"{notice}-day notice period")
    if consulting:
        avail_items.append("consulting-firm career path — JD prefers product-company background")
    if avail_items:
        parts.append("; ".join(avail_items))

    return "; ".join(parts) + "."


def write_csv(top100: pd.DataFrame, out_path: Path, team_id: str):
    rows = []
    for rank_pos, (_, row) in enumerate(top100.iterrows(), start=1):
        rows.append({
            "candidate_id": row["candidate_id"],
            "rank": rank_pos,
            "score": round(float(row["final_score"]), 6),
            "reasoning": generate_reasoning(row),
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument(
        "--artifacts_dir",
        default=".",
        help="Directory containing pre-computed .npy and .parquet files",
    )
    parser.add_argument("--team_id", default="team_xxx", help="Your registered participant ID")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    required = [
        "career_embeddings.npy",
        "skill_embeddings.npy",
        "jd_vectors.npy",
        "candidate_ids.npy",
        "features.parquet",
    ]
    missing = [f for f in required if not (artifacts_dir / f).exists()]
    if missing:
        print(f"ERROR: Missing pre-computed artifacts in {artifacts_dir}: {missing}")
        print("Run feature_extractor.py first.")
        raise SystemExit(1)

    print("Loading pre-computed artifacts...")
    artifacts = load_artifacts(artifacts_dir)
    print(f"  candidates: {len(artifacts['cids']):,}")
    print(f"  embedding shape: {artifacts['career_vecs'].shape}")

    print("Computing scores...")
    scored = compute_scores(artifacts)

    # Sort by final_score descending; tie-break by candidate_id ascending (validator rule)
    scored_sorted = scored.sort_values(
        ["final_score", "candidate_id"], ascending=[False, True]
    )
    top100 = scored_sorted.head(100).reset_index(drop=True)

    print(f"\nTop-10 preview:")
    for i, row in top100.head(10).iterrows():
        print(
            f"  #{i+1:2d}  {row['candidate_id']}  score={row['final_score']:.4f}"
            f"  {row['current_title'][:30]:<30}  {row['location'][:20]}"
        )

    out_path = Path(args.out)
    # Ensure the stem matches team_id if using default name
    if out_path.name == "submission.csv":
        out_path = out_path.parent / f"{args.team_id}.csv"

    write_csv(top100, out_path, args.team_id)
    print(f"\nDone. Validate with: python validate_submission.py {out_path}")


if __name__ == "__main__":
    main()
