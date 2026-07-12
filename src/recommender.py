"""
Recommender utilities for the Personalized Content Recommendation Engine.

This module loads processed data and saved model assets, then exposes helper
functions for:
- popularity recommendations
- content-based similarity lookup
- hybrid user recommendations
- cold-start recommendations from manual preferences

Expected project structure
--------------------------
personalized-content-recommendation-engine/
│
├── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── src/
    └── recommender.py
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set
import pickle

import numpy as np
import pandas as pd


@dataclass
class RecommenderBundle:
    """Container for loaded recommender assets."""
    users: pd.DataFrame
    content_features: pd.DataFrame
    interaction_scores: pd.DataFrame
    top_popular: pd.DataFrame
    similarity_matrix: object
    content_index: Dict[str, int]
    seen_lookup: Dict[str, Set[str]]
    content_popularity_lookup: pd.DataFrame
    asset_source: str = "saved pickle assets"


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Find the project root by checking common locations for data/raw.

    This works whether the caller runs code from:
    - project root
    - src/
    - notebooks/
    """
    base = Path(start_path or Path.cwd()).resolve()
    candidates = [base, *base.parents]

    for candidate in candidates:
        if (candidate / "data" / "raw").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find project root containing 'data/raw'. "
        "Run this from inside the project directory."
    )


def _validate_required_files(project_root: Path) -> None:
    required = [
        project_root / "data" / "raw" / "users.csv",
        project_root / "data" / "processed" / "content_features.csv",
        project_root / "data" / "processed" / "interaction_scores.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required recommender files:\n- " + "\n- ".join(missing)
        )


def load_recommender_bundle(project_root: Optional[Path] = None) -> RecommenderBundle:
    """
    Load processed files and pickled recommender assets.
    """
    root = find_project_root(project_root)
    _validate_required_files(root)

    users = pd.read_csv(root / "data" / "raw" / "users.csv")
    content_features = pd.read_csv(root / "data" / "processed" / "content_features.csv")
    interaction_scores = pd.read_csv(root / "data" / "processed" / "interaction_scores.csv")

    product_columns = {"description", "source_url", "recommendation_surface"}
    if product_columns.issubset(set(content_features.columns)):
        assets = build_recommender_assets(content_features, interaction_scores)
        similarity_matrix = assets["similarity_matrix"]
        asset_source = "rebuilt from enriched CSV files"
    else:
        asset_source = "saved pickle assets"
        try:
            with open(root / "models" / "similarity_matrix.pkl", "rb") as f:
                similarity_matrix = pickle.load(f)

            with open(root / "models" / "recommender_assets.pkl", "rb") as f:
                assets = pickle.load(f)

            if not product_columns.issubset(set(assets["top_popular"].columns)):
                raise ValueError("Saved recommender assets are older than the enriched catalog schema.")
        except (FileNotFoundError, ModuleNotFoundError, AttributeError, ValueError, pickle.UnpicklingError):
            assets = build_recommender_assets(content_features, interaction_scores)
            similarity_matrix = assets["similarity_matrix"]
            asset_source = "rebuilt from CSV files"

    top_popular = assets["top_popular"].copy()

    content_popularity_lookup = top_popular[
        ["content_id", "popularity_rank_score"]
    ].copy()
    max_score = content_popularity_lookup["popularity_rank_score"].max()
    if max_score and max_score != 0:
        content_popularity_lookup["normalized_popularity_score"] = (
            content_popularity_lookup["popularity_rank_score"] / max_score
        )
    else:
        content_popularity_lookup["normalized_popularity_score"] = 0.0

    seen_lookup = assets["seen_lookup"]
    seen_lookup = {k: set(v) for k, v in seen_lookup.items()}

    return RecommenderBundle(
        users=users,
        content_features=content_features,
        interaction_scores=interaction_scores,
        top_popular=top_popular,
        similarity_matrix=similarity_matrix,
        content_index=assets["content_index"],
        seen_lookup=seen_lookup,
        content_popularity_lookup=content_popularity_lookup[
            ["content_id", "normalized_popularity_score"]
        ],
        asset_source=asset_source,
    )


def _tokenize_text(value: object) -> list[str]:
    text = "" if pd.isna(value) else str(value).lower()
    for char in "|,-:/()[]":
        text = text.replace(char, " ")
    return [token for token in text.split() if len(token) > 1]


def _build_similarity_matrix(content_features: pd.DataFrame) -> np.ndarray:
    documents = [Counter(_tokenize_text(text)) for text in content_features["text_features"]]
    vocabulary = sorted({token for doc in documents for token in doc})
    if not vocabulary:
        return np.eye(len(content_features), dtype=float)

    token_index = {token: idx for idx, token in enumerate(vocabulary)}
    matrix = np.zeros((len(documents), len(vocabulary)), dtype=float)

    for row_idx, document in enumerate(documents):
        for token, count in document.items():
            matrix[row_idx, token_index[token]] = count

    document_frequency = (matrix > 0).sum(axis=0)
    idf = np.log((1 + len(documents)) / (1 + document_frequency)) + 1
    tfidf = matrix * idf
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = tfidf / norms
    return normalized @ normalized.T


def build_recommender_assets(
    content_features: pd.DataFrame,
    interaction_scores: pd.DataFrame,
) -> dict[str, object]:
    """Build recommender assets directly from CSVs when pickle files are unavailable."""
    popularity = (
        interaction_scores.groupby("content_id", as_index=False)
        .agg(
            interaction_count=("interaction_id", "count"),
            avg_engagement=("engagement_score", "mean"),
            like_rate=("liked", "mean"),
            save_rate=("saved", "mean"),
            completion_rate=("completed", "mean"),
        )
    )
    popularity = content_features.merge(popularity, on="content_id", how="left")
    for column in ["interaction_count", "avg_engagement", "like_rate", "save_rate", "completion_rate"]:
        popularity[column] = popularity[column].fillna(0)

    popularity["popularity_rank_score"] = (
        popularity["popularity_score"].rank(pct=True) * 0.25
        + popularity["content_quality_score"].rank(pct=True) * 0.20
        + popularity["recency_score"].rank(pct=True) * 0.15
        + popularity["interaction_count"].rank(pct=True) * 0.15
        + popularity["avg_engagement"].rank(pct=True) * 0.15
        + popularity["save_rate"].rank(pct=True) * 0.10
    )

    seen_lookup = (
        interaction_scores.groupby("user_id")["content_id"]
        .apply(lambda values: set(values.dropna()))
        .to_dict()
    )
    content_index = {
        content_id: idx for idx, content_id in enumerate(content_features["content_id"].tolist())
    }

    return {
        "top_popular": popularity.sort_values("popularity_rank_score", ascending=False),
        "content_index": content_index,
        "seen_lookup": seen_lookup,
        "similarity_matrix": _build_similarity_matrix(content_features),
    }


def _content_columns(content_features: pd.DataFrame) -> list[str]:
    preferred = [
        "content_id",
        "title",
        "category",
        "subcategory",
        "tags",
        "format",
        "mood",
        "depth_level",
        "duration_minutes",
        "cost_band",
        "time_of_day_fit",
        "location_scope",
        "description",
        "why_it_matters",
        "source_name",
        "action_label",
        "source_url",
        "recommendation_surface",
        "practicality_score",
        "popularity_score",
        "recency_score",
        "content_quality_score",
    ]
    return [column for column in preferred if column in content_features.columns]


def get_popular_recommendations(
    bundle: RecommenderBundle,
    top_n: int = 10,
    category: Optional[str] = None,
    exclude_content_ids: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Return top popular content, optionally filtered by category.
    """
    exclude_set = set(exclude_content_ids or [])
    recs = bundle.top_popular.copy()

    if category:
        recs = recs[recs["category"] == category].copy()

    if exclude_set:
        recs = recs[~recs["content_id"].isin(exclude_set)].copy()

    recs = recs.sort_values("popularity_rank_score", ascending=False).head(top_n).copy()
    recs["recommendation_reason"] = "Popular content baseline"
    return recs.reset_index(drop=True)


def get_similar_items(
    bundle: RecommenderBundle,
    content_id: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return items similar to a given content item using the saved similarity matrix.
    """
    if content_id not in bundle.content_index:
        return pd.DataFrame(
            columns=["content_id", "title", "category", "format", "mood", "similarity_score"]
        )

    idx = bundle.content_index[content_id]
    sim_scores = list(enumerate(bundle.similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]

    item_indices = [i[0] for i in sim_scores]
    item_scores = [float(i[1]) for i in sim_scores]

    results = bundle.content_features.iloc[item_indices][_content_columns(bundle.content_features)].copy()
    results["similarity_score"] = item_scores
    return results.reset_index(drop=True)


def get_user_seed_items(
    bundle: RecommenderBundle,
    user_id: str,
    top_k: int = 3,
) -> list[str]:
    """
    Select a user's strongest historically engaged items to seed recommendations.
    """
    user_rows = bundle.interaction_scores.loc[
        bundle.interaction_scores["user_id"] == user_id
    ].copy()

    if user_rows.empty:
        return []

    positive_rows = user_rows[
        (user_rows["engagement_score"] >= 6)
        | (user_rows["liked"] == 1)
        | (user_rows["saved"] == 1)
        | (user_rows["completed"] == 1)
    ].copy()

    if positive_rows.empty:
        return []

    seeds = (
        positive_rows.sort_values("engagement_score", ascending=False)[
            ["content_id", "engagement_score"]
        ]
        .drop_duplicates("content_id")
        .head(top_k)
    )
    return seeds["content_id"].tolist()


def _build_profile_sets(user_row: pd.Series) -> tuple[Set[str], Set[str], Set[str]]:
    fav_categories = set(str(user_row.get("favorite_categories", "")).split("|"))
    pref_formats = set(str(user_row.get("preferred_formats", "")).split("|"))
    pref_moods = set(str(user_row.get("preferred_moods", "")).split("|"))

    fav_categories.discard("")
    pref_formats.discard("")
    pref_moods.discard("")

    return fav_categories, pref_formats, pref_moods


def _build_reason(
    row: pd.Series,
    fav_categories: Set[str],
    pref_formats: Set[str],
    pref_moods: Set[str],
) -> str:
    reasons = []

    if row["category"] in fav_categories:
        reasons.append(f"you often choose {row['category']} content")
    if row["format"] in pref_formats:
        reasons.append(f"you prefer {str(row['format']).lower()}s")
    if row["mood"] in pref_moods:
        reasons.append(f"it fits a {str(row['mood']).lower()} mood")

    if not reasons:
        reasons.append("it is close to content you engaged with before")

    why_it_matters = str(row.get("why_it_matters", "")).strip()
    if why_it_matters:
        reasons.append(why_it_matters.rstrip("."))

    return "Recommended because " + ", ".join(reasons[:4]) + "."


def recommend_for_user(
    bundle: RecommenderBundle,
    user_id: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Generate hybrid recommendations for an existing user.

    Hybrid score:
    - 55% content similarity
    - 25% popularity
    - 20% profile match
    """
    if user_id not in set(bundle.users["user_id"]):
        raise ValueError(f"Unknown user_id: {user_id}")

    seen_items = bundle.seen_lookup.get(user_id, set())
    seed_items = get_user_seed_items(bundle, user_id=user_id, top_k=3)

    if not seed_items:
        return get_popular_recommendations(
            bundle=bundle,
            top_n=top_n,
            exclude_content_ids=seen_items,
        )

    candidate_frames = []
    for seed in seed_items:
        sims = get_similar_items(bundle, seed, top_n=50)
        if sims.empty:
            continue
        sims["seed_content_id"] = seed
        candidate_frames.append(sims)

    if not candidate_frames:
        return get_popular_recommendations(
            bundle=bundle,
            top_n=top_n,
            exclude_content_ids=seen_items,
        )

    candidates = pd.concat(candidate_frames, ignore_index=True)
    candidates = candidates[~candidates["content_id"].isin(seen_items)].copy()

    if candidates.empty:
        return get_popular_recommendations(
            bundle=bundle,
            top_n=top_n,
            exclude_content_ids=seen_items,
        )

    candidates = (
        candidates.groupby(_content_columns(bundle.content_features), as_index=False)
        .agg(content_similarity_score=("similarity_score", "mean"))
    )

    candidates = candidates.merge(
        bundle.content_popularity_lookup,
        on="content_id",
        how="left",
    )
    candidates["normalized_popularity_score"] = candidates[
        "normalized_popularity_score"
    ].fillna(0.0)

    user_row = bundle.users.loc[bundle.users["user_id"] == user_id].iloc[0]
    fav_categories, pref_formats, pref_moods = _build_profile_sets(user_row)

    candidates["profile_match_score"] = (
        candidates["category"].isin(fav_categories).astype(int) * 0.5
        + candidates["format"].isin(pref_formats).astype(int) * 0.3
        + candidates["mood"].isin(pref_moods).astype(int) * 0.2
    )

    candidates["final_score"] = (
        0.55 * candidates["content_similarity_score"]
        + 0.25 * candidates["normalized_popularity_score"]
        + 0.20 * candidates["profile_match_score"]
    )

    candidates["recommendation_reason"] = candidates.apply(
        _build_reason,
        axis=1,
        args=(fav_categories, pref_formats, pref_moods),
    )
    output = candidates.sort_values("final_score", ascending=False).head(top_n).copy()
    return output.reset_index(drop=True)


def recommend_from_preferences(
    bundle: RecommenderBundle,
    favorite_categories: Optional[Sequence[str]] = None,
    preferred_formats: Optional[Sequence[str]] = None,
    preferred_moods: Optional[Sequence[str]] = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Cold-start recommendation function using manually selected preferences.

    This is useful in app.py when a user chooses preferences instead of a user_id.
    """
    favorite_categories = set(favorite_categories or [])
    preferred_formats = set(preferred_formats or [])
    preferred_moods = set(preferred_moods or [])

    candidates = bundle.content_features.copy()

    candidates["profile_match_score"] = (
        candidates["category"].isin(favorite_categories).astype(int) * 0.5
        + candidates["format"].isin(preferred_formats).astype(int) * 0.3
        + candidates["mood"].isin(preferred_moods).astype(int) * 0.2
    )

    candidates = candidates.merge(
        bundle.content_popularity_lookup,
        on="content_id",
        how="left",
    )
    candidates["normalized_popularity_score"] = candidates[
        "normalized_popularity_score"
    ].fillna(0.0)

    candidates["final_score"] = (
        0.65 * candidates["profile_match_score"]
        + 0.35 * candidates["normalized_popularity_score"]
    )

    def cold_reason(row: pd.Series) -> str:
        parts = []
        if row["category"] in favorite_categories:
            parts.append(f"it matches your interest in {row['category']}")
        if row["format"] in preferred_formats:
            parts.append(f"it is a {str(row['format']).lower()}")
        if row["mood"] in preferred_moods:
            parts.append(f"it has a {str(row['mood']).lower()} feel")
        if not parts:
            parts.append("it is one of the stronger all-round picks")
        why_it_matters = str(row.get("why_it_matters", "")).strip()
        if why_it_matters:
            parts.append(why_it_matters.rstrip("."))
        return "Recommended because " + ", ".join(parts[:4]) + "."

    candidates["recommendation_reason"] = candidates.apply(cold_reason, axis=1)

    cols = [
        "content_id",
        "title",
        "category",
        "subcategory",
        "tags",
        "format",
        "mood",
        "depth_level",
        "duration_minutes",
        "cost_band",
        "time_of_day_fit",
        "location_scope",
        "description",
        "why_it_matters",
        "source_name",
        "action_label",
        "source_url",
        "recommendation_surface",
        "practicality_score",
        "final_score",
        "recommendation_reason",
    ]
    cols = [col for col in cols if col in candidates.columns]
    return (
        candidates.sort_values(["final_score", "popularity_score"], ascending=False)
        .loc[:, cols]
        .head(top_n)
        .reset_index(drop=True)
    )


def get_available_filters(bundle: RecommenderBundle) -> dict[str, list[str]]:
    """
    Return sorted filter values for app controls.
    """
    return {
        "categories": sorted(bundle.content_features["category"].dropna().unique().tolist()),
        "formats": sorted(bundle.content_features["format"].dropna().unique().tolist()),
        "moods": sorted(bundle.content_features["mood"].dropna().unique().tolist()),
        "personas": sorted(bundle.users["persona"].dropna().unique().tolist()),
        "user_ids": sorted(bundle.users["user_id"].dropna().unique().tolist()),
    }


if __name__ == "__main__":
    project_root = find_project_root()
    bundle = load_recommender_bundle(project_root)

    print("Loaded recommender bundle successfully.")
    print("Users:", bundle.users.shape)
    print("Content features:", bundle.content_features.shape)
    print("Interaction scores:", bundle.interaction_scores.shape)

    sample_user_id = bundle.users["user_id"].iloc[0]
    print(f"\nSample recommendations for {sample_user_id}:")
    print(recommend_for_user(bundle, sample_user_id, top_n=5).to_string(index=False))
