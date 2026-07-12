from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


@dataclass(frozen=True)
class ProjectPaths:
    """Container for commonly used project paths."""

    project_root: Path
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    results_dir: Path
    figures_dir: Path
    metrics_dir: Path


REQUIRED_RAW_COLUMNS: Dict[str, list[str]] = {
    "users": [
        "user_id",
        "persona",
        "favorite_categories",
        "preferred_formats",
        "preferred_moods",
        "session_pattern",
        "activity_level",
    ],
    "content": [
        "content_id",
        "title",
        "category",
        "subcategory",
        "tags",
        "format",
        "mood",
        "depth_level",
        "duration_minutes",
        "popularity_score",
        "recency_score",
        "content_quality_score",
    ],
    "interactions": [
        "interaction_id",
        "user_id",
        "content_id",
        "viewed",
        "liked",
        "saved",
        "shared",
        "completed",
        "dwell_time_percent",
        "rating",
        "engaged_minutes",
        "interaction_timestamp",
        "engagement_score",
    ],
}


TEXT_FEATURE_COLUMNS = [
    "title",
    "category",
    "subcategory",
    "tags",
    "format",
    "mood",
    "depth_level",
    "description",
    "why_it_matters",
    "source_name",
    "recommendation_surface",
]


def get_project_paths(start_path: str | Path | None = None) -> ProjectPaths:
    """
    Resolve project folders whether this module is called from the project root,
    src/, notebooks/, or app.py.
    """
    current = Path(start_path).resolve() if start_path else Path.cwd().resolve()
    search_roots = [current, current.parent, current.parent.parent]

    for candidate in search_roots:
        raw_dir = candidate / "data" / "raw"
        if raw_dir.exists():
            processed_dir = candidate / "data" / "processed"
            models_dir = candidate / "models"
            results_dir = candidate / "results"
            figures_dir = results_dir / "figures"
            metrics_dir = results_dir / "metrics"

            processed_dir.mkdir(parents=True, exist_ok=True)
            models_dir.mkdir(parents=True, exist_ok=True)
            figures_dir.mkdir(parents=True, exist_ok=True)
            metrics_dir.mkdir(parents=True, exist_ok=True)

            return ProjectPaths(
                project_root=candidate,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                models_dir=models_dir,
                results_dir=results_dir,
                figures_dir=figures_dir,
                metrics_dir=metrics_dir,
            )

    raise FileNotFoundError(
        "Could not locate the project root. Expected to find data/raw in the current "
        "directory or one of its parent directories."
    )


def _check_required_columns(df: pd.DataFrame, dataset_name: str) -> None:
    required = REQUIRED_RAW_COLUMNS[dataset_name]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {dataset_name}.csv: {missing}")


def load_raw_data(paths: ProjectPaths) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSV files and validate their key columns."""
    users = pd.read_csv(paths.raw_dir / "users.csv")
    content = pd.read_csv(paths.raw_dir / "content.csv")
    interactions = pd.read_csv(paths.raw_dir / "interactions.csv")

    _check_required_columns(users, "users")
    _check_required_columns(content, "content")
    _check_required_columns(interactions, "interactions")

    interactions["interaction_timestamp"] = pd.to_datetime(
        interactions["interaction_timestamp"], errors="coerce"
    )

    return users, content, interactions


def validate_relationships(
    users: pd.DataFrame,
    content: pd.DataFrame,
    interactions: pd.DataFrame,
) -> dict[str, int]:
    """Check that interaction keys match known users and content items."""
    unknown_users = (~interactions["user_id"].isin(users["user_id"])).sum()
    unknown_content = (~interactions["content_id"].isin(content["content_id"])).sum()
    duplicate_user_ids = users["user_id"].duplicated().sum()
    duplicate_content_ids = content["content_id"].duplicated().sum()
    duplicate_interaction_ids = interactions["interaction_id"].duplicated().sum()

    summary = {
        "unknown_user_references": int(unknown_users),
        "unknown_content_references": int(unknown_content),
        "duplicate_user_ids": int(duplicate_user_ids),
        "duplicate_content_ids": int(duplicate_content_ids),
        "duplicate_interaction_ids": int(duplicate_interaction_ids),
    }

    if unknown_users or unknown_content or duplicate_user_ids or duplicate_content_ids or duplicate_interaction_ids:
        raise ValueError(f"Relationship validation failed: {summary}")

    return summary


def build_content_features(content: pd.DataFrame) -> pd.DataFrame:
    """Create content features for TF-IDF and downstream ranking."""
    df = content.copy()
    for column in TEXT_FEATURE_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df[TEXT_FEATURE_COLUMNS] = df[TEXT_FEATURE_COLUMNS].fillna("")

    df["text_features"] = (
        df["title"].astype(str)
        + " "
        + df["category"].astype(str)
        + " "
        + df["subcategory"].astype(str)
        + " "
        + df["tags"].astype(str).str.replace("|", " ", regex=False)
        + " "
        + df["format"].astype(str)
        + " "
        + df["mood"].astype(str)
        + " "
        + df["depth_level"].astype(str)
        + " "
        + df["description"].astype(str)
        + " "
        + df["why_it_matters"].astype(str)
        + " "
        + df["source_name"].astype(str)
        + " "
        + df["recommendation_surface"].astype(str)
    ).str.strip()

    return df


def build_user_profiles(users: pd.DataFrame) -> pd.DataFrame:
    """Create lightweight user profile features for recommendation logic."""
    df = users.copy()

    df["favorite_category_count"] = df["favorite_categories"].fillna("").apply(
        lambda x: len([item for item in str(x).split("|") if item])
    )
    df["preferred_format_count"] = df["preferred_formats"].fillna("").apply(
        lambda x: len([item for item in str(x).split("|") if item])
    )
    df["preferred_mood_count"] = df["preferred_moods"].fillna("").apply(
        lambda x: len([item for item in str(x).split("|") if item])
    )

    return df


def build_interaction_scores(interactions: pd.DataFrame) -> pd.DataFrame:
    """Create helper columns for evaluation and ranking."""
    df = interactions.copy()

    df["rating_filled"] = df["rating"].fillna(0)
    df["interaction_day_type"] = df["interaction_timestamp"].dt.dayofweek.map(
        lambda x: "Weekend" if pd.notna(x) and x >= 5 else "Weekday"
    )
    df["engagement_bucket"] = pd.cut(
        df["engagement_score"],
        bins=[-0.01, 3, 6, 9, float("inf")],
        labels=["Low", "Medium", "High", "Very High"],
    )

    return df


def build_processed_datasets(
    users: pd.DataFrame,
    content: pd.DataFrame,
    interactions: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build all processed datasets used by notebooks and app.py."""
    content_features = build_content_features(content)
    user_profiles = build_user_profiles(users)
    interaction_scores = build_interaction_scores(interactions)
    return content_features, user_profiles, interaction_scores


def save_processed_datasets(
    paths: ProjectPaths,
    content_features: pd.DataFrame,
    user_profiles: pd.DataFrame,
    interaction_scores: pd.DataFrame,
) -> None:
    """Save processed CSV outputs to data/processed."""
    content_features.to_csv(paths.processed_dir / "content_features.csv", index=False)
    user_profiles.to_csv(paths.processed_dir / "user_profiles.csv", index=False)
    interaction_scores.to_csv(paths.processed_dir / "interaction_scores.csv", index=False)


def prepare_datasets(start_path: str | Path | None = None) -> dict[str, object]:
    """
    Main convenience function.

    Returns a dictionary containing the resolved paths, raw datasets,
    processed datasets, and validation summary.
    """
    paths = get_project_paths(start_path=start_path)
    users, content, interactions = load_raw_data(paths)
    validation_summary = validate_relationships(users, content, interactions)
    content_features, user_profiles, interaction_scores = build_processed_datasets(
        users, content, interactions
    )
    save_processed_datasets(paths, content_features, user_profiles, interaction_scores)

    return {
        "paths": paths,
        "validation_summary": validation_summary,
        "users": users,
        "content": content,
        "interactions": interactions,
        "content_features": content_features,
        "user_profiles": user_profiles,
        "interaction_scores": interaction_scores,
    }


if __name__ == "__main__":
    assets = prepare_datasets()
    print("Processed datasets saved successfully.")
    print(f"Project root: {assets['paths'].project_root}")
    print(f"Processed directory: {assets['paths'].processed_dir}")
    print(f"Validation summary: {assets['validation_summary']}")
