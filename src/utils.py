"""
General utility helpers for the Personalized Content Recommendation Engine.

This module keeps small reusable functions out of notebooks and app.py.
It focuses on:
- parsing pipe-separated fields
- formatting recommendation outputs
- filtering recommendation tables
- building quick user summaries
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence
import math

import pandas as pd


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Find the project root by searching upward for a folder containing data/raw.
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


def parse_pipe_separated(value: object) -> list[str]:
    """
    Convert a pipe-separated string into a clean list.

    Examples
    --------
    'Food|Travel|Events' -> ['Food', 'Travel', 'Events']
    """
    if value is None:
        return []

    if isinstance(value, float) and math.isnan(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    parts = [part.strip() for part in text.split("|")]
    return [part for part in parts if part]


def list_to_pipe(values: Sequence[str]) -> str:
    """
    Convert a list of strings into a pipe-separated string.
    """
    clean_values = [str(v).strip() for v in values if str(v).strip()]
    return "|".join(clean_values)


def deduplicate_preserve_order(values: Iterable[str]) -> list[str]:
    """
    Remove duplicates while preserving first-seen order.
    """
    seen = set()
    output = []

    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)

    return output


def normalise_text_label(value: object) -> str:
    """
    Turn a raw value into a clean display string.
    """
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def format_score(value: object, decimals: int = 3) -> str:
    """
    Format numeric scores for clean display in the app.
    """
    if value is None:
        return ""
    try:
        num = float(value)
        return f"{num:.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def filter_recommendations(
    recommendations: pd.DataFrame,
    categories: Optional[Sequence[str]] = None,
    formats: Optional[Sequence[str]] = None,
    moods: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Filter a recommendation DataFrame by category, format, and mood.
    """
    df = recommendations.copy()

    if categories:
        df = df[df["category"].isin(categories)].copy()

    if formats and "format" in df.columns:
        df = df[df["format"].isin(formats)].copy()

    if moods and "mood" in df.columns:
        df = df[df["mood"].isin(moods)].copy()

    return df.reset_index(drop=True)


def prepare_recommendation_display(
    recommendations: pd.DataFrame,
    score_column: str = "final_score",
) -> pd.DataFrame:
    """
    Create a cleaner display table for Streamlit.

    It keeps the most useful columns and formats the main score.
    """
    if recommendations.empty:
        return recommendations.copy()

    df = recommendations.copy()

    preferred_columns = [
        "content_id",
        "title",
        "category",
        "format",
        "mood",
        score_column,
        "recommendation_reason",
    ]
    available_columns = [col for col in preferred_columns if col in df.columns]
    df = df.loc[:, available_columns].copy()

    if score_column in df.columns:
        df[score_column] = df[score_column].apply(format_score)

    return df


def build_user_summary(user_row: pd.Series) -> dict[str, object]:
    """
    Build a compact summary dictionary from a user profile row.
    """
    summary = {
        "user_id": normalise_text_label(user_row.get("user_id")),
        "persona": normalise_text_label(user_row.get("persona")),
        "favorite_categories": parse_pipe_separated(user_row.get("favorite_categories")),
        "preferred_formats": parse_pipe_separated(user_row.get("preferred_formats")),
        "preferred_moods": parse_pipe_separated(user_row.get("preferred_moods")),
        "session_pattern": normalise_text_label(user_row.get("session_pattern")),
        "activity_level": normalise_text_label(user_row.get("activity_level")),
        "budget_sensitivity": normalise_text_label(user_row.get("budget_sensitivity")),
        "home_region": normalise_text_label(user_row.get("home_region")),
        "discovery_style": normalise_text_label(user_row.get("discovery_style")),
        "avg_session_minutes": user_row.get("avg_session_minutes"),
    }
    return summary


def get_user_row(users: pd.DataFrame, user_id: str) -> pd.Series:
    """
    Return a single user row by ID.
    """
    matches = users.loc[users["user_id"] == user_id]
    if matches.empty:
        raise ValueError(f"Unknown user_id: {user_id}")
    return matches.iloc[0]


def build_user_profile_text(user_row: pd.Series) -> str:
    """
    Build a plain-English summary for the selected user.
    """
    summary = build_user_summary(user_row)

    favorite_categories = ", ".join(summary["favorite_categories"]) or "None"
    preferred_formats = ", ".join(summary["preferred_formats"]) or "None"
    preferred_moods = ", ".join(summary["preferred_moods"]) or "None"

    return (
        f"Persona: {summary['persona']} | "
        f"Favorite categories: {favorite_categories} | "
        f"Preferred formats: {preferred_formats} | "
        f"Preferred moods: {preferred_moods} | "
        f"Session pattern: {summary['session_pattern']} | "
        f"Discovery style: {summary['discovery_style']}"
    )


def top_values_as_string(
    df: pd.DataFrame,
    column: str,
    top_n: int = 5,
) -> str:
    """
    Return the top values in a column as a comma-separated string.
    """
    if column not in df.columns or df.empty:
        return ""

    values = df[column].value_counts().head(top_n).index.tolist()
    return ", ".join(map(str, values))


if __name__ == "__main__":
    print("utils.py loaded successfully.")
