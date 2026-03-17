"""
Streamlit app for the Personalized Content Recommendation Engine.

This app supports:
- existing-user recommendations
- cold-start recommendations from selected preferences
- similar-item lookup
- popularity baseline view
- optional metrics display if evaluation outputs exist

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# -------------------------------------------------------------------
# Project path setup
# -------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from recommender import (  # noqa: E402
    find_project_root,
    get_available_filters,
    get_popular_recommendations,
    get_similar_items,
    load_recommender_bundle,
    recommend_for_user,
    recommend_from_preferences,
)
from utils import (  # noqa: E402
    build_user_profile_text,
    filter_recommendations,
    get_user_row,
    prepare_recommendation_display,
)


# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Personalized Content Recommendation Engine",
    page_icon=None,
    layout="wide",
)

st.title("Personalized Content Recommendation Engine")
st.caption(
    "Hybrid recommendations using content similarity, popularity signals, and user preference matching."
)


# -------------------------------------------------------------------
# Cached loaders
# -------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    root = find_project_root(APP_DIR)
    return load_recommender_bundle(root)


@st.cache_data
def load_optional_metrics(project_root: Path) -> dict[str, pd.DataFrame]:
    metrics_dir = project_root / "results" / "metrics"
    outputs: dict[str, pd.DataFrame] = {}

    metric_files = {
        "offline_summary_metrics": "offline_summary_metrics.csv",
        "diagnostic_metrics": "diagnostic_metrics.csv",
        "sample_recommendations": "sample_recommendations.csv",
        "top_popular_content": "top_popular_content.csv",
        "persona_hit_summary": "persona_hit_summary.csv",
    }

    for key, filename in metric_files.items():
        path = metrics_dir / filename
        if path.exists():
            outputs[key] = pd.read_csv(path)

    return outputs


try:
    PROJECT_ROOT = find_project_root(APP_DIR)
    bundle = load_bundle()
    filter_options = get_available_filters(bundle)
    optional_metrics = load_optional_metrics(PROJECT_ROOT)
except Exception as exc:
    st.error(
        "The app could not load the processed data or model assets.\n\n"
        "Run `02_recommender_build.ipynb` first so the processed files and model pickles are created."
    )
    st.exception(exc)
    st.stop()


# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------
st.sidebar.header("Controls")

mode = st.sidebar.radio(
    "Recommendation mode",
    options=["Existing User", "Preference-Based"],
)

top_n = st.sidebar.slider("Number of recommendations", min_value=5, max_value=20, value=10)

selected_categories = st.sidebar.multiselect(
    "Filter categories",
    options=filter_options["categories"],
    default=[],
)

selected_formats = st.sidebar.multiselect(
    "Filter formats",
    options=filter_options["formats"],
    default=[],
)

selected_moods = st.sidebar.multiselect(
    "Filter moods",
    options=filter_options["moods"],
    default=[],
)


# -------------------------------------------------------------------
# Helper display functions
# -------------------------------------------------------------------
def show_metric_cards(metrics_df: pd.DataFrame) -> None:
    cols = st.columns(len(metrics_df))
    for col, (_, row) in zip(cols, metrics_df.iterrows()):
        label = str(row["metric"])
        value = row["value"]
        if isinstance(value, float):
            display_value = f"{value:.3f}"
        else:
            display_value = str(value)
        col.metric(label=label, value=display_value)


def show_recommendation_table(df: pd.DataFrame, score_column: str = "final_score") -> None:
    if df.empty:
        st.info("No recommendations matched the current filters.")
        return

    display_df = prepare_recommendation_display(df, score_column=score_column)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def build_user_history_snapshot(user_id: str) -> pd.DataFrame:
    history = bundle.interaction_scores.merge(
        bundle.content_features[
            ["content_id", "title", "category", "format", "mood"]
        ],
        on="content_id",
        how="left",
    )

    history = history.loc[history["user_id"] == user_id].copy()
    if history.empty:
        return history

    history = history.sort_values("engagement_score", ascending=False)
    history = history[
        [
            "content_id",
            "title",
            "category",
            "format",
            "mood",
            "engagement_score",
            "liked",
            "saved",
            "completed",
        ]
    ].head(10)

    return history.reset_index(drop=True)


def apply_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    return filter_recommendations(
        recommendations=df,
        categories=selected_categories,
        formats=selected_formats,
        moods=selected_moods,
    )


# -------------------------------------------------------------------
# Top-level tabs
# -------------------------------------------------------------------
main_tab, explore_tab, metrics_tab = st.tabs(
    ["Recommendations", "Explore Content", "Model Diagnostics"]
)


with main_tab:
    if mode == "Existing User":
        persona_filter = st.selectbox(
            "Filter users by persona",
            options=["All"] + filter_options["personas"],
            index=0,
        )

        available_users = bundle.users.copy()
        if persona_filter != "All":
            available_users = available_users.loc[
                available_users["persona"] == persona_filter
            ].copy()

        user_id = st.selectbox(
            "Select user ID",
            options=available_users["user_id"].tolist(),
        )

        user_row = get_user_row(bundle.users, user_id)
        st.subheader("User Profile")
        st.write(build_user_profile_text(user_row))

        profile_cols = st.columns(4)
        profile_cols[0].metric("Persona", str(user_row["persona"]))
        profile_cols[1].metric("Activity Level", str(user_row["activity_level"]))
        profile_cols[2].metric("Session Pattern", str(user_row["session_pattern"]))
        profile_cols[3].metric("Discovery Style", str(user_row["discovery_style"]))

        st.subheader("Recommended For This User")
        recommendations = recommend_for_user(bundle, user_id=user_id, top_n=top_n * 2)
        recommendations = apply_global_filters(recommendations).head(top_n)
        show_recommendation_table(recommendations, score_column="final_score")

        st.subheader("Strongest Historical Interactions")
        user_history = build_user_history_snapshot(user_id)
        if user_history.empty:
            st.info("No interaction history is available for this user.")
        else:
            st.dataframe(user_history, use_container_width=True, hide_index=True)

    else:
        st.subheader("Preference-Based Cold-Start Recommendations")
        selected_pref_categories = st.multiselect(
            "Choose favorite categories",
            options=filter_options["categories"],
        )
        selected_pref_formats = st.multiselect(
            "Choose preferred formats",
            options=filter_options["formats"],
        )
        selected_pref_moods = st.multiselect(
            "Choose preferred moods",
            options=filter_options["moods"],
        )

        recommendations = recommend_from_preferences(
            bundle=bundle,
            favorite_categories=selected_pref_categories,
            preferred_formats=selected_pref_formats,
            preferred_moods=selected_pref_moods,
            top_n=top_n * 2,
        )
        recommendations = apply_global_filters(recommendations).head(top_n)

        st.subheader("Recommended Content")
        show_recommendation_table(recommendations, score_column="final_score")

        st.subheader("Selection Summary")
        summary = pd.DataFrame(
            {
                "Preference Group": ["Categories", "Formats", "Moods"],
                "Selected Values": [
                    ", ".join(selected_pref_categories) or "None selected",
                    ", ".join(selected_pref_formats) or "None selected",
                    ", ".join(selected_pref_moods) or "None selected",
                ],
            }
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Popular Content Baseline")
    popular = get_popular_recommendations(bundle, top_n=top_n * 2)
    popular = apply_global_filters(popular).head(top_n)
    show_recommendation_table(popular, score_column="popularity_rank_score")


with explore_tab:
    st.subheader("Find Similar Content")

    content_search_category = st.selectbox(
        "Limit content explorer to category",
        options=["All"] + filter_options["categories"],
        index=0,
    )

    content_pool = bundle.content_features.copy()
    if content_search_category != "All":
        content_pool = content_pool.loc[
            content_pool["category"] == content_search_category
        ].copy()

    content_choice = st.selectbox(
        "Select a content item",
        options=content_pool["content_id"].tolist(),
        format_func=lambda cid: f"{cid} — {content_pool.loc[content_pool['content_id'] == cid, 'title'].iloc[0]}",
    )

    selected_item = content_pool.loc[content_pool["content_id"] == content_choice].iloc[0]
    st.write(
        f"Selected item: {selected_item['title']} | "
        f"Category: {selected_item['category']} | "
        f"Format: {selected_item['format']} | "
        f"Mood: {selected_item['mood']}"
    )

    similar_items = get_similar_items(bundle, content_id=content_choice, top_n=top_n * 2)
    similar_items = apply_global_filters(similar_items).head(top_n)
    show_recommendation_table(similar_items, score_column="similarity_score")

    st.subheader("Content Catalog Snapshot")
    catalog_view = bundle.content_features[
        [
            "content_id",
            "title",
            "category",
            "format",
            "mood",
            "popularity_score",
            "recency_score",
        ]
    ].copy()
    catalog_view = apply_global_filters(catalog_view)
    st.dataframe(catalog_view.head(50), use_container_width=True, hide_index=True)


with metrics_tab:
    st.subheader("Saved Evaluation Outputs")

    if "offline_summary_metrics" in optional_metrics:
        st.markdown("**Offline Summary Metrics**")
        show_metric_cards(optional_metrics["offline_summary_metrics"])

    if "diagnostic_metrics" in optional_metrics:
        st.markdown("**Diagnostic Metrics**")
        st.dataframe(
            optional_metrics["diagnostic_metrics"],
            use_container_width=True,
            hide_index=True,
        )

    if "persona_hit_summary" in optional_metrics:
        st.markdown("**Persona-Level Performance**")
        st.dataframe(
            optional_metrics["persona_hit_summary"],
            use_container_width=True,
            hide_index=True,
        )

    if "sample_recommendations" in optional_metrics:
        st.markdown("**Sample Saved Recommendations**")
        st.dataframe(
            optional_metrics["sample_recommendations"].head(20),
            use_container_width=True,
            hide_index=True,
        )

    if "top_popular_content" in optional_metrics:
        st.markdown("**Top Popular Content Snapshot**")
        st.dataframe(
            optional_metrics["top_popular_content"].head(20),
            use_container_width=True,
            hide_index=True,
        )

    if not optional_metrics:
        st.info(
            "No evaluation outputs were found yet. Run `03_model_evaluation.ipynb` "
            "to populate the diagnostics section."
        )
