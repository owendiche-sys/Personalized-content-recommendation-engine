"""
Streamlit app for the Personalized Content Recommendation Engine.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st


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


st.set_page_config(
    page_title="Personalized Content Recommendation Engine",
    page_icon=None,
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f8fafc;
        color: #0f172a;
    }
    .main .block-container {padding-top: 1.25rem; max-width: 1240px;}
    h1, h2, h3 {letter-spacing: 0;}
    p, label, span, div {
        color: inherit;
    }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #0f172a;
    }
    [data-baseweb="tab"],
    [data-baseweb="tab"] p,
    [data-testid="stSelectbox"] label,
    [data-testid="stMultiSelect"] label,
    [data-testid="stSlider"] label,
    [data-testid="stRadio"] label {
        color: #0f172a;
    }
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="tag"] {
        background: #ffffff;
        color: #0f172a;
        border-color: #cbd5e1;
    }
    [data-baseweb="select"] span,
    [data-baseweb="input"] input,
    [data-baseweb="tag"] span {
        color: #0f172a;
    }
    .stButton > button,
    .stLinkButton > a {
        background: #ffffff;
        color: #0f172a;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
    }
    .stButton > button:hover,
    .stLinkButton > a:hover {
        background: #e0f2fe;
        color: #075985;
        border-color: #7dd3fc;
    }
    .stButton > button:disabled {
        background: #f1f5f9;
        color: #64748b;
        border-color: #cbd5e1;
    }
    .stDataFrame,
    [data-testid="stDataFrame"] {
        color: #0f172a;
    }
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #dbe3ee;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        color: #0f172a;
    }
    .hero {
        border: 1px solid #dbe3ee;
        border-radius: 8px;
        background: #f8fafc;
        padding: 1.15rem 1.25rem;
        margin-bottom: 1rem;
    }
    .hero h1 {
        color: #0f172a;
        font-size: 2rem;
        line-height: 1.15;
        margin: 0 0 0.35rem 0;
    }
    .hero p {
        color: #475569;
        font-size: 1rem;
        line-height: 1.55;
        margin: 0;
        max-width: 900px;
    }
    .feed-note {
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.5;
        margin: 0.2rem 0 0.8rem 0;
    }
    .rec-card {
        border: 1px solid #dbe3ee;
        border-radius: 8px;
        padding: 1rem;
        background: #ffffff;
        min-height: 335px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }
    .rec-topline {
        display: flex;
        gap: 0.4rem;
        flex-wrap: wrap;
        margin-bottom: 0.65rem;
    }
    .pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.18rem 0.5rem;
        background: #e0f2fe;
        color: #075985;
        font-size: 0.76rem;
        font-weight: 700;
    }
    .pill-alt {
        display: inline-block;
        border-radius: 999px;
        padding: 0.18rem 0.5rem;
        background: #ecfdf5;
        color: #166534;
        font-size: 0.76rem;
        font-weight: 700;
    }
    .rec-title {
        font-size: 1.04rem;
        font-weight: 750;
        color: #0f172a;
        line-height: 1.32;
        margin-bottom: 0.45rem;
    }
    .rec-description {
        color: #334155;
        font-size: 0.9rem;
        line-height: 1.52;
        margin-bottom: 0.7rem;
    }
    .rec-meta, .rec-reason, .rec-source {
        color: #475569;
        font-size: 0.84rem;
        line-height: 1.45;
        margin-bottom: 0.45rem;
    }
    .rec-reason {
        color: #1e293b;
        border-top: 1px solid #e2e8f0;
        padding-top: 0.65rem;
        margin-top: 0.7rem;
    }
    .muted-box {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        background: #f8fafc;
        padding: 0.85rem 1rem;
        color: #334155;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Student Content Finder</h1>
      <p>Get a short list of content picks that match a student's interests, current mood, time, budget, and previous engagement.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


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
    st.error("The app could not load the processed data or recommender assets.")
    st.exception(exc)
    st.stop()


for state_key, default in {
    "saved_items": [],
    "dismissed_items": [],
    "more_like_content_id": None,
}.items():
    st.session_state.setdefault(state_key, default)


def safe_value(row: pd.Series, column: str, fallback: str = "") -> str:
    value = row.get(column, fallback)
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    return text or fallback


def score_text(row: pd.Series, score_column: str) -> str:
    try:
        return f"{float(row[score_column]):.2f}"
    except (KeyError, TypeError, ValueError):
        return "--"


def cost_friendly_percent(df: pd.DataFrame) -> str:
    if df.empty or "cost_band" not in df.columns:
        return "0%"
    rate = df["cost_band"].isin(["Free", "Low"]).mean()
    return f"{float(rate * 100):.0f}%"


def apply_intent_filters(
    recommendations: pd.DataFrame,
    categories: list[str],
    formats: list[str],
    moods: list[str],
    max_minutes: int,
    cost_bands: list[str],
    surface: str,
    hide_dismissed: bool = True,
) -> pd.DataFrame:
    df = filter_recommendations(recommendations, categories, formats, moods)

    if "duration_minutes" in df.columns:
        df = df[df["duration_minutes"] <= max_minutes].copy()
    if cost_bands and "cost_band" in df.columns:
        df = df[df["cost_band"].isin(cost_bands)].copy()
    if surface != "Any" and "recommendation_surface" in df.columns:
        df = df[df["recommendation_surface"] == surface].copy()
    if hide_dismissed:
        df = df[~df["content_id"].isin(st.session_state.dismissed_items)].copy()

    return df.reset_index(drop=True)


def boost_saved_and_practicality(df: pd.DataFrame, score_column: str) -> pd.DataFrame:
    if df.empty:
        return df
    output = df.copy()
    if score_column not in output.columns:
        output[score_column] = 0.0
    output["_display_score"] = output[score_column].astype(float)
    if "practicality_score" in output.columns:
        output["_display_score"] += output["practicality_score"].fillna(0).astype(float) * 0.08
    output = output.sort_values("_display_score", ascending=False)
    return output.drop(columns=["_display_score"]).reset_index(drop=True)


def show_metric_cards(metrics_df: pd.DataFrame) -> None:
    cols = st.columns(len(metrics_df))
    for col, (_, row) in zip(cols, metrics_df.iterrows()):
        value = row["value"]
        display_value = f"{value:.3f}" if isinstance(value, float) else str(value)
        col.metric(label=str(row["metric"]), value=display_value)


def show_recommendation_table(df: pd.DataFrame, score_column: str = "final_score") -> None:
    if df.empty:
        st.info("No recommendations matched the current filters.")
        return
    display_df = prepare_recommendation_display(df, score_column=score_column)
    extra_cols = [
        "description",
        "why_it_matters",
        "source_name",
        "action_label",
        "recommendation_surface",
        "source_url",
    ]
    for col in extra_cols:
        if col in df.columns and col not in display_df.columns:
            display_df[col] = df[col].values
    st.dataframe(display_df, width="stretch", hide_index=True)


def remember_saved(content_id: str) -> None:
    if content_id not in st.session_state.saved_items:
        st.session_state.saved_items.append(content_id)


def remember_dismissed(content_id: str) -> None:
    if content_id not in st.session_state.dismissed_items:
        st.session_state.dismissed_items.append(content_id)


def show_recommendation_cards(
    df: pd.DataFrame,
    score_column: str = "final_score",
    key_prefix: str = "feed",
) -> None:
    if df.empty:
        st.info("No recommendations matched the current filters.")
        return

    rows = df.reset_index(drop=True)
    for start in range(0, len(rows), 2):
        columns = st.columns(2)
        for col, (_, row) in zip(columns, rows.iloc[start : start + 2].iterrows()):
            content_id = safe_value(row, "content_id")
            source_url = safe_value(row, "source_url")
            action_label = safe_value(row, "action_label", "Open")
            tags = safe_value(row, "tags").replace("|", ", ")
            duration = safe_value(row, "duration_minutes", "Flexible")
            if duration != "Flexible":
                duration = f"{duration} min"
            meta = " | ".join(
                part
                for part in [
                    safe_value(row, "format"),
                    safe_value(row, "mood"),
                    safe_value(row, "depth_level"),
                    duration,
                    safe_value(row, "cost_band"),
                    safe_value(row, "time_of_day_fit"),
                ]
                if part
            )
            context = " | ".join(
                part
                for part in [
                    safe_value(row, "category"),
                    safe_value(row, "subcategory"),
                    safe_value(row, "location_scope"),
                ]
                if part
            )

            col.markdown(
                f"""
                <div class="rec-card">
                    <div class="rec-topline">
                        <span class="pill">Score {score_text(row, score_column)}</span>
                        <span class="pill-alt">{escape(safe_value(row, "recommendation_surface", "Pick"))}</span>
                    </div>
                    <div class="rec-title">{escape(safe_value(row, "title"))}</div>
                    <div class="rec-description">{escape(safe_value(row, "description", "A relevant content pick for this profile."))}</div>
                    <div class="rec-meta">{escape(context)}</div>
                    <div class="rec-meta">{escape(meta)}</div>
                    <div class="rec-meta">{escape(tags)}</div>
                    <div class="rec-reason">{escape(safe_value(row, "recommendation_reason", safe_value(row, "why_it_matters")))}</div>
                    <div class="rec-source">{escape(safe_value(row, "source_name", "Content catalog"))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            action_cols = col.columns([1.1, 0.8, 1.0, 1.0])
            if source_url:
                action_cols[0].link_button(action_label, source_url, width="stretch")
            else:
                action_cols[0].button(
                    action_label,
                    disabled=True,
                    key=f"{key_prefix}_open_{content_id}_{start}",
                    width="stretch",
                )
            if action_cols[1].button("Save", key=f"{key_prefix}_save_{content_id}_{start}", width="stretch"):
                remember_saved(content_id)
                st.toast("Saved")
            if action_cols[2].button("More like this", key=f"{key_prefix}_more_{content_id}_{start}", width="stretch"):
                st.session_state.more_like_content_id = content_id
            if action_cols[3].button("Not for me", key=f"{key_prefix}_dismiss_{content_id}_{start}", width="stretch"):
                remember_dismissed(content_id)
                st.rerun()


def build_user_history_snapshot(user_id: str) -> pd.DataFrame:
    history = bundle.interaction_scores.merge(
        bundle.content_features[["content_id", "title", "category", "format", "mood"]],
        on="content_id",
        how="left",
    )
    history = history.loc[history["user_id"] == user_id].copy()
    if history.empty:
        return history
    return (
        history.sort_values("engagement_score", ascending=False)[
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
        ]
        .head(10)
        .reset_index(drop=True)
    )


with st.sidebar:
    st.header("Intent")
    st.caption(f"Assets: {bundle.asset_source}")

    mode = st.radio("Starting point", options=["Existing User", "New Taste Profile"])
    top_n = st.slider("Feed size", min_value=4, max_value=16, value=8)
    max_minutes = st.slider("Time available", min_value=5, max_value=60, value=25, step=5)

    selected_categories = st.multiselect("Interests", options=filter_options["categories"], default=[])
    selected_formats = st.multiselect("Formats", options=filter_options["formats"], default=[])
    selected_moods = st.multiselect("Mood", options=filter_options["moods"], default=[])
    selected_costs = st.multiselect(
        "Budget",
        options=sorted(bundle.content_features["cost_band"].dropna().unique().tolist()),
        default=[],
    )
    surface_options = ["Any"] + sorted(
        bundle.content_features["recommendation_surface"].dropna().unique().tolist()
    )
    selected_surface = st.selectbox("Content depth", options=surface_options, index=0)

    if st.button("Reset feedback", width="stretch"):
        st.session_state.saved_items = []
        st.session_state.dismissed_items = []
        st.session_state.more_like_content_id = None
        st.rerun()


feed_tab, saved_tab, explore_tab, metrics_tab = st.tabs(
    ["For You", "Saved", "Explore", "Diagnostics"]
)


with feed_tab:
    if mode == "Existing User":
        persona_filter = st.selectbox(
            "Persona",
            options=["All"] + filter_options["personas"],
            index=0,
        )
        available_users = bundle.users.copy()
        if persona_filter != "All":
            available_users = available_users.loc[available_users["persona"] == persona_filter].copy()

        user_id = st.selectbox("Student profile", options=available_users["user_id"].tolist())
        user_row = get_user_row(bundle.users, user_id)

        profile_cols = st.columns(4)
        profile_cols[0].metric("Persona", str(user_row["persona"]))
        profile_cols[1].metric("Activity", str(user_row["activity_level"]))
        profile_cols[2].metric("Pattern", str(user_row["session_pattern"]))
        profile_cols[3].metric("Discovery", str(user_row["discovery_style"]))

        st.markdown(f"<div class='muted-box'>{escape(build_user_profile_text(user_row))}</div>", unsafe_allow_html=True)

        base_recs = recommend_for_user(bundle, user_id=user_id, top_n=max(top_n * 4, 30))
    else:
        st.subheader("Build a Taste Profile")
        st.markdown(
            "<p class='feed-note'>Choose at least one interest, format, or mood to make the feed more personal.</p>",
            unsafe_allow_html=True,
        )
        base_recs = recommend_from_preferences(
            bundle=bundle,
            favorite_categories=selected_categories,
            preferred_formats=selected_formats,
            preferred_moods=selected_moods,
            top_n=max(top_n * 5, 40),
        )

    recommendations = apply_intent_filters(
        base_recs,
        categories=selected_categories,
        formats=selected_formats,
        moods=selected_moods,
        max_minutes=max_minutes,
        cost_bands=selected_costs,
        surface=selected_surface,
    )
    recommendations = boost_saved_and_practicality(recommendations, "final_score").head(top_n)

    if len(recommendations) < top_n:
        backfill = get_popular_recommendations(bundle, top_n=top_n * 6)
        backfill = apply_intent_filters(
            backfill,
            categories=selected_categories,
            formats=selected_formats,
            moods=selected_moods,
            max_minutes=max_minutes,
            cost_bands=selected_costs,
            surface=selected_surface,
        )
        backfill = backfill[~backfill["content_id"].isin(recommendations["content_id"])].copy()
        backfill["final_score"] = backfill["popularity_rank_score"]
        backfill["recommendation_reason"] = backfill["why_it_matters"].apply(
            lambda value: f"Recommended as a strong backup pick: {str(value).rstrip('.')}."
        )
        recommendations = pd.concat(
            [recommendations, backfill.head(top_n - len(recommendations))],
            ignore_index=True,
        )

    st.subheader("Recommended Now")
    if not recommendations.empty:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Items", len(recommendations))
        metric_cols[1].metric("Avg duration", f"{recommendations['duration_minutes'].mean():.0f} min")
        metric_cols[2].metric("Free or low cost", cost_friendly_percent(recommendations))
        metric_cols[3].metric("Saved this session", len(st.session_state.saved_items))

    show_recommendation_cards(recommendations, score_column="final_score", key_prefix="primary")

    if st.session_state.more_like_content_id:
        st.subheader("More Like Your Selected Item")
        similar = get_similar_items(bundle, st.session_state.more_like_content_id, top_n=12)
        similar["recommendation_reason"] = "Recommended because it shares a similar topic, format, mood, and tag profile."
        similar = apply_intent_filters(
            similar,
            categories=[],
            formats=[],
            moods=[],
            max_minutes=max_minutes,
            cost_bands=selected_costs,
            surface=selected_surface,
            hide_dismissed=True,
        ).head(6)
        show_recommendation_cards(similar, score_column="similarity_score", key_prefix="more_like")

    with st.expander("View feed as data"):
        show_recommendation_table(recommendations, score_column="final_score")

    if mode == "Existing User":
        st.subheader("Strongest Historical Signals")
        history = build_user_history_snapshot(user_id)
        st.dataframe(history, width="stretch", hide_index=True)

    st.subheader("Popular Baseline")
    popular = get_popular_recommendations(bundle, top_n=top_n * 3)
    popular = apply_intent_filters(
        popular,
        categories=selected_categories,
        formats=selected_formats,
        moods=selected_moods,
        max_minutes=max_minutes,
        cost_bands=selected_costs,
        surface=selected_surface,
    ).head(min(top_n, 6))
    show_recommendation_cards(popular, score_column="popularity_rank_score", key_prefix="popular")


with saved_tab:
    st.subheader("Saved Picks")
    saved = bundle.content_features[
        bundle.content_features["content_id"].isin(st.session_state.saved_items)
    ].copy()
    if saved.empty:
        st.info("Save recommendations from the feed to build a short list.")
    else:
        saved["recommendation_reason"] = saved["why_it_matters"]
        saved["final_score"] = saved.get("practicality_score", 0)
        show_recommendation_cards(saved, score_column="final_score", key_prefix="saved")
        st.dataframe(
            saved[["title", "category", "format", "duration_minutes", "source_url"]],
            width="stretch",
            hide_index=True,
        )


with explore_tab:
    st.subheader("Explore the Catalog")
    explorer_cols = st.columns([1, 1, 1])
    content_search_category = explorer_cols[0].selectbox(
        "Category",
        options=["All"] + filter_options["categories"],
        index=0,
    )
    content_search_surface = explorer_cols[1].selectbox(
        "Depth",
        options=surface_options,
        index=0,
    )
    query = explorer_cols[2].text_input("Search title or tags")

    content_pool = bundle.content_features.copy()
    if content_search_category != "All":
        content_pool = content_pool.loc[content_pool["category"] == content_search_category].copy()
    if content_search_surface != "Any":
        content_pool = content_pool.loc[
            content_pool["recommendation_surface"] == content_search_surface
        ].copy()
    if query:
        query_lower = query.lower()
        content_pool = content_pool[
            content_pool["title"].str.lower().str.contains(query_lower, na=False)
            | content_pool["tags"].str.lower().str.contains(query_lower, na=False)
            | content_pool["description"].str.lower().str.contains(query_lower, na=False)
        ].copy()

    if content_pool.empty:
        st.info("No catalog items match the current search.")
    else:
        content_choice = st.selectbox(
            "Content item",
            options=content_pool["content_id"].tolist(),
            format_func=lambda cid: (
                f"{cid} - {content_pool.loc[content_pool['content_id'] == cid, 'title'].iloc[0]}"
            ),
        )

        selected_item = content_pool.loc[content_pool["content_id"] == content_choice].iloc[0]
        selected_df = pd.DataFrame([selected_item])
        selected_df["recommendation_reason"] = selected_df["why_it_matters"]
        selected_df["final_score"] = selected_df["practicality_score"]
        show_recommendation_cards(selected_df, score_column="final_score", key_prefix="selected")

        st.subheader("Similar Picks")
        similar_items = get_similar_items(bundle, content_id=content_choice, top_n=8)
        similar_items["recommendation_reason"] = "Recommended because it shares a similar topic, format, mood, and tag profile."
        show_recommendation_cards(similar_items, score_column="similarity_score", key_prefix="explore_similar")

        with st.expander("Catalog table"):
            catalog_cols = [
                "content_id",
                "title",
                "category",
                "format",
                "mood",
                "duration_minutes",
                "cost_band",
                "recommendation_surface",
                "practicality_score",
            ]
            st.dataframe(content_pool[catalog_cols].head(100), width="stretch", hide_index=True)


with metrics_tab:
    st.subheader("Model Diagnostics")
    if "offline_summary_metrics" in optional_metrics:
        show_metric_cards(optional_metrics["offline_summary_metrics"])
    if "diagnostic_metrics" in optional_metrics:
        st.dataframe(optional_metrics["diagnostic_metrics"], width="stretch", hide_index=True)
    if "persona_hit_summary" in optional_metrics:
        st.subheader("Persona-Level Performance")
        st.dataframe(optional_metrics["persona_hit_summary"], width="stretch", hide_index=True)
    if "sample_recommendations" in optional_metrics:
        st.subheader("Saved Recommendation Samples")
        st.dataframe(optional_metrics["sample_recommendations"].head(20), width="stretch", hide_index=True)
    if "top_popular_content" in optional_metrics:
        st.subheader("Top Popular Content")
        st.dataframe(optional_metrics["top_popular_content"].head(20), width="stretch", hide_index=True)
    if not optional_metrics:
        st.info("No evaluation outputs were found.")
