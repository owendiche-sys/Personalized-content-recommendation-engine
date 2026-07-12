from __future__ import annotations

from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_CONTENT_PATH = ROOT / "data" / "raw" / "content.csv"


CATEGORY_DETAILS = {
    "Study": {
        "source": "Campus Study Hub",
        "action": "Open study guide",
        "benefit": "turn a scattered study session into a focused next step",
        "audience": "students balancing lectures, deadlines, and revision",
    },
    "Productivity": {
        "source": "Focus Lab",
        "action": "Try the workflow",
        "benefit": "make the next hour easier to plan and finish",
        "audience": "students who want lighter systems that still work",
    },
    "Career": {
        "source": "Graduate Launchpad",
        "action": "View career resource",
        "benefit": "move one step closer to a stronger application or interview",
        "audience": "students preparing for placements, internships, and first roles",
    },
    "Finance": {
        "source": "Student Money Desk",
        "action": "Read money tip",
        "benefit": "make a practical money decision without heavy jargon",
        "audience": "students trying to stretch their budget",
    },
    "Food": {
        "source": "Quick Bite Campus",
        "action": "See food idea",
        "benefit": "find a realistic meal option around time and budget",
        "audience": "students who want easy food choices",
    },
    "Travel": {
        "source": "Local Weekend Finder",
        "action": "Plan this trip",
        "benefit": "find a low-friction break from routine",
        "audience": "students looking for nearby things to do",
    },
    "Events": {
        "source": "Campus Whatson",
        "action": "Check event details",
        "benefit": "discover something worth showing up for",
        "audience": "students who want social plans without endless searching",
    },
    "Wellness": {
        "source": "Student Wellbeing Studio",
        "action": "Start wellbeing pick",
        "benefit": "reset your energy without making it a huge project",
        "audience": "students managing stress, sleep, and motivation",
    },
    "Fitness": {
        "source": "Small Wins Fitness",
        "action": "Start routine",
        "benefit": "fit movement around a student schedule",
        "audience": "students who want simple, doable exercise",
    },
    "Entertainment": {
        "source": "Study Break Queue",
        "action": "Open recommendation",
        "benefit": "choose a break that will not swallow the whole evening",
        "audience": "students looking for a better study break",
    },
    "Music": {
        "source": "Mood Mix Desk",
        "action": "Play this mix",
        "benefit": "match the soundtrack to your current energy",
        "audience": "students who use music to focus, relax, or reset",
    },
    "Lifestyle": {
        "source": "Student Life Edit",
        "action": "Read lifestyle pick",
        "benefit": "make everyday student life feel a little more intentional",
        "audience": "students improving routines, spaces, and habits",
    },
    "Tech": {
        "source": "Useful Tech Shelf",
        "action": "Explore tool",
        "benefit": "find a practical tool before falling into app overload",
        "audience": "students who want useful digital shortcuts",
    },
    "Deals": {
        "source": "Student Deals Watch",
        "action": "View deal",
        "benefit": "spot a useful saving before it disappears",
        "audience": "students watching price and value",
    },
}


FORMAT_NOUN = {
    "Article": "read",
    "Guide": "step-by-step guide",
    "Video": "video",
    "Short Video": "quick watch",
    "Long Video": "deep-dive video",
    "Podcast": "listen",
    "Playlist": "playlist",
    "Carousel": "swipeable checklist",
}


def build_description(row: pd.Series) -> str:
    details = CATEGORY_DETAILS.get(row["category"], CATEGORY_DETAILS["Lifestyle"])
    format_noun = FORMAT_NOUN.get(row["format"], str(row["format"]).lower())
    tags = str(row["tags"]).replace("|", ", ")
    return (
        f"A {str(row['depth_level']).lower()} {format_noun} for {details['audience']}. "
        f"It uses {tags} to help you {details['benefit']}."
    )


def build_why(row: pd.Series) -> str:
    duration = int(row["duration_minutes"])
    cost = str(row["cost_band"]).lower()
    mood = str(row["mood"]).lower()
    return (
        f"Good when you have about {duration} minutes, want something {mood}, "
        f"and prefer a {cost} option."
    )


def build_source_url(row: pd.Series) -> str:
    query = quote_plus(f"{row['category']} {row['subcategory']} {row['title']}")
    return f"https://www.google.com/search?q={query}"


def enrich_catalog() -> None:
    content = pd.read_csv(RAW_CONTENT_PATH)

    content["description"] = content.apply(build_description, axis=1)
    content["why_it_matters"] = content.apply(build_why, axis=1)
    content["source_name"] = content["category"].map(
        lambda category: CATEGORY_DETAILS.get(category, CATEGORY_DETAILS["Lifestyle"])["source"]
    )
    content["action_label"] = content["category"].map(
        lambda category: CATEGORY_DETAILS.get(category, CATEGORY_DETAILS["Lifestyle"])["action"]
    )
    content["source_url"] = content.apply(build_source_url, axis=1)
    content["recommendation_surface"] = content.apply(
        lambda row: "Quick Pick"
        if int(row["duration_minutes"]) <= 10
        else ("Deep Dive" if int(row["duration_minutes"]) >= 30 else "Worth Your Time"),
        axis=1,
    )
    content["practicality_score"] = (
        content["content_quality_score"].rank(pct=True) * 0.45
        + content["popularity_score"].rank(pct=True) * 0.25
        + content["recency_score"].rank(pct=True) * 0.20
        + (content["cost_band"].isin(["Free", "Low"]).astype(float) * 0.10)
    ).round(3)

    content.to_csv(RAW_CONTENT_PATH, index=False)
    print(f"Enriched {len(content)} content rows at {RAW_CONTENT_PATH}")


if __name__ == "__main__":
    enrich_catalog()
