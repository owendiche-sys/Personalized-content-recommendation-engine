# Data Notes

This project uses synthetic student lifestyle data designed for a recommender-system portfolio project.

## Files

- `raw/users.csv`: student profiles, personas, preferred categories, formats, moods, session habits, and discovery style.
- `raw/content.csv`: content catalog with titles, tags, category metadata, format, mood, duration, cost band, freshness, popularity, recency, and quality signals.
- `raw/interactions.csv`: behavioral events connecting users to content, including likes, saves, shares, completions, dwell time, ratings, and engagement score.
- `processed/content_features.csv`: catalog plus text features used by the recommender.
- `processed/user_profiles.csv`: user profile features derived from preference fields.
- `processed/interaction_scores.csv`: interaction table with helper fields for evaluation and diagnostics.

## Reproducibility

Run the preprocessing step from the repository root:

```bash
python -m src.data_prep
```

The Streamlit app can load saved model assets from `models/`, but `src/recommender.py` can also rebuild the core similarity, popularity, and seen-item lookup directly from the CSV files when pickled assets are unavailable or incompatible.
