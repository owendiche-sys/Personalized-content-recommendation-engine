# Personalized Content Recommendation Engine

A Streamlit recommendation product that suggests student lifestyle content from real user preferences, historical engagement, content metadata, and popularity signals.

The app supports two practical flows:

- **Existing user recommendations**: choose a student profile and receive unseen content ranked for that user's interests and past engagement.
- **Preference-based recommendations**: choose categories, formats, and moods for a cold-start user and get immediate content suggestions.

Each recommendation includes a score, category, format, duration, cost band, tags, and a plain-English reason explaining why it was selected.

## Why This Project Matters

Many recommender demos stop at a table of IDs. This project is built to feel closer to a real product surface: users can filter recommendations, inspect the catalog, compare against a popularity baseline, and view model diagnostics from one dashboard.

The synthetic dataset models a student content platform with study resources, career guidance, finance tips, events, wellness content, food ideas, music, entertainment, travel, fitness, deals, productivity, tech, and lifestyle content.

## Features

- Hybrid ranking that blends content similarity, popularity, quality, recency, and profile matching.
- Existing-user flow that excludes content the user has already seen.
- Cold-start flow for users without interaction history.
- Similar-content explorer for item-to-item discovery.
- Recommendation cards with explainable reasons and useful content details.
- Popularity baseline for comparison.
- Offline metrics and diagnostic outputs stored in `results/metrics/`.
- Reproducible preprocessing through `src/data_prep.py`.
- Robust CSV fallback when saved pickle assets are missing or incompatible.
- Unit tests for asset rebuilding and recommendation behavior.

## Dataset

The project includes:

- `450` synthetic users
- `900` content items
- `22,000` interaction events

Core files live in `data/raw/`:

- `users.csv`: student personas, interests, preferred formats, moods, session habits, and discovery style.
- `content.csv`: titles, categories, tags, formats, mood, depth, duration, cost band, freshness, popularity, recency, and quality signals.
- `interactions.csv`: views, likes, saves, shares, completions, dwell time, ratings, timestamps, and engagement scores.

Processed files live in `data/processed/`. See `data/README.md` for details.

## Project Structure

```text
Personalized-content-recommendation-engine/
|-- app.py
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- README.md
|-- models/
|-- notebooks/
|-- results/
|   |-- figures/
|   `-- metrics/
|-- src/
|   |-- data_prep.py
|   |-- recommender.py
|   `-- utils.py
|-- tests/
|   `-- test_recommender.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Rebuild processed datasets:

```bash
python -m src.data_prep
```

Run a recommender smoke test:

```bash
python -m src.recommender
```

Launch the dashboard:

```bash
streamlit run app.py
```

Run tests:

```bash
python -m unittest discover -s tests
```

## Recommendation Logic

For existing users, the engine:

1. Finds each user's strongest historical engagement events.
2. Uses those items as seeds for content similarity.
3. Removes content the user has already interacted with.
4. Scores candidates using similarity, popularity, and profile match.
5. Returns ranked recommendations with an explanation.

For cold-start users, the engine:

1. Scores content against selected categories, formats, and moods.
2. Blends preference fit with popularity.
3. Returns immediately useful recommendations without requiring user history.

## Dashboard

The Streamlit app includes:

- recommendation cards for user-specific and cold-start suggestions
- category, format, and mood filters
- user profile summaries
- strongest historical interactions
- similar-item lookup
- content catalog browser
- saved offline diagnostics and baseline outputs

## Validation

Current automated checks cover:

- rebuilding recommender assets directly from CSV files
- returning unseen recommendations for existing users
- generating explained cold-start recommendations

The included offline evaluation artifacts are stored in `results/metrics/` and notebook outputs are available in `notebooks/`.

## Portfolio Notes

This project demonstrates:

- recommender-system product thinking
- synthetic behavioral data design
- feature engineering for profiles, content, and interactions
- hybrid ranking logic
- explainable recommendation UX
- reproducible preprocessing
- practical Streamlit app delivery
