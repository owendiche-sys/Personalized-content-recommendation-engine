# Personalized Content Recommendation Engine



A hybrid recommendation system that personalizes student lifestyle content suggestions using user preferences, historical engagement, and content similarity, delivered through an interactive Streamlit app.



## Overview



Content overload makes it difficult for users to discover the most relevant items quickly. This project builds a recommendation engine that ranks diverse digital content such as study resources, career content, food guides, events, wellness suggestions, entertainment picks, and lifestyle recommendations.



The system combines:
 - **content-based filtering** using TF-IDF and cosine similarity
 - **popularity-based ranking** as a baseline and fallback
 - **hybrid recommendation logic** that blends similarity, popularity, and profile matching
 - **cold-start recommendations** based on manually selected preferences

The final solution is supported by an interactive **Streamlit app** for both existing-user and preference-based recommendation flows.


## Project Objectives

 - Build a realistic recommendation pipeline on top of synthetic but behavior-driven student lifestyle data
 - Personalize content suggestions using user profiles and interaction history
 - Compare popularity-based and hybrid recommendation strategies
 - Evaluate the recommender using offline metrics and diagnostic checks
 - Deploy the recommendation workflow in a clean interactive app

## Dataset

The project uses three core CSV files stored in `data/raw/`:
 - `users.csv` — user profiles, personas, interests, formats, moods, and behavior preferences
 - `content.csv` — content catalog with categories, tags, formats, moods, freshness, and popularity signals
 - `interactions.csv` — user-content interactions including likes, saves, shares, completion, dwell time, and engagement score


### Content coverage
The catalog is intentionally diverse and includes categories such as:
 - Study
 - Productivity
 - Career
 - Finance
 - Food
 - Travel
 - Events
 - Wellness
 - Fitness
 - Entertainment
 - Music
 - Lifestyle
 - Tech
 - Deals

 ## Project Structure
```text
personalized-content-recommendation-engine/
│
├── app.py
│
├── data/
│   ├── raw/
│   │   ├── users.csv
│   │   ├── content.csv
│   │   └── interactions.csv
│   └── processed/
│       ├── content_features.csv
│       ├── user_profiles.csv
│       └── interaction_scores.csv
│
├── notebooks/
│   ├── 01_data_check_and_eda.ipynb
│   ├── 02_recommender_build.ipynb
│   └── 03_model_evaluation.ipynb
│
├── src/
│   ├── data_prep.py
│   ├── recommender.py
│   └── utils.py
│
├── models/
│   ├── tfidf_matrix.pkl
│   ├── similarity_matrix.pkl
│   └── recommender_assets.pkl
│
├── results/
│   ├── figures/
│   └── metrics/
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

