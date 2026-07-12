from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_prep import prepare_datasets
from recommender import build_recommender_assets, load_recommender_bundle, recommend_for_user, recommend_from_preferences


class RecommenderBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepared = prepare_datasets(ROOT)
        cls.users = prepared["users"]
        cls.content_features = prepared["content_features"]
        cls.interaction_scores = prepared["interaction_scores"]
        cls.bundle = load_recommender_bundle(ROOT)

    def test_assets_can_be_rebuilt_from_csvs(self):
        assets = build_recommender_assets(self.content_features, self.interaction_scores)

        self.assertEqual(
            assets["similarity_matrix"].shape[0],
            len(self.content_features),
        )
        self.assertIn("top_popular", assets)
        self.assertGreater(len(assets["seen_lookup"]), 0)

    def test_catalog_has_product_facing_content_fields(self):
        expected_columns = {
            "description",
            "why_it_matters",
            "source_name",
            "action_label",
            "source_url",
            "recommendation_surface",
            "practicality_score",
        }

        self.assertTrue(expected_columns.issubset(set(self.content_features.columns)))
        self.assertTrue(self.content_features["description"].str.len().gt(40).all())
        self.assertTrue(self.content_features["source_url"].str.startswith("https://").all())

    def test_existing_user_recommendations_are_unseen_and_explained(self):
        user_id = self.users["user_id"].iloc[0]
        seen = self.bundle.seen_lookup.get(user_id, set())

        recs = recommend_for_user(self.bundle, user_id=user_id, top_n=8)

        self.assertFalse(recs.empty)
        self.assertTrue(set(recs["content_id"]).isdisjoint(seen))
        self.assertTrue(recs["recommendation_reason"].str.startswith("Recommended because").all())
        self.assertIn("description", recs.columns)
        self.assertIn("action_label", recs.columns)

    def test_preference_recommendations_prioritize_selected_category(self):
        recs = recommend_from_preferences(
            self.bundle,
            favorite_categories=["Career"],
            preferred_formats=["Guide"],
            preferred_moods=["Focused"],
            top_n=10,
        )

        self.assertFalse(recs.empty)
        self.assertIn("Career", set(recs.head(5)["category"]))
        self.assertTrue(recs["final_score"].is_monotonic_decreasing)


if __name__ == "__main__":
    unittest.main()
