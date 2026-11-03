"""Outcome-derived team features must not encode their own training labels."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from pokemon_challenge.features.winning_teams import WinningTeamFlags, team_key_features
from pokemon_challenge.models.classifiers import make_pipeline


class WinningTeamTests(unittest.TestCase):
    def test_roster_order_and_numeric_key(self):
        roster = [{"name": name} for name in "abcdef"]
        a = team_key_features({"p1_team_details": roster})
        b = team_key_features({"p1_team_details": roster[::-1], "player_won": 1})
        self.assertEqual(a, b)
        self.assertEqual(float(a["team_key"]), a["team_key"])
        self.assertTrue(np.isnan(team_key_features({})["team_key"]))

    def test_ranking_unseen_and_nested_flags(self):
        X = pd.DataFrame({"stat": np.arange(18), "roster__team_key": np.repeat(np.arange(9), 2)})
        y = np.array([1, 1] * 7 + [0, 0] * 2)
        encoder = WinningTeamFlags().fit(X, y)
        result = encoder.transform(X)
        self.assertNotIn("roster__team_key", result)
        self.assertEqual(result.p1_top_2_winning_team.sum(), 4)
        self.assertEqual(result.p1_top_5_winning_team.sum(), 10)
        self.assertEqual(result.p1_top_6_winning_team.sum(), 12)
        self.assertTrue((result.p1_top_2_winning_team <= result.p1_top_5_winning_team).all())
        unseen = pd.DataFrame({"stat": [0, 0], "roster__team_key": [99, np.nan]})
        self.assertEqual(encoder.transform(unseen).iloc[:, 1:].to_numpy().sum(), 0)

    def test_training_rows_are_cross_fitted(self):
        X = pd.DataFrame({"stat": np.arange(30), "roster__team_key": np.arange(30)})
        y = np.tile([0, 1], 15)
        encoder = WinningTeamFlags()
        training = encoder.fit_transform(X, y)
        # Each unique team is absent from its internal training fold.
        self.assertEqual(training.iloc[:, 1:].to_numpy().sum(), 0)
        self.assertGreater(encoder.transform(X).iloc[:, 1:].to_numpy().sum(), 0)
        np.testing.assert_array_equal(training.stat, X.stat)

    def test_xgboost_factory(self):
        X = pd.DataFrame({"x": np.tile([0., 1.], 10), "constant": 1.})
        model = make_pipeline("xgboost", params={"n_estimators": 3, "max_depth": 2})
        model.fit(X, np.tile([0, 1], 10))
        self.assertEqual(model.predict(X).shape, (20,))
        self.assertEqual(model.named_steps["model"].n_jobs, 1)


if __name__ == "__main__":
    unittest.main()
