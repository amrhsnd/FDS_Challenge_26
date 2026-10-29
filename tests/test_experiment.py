"""Workflow tests use synthetic files and never read the official test data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib
import numpy as np
import pandas as pd
from pokemon_challenge.experiment import BattleExperiment, Candidate
from pokemon_challenge.models.classifiers import make_pipeline


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        def battle(i, labelled=True):
            y = i % 2
            pokemon = {f"base_{s}": 60 + 20*y + i % (j+3) for j, s in enumerate(("hp", "atk", "def", "spa", "spd", "spe"))}
            b = {"battle_id": i, "p1_team_details": [pokemon], "p2_lead_details": pokemon.copy(), "battle_timeline": []}
            if labelled: b["player_won"] = y
            return b
        for name, count in [("train", 100), ("test", 20)]:
            (self.data / f"{name}.jsonl").write_text("".join(json.dumps(battle(i, name == "train"))+"\n" for i in range(count)))
        pd.DataFrame({"battle_id": list(reversed(range(20))), "player_won": 0}).to_csv(self.data / "sample_submission.csv", index=False)
        def destination(owner, run):
            path = self.root / "outputs" / owner / run
            path.mkdir(parents=True, exist_ok=True)
            return path
        patcher = patch("pokemon_challenge.experiment.output_dir", side_effect=destination)
        patcher.start()
        self.addCleanup(patcher.stop)

    def experiment(self, run="one", **kwargs):
        return BattleExperiment("test", run, data_path=self.data, **kwargs)

    def test_repeatable_disjoint_partitions(self):
        a, b = self.experiment(), self.experiment("two")
        pd.testing.assert_frame_equal(a.assignments, b.assignments)
        self.assertEqual(a.manifest["splits_sha256"], b.manifest["splits_sha256"])
        self.assertFalse(set(a.development) & set(a._holdout))
        seen = []
        for training, validation in a.folds:
            self.assertFalse(set(training) & set(validation))
            seen.extend(validation)
        self.assertEqual(sorted(seen), list(range(len(a.development))))
        with self.assertRaises(FileExistsError): self.experiment()

    def test_compare_holdout_lock_and_submission_alignment(self):
        experiment = self.experiment()
        with self.assertRaises(RuntimeError): experiment.make_submission()
        parameters = {"C": 0.5}
        result = experiment.compare([
            Candidate("logistic", params=parameters), Candidate("selected", percentile=50),
            Candidate("forest", model="random_forest", params={"n_estimators": 3, "max_depth": 3}),
            Candidate("boosting", model="hist_gradient_boosting", params={"max_iter": 3, "max_leaf_nodes": 3}),
        ])
        self.assertEqual(len(result), 4)
        parameters["C"] = 999
        self.assertEqual(experiment._candidates["logistic"].params["C"], 0.5)
        self.assertTrue(result["fold_5"].notna().all())
        self.assertEqual(experiment.best_candidate, result.iloc[0]["candidate"])
        metrics = experiment.evaluate_holdout()
        self.assertEqual(metrics["n_battles"], 20)
        self.assertTrue(metrics["selected_features"])
        self.assertEqual(metrics, experiment.evaluate_holdout())
        with self.assertRaises(RuntimeError): experiment.compare([Candidate("later")])
        other = next(n for n in result["candidate"] if n != metrics["candidate"])
        with self.assertRaises(RuntimeError): experiment.evaluate_holdout(other)
        submission = experiment.make_submission()
        self.assertEqual(submission.battle_id.tolist(), list(reversed(range(20))))
        self.assertTrue(submission.player_won.isin([0, 1]).all())
        final = json.loads((experiment.output_path / "final_fit.json").read_text())
        self.assertEqual(final["fit_rows"], 100)
        model = joblib.load(experiment.output_path / "model.joblib")
        self.assertEqual(model.feature_names_in_.tolist(), final["feature_columns"])
        self.assertEqual(len(list(experiment.output_path.glob("selection_frequency_*.csv"))), 4)
        pd.DataFrame({"battle_id": [999], "player_won": [0]}).to_csv(self.data / "sample_submission.csv", index=False)
        with self.assertRaisesRegex(ValueError, "Sample IDs"): experiment.make_submission()

    def test_feature_contract_removes_identifiers_and_label(self):
        def custom(battle):
            self.assertNotIn("player_won", battle)
            self.assertNotIn("battle_id", battle)
            return {"hp": battle["p1_team_details"][0]["base_hp"]}
        experiment = self.experiment(feature_functions={"custom": custom})
        self.assertEqual(list(experiment.X), ["custom__hp"])
        with self.assertRaisesRegex(ValueError, "Unknown feature groups"):
            experiment.compare([Candidate("wrong")])
        def categorical(battle): return {"species": "snorlax"}
        with self.assertRaisesRegex(ValueError, "not numeric"):
            self.experiment("bad", feature_functions={"category": categorical})

    def test_preprocessing_learns_training_rows_only(self):
        X = pd.DataFrame({"variable": [0., 1., np.nan, 3., 4., 5.], "constant_in_train": [1.]*6})
        pipeline = make_pipeline()
        pipeline.fit(X, [0, 0, 0, 1, 1, 1])
        self.assertEqual(pipeline.named_steps["imputer"].statistics_[0], 3.)
        pipeline.predict(pd.DataFrame({"variable": [10000.], "constant_in_train": [999.]}))
        self.assertEqual(pipeline.named_steps["imputer"].statistics_[0], 3.)
        self.assertEqual(pipeline[:-1].get_feature_names_out().tolist(), ["variable"])


if __name__ == "__main__": unittest.main()
