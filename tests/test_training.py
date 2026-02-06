import unittest

import pandas as pd

from src.models.train_baseline import BaselineTraining


class TestBaselineTraining(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "Attack": ["Benign"] * 50 + ["DoS"] * 50,
                "f1": range(100),
            }
        )

    def test_training_runs(self):
        trainer = BaselineTraining(df=self.df)
        trainer.load_and_prepare_data()
        trainer.train_knn()

        self.assertIn("KNN", trainer.trained_models)
