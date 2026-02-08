import unittest

import pandas as pd

from src.models.train_baseline import BaselineTraining


class TestBaselineTraining(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "Attack": ["Benign"] * 50 + ["DoS"] * 50,
                "SRC_TO_DST_IAT_MIN": range(100),
                "SRC_TO_DST_IAT_AVG": range(100),
                "SRC_TO_DST_IAT_MAX": range(100),
                "SRC_TO_DST_IAT_STDDEV": range(100),
                "DST_TO_SRC_IAT_MIN": range(100),
                "DST_TO_SRC_IAT_AVG": range(100),
                "DST_TO_SRC_IAT_MAX": range(100),
                "DST_TO_SRC_IAT_STDDEV": range(100),
                "SRC_TO_DST_AVG_THROUGHPUT": range(100),
                "DST_TO_SRC_AVG_THROUGHPUT": range(100),
            }
        )

    def test_training_runs(self):
        trainer = BaselineTraining(df=self.df)
        trainer.load_and_prepare_data()
        trainer.train_knn()

        self.assertIn("KNN", trainer.trained_models)
