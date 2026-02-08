import unittest

import numpy as np
import pandas as pd

from src.preprocessing.clean_data import ProcessDataset


class TestDataCleaning(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "Attack": ["Benign", "DoS"],
                "SRC_TO_DST_IAT_MIN": [1, 2],
                "SRC_TO_DST_IAT_AVG": [1, 2],
                "SRC_TO_DST_IAT_MAX": [1, 2],
                "SRC_TO_DST_IAT_STDDEV": [1, 2],
                "DST_TO_SRC_IAT_MIN": [1, 2],
                "DST_TO_SRC_IAT_AVG": [1, 2],
                "DST_TO_SRC_IAT_MAX": [1, 2],
                "DST_TO_SRC_IAT_STDDEV": [1, 2],
                "SRC_TO_DST_AVG_THROUGHPUT": [1, 2],
                "DST_TO_SRC_AVG_THROUGHPUT": [1, 2],
            }
        )

    def test_label_creation(self):
        dataset = ProcessDataset(csv_path=None)
        dataset.df = self.df

        dataset.clean_data()

        expected_columns = dataset.cwt_columns + ["Attack"]
        for col in expected_columns:
            self.assertIn(col, dataset.df_clean.columns)

        self.assertFalse(dataset.df_clean.isna().any().any())

        self.assertTrue(
            dataset.df_clean["Attack"].dtype == object
            or np.issubdtype(dataset.df_clean["Attack"].dtype, np.str_)
        )


if __name__ == "__main__":
    unittest.main()
