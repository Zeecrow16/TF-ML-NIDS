import unittest

import pandas as pd

from src.preprocessing.clean_data import UNSWNB15Dataset


class TestDataCleaning(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "Attack": ["Benign", "DoS"],
                "NUM_PKTS_UP_TO_128_BYTES": [1, 2],
                "NUM_PKTS_128_TO_256_BYTES": [1, 2],
                "NUM_PKTS_256_TO_512_BYTES": [1, 2],
                "NUM_PKTS_512_TO_1024_BYTES": [1, 2],
                "NUM_PKTS_1024_TO_1514_BYTES": [1, 2],
                "SRC_TO_DST_IAT_MIN": [1, 2],
                "SRC_TO_DST_IAT_AVG": [1, 2],
                "SRC_TO_DST_IAT_MAX": [1, 2],
                "SRC_TO_DST_IAT_STDDEV": [1, 2],
                "DST_TO_SRC_IAT_MIN": [1, 2],
                "DST_TO_SRC_IAT_AVG": [1, 2],
                "DST_TO_SRC_IAT_MAX": [1, 2],
                "DST_TO_SRC_IAT_STDDEV": [1, 2],
                "IN_BYTES": [1, 2],
                "OUT_BYTES": [1, 2],
                "SRC_TO_DST_AVG_THROUGHPUT": [1, 2],
                "DST_TO_SRC_AVG_THROUGHPUT": [1, 2],
            }
        )

    def test_label_creation(self):
        dataset = UNSWNB15Dataset(csv_path=None)
        dataset.df = self.df

        dataset.clean_data()

        self.assertIn("Attack", dataset.df_clean.columns)
