import unittest

import pandas as pd

from src.features.cwt import CalculateCwt


class TestCWT(unittest.TestCase):

    def test_cwt_features_created(self):
        df = pd.DataFrame(
            {
                "a": [1, 2],
                "b": [3, 4],
            }
        )

        cwt = CalculateCwt()
        result = cwt.calculate_cwt_category(df, ["a", "b"])

        self.assertEqual(len(result), 2)
