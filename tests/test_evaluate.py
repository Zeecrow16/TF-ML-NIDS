import os
import unittest

from src.models.evaluate import ModelEvaluator


class TestEvaluator(unittest.TestCase):

    def test_results_folder_created(self):
        evaluator = ModelEvaluator("test_results")
        self.assertTrue(os.path.exists("test_results"))
