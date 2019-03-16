from unittest import TestCase

import pandas as pd

from analysis.machine_learning.random_forest import random_forest


class TestRandom_forest(TestCase):
    def test_random_forest_fails_with_one_class_of_data_all_zero(self):
        zero = pd.read_csv(
            r'/home/beneen/PycharmProjects/p2p_lending/analysis/machine_learning/data_cleaning/zero.csv')
        with self.assertRaises(Exception):
            random_forest(zero)

    def test_random_forest_fails_with_one_class_of_data_all_one(self):
        one = pd.read_csv(
            r'/home/beneen/PycharmProjects/p2p_lending/analysis/machine_learning/data_cleaning/one.csv')
        with self.assertRaises(Exception):
            random_forest(one)
