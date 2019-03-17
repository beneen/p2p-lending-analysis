"""
-----
test cases for knn algorithm
-----
"""

from unittest import TestCase

import pandas as pd

from analysis.machine_learning.KNN import k_nearest_neighbour


class TestK_nearest_neighbour(TestCase):

    def test_k_nearest_neighbour_fails_with_one_class_of_data_all_zero(self):
        zero = pd.read_csv(
            r'/home/beneen/PycharmProjects/p2p_lending/analysis/machine_learning/data_cleaning/zero.csv')
        with self.assertRaises(Exception):
            k_nearest_neighbour(zero)

    def test_k_nearest_neighbour_fails_with_one_class_of_data_all_one(self):
        one = pd.read_csv(
            r'/home/beneen/PycharmProjects/p2p_lending/analysis/machine_learning/data_cleaning/one.csv')
        with self.assertRaises(Exception):
            k_nearest_neighbour(one)
