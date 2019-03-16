from unittest import TestCase
import pandas as pd

from analysis.machine_learning.logistic_regression import logistic_regression


class TestLogistic_regression(TestCase):
    def test_logistic_regression(self):
        zero = pd.read_csv(r'/home/beneen/PycharmProjects/p2p_lending/analysis/machine_learning/data_cleaning/zero.csv')
        with self.assertRaises(Exception):
            logistic_regression(zero)

    def test_logistic_regression_one(self):
        one = pd.read_csv(r'/home/beneen/PycharmProjects/p2p_lending/analysis/machine_learning/data_cleaning/one.csv')
        with self.assertRaises(Exception):
            logistic_regression(one)
