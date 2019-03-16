from unittest import TestCase

import pandas as pd

from analysis.machine_learning.support_vector_machine import support_vector_machine


class TestSupport_vector_machine(TestCase):
    def test_support_vector_machine_fails_with_one_class_of_data_all_zero(self):
        zero = pd.read_csv(r'/home/beneen/PycharmProjects/p2p_lending/analysis/machine_learning/data_cleaning/zero.csv')
        with self.assertRaises(Exception):
            support_vector_machine(zero)

    def test_support_vector_machine_fails_with_one_class_of_data_all_one(self):
        one = pd.read_csv(r'/home/beneen/PycharmProjects/p2p_lending/analysis/machine_learning/data_cleaning/one.csv')
        with self.assertRaises(Exception):
            support_vector_machine(one)
