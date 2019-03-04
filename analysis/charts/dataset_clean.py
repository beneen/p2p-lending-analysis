"""
Loads the huge dataset of accepted loans

csv read not ideal
----------
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
import seaborn as sns
from sklearn import tree
#matplotlib inline

def dataset_clean_year():
    df = pd.read_csv(r'/home/beneen/PycharmProjects/p2p_lending/analysis/data_sample/accepted.csv')


    # renaming columns to make it easier
    df = df.rename(columns={"loan_amnt": "loan_amount", "funded_amnt": "funded_amount", "funded_amnt_inv": "investor_funds",
                           "int_rate": "interest_rate", "annual_inc": "annual_income"})


    # dates to year only
    df['issue_d'].head()
    dt_series = pd.to_datetime(df['issue_d'])
    df['year'] = dt_series.dt.year

    return df


def loan_condition(status):
    if status in bad_loan:
        return 'Bad Loan'
    else:
        return 'Good Loan'