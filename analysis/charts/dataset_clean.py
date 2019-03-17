"""
----------
Loads the huge dataset of accepted loans

csv read not ideal
----------
"""

import pandas as pd


def dataset_clean_year():
    df = pd.read_csv(r'/home/beneen/PycharmProjects/p2p_lending/analysis/data_sample/accepted.csv')

    # renaming columns to make it easier
    df = df.rename(
        columns={"loan_amnt": "loan_amount", "funded_amnt": "funded_amount", "funded_amnt_inv": "investor_funds",
                 "int_rate": "interest_rate", "annual_inc": "annual_income"})

    # dates to year only
    df['issue_d'].head()
    dt_series = pd.to_datetime(df['issue_d'])
    df['year'] = dt_series.dt.year

    return df


if __name__ == "__main__":
    dataset_clean_year()
