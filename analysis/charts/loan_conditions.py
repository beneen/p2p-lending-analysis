"""
Produces charts showing loan conditions (bad v good)
doesnt return anything
takes ages to run because huge csv read
----------
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
import seaborn as sns
from sklearn import tree
#matplotlib inline
from dataset_clean import dataset_clean_year

bad_loan = ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off", "In Grace Period",
            "Late (16-30 days)", "Late (31-120 days)"]

def loan_condition(status):
    if status in bad_loan:
        return 'Bad Loan'
    else:
        return 'Good Loan'

def loan_conditions(df):
    df['loan_condition'] = np.nan
    df['loan_condition'] = df['loan_status'].apply(loan_condition)
    return df


if __name__ == "__main__":

    df = dataset_clean_year()
    df = loan_conditions(df)

    f, ax = plt.subplots(1,2, figsize=(16,8))

    colors = ["#3791D7", "#D72626"]
    labels ="Good Loans", "Bad Loans"

    plt.suptitle('Information on Loan Conditions', fontsize=20)

    df["loan_condition"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', ax=ax[0],
                                                 shadow=True, colors = colors,
                                                 labels=labels, fontsize=12, startangle=70)


    # ax[0].set_title('State of Loan', fontsize=16)
    ax[0].set_ylabel('% of Condition of Loans', fontsize=14)

    # sns.countplot('loan_condition', data=df, ax=ax[1], palette=colors)
    # ax[1].set_title('Condition of Loans', fontsize=20)
    # ax[1].set_xticklabels(['Good', 'Bad'], rotation='horizontal')
    palette = ["#3791D7", "#E01E1B"]

    sns.barplot(x="year", y="loan_amount", hue="loan_condition", data=df, palette=palette,
                estimator=lambda x: len(x) / len(df) * 100)
    ax[1].set(ylabel="(%)")

    plt.show()

