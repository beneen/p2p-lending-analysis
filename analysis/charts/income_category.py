"""
Produces chart showing the issuance of loans per income category

----------
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from analysis.charts.dataset_clean import dataset_clean_year
from analysis.charts.loan_conditions import loan_conditions

if __name__ == "__main__":

    df = dataset_clean_year()
    df = loan_conditions(df)

    employment_length = ['10+ years', '< 1 year', '1 year', '3 years', '8 years', '9 years',
                         '4 years', '5 years', '6 years', '2 years', '7 years', 'n/a']

    # work in progress need to clean this part up

    lst = [df]
    df['emp_length_int'] = np.nan

    for col in lst:
        col.loc[col['emp_length'] == '10+ years', "emp_length_int"] = 10
        col.loc[col['emp_length'] == '9 years', "emp_length_int"] = 9
        col.loc[col['emp_length'] == '8 years', "emp_length_int"] = 8
        col.loc[col['emp_length'] == '7 years', "emp_length_int"] = 7
        col.loc[col['emp_length'] == '6 years', "emp_length_int"] = 6
        col.loc[col['emp_length'] == '5 years', "emp_length_int"] = 5
        col.loc[col['emp_length'] == '4 years', "emp_length_int"] = 4
        col.loc[col['emp_length'] == '3 years', "emp_length_int"] = 3
        col.loc[col['emp_length'] == '2 years', "emp_length_int"] = 2
        col.loc[col['emp_length'] == '1 year', "emp_length_int"] = 1
        col.loc[col['emp_length'] == '< 1 year', "emp_length_int"] = 0.5
        col.loc[col['emp_length'] == 'n/a', "emp_length_int"] = 0

    df['income_category'] = np.nan
    lst = [df]

    for col in lst:
        col.loc[col['annual_income'] <= 100000, 'income_category'] = 'Low'
        col.loc[(col['annual_income'] > 100000) & (col['annual_income'] <= 200000), 'income_category'] = 'Medium'
        col.loc[col['annual_income'] > 200000, 'income_category'] = 'High'

    # changing string indicators to binary

    lst = [df]
    df['loan_condition_int'] = np.nan

    for col in lst:
        col.loc[df['loan_condition'] == 'Good Loan', 'loan_condition_int'] = 0  # bad loan
        col.loc[df['loan_condition'] == 'Bad Loan', 'loan_condition_int'] = 1  # good loan

    # for labelling, this conversion
    df['loan_condition_int'] = df['loan_condition_int'].astype(int)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(14, 6))

    # plotting

    sns.violinplot(x="income_category", y="loan_amount", data=df, palette="Set2", ax=ax1)
    sns.violinplot(x="income_category", y="loan_condition_int", data=df, palette="Set2", ax=ax2)
    sns.boxplot(x="income_category", y="emp_length_int", data=df, palette="Set2", ax=ax3)
    sns.boxplot(x="income_category", y="interest_rate", data=df, palette="Set2", ax=ax4)

    plt.show()
