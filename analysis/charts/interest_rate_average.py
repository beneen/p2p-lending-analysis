"""
----------
Produces chart showing the issuance of loans per year
----------
"""

import matplotlib.pyplot as plt

from analysis.charts.dataset_clean import dataset_clean_year
from analysis.charts.loan_conditions import loan_conditions

if __name__ == "__main__":
    df = dataset_clean_year()
    df = loan_conditions(df)

    # printing average interest rates and annual income

    print(df['interest_rate'].mean())
    print(df['annual_income'].mean())

    # plotting interest by year, loans by region and loans by grade

    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(212)

    cmap = plt.cm.coolwarm_r

    loans_by_region = df.groupby(['grade', 'loan_condition']).size()
    loans_by_region.unstack().plot(kind='bar', stacked=True, colormap=cmap, ax=ax1, grid=False)
    ax1.set_title('Type of Loans by Grade', fontsize=14)

    loans_by_grade = df.groupby(['sub_grade', 'loan_condition']).size()
    loans_by_grade.unstack().plot(kind='bar', stacked=True, colormap=cmap, ax=ax2, grid=False)
    ax2.set_title('Type of Loans by Sub-Grade', fontsize=14)

    by_interest = df.groupby(['year', 'loan_condition']).interest_rate.mean()
    by_interest.unstack().plot(ax=ax3, colormap=cmap)
    ax3.set_title('Average Interest rate by Loan Condition', fontsize=14)
    ax3.set_ylabel('Interest Rate (%)', fontsize=12)

    plt.show()
