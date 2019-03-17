"""
----------
Produces chart showing the issuance of by credit score
----------
"""

import matplotlib.pyplot as plt
from analysis.charts.dataset_clean import dataset_clean_year

if __name__ == "__main__":
    df = dataset_clean_year()

    f, ((ax1, ax2)) = plt.subplots(1, 2)
    cmap = plt.cm.coolwarm

    by_credit_score = df.groupby(['year', 'grade']).loan_amount.mean()
    by_credit_score.unstack().plot(legend=False, ax=ax1, figsize=(14, 4), colormap=cmap)
    ax1.set_title('Issuance of loans by credit score', fontsize=14)

    by_inc = df.groupby(['year', 'grade']).interest_rate.mean()
    by_inc.unstack().plot(ax=ax2, figsize=(14, 4), colormap=cmap)
    ax2.set_title('Interest rate assigned by credit score', fontsize=14)

    ax2.legend(bbox_to_anchor=(-1.0, -0.3, 1.7, 0.1), loc=5, prop={'size': 12},
               ncol=7, mode="expand", borderaxespad=0.)

    plt.show()
