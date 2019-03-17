"""
----------
Produces chart showing the issuance of loans per year
----------
"""

import matplotlib.pyplot as plt
import seaborn as sns

from analysis.charts.dataset_clean import dataset_clean_year

if __name__ == "__main__":

    df = dataset_clean_year()

    # setting up plot of loan amount and issuance by year
    plt.figure()
    sns.barplot('year', 'loan_amount', data=df, palette='husl')
    plt.title('Issuance of Loans', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Average loan amount issued', fontsize=14)

    plt.show()
