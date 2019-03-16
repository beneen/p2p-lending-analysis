"""
Produces chart showing the issuance of loans per year
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




    df = dataset_clean_year()

    # setting up plot
    plt.figure()
    sns.barplot('year', 'loan_amount', data=df, palette='husl')
    plt.title('Issuance of Loans', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Average loan amount issued', fontsize=14)

    plt.show()