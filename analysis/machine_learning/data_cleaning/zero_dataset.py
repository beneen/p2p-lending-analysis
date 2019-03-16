import numpy as np
import pandas as pd

from analysis.machine_learning.data_cleaning import data_cleaning

"""
-----
generating a zero filled dataset for testing purposes
-----
"""


def zero_dataset(dataset):
    return pd.DataFrame(0, index=np.arange(len(dataset)), columns=dataset.columns)


def main():
    dataset, _ = data_cleaning.dataset_clean()
    print(zero_dataset(dataset.head(600)))


if __name__ == "__main__":
    main()
