import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

from analysis.machine_learning.data_cleaning import data_cleaning
from analysis.machine_learning.get_test_and_train import get_test_and_train
from analysis.machine_learning.plotting_confusion_matrix import plotting_confusion_matrix
from analysis.machine_learning.roc_curve_plot import roc_curve_plot

"""
-----
random forest implementation and plotting
-----
"""


def main():
    dataset, length_of_features = data_cleaning.dataset_clean()

    # for testing
    # dataset = dataset.head(500)

    X_train, X_test, y_train, y_test = get_test_and_train(dataset)

    random_forest = RandomForestClassifier(criterion='gini', random_state=0)
    maximum_features = range(1, dataset.shape[1] - 1)
    parameters = dict(max_features=maximum_features)
    randomised_search = RandomizedSearchCV(random_forest, parameters, cv=10, scoring='accuracy',
                                           n_iter=len(maximum_features), random_state=10)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    randomised_search.fit(X, y)

    print(randomised_search.best_estimator_)

    random_forest = RandomForestClassifier(bootstrap=True, criterion="gini",
                                           max_features=randomised_search.best_estimator_.max_features, random_state=0)
    random_forest.fit(X_train, y_train)
    random_forest_predict = random_forest.predict(X_test)
    random_forest_predict_probabilities = random_forest.predict_proba(X_test)[:, 1]
    random_forest_accuracy = accuracy_score(y_test, random_forest_predict)
    roc_score = roc_auc_score(y_test, random_forest_predict)
    print(random_forest_accuracy)

    fig, ax = plt.subplots()
    width = 0.35
    ax.bar(np.arange(length_of_features - 1), random_forest.feature_importances_, width, color='r')
    ax.set_xticks(np.arange(len(random_forest.feature_importances_)))
    ax.set_xticklabels(X_train.columns.values, rotation=90)
    plt.title('importance of features')
    ax.set_ylabel('normalised importance (gini)')
    plt.show()

    roc_curve_plot(y_test, random_forest_predict_probabilities, 'Random Forest')
    plt.show()
    plt.figure(figsize=(6, 6))
    plotting_confusion_matrix(y_test, random_forest_predict, normalize=True)
    plt.show()


if __name__ == "__main__":
    main()
