import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from analysis.machine_learning.data_cleaning import data_cleaning
from analysis.machine_learning.get_test_and_train import get_test_and_train
from analysis.machine_learning.plotting_confusion_matrix import plotting_confusion_matrix
from analysis.machine_learning.roc_curve_plot import roc_curve_plot

"""
-----
Logistic regression implementation and plots
-----
"""


def logistic_regression(dataset):
    # for testing
    # dataset = dataset.head(500)

    X_train, X_test, y_train, y_test = get_test_and_train(dataset)

    logistic_regression = linear_model.LogisticRegression(random_state=0)
    c = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    parameters = dict(C=c)

    # cross validation - grid search
    grid = GridSearchCV(logistic_regression, parameters, cv=10, scoring='accuracy')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    grid.fit(X, y)
    best_accuracy = grid.best_score_
    best_classifier = grid.best_estimator_
    print("best accuracy is " + str(best_accuracy))
    print(best_classifier)

    logistic_regression_classifier = linear_model.LogisticRegression(C=best_classifier.C)
    logistic_regression_classifier.fit(X_train, y_train)
    logistic_regression_prediction = logistic_regression_classifier.predict_proba(X_test)[:, 1]
    logistic_regression_predict_bin = logistic_regression_classifier.predict(X_test)

    # accuracy
    logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_prediction.round())
    print("logistic regression accuracy ", logistic_regression_accuracy)

    # plotting
    roc_curve_plot(y_test, logistic_regression_prediction, 'logistic regression')
    plt.show()
    plt.figure(figsize=(6, 6))
    plotting_confusion_matrix(y_test, logistic_regression_predict_bin, normalize=True)
    plt.show()


def main():
    dataset, _ = data_cleaning.dataset_clean()
    logistic_regression(dataset)


if __name__ == "__main__":
    main()
