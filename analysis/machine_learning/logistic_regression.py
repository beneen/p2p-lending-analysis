from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

from analysis.machine_learning.get_test_and_train import get_test_and_train
from analysis.machine_learning.plotting_confusion_matrix import plotting_confusion_matrix
from analysis.machine_learning.cross_validation import cross_validation_best_parameters
from analysis.machine_learning.data_cleaning import data_cleaning
from analysis.machine_learning.roc_curve_plot import roc_curve_plot

"""
-----
Logistic regression implementation and plots
-----
"""


def main():
    dataset, length_of_features = data_cleaning.dataset_clean()

    X_train, X_test, y_train, y_test = get_test_and_train(dataset)


    logistic_regression = linear_model.LogisticRegression(random_state=0)
    c = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    parameters = dict(C=c)
    best_accuracy, best_classifier = cross_validation_best_parameters(dataset, logistic_regression, parameters)
    print("Best accuracy is " + str(best_accuracy))
    print(best_classifier)

    logistic_regression_classifier = linear_model.LogisticRegression(C=best_classifier.C)
    logistic_regression_classifier.fit(X_train, y_train)
    logistic_regression_prediction = logistic_regression_classifier.predict_proba(X_test)[:, 1]
    logistic_regression_predict_bin = logistic_regression_classifier.predict(X_test)
    logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_prediction.round())
    print("logistic regression accuracy ", logistic_regression_accuracy)
    roc_curve_plot(y_test, logistic_regression_prediction, 'logistic regression')
    plt.show()
    plt.figure(figsize=(6, 6))
    plotting_confusion_matrix(y_test, logistic_regression_predict_bin, normalize=True)
    plt.show()


if __name__ == "__main__":
    main()
