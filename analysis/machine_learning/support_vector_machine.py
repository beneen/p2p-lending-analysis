from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from analysis.machine_learning.get_test_and_train import get_test_and_train
from analysis.machine_learning.plotting_confusion_matrix import plotting_confusion_matrix
from analysis.machine_learning.data_cleaning import data_cleaning
from analysis.machine_learning.roc_curve_plot import roc_curve_plot
from sklearn.svm import SVC

"""
-----
support vector machine is really slow to run on the full dataset but when testing it does run.
-----
"""

def main():

    dataset, length_of_features = data_cleaning.dataset_clean()
    # for testing
    #dataset = dataset.head(500)

    X_train, X_test, y_train, y_test = get_test_and_train(dataset)

    support_vector_machine_classifier = SVC()
    powers = range(0, 5)
    cs = [10 ** i for i in powers]
    parameters = dict(C=cs)
    grid = GridSearchCV(support_vector_machine_classifier, parameters, cv=10, scoring='accuracy')
    grid.fit(dataset.iloc[:, :-1].values, dataset.iloc[:, -1].values)

    print(grid.best_params_)
    print("---------------")
    print(grid.best_estimator_)

    support_vector_machine_classifier = SVC(kernel="rbf", C=grid.best_estimator_.C)
    support_vector_machine_classifier.fit(X_train.iloc[:, :], y_train)
    support_vector_machine_predictions = support_vector_machine_classifier.predict(X_test.iloc[:, :])
    support_vector_machine_prediction_probabilities = support_vector_machine_classifier.decision_function(X_test.iloc[:, :])
    support_vector_machine_accuracy = accuracy_score(y_test, support_vector_machine_predictions)
    print("support vector machine accuracy ", support_vector_machine_accuracy)
    roc_curve_plot(y_test, support_vector_machine_prediction_probabilities, 'support vector machine')
    plt.show()
    plt.figure(figsize=(6, 6))
    plotting_confusion_matrix(support_vector_machine_predictions, normalize=True)
    plt.show()

if __name__ == "__main__":
    main()
