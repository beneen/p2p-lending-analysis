from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from analysis.machine_learning.get_test_and_train import get_test_and_train
from analysis.machine_learning.plotting_confusion_matrix import plotting_confusion_matrix
from analysis.machine_learning.data_cleaning import data_cleaning
from sklearn.model_selection import GridSearchCV
from analysis.machine_learning.roc_curve_plot import roc_curve_plot

"""
-----
KNN implementation and plot
-----
"""


def main():

    dataset, length_of_features = data_cleaning.dataset_clean()
    # for testing

    X_train, X_test, y_train, y_test = get_test_and_train(dataset)

    knn_classsifier = KNeighborsClassifier()
    k_range = list(range(35, 50))
    parameter_grid = dict(n_neighbors=k_range)
    grid = GridSearchCV(knn_classsifier, parameter_grid, cv=10, scoring='accuracy')
    grid.fit(dataset.iloc[:, :-1].values, dataset.iloc[:, -1].values)
    print(grid.best_params_)
    print(grid.best_estimator_)
    print("", grid.best_params_['n_neighbors'])
    knn_final_classifier = KNeighborsClassifier(n_neighbors=grid.best_params_[
        'n_neighbors'])

    knn_final_classifier.fit(X_train, y_train)
    knn_prediction = knn_final_classifier.predict(X_test)
    knn_prediction_probability = knn_final_classifier.predict_proba(X_test)[:, 1]
    knn_accuracy = accuracy_score(y_test, knn_prediction)
    print("knn accuracy ", knn_accuracy)
    roc_curve_plot(y_test, knn_prediction_probability, 'KNN')
    plt.show()
    plt.figure(figsize=(6, 6))
    plotting_confusion_matrix(y_test, knn_prediction, normalize=True)
    plt.show()


if __name__ == "__main__":
    main()

