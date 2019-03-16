import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

"""
-----
Function that plots the confusion matrix rather than have this code repeated in each file
-----
"""


def plotting_confusion_matrix(y_test, model, normalize=False):
    # getting the confusion matrix
    analysis_confusion_matrix = confusion_matrix(y_test, model, labels=[0, 1])
    outcomes = ["pay in full", "default"]
    cmap = plt.cm.plasma
    title = "confusion matrix"
    if normalize:
        analysis_confusion_matrix = analysis_confusion_matrix.astype('float') / analysis_confusion_matrix.sum(axis=1)[:,
                                                                                np.newaxis]
        analysis_confusion_matrix = np.around(analysis_confusion_matrix, decimals=3)

    # setting up plot
    plt.imshow(analysis_confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(outcomes))
    plt.xticks(tick_marks, outcomes, rotation=45)
    plt.yticks(tick_marks, outcomes)
    thresh = analysis_confusion_matrix.max() / 2.
    for i, j in itertools.product(range(analysis_confusion_matrix.shape[0]), range(analysis_confusion_matrix.shape[1])):
        plt.text(j, i, analysis_confusion_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if analysis_confusion_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('actual')
    plt.xlabel('predicted')
