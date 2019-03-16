import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

"""
-----
roc curve plot function
-----
"""


def roc_curve_plot(truth, pred, lab):
    fpr, tpr, _ = metrics.roc_curve(truth, pred)
    roc_auc = metrics.auc(fpr, tpr)
    lw = 2
    c = (np.random.rand(), np.random.rand(), np.random.rand())
    plt.plot(fpr, tpr, color=c, lw=lw, label=lab + '(AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')  # Receiver Operating Characteristic
    plt.legend(loc="lower right")
