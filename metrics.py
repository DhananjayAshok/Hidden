from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def compute_accuracies(y_true, y_pred_proba):
    accuracies = []
    thresholds = []
    for threshold in np.linspace(0, 1, 100):
        y_pred = (y_pred_proba > threshold).astype(int)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        accuracies.append(accuracy)
        thresholds.append(threshold)
    return accuracies, thresholds

def get_best_accuracy(y_true, y_pred_proba):
    accuracies, thresholds = compute_accuracies(y_true, y_pred_proba)
    best_idx = np.argmax(accuracies)
    return accuracies[best_idx], thresholds[best_idx]
