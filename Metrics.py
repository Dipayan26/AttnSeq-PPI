import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
## Evaliation merics ###########################
def accuracy(y_true, y_prob, thresh=0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.accuracy_score(y_true, y_prob)
def sensitivity(y_true, y_prob, thresh=0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_prob).ravel()
    return tp / (tp + fn)
def specificity(y_true, y_prob, thresh=0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_prob).ravel()
    return tn / (tn + fp)
def precision(y_true, y_prob, thresh = 0.5): 
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.precision_score(y_true,y_prob)
def negative_predictive_value(y_true, y_prob, thresh = 0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    tn, fp, fn, tp = confusion_matrix(y_true, y_prob).ravel()
    npv = tn / (tn + fn)
    return npv
def recall(y_true, y_prob, thresh = 0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.recall_score(y_true,y_prob)
def mcc(y_true, y_prob, thresh=0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.matthews_corrcoef(y_true, y_prob)
def f1(y_true, y_prob, thresh = 0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.f1_score(y_true,y_prob)
def AUPRC(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return metrics.average_precision_score(y_true, y_prob)
def auc(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return metrics.roc_auc_score(y_true, y_prob)
def conf_matrix(y_true,y_prob, thresh = 0.5):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    tn, fp, fn, tp = confusion_matrix(y_true, y_prob).ravel()

    return tn, fp, fn, tp

metrics_dict = {"accuracy":accuracy,"sensitivity":sensitivity, "specificity":specificity ,"precision":precision,'NPV':negative_predictive_value,"recall":recall,"mcc":mcc,"f1":f1,"auc":auc,"AUPRC":AUPRC}
