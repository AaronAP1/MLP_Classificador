import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score

def compute_metrics(y_true, y_pred, y_proba=None, average="weighted"):
    m = {}
    m['accuracy'] = accuracy_score(y_true, y_pred)
    m['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    m['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    m['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    m['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    m['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    m['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    m['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                m['roc_auc'] = roc_auc_score(y_true, y_proba[:,1])
                m['pr_auc'] = average_precision_score(y_true, y_proba[:,1])
            else:
                m['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
        except Exception:
            m['roc_auc'] = None
    return m
