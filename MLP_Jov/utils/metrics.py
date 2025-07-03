from sklearn.metrics import precision_score, f1_score, confusion_matrix

def compute_metrics(y_true, y_pred, average="weighted"):
    prec = precision_score(y_true, y_pred, average=average)
    f1   = f1_score(y_true, y_pred, average=average)
    cm   = confusion_matrix(y_true, y_pred)
    return prec, f1, cm
