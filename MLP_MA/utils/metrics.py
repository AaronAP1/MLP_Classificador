import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
)
import matplotlib.pyplot as plt
import os


def compute_metrics(y_true, y_pred, y_proba=None, average="weighted"):
    m = {}
    m['accuracy'] = accuracy_score(y_true, y_pred)
    m['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    m['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    m['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    # Macro metrics para sensibilidad a clases raras
    m['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    m['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    m['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    m['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    m['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    m['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    m['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)

    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                m['roc_auc'] = roc_auc_score(y_true, y_proba[:,1])
                m['pr_auc'] = average_precision_score(y_true, y_proba[:,1])
            else:
                m['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                # PR AUC por clase y weighted
                from sklearn.preprocessing import label_binarize
                classes = np.unique(y_true)
                y_bin = label_binarize(y_true, classes=classes)
                pr_aucs = []
                for i in range(len(classes)):
                    try:
                        pr_aucs.append(average_precision_score(y_bin[:, i], y_proba[:, i]))
                    except Exception:
                        pr_aucs.append(np.nan)
                m['pr_auc_per_class'] = pr_aucs
                m['pr_auc_weighted'] = np.nanmean(pr_aucs)
        except Exception:
            m['roc_auc'] = None
    return m


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', save_path=None):
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_proba, class_names, save_path=None):
    n_classes = len(class_names)
    plt.figure(figsize=(8,6))
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])
        auc = roc_auc_score(y_true, y_proba[:,1])
        plt.plot(fpr, tpr, label=f'ROC (AUC={auc:.3f})')
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for i in range(n_classes):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
                plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC={auc:.3f})')
            except Exception:
                continue
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_classification_report(metrics, class_names):
    print("\n" + "="*60)
    print("REPORTE DE CLASIFICACIÓN")
    print("="*60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (weighted): {metrics['precision']:.4f}")
    print(f"Recall (weighted): {metrics['recall']:.4f}")
    print(f"F1 (weighted): {metrics['f1']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")
    if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
        print(f"ROC AUC (weighted): {metrics['roc_auc']:.4f}")
    if 'pr_auc_weighted' in metrics:
        print(f"PR AUC (weighted): {metrics['pr_auc_weighted']:.4f}")
    print("\nPor clase:")
    for i, c in enumerate(class_names):
        print(f" - {c}: P={metrics['precision_per_class'][i]:.3f} R={metrics['recall_per_class'][i]:.3f} F1={metrics['f1_per_class'][i]:.3f}")
    print("="*60)
