import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import os

def compute_metrics(y_true, y_pred, y_proba=None, average="weighted"):
    """
    Calcula métricas completas de clasificación
    """
    metrics = {}
    
    # Métricas básicas
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # Matriz de confusión
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # Métricas por clase
    metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # ROC AUC (si se proporcionan probabilidades)
    if y_proba is not None:
        if len(np.unique(y_true)) == 2:
            # Clasificación binaria
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            # Clasificación multiclase - One-vs-Rest
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            except ValueError:
                metrics['roc_auc'] = None
    
    return metrics

def plot_roc_curve(y_true, y_proba, class_names, save_path=None):
    """
    Grafica curva ROC para clasificación multiclase
    """
    n_classes = len(class_names)
    
    plt.figure(figsize=(10, 8))
    
    if n_classes == 2:
        # Clasificación binaria
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        auc = roc_auc_score(y_true, y_proba[:, 1])
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    else:
        # Clasificación multiclase - One-vs-Rest
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=range(n_classes))
        
        for i in range(n_classes):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
                plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc:.3f})')
            except:
                continue
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def print_classification_report(metrics, class_names):
    """
    Imprime un reporte detallado de clasificación
    """
    print("\n" + "="*60)
    print("REPORTE DE CLASIFICACIÓN")
    print("="*60)
    
    print(f"\nMÉTRICAS GLOBALES:")
    print(f"Accuracy Global: {metrics['accuracy']:.4f}")
    print(f"Precision (weighted): {metrics['precision']:.4f}")
    print(f"Recall (weighted): {metrics['recall']:.4f}")
    print(f"F1-Score (weighted): {metrics['f1']:.4f}")
    
    if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
        print(f"ROC AUC (weighted): {metrics['roc_auc']:.4f}")
    
    print(f"\nMÉTRICAS POR CLASE:")
    print(f"{'Clase':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 56)
    
    for i, class_name in enumerate(class_names):
        prec = metrics['precision_per_class'][i]
        rec = metrics['recall_per_class'][i]
        f1 = metrics['f1_per_class'][i]
        print(f"{class_name:<20} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")
    
    print(f"\nMATRIZ DE CONFUSIÓN:")
    print(metrics['confusion_matrix'])
    print("="*60)
