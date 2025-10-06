"""
Módulo de Métricas para Clasificadores
======================================

Funciones para calcular métricas de evaluación de clasificadores.

Autor: Aaron (AaronAP1)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report
)
from typing import Dict, Optional


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    labels: Optional[list] = None
) -> Dict[str, float]:
    """
    Calcula métricas comprehensivas de clasificación.
    
    Args:
        y_true: Labels verdaderos
        y_pred: Predicciones del clasificador
        y_probs: Probabilidades por clase (opcional, para ROC AUC)
        labels: Nombres de clases (opcional)
        
    Returns:
        metrics: Diccionario con métricas calculadas
    """
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['precision_weighted'] = precision
    metrics['recall_weighted'] = recall
    metrics['f1_weighted'] = f1
    
    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics['precision_macro'] = precision_macro
    metrics['recall_macro'] = recall_macro
    metrics['f1_macro'] = f1_macro
    
    # ROC AUC (si se proporcionan probabilidades)
    if y_probs is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(
                y_true, y_probs, multi_class='ovr', average='weighted'
            )
        except:
            metrics['roc_auc'] = None
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list] = None
) -> np.ndarray:
    """
    Calcula la matriz de confusión.
    
    Args:
        y_true: Labels verdaderos
        y_pred: Predicciones
        labels: Nombres de clases
        
    Returns:
        cm: Matriz de confusión
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def compute_roc_auc(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    multi_class: str = 'ovr',
    average: str = 'weighted'
) -> float:
    """
    Calcula ROC AUC para clasificación multiclase.
    
    Args:
        y_true: Labels verdaderos
        y_probs: Probabilidades por clase
        multi_class: Estrategia ('ovr' o 'ovo')
        average: Tipo de promedio
        
    Returns:
        roc_auc: Valor de ROC AUC
    """
    try:
        return roc_auc_score(y_true, y_probs, multi_class=multi_class, average=average)
    except:
        return None
