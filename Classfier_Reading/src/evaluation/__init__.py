"""
Módulo de Evaluación - Clasificador de Hábitos de Lectura
==========================================================

Este módulo contiene funciones para evaluar y comparar clasificadores
de redes neuronales profundas.

Componentes:
- metrics.py: Cálculo de métricas de clasificación
- copeland.py: Algoritmo de Copeland para comparación
- visualization.py: Visualizaciones de resultados
- reports.py: Generación de informes

Autor: Aaron (AaronAP1)
Proyecto: Clasificador de Hábitos de Lectura
Universidad Peruana Unión
"""

__version__ = "1.0.0"
__author__ = "Aaron (AaronAP1)"

from .metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_roc_auc,
)

from .copeland import (
    copeland_ranking,
    compare_classifiers,
    generate_comparison_table,
)

__all__ = [
    "compute_classification_metrics",
    "compute_confusion_matrix",
    "compute_roc_auc",
    "copeland_ranking",
    "compare_classifiers",
    "generate_comparison_table",
]
