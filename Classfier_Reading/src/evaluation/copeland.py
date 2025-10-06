"""
Algoritmo de Copeland para Comparación de Clasificadores
=========================================================

Implementación del método de Copeland para ranking de clasificadores.

El algoritmo de Copeland compara clasificadores por pares y asigna puntos:
- +1 punto si el clasificador A es mejor que B en una métrica
- -1 punto si el clasificador A es peor que B
- 0 puntos si hay empate

El clasificador con mayor puntuación Copeland es declarado ganador.

Referencia:
-----------
Copeland, A. H. (1951). A 'reasonable' social welfare function.
Seminar on Mathematics in Social Sciences, University of Michigan.

Uso:
----
>>> from src.evaluation.copeland import copeland_ranking
>>> 
>>> # DataFrame con métricas de clasificadores
>>> results = pd.DataFrame({
...     'classifier': ['MLP_MA', 'TabNet', 'FT-Transformer'],
...     'f1_weighted': [0.85, 0.87, 0.83],
...     'precision': [0.84, 0.86, 0.82],
...     'recall': [0.86, 0.88, 0.84],
...     'accuracy': [0.85, 0.87, 0.83]
... })
>>> 
>>> ranking = copeland_ranking(results, metrics=['f1_weighted', 'precision', 'recall', 'accuracy'])
>>> print(ranking)

Autor: Aaron (AaronAP1)
Fecha: Octubre 2025
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json


class CopelandRanking:
    """
    Clase para implementar el algoritmo de Copeland para ranking de clasificadores.
    
    Attributes:
        results_df (pd.DataFrame): DataFrame con resultados de clasificadores
        metrics (List[str]): Lista de métricas a considerar
        weights (Dict[str, float]): Pesos opcionales por métrica
    """
    
    def __init__(
        self,
        results_df: pd.DataFrame,
        metrics: List[str],
        weights: Optional[Dict[str, float]] = None,
        classifier_col: str = 'classifier'
    ):
        """
        Inicializa el ranking de Copeland.
        
        Args:
            results_df: DataFrame con columnas [classifier, metric1, metric2, ...]
            metrics: Lista de nombres de métricas a comparar
            weights: Diccionario opcional de pesos por métrica (default: igual peso)
            classifier_col: Nombre de la columna con nombres de clasificadores
        """
        self.results_df = results_df.copy()
        self.metrics = metrics
        self.classifier_col = classifier_col
        
        # Validar que las métricas existen
        missing_metrics = set(metrics) - set(results_df.columns)
        if missing_metrics:
            raise ValueError(f"Métricas faltantes en DataFrame: {missing_metrics}")
        
        # Pesos por defecto (iguales)
        if weights is None:
            self.weights = {metric: 1.0 for metric in metrics}
        else:
            self.weights = weights
            
        # Normalizar pesos
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        self.classifiers = results_df[classifier_col].tolist()
        self.n_classifiers = len(self.classifiers)
        
        # Inicializar matriz de comparaciones
        self.comparison_matrix = np.zeros((self.n_classifiers, self.n_classifiers))
        self.scores = {clf: 0.0 for clf in self.classifiers}
        
    def _compare_pair(self, clf_a: str, clf_b: str) -> float:
        """
        Compara dos clasificadores en todas las métricas.
        
        Args:
            clf_a: Nombre del clasificador A
            clf_b: Nombre del clasificador B
            
        Returns:
            score: Puntuación de A vs B (-1, 0, o +1 por métrica, ponderado)
        """
        score = 0.0
        
        for metric in self.metrics:
            val_a = self.results_df.loc[
                self.results_df[self.classifier_col] == clf_a, metric
            ].values[0]
            
            val_b = self.results_df.loc[
                self.results_df[self.classifier_col] == clf_b, metric
            ].values[0]
            
            # Comparación con tolerancia para empates
            tolerance = 1e-4
            if abs(val_a - val_b) < tolerance:
                metric_score = 0.0  # Empate
            elif val_a > val_b:
                metric_score = 1.0  # A gana
            else:
                metric_score = -1.0  # B gana
            
            # Ponderar por peso de la métrica
            score += metric_score * self.weights[metric]
        
        return score
    
    def compute_ranking(self) -> pd.DataFrame:
        """
        Calcula el ranking de Copeland para todos los clasificadores.
        
        Returns:
            ranking_df: DataFrame con ranking ordenado
        """
        # Comparaciones por pares
        for i, clf_a in enumerate(self.classifiers):
            for j, clf_b in enumerate(self.classifiers):
                if i != j:
                    score = self._compare_pair(clf_a, clf_b)
                    self.comparison_matrix[i, j] = score
                    self.scores[clf_a] += score
        
        # Crear DataFrame de ranking
        ranking_data = []
        for clf in self.classifiers:
            row_data = {
                'classifier': clf,
                'copeland_score': self.scores[clf],
            }
            
            # Agregar métricas originales
            for metric in self.metrics:
                row_data[metric] = self.results_df.loc[
                    self.results_df[self.classifier_col] == clf, metric
                ].values[0]
            
            ranking_data.append(row_data)
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Ordenar por puntuación Copeland (descendente)
        ranking_df = ranking_df.sort_values('copeland_score', ascending=False)
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        
        # Reordenar columnas
        cols = ['rank', 'classifier', 'copeland_score'] + self.metrics
        ranking_df = ranking_df[cols]
        
        return ranking_df
    
    def get_winner(self) -> Tuple[str, float]:
        """
        Obtiene el clasificador ganador.
        
        Returns:
            winner: Tupla (nombre_clasificador, copeland_score)
        """
        ranking = self.compute_ranking()
        winner_row = ranking.iloc[0]
        return (winner_row['classifier'], winner_row['copeland_score'])
    
    def get_comparison_matrix_df(self) -> pd.DataFrame:
        """
        Obtiene la matriz de comparaciones como DataFrame.
        
        Returns:
            comparison_df: DataFrame con comparaciones por pares
        """
        self.compute_ranking()  # Asegurar que está calculado
        
        comparison_df = pd.DataFrame(
            self.comparison_matrix,
            index=self.classifiers,
            columns=self.classifiers
        )
        
        return comparison_df
    
    def generate_detailed_report(self) -> Dict:
        """
        Genera un reporte detallado con toda la información del ranking.
        
        Returns:
            report: Diccionario con información completa
        """
        ranking_df = self.compute_ranking()
        
        report = {
            'ranking': ranking_df.to_dict('records'),
            'winner': {
                'classifier': ranking_df.iloc[0]['classifier'],
                'copeland_score': float(ranking_df.iloc[0]['copeland_score'])
            },
            'comparison_matrix': self.get_comparison_matrix_df().to_dict(),
            'metrics_used': self.metrics,
            'weights': self.weights,
            'n_classifiers': self.n_classifiers
        }
        
        return report


def copeland_ranking(
    results_df: pd.DataFrame,
    metrics: List[str],
    weights: Optional[Dict[str, float]] = None,
    classifier_col: str = 'classifier'
) -> pd.DataFrame:
    """
    Función de conveniencia para obtener ranking de Copeland.
    
    Args:
        results_df: DataFrame con resultados de clasificadores
        metrics: Lista de métricas a considerar
        weights: Pesos opcionales por métrica
        classifier_col: Nombre de columna de clasificadores
        
    Returns:
        ranking_df: DataFrame con ranking ordenado
        
    Example:
        >>> results = pd.DataFrame({
        ...     'classifier': ['MLP_MA', 'TabNet', 'FT-Transformer'],
        ...     'f1_weighted': [0.85, 0.87, 0.83],
        ...     'precision': [0.84, 0.86, 0.82],
        ...     'recall': [0.86, 0.88, 0.84]
        ... })
        >>> ranking = copeland_ranking(results, metrics=['f1_weighted', 'precision', 'recall'])
    """
    copeland = CopelandRanking(results_df, metrics, weights, classifier_col)
    return copeland.compute_ranking()


def compare_classifiers(
    results_dict: Dict[str, Dict[str, float]],
    metrics: List[str],
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Compara clasificadores desde un diccionario de resultados.
    
    Args:
        results_dict: Diccionario {classifier_name: {metric: value, ...}}
        metrics: Lista de métricas a comparar
        weights: Pesos opcionales
        
    Returns:
        ranking_df: DataFrame con ranking
        
    Example:
        >>> results = {
        ...     'MLP_MA': {'f1_weighted': 0.85, 'precision': 0.84},
        ...     'TabNet': {'f1_weighted': 0.87, 'precision': 0.86},
        ...     'FT-Transformer': {'f1_weighted': 0.83, 'precision': 0.82}
        ... }
        >>> ranking = compare_classifiers(results, metrics=['f1_weighted', 'precision'])
    """
    # Convertir dict a DataFrame
    rows = []
    for clf_name, metrics_dict in results_dict.items():
        row = {'classifier': clf_name}
        row.update(metrics_dict)
        rows.append(row)
    
    results_df = pd.DataFrame(rows)
    
    return copeland_ranking(results_df, metrics, weights)


def generate_comparison_table(
    ranking_df: pd.DataFrame,
    title: str = "Ranking de Clasificadores - Algoritmo de Copeland",
    highlight_winner: bool = True
) -> str:
    """
    Genera una tabla formateada en Markdown con el ranking.
    
    Args:
        ranking_df: DataFrame con ranking de Copeland
        title: Título de la tabla
        highlight_winner: Resaltar el ganador con ⭐
        
    Returns:
        markdown_table: Tabla en formato Markdown
    """
    lines = [f"# {title}", ""]
    
    # Encabezado de tabla
    headers = list(ranking_df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    lines.append(header_line)
    lines.append(separator_line)
    
    # Filas de datos
    for idx, row in ranking_df.iterrows():
        values = []
        for col in headers:
            val = row[col]
            
            # Formatear números
            if isinstance(val, (int, np.integer)):
                formatted_val = str(val)
            elif isinstance(val, (float, np.floating)):
                formatted_val = f"{val:.4f}"
            else:
                formatted_val = str(val)
            
            values.append(formatted_val)
        
        row_line = "| " + " | ".join(values) + " |"
        
        # Resaltar ganador
        if highlight_winner and row['rank'] == 1:
            row_line += " ⭐ **GANADOR**"
        
        lines.append(row_line)
    
    return "\n".join(lines)


def save_ranking_results(
    ranking_df: pd.DataFrame,
    output_dir: Path,
    group_name: str,
    format: str = 'all'
):
    """
    Guarda los resultados del ranking en múltiples formatos.
    
    Args:
        ranking_df: DataFrame con ranking
        output_dir: Directorio de salida
        group_name: Nombre del grupo (adultos, jovenes, ninos)
        format: Formato de salida ('csv', 'json', 'markdown', 'all')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_filename = f"{group_name}_copeland_ranking"
    
    if format in ['csv', 'all']:
        csv_path = output_dir / f"{base_filename}.csv"
        ranking_df.to_csv(csv_path, index=False)
        print(f"✅ Guardado CSV: {csv_path}")
    
    if format in ['json', 'all']:
        json_path = output_dir / f"{base_filename}.json"
        ranking_df.to_json(json_path, orient='records', indent=2)
        print(f"✅ Guardado JSON: {json_path}")
    
    if format in ['markdown', 'all']:
        md_path = output_dir / f"{base_filename}.md"
        markdown_table = generate_comparison_table(
            ranking_df,
            title=f"Ranking de Clasificadores - {group_name.capitalize()}"
        )
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_table)
        print(f"✅ Guardado Markdown: {md_path}")


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Ejemplo con datos simulados
    print("=" * 70)
    print("EJEMPLO: Algoritmo de Copeland para Comparación de Clasificadores")
    print("=" * 70)
    print()
    
    # Datos de ejemplo
    results = pd.DataFrame({
        'classifier': ['MLP_MA', 'TabNet', 'FT-Transformer'],
        'f1_weighted': [0.8534, 0.8712, 0.8456],
        'precision_weighted': [0.8501, 0.8698, 0.8423],
        'recall_weighted': [0.8567, 0.8726, 0.8489],
        'accuracy': [0.8534, 0.8712, 0.8456],
        'roc_auc': [0.9123, 0.9256, 0.9089]
    })
    
    print("Datos de entrada:")
    print(results.to_string(index=False))
    print()
    
    # Aplicar Copeland
    metrics_to_compare = ['f1_weighted', 'precision_weighted', 'recall_weighted', 'accuracy']
    
    ranking = copeland_ranking(
        results,
        metrics=metrics_to_compare,
        weights=None  # Pesos iguales
    )
    
    print("\n" + "=" * 70)
    print("RESULTADO: Ranking de Copeland")
    print("=" * 70)
    print(ranking.to_string(index=False))
    print()
    
    # Ganador
    winner = ranking.iloc[0]
    print(f"\n🏆 GANADOR: {winner['classifier']}")
    print(f"   Puntuación Copeland: {winner['copeland_score']:.4f}")
    print(f"   F1-Score: {winner['f1_weighted']:.4f}")
    print()
    
    # Tabla Markdown
    print("\n" + "=" * 70)
    print("TABLA EN MARKDOWN")
    print("=" * 70)
    print(generate_comparison_table(ranking))
