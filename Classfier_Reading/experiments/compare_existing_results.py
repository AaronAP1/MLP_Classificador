"""
Script para Comparar Resultados de Clasificadores Existentes
============================================================

Este script carga los resultados de los modelos ya entrenados desde
V0.1/Resultados_prev/ y aplica el algoritmo de Copeland para
determinar qué clasificador es el mejor.

Uso:
----
python experiments/compare_existing_results.py --grupo adultos

Autor: Aaron (AaronAP1)
Fecha: Octubre 2025
"""

import sys
from pathlib import Path
import pandas as pd
import argparse
import re
from typing import Dict, List, Tuple

# Agregar src/ al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.copeland import (
    copeland_ranking,
    generate_comparison_table,
    save_ranking_results,
    CopelandRanking
)


def parse_classification_report(report_path: Path) -> Dict[str, float]:
    """
    Parsea un archivo classification_report.txt y extrae métricas.
    
    Args:
        report_path: Path al archivo de reporte
        
    Returns:
        metrics: Diccionario con métricas extraídas
    """
    metrics = {}
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar accuracy
    accuracy_match = re.search(r'accuracy\s+(\d+\.\d+)', content)
    if accuracy_match:
        metrics['accuracy'] = float(accuracy_match.group(1))
    
    # Buscar weighted avg
    weighted_pattern = r'weighted avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)'
    weighted_match = re.search(weighted_pattern, content)
    if weighted_match:
        metrics['precision_weighted'] = float(weighted_match.group(1))
        metrics['recall_weighted'] = float(weighted_match.group(2))
        metrics['f1_weighted'] = float(weighted_match.group(3))
    
    # Buscar macro avg
    macro_pattern = r'macro avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)'
    macro_match = re.search(macro_pattern, content)
    if macro_match:
        metrics['precision_macro'] = float(macro_match.group(1))
        metrics['recall_macro'] = float(macro_match.group(2))
        metrics['f1_macro'] = float(macro_match.group(3))
    
    return metrics


def parse_metrics_csv(csv_path: Path) -> Dict[str, float]:
    """
    Parsea metrics_by_epoch.csv y obtiene métricas de la mejor época.
    
    Args:
        csv_path: Path al CSV de métricas
        
    Returns:
        metrics: Métricas de la mejor época (val_f1 más alto)
    """
    df = pd.read_csv(csv_path)
    
    # Encontrar época con mejor val_f1
    if 'val_f1' in df.columns:
        best_epoch_idx = df['val_f1'].idxmax()
    elif 'val_loss' in df.columns:
        best_epoch_idx = df['val_loss'].idxmin()
    else:
        best_epoch_idx = len(df) - 1  # Última época
    
    best_row = df.iloc[best_epoch_idx]
    
    metrics = {}
    for col in df.columns:
        if col != 'epoch':
            metrics[col] = float(best_row[col])
    
    return metrics


def load_model_results(
    results_dir: Path,
    model_name: str
) -> Dict[str, float]:
    """
    Carga todos los resultados de un modelo específico.
    
    Args:
        results_dir: Directorio de resultados (ej: Resultados_prev/MLP/)
        model_name: Nombre del modelo (MLP_MA, TabNet, FT-Transformer)
        
    Returns:
        metrics: Diccionario con todas las métricas
    """
    all_metrics = {}
    
    # Cargar desde classification_report.txt
    report_path = results_dir / "classification_report.txt"
    if report_path.exists():
        metrics_from_report = parse_classification_report(report_path)
        all_metrics.update(metrics_from_report)
    
    # Cargar desde metrics_by_epoch.csv
    csv_path = results_dir / "metrics_by_epoch.csv"
    if csv_path.exists():
        metrics_from_csv = parse_metrics_csv(csv_path)
        # Solo agregar métricas que no están en report
        for key, val in metrics_from_csv.items():
            if key not in all_metrics:
                all_metrics[key] = val
    
    # Agregar nombre del modelo
    all_metrics['classifier'] = model_name
    
    return all_metrics


def load_all_results_for_group(
    base_dir: Path,
    grupo: str = None
) -> pd.DataFrame:
    """
    Carga resultados de todos los modelos.
    
    Args:
        base_dir: Directorio base (ej: V0.1/Resultados_prev/)
        grupo: Grupo etario (opcional, actualmente no usado)
        
    Returns:
        results_df: DataFrame con resultados de todos los modelos
    """
    base_dir = Path(base_dir)
    
    # Mapeo de carpetas a nombres de modelos
    model_mapping = {
        'MLP': 'MLP_MA',
        'TabNet': 'TabNet',
        'FT-Transformer': 'FT-Transformer'
    }
    
    results_list = []
    
    for folder_name, model_name in model_mapping.items():
        model_dir = base_dir / folder_name
        
        if model_dir.exists():
            print(f"📊 Cargando resultados de {model_name}...")
            try:
                metrics = load_model_results(model_dir, model_name)
                results_list.append(metrics)
                print(f"   ✅ {len(metrics)} métricas cargadas")
            except Exception as e:
                print(f"   ⚠️  Error: {e}")
        else:
            print(f"   ⚠️  Carpeta no encontrada: {model_dir}")
    
    if not results_list:
        raise ValueError("No se pudieron cargar resultados de ningún modelo")
    
    results_df = pd.DataFrame(results_list)
    
    return results_df


def compare_models(
    results_df: pd.DataFrame,
    metrics_to_compare: List[str] = None,
    weights: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Compara modelos usando Copeland.
    
    Args:
        results_df: DataFrame con resultados
        metrics_to_compare: Lista de métricas a comparar
        weights: Pesos por métrica
        
    Returns:
        ranking_df: Ranking de Copeland
    """
    # Métricas por defecto
    if metrics_to_compare is None:
        available_metrics = [col for col in results_df.columns if col != 'classifier']
        
        # Priorizar métricas importantes
        priority_metrics = [
            'f1_weighted', 'f1_macro', 
            'accuracy',
            'precision_weighted', 'precision_macro',
            'recall_weighted', 'recall_macro'
        ]
        
        metrics_to_compare = [m for m in priority_metrics if m in available_metrics]
        
        if not metrics_to_compare:
            metrics_to_compare = available_metrics[:5]  # Primeras 5 métricas
    
    print(f"\n📏 Métricas utilizadas para comparación: {metrics_to_compare}")
    
    # Aplicar Copeland
    ranking = copeland_ranking(
        results_df,
        metrics=metrics_to_compare,
        weights=weights
    )
    
    return ranking


def main():
    parser = argparse.ArgumentParser(
        description="Comparar clasificadores usando Algoritmo de Copeland"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='../V0.1/Resultados_prev',
        help='Directorio con resultados de modelos'
    )
    parser.add_argument(
        '--grupo',
        type=str,
        choices=['adultos', 'jovenes', 'ninos', 'all'],
        default='adultos',
        help='Grupo etario a analizar'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/comparisons',
        help='Directorio de salida para resultados'
    )
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=None,
        help='Métricas específicas a comparar'
    )
    
    args = parser.parse_args()
    
    # Configuración
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 80)
    print("🏆 COMPARACIÓN DE CLASIFICADORES - ALGORITMO DE COPELAND")
    print("=" * 80)
    print(f"\n📂 Directorio de resultados: {results_dir}")
    print(f"👥 Grupo etario: {args.grupo}")
    print()
    
    # Cargar resultados
    print("=" * 80)
    print("1️⃣ CARGANDO RESULTADOS DE MODELOS")
    print("=" * 80)
    
    results_df = load_all_results_for_group(results_dir, args.grupo)
    
    print(f"\n✅ Resultados cargados: {len(results_df)} modelos")
    print(f"📊 Métricas disponibles: {len(results_df.columns) - 1}")
    print()
    
    # Mostrar datos cargados
    print("\n" + "=" * 80)
    print("2️⃣ DATOS CARGADOS")
    print("=" * 80)
    
    # Mostrar solo columnas importantes
    display_cols = ['classifier']
    metric_cols = [col for col in results_df.columns if col != 'classifier']
    display_cols.extend(metric_cols[:8])  # Primeras 8 métricas
    
    print(results_df[display_cols].to_string(index=False))
    print()
    
    # Comparar con Copeland
    print("\n" + "=" * 80)
    print("3️⃣ APLICANDO ALGORITMO DE COPELAND")
    print("=" * 80)
    
    ranking = compare_models(
        results_df,
        metrics_to_compare=args.metrics
    )
    
    print("\n✅ Ranking calculado")
    print()
    
    # Mostrar ranking
    print("\n" + "=" * 80)
    print("4️⃣ RANKING FINAL - ALGORITMO DE COPELAND")
    print("=" * 80)
    print()
    
    print(ranking.to_string(index=False))
    print()
    
    # Ganador
    winner = ranking.iloc[0]
    print("\n" + "=" * 80)
    print("🏆 GANADOR")
    print("=" * 80)
    print(f"\n   Clasificador: {winner['classifier']}")
    print(f"   Puntuación Copeland: {winner['copeland_score']:.4f}")
    
    if 'f1_weighted' in winner:
        print(f"   F1-Score (weighted): {winner['f1_weighted']:.4f}")
    if 'accuracy' in winner:
        print(f"   Accuracy: {winner['accuracy']:.4f}")
    print()
    
    # Guardar resultados
    print("\n" + "=" * 80)
    print("5️⃣ GUARDANDO RESULTADOS")
    print("=" * 80)
    print()
    
    save_ranking_results(
        ranking,
        output_dir=output_dir,
        group_name=args.grupo,
        format='all'
    )
    
    # Generar tabla markdown
    markdown_table = generate_comparison_table(
        ranking,
        title=f"Ranking de Clasificadores - {args.grupo.capitalize()}"
    )
    
    print("\n" + "=" * 80)
    print("6️⃣ TABLA MARKDOWN GENERADA")
    print("=" * 80)
    print()
    print(markdown_table)
    print()
    
    print("\n" + "=" * 80)
    print("✅ PROCESO COMPLETADO")
    print("=" * 80)
    print(f"\n📁 Resultados guardados en: {output_dir}")
    print()


if __name__ == "__main__":
    main()
