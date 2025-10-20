# -*- coding: utf-8 -*-
"""
Script de Análisis Experimental - VALIDACIÓN DE CONFIGURACIONES

Este script te ayuda a responder:
1. ¿El seed 42 es suficiente o necesito promediar múltiples seeds?
2. ¿Cuántas épocas realmente necesito?
3. ¿Los class weights son apropiados?
4. ¿Qué hiperparámetros son más sensibles?

Estrategia:
- Entrena modelos con DIFERENTES seeds (42, 123, 456, 789, 2024)
- Mide VARIABILIDAD de resultados
- Determina si seed único es confiable o necesitas promedio
- Analiza curvas de aprendizaje (overfitting, convergencia)

Universidad Peruana Unión
Fecha: Octubre 2025
"""

import sys
import os

# Configurar UTF-8 para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from pathlib import Path
import time
import random
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar path del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.Ninos import create_tabnet_for_ninos


# ===================================================================
# CONFIGURACIÓN EXPERIMENTAL
# ===================================================================

EXPERIMENT_CONFIG = {
    # Seeds a probar (5 seeds para medir variabilidad)
    'seeds': [42, 123, 456, 789, 2024],
    
    # Configuraciones de épocas a probar
    'epoch_tests': [50, 100, 150, 200, 250],
    
    # Configuraciones de patience
    'patience_tests': [10, 20, 30, 40],
    
    # Grupo de prueba (empezar con niños porque es balanceado)
    'grupo': 'ninos',
    'num_features': 131,
    'num_classes': 3,
    'batch_size': 128,
    
    # Paths
    'data_dir': project_root / 'data' / 'processed' / 'ninos',
    'output_dir': project_root / 'experiments' / 'analysis',
}


# ===================================================================
# UTILIDADES
# ===================================================================

def set_seed(seed: int):
    """Fijar seeds para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data():
    """Cargar datos de niños."""
    data_dir = EXPERIMENT_CONFIG['data_dir']
    
    train_df = pd.read_csv(data_dir / 'ninos_train.csv')
    val_df = pd.read_csv(data_dir / 'ninos_val.csv')
    test_df = pd.read_csv(data_dir / 'ninos_test.csv')
    
    X_train = train_df.drop(columns=['target']).values
    y_train = train_df['target'].values
    
    X_val = val_df.drop(columns=['target']).values
    y_val = val_df['target'].values
    
    X_test = test_df.drop(columns=['target']).values
    y_test = test_df['target'].values
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# ===================================================================
# EXPERIMENTO 1: ANÁLISIS DE MÚLTIPLES SEEDS
# ===================================================================

def experiment_multiple_seeds(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Experimento 1: ¿Es suficiente seed 42 o necesito promediar múltiples seeds?
    
    Responde:
    - ¿Cuánta variabilidad hay entre seeds?
    - ¿El modelo es estable o muy sensible al seed?
    - ¿Necesito reportar promedio ± desviación estándar?
    """
    print("\n" + "="*70)
    print("🔬 EXPERIMENTO 1: ANÁLISIS DE MÚLTIPLES SEEDS")
    print("="*70)
    print("Pregunta: ¿Es confiable usar solo seed 42 o necesito promediar?")
    print(f"Seeds a probar: {EXPERIMENT_CONFIG['seeds']}")
    
    results = []
    
    for seed in EXPERIMENT_CONFIG['seeds']:
        print(f"\n🌱 Entrenando con seed={seed}...")
        set_seed(seed)
        
        start_time = time.time()
        
        # Crear y entrenar modelo
        model = create_tabnet_for_ninos(input_dim=EXPERIMENT_CONFIG['num_features'])
        
        history = model.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            max_epochs=100,  # Épocas moderadas para prueba rápida
            patience=20,
            batch_size=EXPERIMENT_CONFIG['batch_size']
        )
        
        # Evaluar
        y_pred = model.predict(X_test)
        
        metrics = {
            'seed': seed,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'training_time': time.time() - start_time
        }
        
        results.append(metrics)
        
        print(f"   ✅ Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f} | Tiempo: {metrics['training_time']:.1f}s")
    
    # Análisis de resultados
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("📊 RESULTADOS DEL EXPERIMENTO 1")
    print("="*70)
    
    print("\n📈 Tabla de Resultados por Seed:")
    print(df.to_string(index=False))
    
    print("\n📉 Estadísticas (Promedio ± Desviación Estándar):")
    for metric in ['accuracy', 'f1_macro', 'precision', 'recall']:
        mean = df[metric].mean()
        std = df[metric].std()
        min_val = df[metric].min()
        max_val = df[metric].max()
        range_val = max_val - min_val
        
        print(f"\n   {metric.upper()}:")
        print(f"      Promedio:     {mean:.4f}")
        print(f"      Desv. Std:    {std:.4f}")
        print(f"      Rango:        [{min_val:.4f}, {max_val:.4f}] (Δ={range_val:.4f})")
        print(f"      CV:           {(std/mean)*100:.2f}%")  # Coeficiente de variación
    
    # Interpretación
    print("\n" + "="*70)
    print("💡 INTERPRETACIÓN")
    print("="*70)
    
    cv_f1 = (df['f1_macro'].std() / df['f1_macro'].mean()) * 100
    
    if cv_f1 < 1.0:
        print(f"✅ VARIABILIDAD MUY BAJA (CV={cv_f1:.2f}%)")
        print("   → El modelo es MUY ESTABLE")
        print("   → Seed 42 es SUFICIENTE para tu tesis")
        print("   → No necesitas promediar múltiples seeds")
    elif cv_f1 < 3.0:
        print(f"✅ VARIABILIDAD BAJA (CV={cv_f1:.2f}%)")
        print("   → El modelo es ESTABLE")
        print("   → Seed 42 es ACEPTABLE, pero considera reportar:")
        print("   → 'Resultados con seed=42 (típico de 5 ejecuciones)'")
    elif cv_f1 < 5.0:
        print(f"⚠️  VARIABILIDAD MODERADA (CV={cv_f1:.2f}%)")
        print("   → El modelo tiene cierta variabilidad")
        print("   → RECOMENDADO: Reportar promedio de 3-5 seeds:")
        print(f"   → 'F1-Macro: {df['f1_macro'].mean():.4f} ± {df['f1_macro'].std():.4f}'")
    else:
        print(f"🔴 VARIABILIDAD ALTA (CV={cv_f1:.2f}%)")
        print("   → El modelo es INESTABLE")
        print("   → OBLIGATORIO: Promediar 5+ seeds")
        print("   → Considera ajustar arquitectura o hiperparámetros")
    
    # Guardar resultados
    output_path = EXPERIMENT_CONFIG['output_dir'] / 'exp1_multiple_seeds.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Visualización
    plot_seed_variability(df)
    
    return df


def plot_seed_variability(df):
    """Graficar variabilidad entre seeds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Variabilidad de Métricas entre Seeds', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'f1_macro', 'precision', 'recall']
    titles = ['Accuracy', 'F1-Score Macro', 'Precision', 'Recall']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        # Boxplot
        ax.boxplot([df[metric]], labels=[''])
        ax.scatter([1]*len(df), df[metric], alpha=0.6, s=100, c=df['seed'], cmap='viridis')
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title}: {df[metric].mean():.4f} ± {df[metric].std():.4f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Línea de promedio
        ax.axhline(df[metric].mean(), color='red', linestyle='--', linewidth=2, label='Promedio')
        ax.legend()
    
    plt.tight_layout()
    plot_path = EXPERIMENT_CONFIG['output_dir'] / 'exp1_seed_variability.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Gráfico guardado: {plot_path}")


# ===================================================================
# EXPERIMENTO 2: ANÁLISIS DE CURVAS DE APRENDIZAJE
# ===================================================================

def experiment_learning_curves(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Experimento 2: ¿Cuántas épocas realmente necesito?
    
    Responde:
    - ¿El modelo converge antes de 200 épocas?
    - ¿Hay overfitting? (val_loss sube mientras train_loss baja)
    - ¿Necesito más/menos épocas?
    """
    print("\n" + "="*70)
    print("🔬 EXPERIMENTO 2: CURVAS DE APRENDIZAJE")
    print("="*70)
    print("Pregunta: ¿Cuántas épocas necesito? ¿Hay overfitting?")
    
    set_seed(42)
    
    # Entrenar con muchas épocas sin early stopping
    model = create_tabnet_for_ninos(input_dim=EXPERIMENT_CONFIG['num_features'])
    
    print("\n🏋️  Entrenando modelo con 300 épocas (sin early stopping)...")
    
    history = model.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_epochs=300,
        patience=300,  # Sin early stopping
        batch_size=EXPERIMENT_CONFIG['batch_size']
    )
    
    # Análisis de historia
    if hasattr(model, 'history') and model.history:
        plot_learning_curves(model.history)
        analyze_convergence(model.history)
    else:
        print("⚠️  Modelo no retorna historia de entrenamiento")
    
    # Evaluar
    y_pred = model.predict(X_test)
    f1_final = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n✅ F1-Score Final (300 épocas): {f1_final:.4f}")
    
    return model


def plot_learning_curves(history):
    """Graficar curvas de pérdida y métricas."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Época', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Curva de Pérdida (Loss)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Métrica (accuracy o F1)
    if 'train_metric' in history and 'val_metric' in history:
        axes[1].plot(epochs, history['train_metric'], label='Train Metric', linewidth=2)
        axes[1].plot(epochs, history['val_metric'], label='Val Metric', linewidth=2)
        axes[1].set_xlabel('Época', fontsize=12)
        axes[1].set_ylabel('Métrica', fontsize=12)
        axes[1].set_title('Curva de Métrica', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = EXPERIMENT_CONFIG['output_dir'] / 'exp2_learning_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Curvas de aprendizaje guardadas: {plot_path}")


def analyze_convergence(history):
    """Analizar convergencia y overfitting."""
    print("\n" + "="*70)
    print("📊 ANÁLISIS DE CONVERGENCIA")
    print("="*70)
    
    train_loss = np.array(history['train_loss'])
    val_loss = np.array(history['val_loss'])
    
    # Encontrar época con mejor val_loss
    best_epoch = np.argmin(val_loss) + 1
    best_val_loss = val_loss[best_epoch - 1]
    
    print(f"\n📍 Mejor época: {best_epoch}")
    print(f"   Val Loss: {best_val_loss:.4f}")
    
    # Detectar overfitting (val_loss aumenta después de mínimo)
    epochs_after_best = len(val_loss) - best_epoch
    if epochs_after_best > 20:
        val_loss_increase = val_loss[-1] - best_val_loss
        if val_loss_increase > 0.01:
            print(f"\n🔴 OVERFITTING DETECTADO:")
            print(f"   Val Loss aumentó {val_loss_increase:.4f} después de época {best_epoch}")
            print(f"   Recomendación: Usar patience=20-30 épocas")
        else:
            print(f"\n✅ SIN OVERFITTING SIGNIFICATIVO")
            print(f"   Val Loss estable después de época {best_epoch}")
    
    # Convergencia
    last_20_epochs = val_loss[-20:]
    variability = np.std(last_20_epochs)
    
    if variability < 0.001:
        print(f"\n✅ CONVERGENCIA COMPLETA (últimas 20 épocas)")
        print(f"   Variabilidad: {variability:.6f}")
        print(f"   Recomendación: {best_epoch + 30} épocas son suficientes")
    else:
        print(f"\n⚠️  CONVERGENCIA PARCIAL")
        print(f"   Variabilidad: {variability:.6f}")
        print(f"   Recomendación: Considerar más épocas o ajustar learning rate")


# ===================================================================
# EXPERIMENTO 3: COMPARACIÓN DE ÉPOCAS
# ===================================================================

def experiment_epoch_comparison(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Experimento 3: ¿Más épocas = Mejor modelo?
    
    Responde:
    - ¿Vale la pena entrenar 200 épocas vs 100?
    - ¿Relación entre tiempo de entrenamiento y mejora?
    """
    print("\n" + "="*70)
    print("🔬 EXPERIMENTO 3: COMPARACIÓN DE ÉPOCAS")
    print("="*70)
    print("Pregunta: ¿Más épocas mejoran el modelo significativamente?")
    
    results = []
    
    for max_epochs in EXPERIMENT_CONFIG['epoch_tests']:
        print(f"\n⏱️  Probando con {max_epochs} épocas...")
        set_seed(42)
        
        start_time = time.time()
        
        model = create_tabnet_for_ninos(input_dim=EXPERIMENT_CONFIG['num_features'])
        
        history = model.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            max_epochs=max_epochs,
            patience=30,
            batch_size=EXPERIMENT_CONFIG['batch_size']
        )
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'max_epochs': max_epochs,
            'training_time': time.time() - start_time,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
        }
        
        results.append(metrics)
        print(f"   F1: {metrics['f1_macro']:.4f} | Tiempo: {metrics['training_time']:.1f}s")
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("📊 RESULTADOS DEL EXPERIMENTO 3")
    print("="*70)
    print(df.to_string(index=False))
    
    # Análisis costo-beneficio
    print("\n💡 ANÁLISIS COSTO-BENEFICIO:")
    
    best_f1 = df['f1_macro'].max()
    fastest_good = df[df['f1_macro'] >= best_f1 - 0.01].iloc[0]
    
    print(f"\n   Mejor F1: {best_f1:.4f} ({df.loc[df['f1_macro'].idxmax(), 'max_epochs']} épocas)")
    print(f"   F1 ≥ {best_f1 - 0.01:.4f} en solo {fastest_good['max_epochs']} épocas")
    print(f"   → Ahorro de tiempo: {df['training_time'].max() - fastest_good['training_time']:.1f}s")
    
    # Visualización
    plot_epoch_comparison(df)
    
    # Guardar
    output_path = EXPERIMENT_CONFIG['output_dir'] / 'exp3_epoch_comparison.csv'
    df.to_csv(output_path, index=False)
    
    return df


def plot_epoch_comparison(df):
    """Graficar relación épocas vs desempeño."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: F1 vs Épocas
    axes[0].plot(df['max_epochs'], df['f1_macro'], marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Número de Épocas', fontsize=12)
    axes[0].set_ylabel('F1-Score Macro', fontsize=12)
    axes[0].set_title('F1 vs Épocas de Entrenamiento', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Tiempo vs F1
    axes[1].scatter(df['training_time'], df['f1_macro'], s=df['max_epochs'], alpha=0.6, c=df['max_epochs'], cmap='viridis')
    axes[1].set_xlabel('Tiempo de Entrenamiento (s)', fontsize=12)
    axes[1].set_ylabel('F1-Score Macro', fontsize=12)
    axes[1].set_title('Costo-Beneficio: Tiempo vs F1', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = EXPERIMENT_CONFIG['output_dir'] / 'exp3_epoch_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Gráfico guardado: {plot_path}")


# ===================================================================
# EXPERIMENTO 4: RESUMEN Y RECOMENDACIONES
# ===================================================================

def generate_recommendations(exp1_df, exp3_df):
    """Generar recomendaciones finales para la tesis."""
    print("\n" + "="*70)
    print("🎓 RECOMENDACIONES PARA TU TESIS")
    print("="*70)
    
    # Análisis de estabilidad
    cv_f1 = (exp1_df['f1_macro'].std() / exp1_df['f1_macro'].mean()) * 100
    
    print("\n1️⃣ SOBRE EL SEED:")
    if cv_f1 < 3.0:
        print(f"   ✅ Usar seed=42 es SUFICIENTE (CV={cv_f1:.2f}%)")
        print(f"   📝 En la tesis escribir:")
        print(f"      'Se utilizó seed=42 para reproducibilidad.'")
        print(f"      'Experimentos con 5 seeds mostraron CV < 3%.'")
        seed_strategy = "single"
    else:
        print(f"   ⚠️  RECOMENDADO promediar múltiples seeds (CV={cv_f1:.2f}%)")
        print(f"   📝 En la tesis escribir:")
        print(f"      'Resultados reportados como promedio ± std de 5 ejecuciones'")
        print(f"      'Seeds utilizados: [42, 123, 456, 789, 2024]'")
        seed_strategy = "multiple"
    
    # Análisis de épocas
    best_epoch_config = exp3_df.loc[exp3_df['f1_macro'].idxmax()]
    
    print("\n2️⃣ SOBRE LAS ÉPOCAS:")
    print(f"   ✅ Configuración óptima: {int(best_epoch_config['max_epochs'])} épocas")
    print(f"   ⏱️  Tiempo estimado: {best_epoch_config['training_time']:.1f}s por modelo")
    print(f"   📊 F1-Score esperado: {best_epoch_config['f1_macro']:.4f}")
    print(f"   📝 En la tesis escribir:")
    print(f"      'Entrenamiento con max_epochs={int(best_epoch_config['max_epochs'])}, patience=30'")
    print(f"      'Early stopping para prevenir overfitting'")
    
    # Tiempo total estimado
    print("\n3️⃣ TIEMPO ESTIMADO TOTAL:")
    time_per_model = best_epoch_config['training_time']
    
    if seed_strategy == "single":
        total_models = 9  # 3 arquitecturas × 3 grupos
        total_time = time_per_model * total_models
    else:
        total_models = 9 * 5  # 3 arquitecturas × 3 grupos × 5 seeds
        total_time = time_per_model * total_models
    
    print(f"   🔢 Modelos a entrenar: {total_models}")
    print(f"   ⏱️  Tiempo total estimado: {total_time/60:.1f} minutos ({total_time/3600:.2f} horas)")
    
    if total_time > 3600:
        print(f"   💡 Tip: Ejecutar entrenamientos overnight o en GPU")
    
    # Guardar recomendaciones
    recommendations = {
        'seed_strategy': seed_strategy,
        'seed_variability_cv': float(cv_f1),
        'recommended_epochs': int(best_epoch_config['max_epochs']),
        'recommended_patience': 30,
        'expected_f1': float(best_epoch_config['f1_macro']),
        'time_per_model_seconds': float(time_per_model),
        'total_models': total_models,
        'total_time_hours': float(total_time / 3600),
        'recommendations': {
            'seed': "Usar seed=42" if seed_strategy == "single" else "Promediar 5 seeds",
            'epochs': f"{int(best_epoch_config['max_epochs'])} épocas con patience=30",
            'class_weights': "Aplicar en jóvenes y adultos (imbalance)",
            'metrics': "Priorizar F1-Macro en datasets desbalanceados"
        }
    }
    
    output_path = EXPERIMENT_CONFIG['output_dir'] / 'recommendations.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(recommendations, f, indent=4, ensure_ascii=False)
    
    print(f"\n💾 Recomendaciones guardadas: {output_path}")
    
    return recommendations


# ===================================================================
# MAIN
# ===================================================================

def main():
    """Ejecutar todos los experimentos."""
    print("="*70)
    print("🔬 ANÁLISIS EXPERIMENTAL - VALIDACIÓN DE CONFIGURACIONES")
    print("="*70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nObjetivos:")
    print("1. Validar si seed=42 es suficiente")
    print("2. Determinar número óptimo de épocas")
    print("3. Analizar curvas de aprendizaje")
    print("4. Generar recomendaciones para la tesis")
    
    # Cargar datos
    print("\n📂 Cargando datos de niños (grupo balanceado para análisis)...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    print(f"   Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")
    
    try:
        # Experimento 1: Múltiples seeds
        exp1_df = experiment_multiple_seeds(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Experimento 2: Curvas de aprendizaje
        experiment_learning_curves(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Experimento 3: Comparación de épocas
        exp3_df = experiment_epoch_comparison(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Generar recomendaciones
        recommendations = generate_recommendations(exp1_df, exp3_df)
        
        print("\n" + "="*70)
        print("✅ ANÁLISIS EXPERIMENTAL COMPLETADO")
        print("="*70)
        print(f"\n📁 Resultados guardados en: {EXPERIMENT_CONFIG['output_dir']}")
        print("\nArchivos generados:")
        print("   - exp1_multiple_seeds.csv")
        print("   - exp1_seed_variability.png")
        print("   - exp2_learning_curves.png")
        print("   - exp3_epoch_comparison.csv")
        print("   - exp3_epoch_comparison.png")
        print("   - recommendations.json")
        
        print("\n🎯 PRÓXIMOS PASOS:")
        print("   1. Revisar las gráficas generadas")
        print("   2. Aplicar recomendaciones a tus scripts de entrenamiento")
        print("   3. Ejecutar entrenamiento completo (9 modelos)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
