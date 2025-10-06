#!/usr/bin/env python3
"""
Script para generar gráficos de curvas de aprendizaje del modelo.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from datetime import datetime

def crear_grafico_aprendizaje(model_dir=None):
    """
    Crea gráficos de curvas de aprendizaje mostrando loss, accuracy, y otras métricas.
    """
    
    # Si no se especifica directorio, usar el más reciente
    if model_dir is None:
        model_dirs = glob.glob("outputs/models/202*")
        if not model_dirs:
            print("❌ No se encontró ningún modelo entrenado")
            return
        model_dir = max(model_dirs)
    
    print(f"📈 GENERANDO GRÁFICOS DE CURVAS DE APRENDIZAJE")
    print(f"📁 Directorio: {model_dir}")
    print("=" * 60)
    
    # Cargar datos de entrenamiento
    metrics_file = os.path.join(model_dir, "metrics_by_epoch.csv")
    if not os.path.exists(metrics_file):
        print("❌ No se encontró archivo de métricas por época")
        return
    
    df = pd.read_csv(metrics_file)
    print(f"✅ Datos cargados: {len(df)} épocas")
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('📈 Curvas de Aprendizaje del Modelo MLP', fontsize=16, fontweight='bold')
    
    # Configurar colores
    train_color = '#1f77b4'  # Azul
    val_color = '#ff7f0e'    # Naranja
    
    # ================================
    # 1. CURVA DE PÉRDIDA (LOSS)
    # ================================
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['train_loss'], color=train_color, linewidth=2, 
             marker='o', markersize=4, label='Training Loss')
    
    ax1.set_title('📉 Curva de Pérdida (Loss)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Marcar convergencia
    convergence_epoch = df.loc[df['train_loss'] < 0.05].index[0] + 1 if len(df.loc[df['train_loss'] < 0.05]) > 0 else len(df)
    ax1.axvline(x=convergence_epoch, color='red', linestyle='--', alpha=0.7, label=f'Convergencia (Época {convergence_epoch})')
    ax1.legend()
    
    # ================================
    # 2. CURVA DE ACCURACY
    # ================================
    ax2 = axes[0, 1]
    ax2.plot(df['epoch'], df['val_accuracy'], color=val_color, linewidth=2, 
             marker='s', markersize=4, label='Validation Accuracy')
    
    ax2.set_title('📊 Curva de Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Línea objetivo (90%)
    ax2.axhline(y=0.90, color='green', linestyle=':', alpha=0.7, label='Objetivo 90%')
    ax2.legend()
    
    # ================================
    # 3. MÉTRICAS COMBINADAS
    # ================================
    ax3 = axes[1, 0]
    ax3.plot(df['epoch'], df['val_accuracy'], 'g-', linewidth=2, label='Accuracy', marker='o', markersize=3)
    ax3.plot(df['epoch'], df['val_precision'], 'b-', linewidth=2, label='Precision', marker='s', markersize=3)
    ax3.plot(df['epoch'], df['val_recall'], 'r-', linewidth=2, label='Recall', marker='^', markersize=3)
    ax3.plot(df['epoch'], df['val_f1'], 'purple', linewidth=2, label='F1-Score', marker='d', markersize=3)
    
    ax3.set_title('🎯 Evolución de Métricas de Validación', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Época')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.5, 1.05)
    
    # ================================
    # 4. ANÁLISIS DE CONVERGENCIA
    # ================================
    ax4 = axes[1, 1]
    
    # Calcular mejora por época en F1-Score
    f1_scores = df['val_f1'].values
    improvements = np.diff(f1_scores, prepend=f1_scores[0])
    
    # Gráfico de barras de mejora
    colors = ['green' if imp > 0 else 'red' if imp < 0 else 'gray' for imp in improvements]
    bars = ax4.bar(df['epoch'], improvements, color=colors, alpha=0.7, width=0.8)
    
    ax4.set_title('⚡ Mejora por Época (ΔF1-Score)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Época')
    ax4.set_ylabel('Cambio en F1-Score')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Añadir línea de tendencia
    z = np.polyfit(df['epoch'], improvements, 1)
    p = np.poly1d(z)
    ax4.plot(df['epoch'], p(df['epoch']), "k--", alpha=0.8, linewidth=1, label='Tendencia')
    ax4.legend()
    
    # Ajustar espaciado
    plt.tight_layout()
    
    # Crear directorio de salida
    os.makedirs("outputs/plots", exist_ok=True)
    
    # Guardar gráfico
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outputs/plots/curvas_aprendizaje_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"✅ Gráfico guardado en: {output_file}")
    
    # Mostrar estadísticas del aprendizaje
    print(f"\n📊 ANÁLISIS DE APRENDIZAJE:")
    print(f"  Épocas totales: {len(df)}")
    print(f"  Loss inicial: {df['train_loss'].iloc[0]:.4f}")
    print(f"  Loss final: {df['train_loss'].iloc[-1]:.4f}")
    print(f"  Reducción de loss: {((df['train_loss'].iloc[0] - df['train_loss'].iloc[-1]) / df['train_loss'].iloc[0] * 100):.1f}%")
    print(f"  Accuracy inicial: {df['val_accuracy'].iloc[0]:.4f}")
    print(f"  Accuracy final: {df['val_accuracy'].iloc[-1]:.4f}")
    print(f"  Mejor F1-Score: {df['val_f1'].max():.4f} (Época {df['val_f1'].idxmax() + 1})")
    
    # Detectar overfitting
    best_epoch = df['val_f1'].idxmax() + 1
    final_epoch = len(df)
    if final_epoch - best_epoch > 5:
        print(f"  ⚠️  Posible overfitting: Mejor modelo en época {best_epoch}, entrenamiento continuó hasta época {final_epoch}")
    else:
        print(f"  ✅ Sin overfitting detectado")
    
    # Mostrar gráfico
    plt.show()
    
    return output_file

def crear_grafico_loss_detallado():
    """
    Crea un gráfico detallado solo del loss para análisis profundo.
    """
    
    print(f"\n📉 GRÁFICO DETALLADO DE PÉRDIDA")
    print("=" * 40)
    
    # Encontrar modelo más reciente
    model_dirs = glob.glob("outputs/models/202*")
    if not model_dirs:
        print("❌ No se encontró ningún modelo")
        return
    
    model_dir = max(model_dirs)
    metrics_file = os.path.join(model_dir, "metrics_by_epoch.csv")
    df = pd.read_csv(metrics_file)
    
    # Crear gráfico de loss detallado
    plt.figure(figsize=(12, 6))
    
    plt.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, marker='o', 
             markersize=4, alpha=0.8, label='Training Loss')
    
    plt.title('📉 Análisis Detallado de la Curva de Pérdida', fontsize=14, fontweight='bold')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    # Añadir anotaciones importantes
    min_loss_epoch = df['train_loss'].idxmin() + 1
    min_loss_value = df['train_loss'].min()
    
    plt.annotate(f'Mínimo Loss\nÉpoca {min_loss_epoch}: {min_loss_value:.4f}',
                xy=(min_loss_epoch, min_loss_value),
                xytext=(min_loss_epoch + 3, min_loss_value + 0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, ha='left')
    
    # Líneas de referencia
    plt.axhline(y=0.1, color='orange', linestyle=':', alpha=0.7, label='Convergencia (0.1)')
    plt.axhline(y=0.05, color='green', linestyle=':', alpha=0.7, label='Buena convergencia (0.05)')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Guardar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outputs/plots/loss_detallado_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"✅ Gráfico de loss detallado guardado en: {output_file}")
    plt.show()
    
    return output_file

def main():
    """Función principal para generar todos los gráficos de aprendizaje."""
    
    print("📈 GENERADOR DE GRÁFICOS DE CURVAS DE APRENDIZAJE")
    print("=" * 60)
    
    # Gráfico principal de aprendizaje
    archivo1 = crear_grafico_aprendizaje()
    
    # Gráfico detallado de loss
    archivo2 = crear_grafico_loss_detallado()
    
    print(f"\n🎯 ¡Gráficos de aprendizaje generados exitosamente!")
    print(f"📁 Archivos:")
    print(f"  - {archivo1}")
    print(f"  - {archivo2}")

if __name__ == "__main__":
    main()
