"""
Visualización Comparativa: Datos Sintéticos vs Datos Reales
Matriz de Calor 3×3 con Métricas de Rendimiento

Este script genera visualizaciones avanzadas para comparar:
- Clasificaciones sintéticas (generadas)
- Clasificaciones reales (cuando estén disponibles)

Genera:
1. Heatmap de Accuracies por Grupo×Modelo
2. Heatmap de Distribución de Clases
3. Heatmap de Confianza Promedio
4. Matriz de Confusión Agregada
5. Comparación de Precisión por Clase

Universidad Peruana Unión - 2025
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurar UTF-8 para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from datetime import datetime

# ===================================================================
# CONFIGURACIÓN
# ===================================================================

classfier_root = Path(__file__).parent

# Paths
SYNTHETIC_DIR = classfier_root / 'outputs' / 'classifications_3x3'
REAL_DIR = classfier_root / 'outputs' / 'classifications_3x3_real'  # Cuando estén los modelos reales
OUTPUT_DIR = classfier_root / 'outputs' / 'visualizations_comparison'

# Crear directorio de salida
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuración
GRUPOS = ['ninos', 'jovenes', 'adultos']
MODELOS = ['ft_transformer', 'tabnet', 'mlp_attention']
CLASS_NAMES = ['No Lector', 'Lector Medio', 'Lector Frecuente']

# Estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ===================================================================
# FUNCIONES DE CARGA
# ===================================================================

def load_metrics(base_dir, grupo, modelo):
    """Cargar métricas de una clasificación."""
    path = base_dir / f'{grupo}_{modelo}'
    
    if not path.exists():
        return None
    
    try:
        # Cargar predictions
        pred_df = pd.read_csv(path / 'predictions.csv')
        
        # Calcular métricas
        y_true = pred_df['true_label'].values
        y_pred = pred_df['predicted_label'].values
        
        accuracy = (y_true == y_pred).mean()
        
        # Confianza promedio
        max_prob = pred_df['max_probability'].mean()
        
        # Distribución de clases predichas
        class_dist = [
            (y_pred == 0).mean(),
            (y_pred == 1).mean(),
            (y_pred == 2).mean()
        ]
        
        # Precisión por clase
        precision_per_class = []
        for class_idx in range(3):
            mask_pred = y_pred == class_idx
            if mask_pred.sum() > 0:
                precision = ((y_pred == class_idx) & (y_true == class_idx)).sum() / mask_pred.sum()
            else:
                precision = 0
            precision_per_class.append(precision)
        
        # Matriz de confusión
        conf_matrix = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                conf_matrix[i, j] = ((y_true == i) & (y_pred == j)).sum()
        
        return {
            'accuracy': accuracy,
            'confidence': max_prob,
            'class_distribution': class_dist,
            'precision_per_class': precision_per_class,
            'confusion_matrix': conf_matrix,
            'n_samples': len(y_true)
        }
    except Exception as e:
        print(f"⚠️  Error cargando {grupo}_{modelo}: {e}")
        return None


def load_all_metrics(base_dir):
    """Cargar todas las métricas de una fuente."""
    metrics = {}
    
    for grupo in GRUPOS:
        metrics[grupo] = {}
        for modelo in MODELOS:
            m = load_metrics(base_dir, grupo, modelo)
            if m:
                metrics[grupo][modelo] = m
    
    return metrics


# ===================================================================
# VISUALIZACIONES
# ===================================================================

def create_heatmap_accuracy(synthetic_metrics, real_metrics=None):
    """Crear heatmap de accuracy comparando sintético vs real."""
    
    fig, axes = plt.subplots(1, 2 if real_metrics else 1, figsize=(14, 5))
    if not real_metrics:
        axes = [axes]
    
    # Preparar datos sintéticos
    synthetic_data = np.zeros((len(GRUPOS), len(MODELOS)))
    for i, grupo in enumerate(GRUPOS):
        for j, modelo in enumerate(MODELOS):
            if modelo in synthetic_metrics.get(grupo, {}):
                synthetic_data[i, j] = synthetic_metrics[grupo][modelo]['accuracy']
    
    # Heatmap sintético
    sns.heatmap(
        synthetic_data,
        annot=True,
        fmt='.3f',
        cmap='YlGnBu',
        xticklabels=[m.replace('_', '\n') for m in MODELOS],
        yticklabels=[g.capitalize() for g in GRUPOS],
        cbar_kws={'label': 'Accuracy'},
        vmin=0.7,
        vmax=0.9,
        ax=axes[0]
    )
    axes[0].set_title('📊 Datos Sintéticos', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Modelo', fontsize=12)
    axes[0].set_ylabel('Grupo de Edad', fontsize=12)
    
    # Heatmap real (si existe)
    if real_metrics:
        real_data = np.zeros((len(GRUPOS), len(MODELOS)))
        for i, grupo in enumerate(GRUPOS):
            for j, modelo in enumerate(MODELOS):
                if modelo in real_metrics.get(grupo, {}):
                    real_data[i, j] = real_metrics[grupo][modelo]['accuracy']
                else:
                    real_data[i, j] = np.nan
        
        sns.heatmap(
            real_data,
            annot=True,
            fmt='.3f',
            cmap='YlGnBu',
            xticklabels=[m.replace('_', '\n') for m in MODELOS],
            yticklabels=[g.capitalize() for g in GRUPOS],
            cbar_kws={'label': 'Accuracy'},
            vmin=0.7,
            vmax=0.9,
            ax=axes[1],
            mask=np.isnan(real_data)
        )
        axes[1].set_title('🎯 Datos Reales', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Modelo', fontsize=12)
        axes[1].set_ylabel('Grupo de Edad', fontsize=12)
    
    plt.suptitle('Comparación de Accuracy: Sintético vs Real', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'heatmap_accuracy_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()


def create_heatmap_confidence(synthetic_metrics, real_metrics=None):
    """Crear heatmap de confianza promedio."""
    
    fig, axes = plt.subplots(1, 2 if real_metrics else 1, figsize=(14, 5))
    if not real_metrics:
        axes = [axes]
    
    # Preparar datos sintéticos
    synthetic_data = np.zeros((len(GRUPOS), len(MODELOS)))
    for i, grupo in enumerate(GRUPOS):
        for j, modelo in enumerate(MODELOS):
            if modelo in synthetic_metrics.get(grupo, {}):
                synthetic_data[i, j] = synthetic_metrics[grupo][modelo]['confidence']
    
    # Heatmap sintético
    sns.heatmap(
        synthetic_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        xticklabels=[m.replace('_', '\n') for m in MODELOS],
        yticklabels=[g.capitalize() for g in GRUPOS],
        cbar_kws={'label': 'Confianza Promedio'},
        vmin=0.6,
        vmax=0.9,
        ax=axes[0]
    )
    axes[0].set_title('📊 Datos Sintéticos', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Modelo', fontsize=12)
    axes[0].set_ylabel('Grupo de Edad', fontsize=12)
    
    # Heatmap real (si existe)
    if real_metrics:
        real_data = np.zeros((len(GRUPOS), len(MODELOS)))
        for i, grupo in enumerate(GRUPOS):
            for j, modelo in enumerate(MODELOS):
                if modelo in real_metrics.get(grupo, {}):
                    real_data[i, j] = real_metrics[grupo][modelo]['confidence']
                else:
                    real_data[i, j] = np.nan
        
        sns.heatmap(
            real_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            xticklabels=[m.replace('_', '\n') for m in MODELOS],
            yticklabels=[g.capitalize() for g in GRUPOS],
            cbar_kws={'label': 'Confianza Promedio'},
            vmin=0.6,
            vmax=0.9,
            ax=axes[1],
            mask=np.isnan(real_data)
        )
        axes[1].set_title('🎯 Datos Reales', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Modelo', fontsize=12)
        axes[1].set_ylabel('Grupo de Edad', fontsize=12)
    
    plt.suptitle('Comparación de Confianza Promedio: Sintético vs Real', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'heatmap_confidence_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()


def create_class_distribution_heatmap(synthetic_metrics, real_metrics=None):
    """Crear heatmap de distribución de clases predichas."""
    
    n_classes = len(CLASS_NAMES)
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(n_classes, 2 if real_metrics else 1, figure=fig, hspace=0.3)
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        # Sintético
        synthetic_data = np.zeros((len(GRUPOS), len(MODELOS)))
        for i, grupo in enumerate(GRUPOS):
            for j, modelo in enumerate(MODELOS):
                if modelo in synthetic_metrics.get(grupo, {}):
                    synthetic_data[i, j] = synthetic_metrics[grupo][modelo]['class_distribution'][class_idx]
        
        ax_syn = fig.add_subplot(gs[class_idx, 0])
        sns.heatmap(
            synthetic_data,
            annot=True,
            fmt='.2%',
            cmap='Purples',
            xticklabels=[m.replace('_', '\n') for m in MODELOS],
            yticklabels=[g.capitalize() for g in GRUPOS],
            cbar_kws={'label': 'Proporción'},
            vmin=0,
            vmax=0.5,
            ax=ax_syn
        )
        ax_syn.set_title(f'📊 Sintético: {class_name}', fontsize=12, fontweight='bold')
        ax_syn.set_xlabel('Modelo' if class_idx == n_classes - 1 else '', fontsize=10)
        ax_syn.set_ylabel('Grupo de Edad', fontsize=10)
        
        # Real (si existe)
        if real_metrics:
            real_data = np.zeros((len(GRUPOS), len(MODELOS)))
            for i, grupo in enumerate(GRUPOS):
                for j, modelo in enumerate(MODELOS):
                    if modelo in real_metrics.get(grupo, {}):
                        real_data[i, j] = real_metrics[grupo][modelo]['class_distribution'][class_idx]
                    else:
                        real_data[i, j] = np.nan
            
            ax_real = fig.add_subplot(gs[class_idx, 1])
            sns.heatmap(
                real_data,
                annot=True,
                fmt='.2%',
                cmap='Purples',
                xticklabels=[m.replace('_', '\n') for m in MODELOS],
                yticklabels=[g.capitalize() for g in GRUPOS],
                cbar_kws={'label': 'Proporción'},
                vmin=0,
                vmax=0.5,
                ax=ax_real,
                mask=np.isnan(real_data)
            )
            ax_real.set_title(f'🎯 Real: {class_name}', fontsize=12, fontweight='bold')
            ax_real.set_xlabel('Modelo' if class_idx == n_classes - 1 else '', fontsize=10)
            ax_real.set_ylabel('Grupo de Edad', fontsize=10)
    
    plt.suptitle('Distribución de Clases Predichas por Grupo y Modelo', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    output_path = OUTPUT_DIR / 'heatmap_class_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()


def create_confusion_matrices_comparison(synthetic_metrics, real_metrics=None):
    """Crear matrices de confusión agregadas por modelo."""
    
    fig, axes = plt.subplots(len(MODELOS), 2 if real_metrics else 1, figsize=(12, 14))
    if not real_metrics:
        axes = axes.reshape(-1, 1)
    
    for idx, modelo in enumerate(MODELOS):
        # Agregar todas las matrices de confusión del modelo (sintético)
        synthetic_cm = np.zeros((3, 3))
        for grupo in GRUPOS:
            if modelo in synthetic_metrics.get(grupo, {}):
                synthetic_cm += synthetic_metrics[grupo][modelo]['confusion_matrix']
        
        # Normalizar por filas (true labels)
        synthetic_cm_norm = synthetic_cm / synthetic_cm.sum(axis=1, keepdims=True)
        
        sns.heatmap(
            synthetic_cm_norm,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Proporción'},
            vmin=0,
            vmax=1,
            ax=axes[idx, 0]
        )
        axes[idx, 0].set_title(f'📊 Sintético: {modelo.replace("_", " ").title()}', 
                               fontsize=12, fontweight='bold')
        axes[idx, 0].set_xlabel('Predicho', fontsize=10)
        axes[idx, 0].set_ylabel('Real', fontsize=10)
        
        # Real (si existe)
        if real_metrics:
            real_cm = np.zeros((3, 3))
            for grupo in GRUPOS:
                if modelo in real_metrics.get(grupo, {}):
                    real_cm += real_metrics[grupo][modelo]['confusion_matrix']
            
            real_cm_norm = real_cm / real_cm.sum(axis=1, keepdims=True)
            
            sns.heatmap(
                real_cm_norm,
                annot=True,
                fmt='.2%',
                cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Proporción'},
                vmin=0,
                vmax=1,
                ax=axes[idx, 1]
            )
            axes[idx, 1].set_title(f'🎯 Real: {modelo.replace("_", " ").title()}', 
                                   fontsize=12, fontweight='bold')
            axes[idx, 1].set_xlabel('Predicho', fontsize=10)
            axes[idx, 1].set_ylabel('Real', fontsize=10)
    
    plt.suptitle('Matrices de Confusión Agregadas por Modelo', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'confusion_matrices_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()


def create_precision_comparison(synthetic_metrics, real_metrics=None):
    """Crear gráfico de comparación de precisión por clase."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        # Preparar datos
        x_labels = []
        synthetic_values = []
        real_values = []
        
        for grupo in GRUPOS:
            for modelo in MODELOS:
                label = f"{grupo[:3].capitalize()}\n{modelo.split('_')[0].upper()}"
                x_labels.append(label)
                
                # Sintético
                if modelo in synthetic_metrics.get(grupo, {}):
                    synthetic_values.append(
                        synthetic_metrics[grupo][modelo]['precision_per_class'][class_idx]
                    )
                else:
                    synthetic_values.append(0)
                
                # Real
                if real_metrics and modelo in real_metrics.get(grupo, {}):
                    real_values.append(
                        real_metrics[grupo][modelo]['precision_per_class'][class_idx]
                    )
                else:
                    real_values.append(None)
        
        x_pos = np.arange(len(x_labels))
        width = 0.35
        
        # Barras sintético
        axes[class_idx].bar(x_pos - width/2, synthetic_values, width, 
                           label='Sintético', alpha=0.8, color='skyblue')
        
        # Barras real (si existe)
        if real_metrics and any(v is not None for v in real_values):
            real_values_clean = [v if v is not None else 0 for v in real_values]
            axes[class_idx].bar(x_pos + width/2, real_values_clean, width,
                              label='Real', alpha=0.8, color='coral')
        
        axes[class_idx].set_xlabel('Grupo - Modelo', fontsize=10)
        axes[class_idx].set_ylabel('Precisión', fontsize=10)
        axes[class_idx].set_title(f'{class_name}', fontsize=12, fontweight='bold')
        axes[class_idx].set_xticks(x_pos)
        axes[class_idx].set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        axes[class_idx].set_ylim(0, 1)
        axes[class_idx].grid(axis='y', alpha=0.3)
        axes[class_idx].legend()
        axes[class_idx].axhline(y=0.8, color='red', linestyle='--', alpha=0.3, label='Umbral 80%')
    
    plt.suptitle('Comparación de Precisión por Clase: Sintético vs Real', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'precision_comparison_by_class.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Guardado: {output_path}")
    plt.close()


# ===================================================================
# MAIN
# ===================================================================

def main():
    """Generar todas las visualizaciones comparativas."""
    
    print("=" * 80)
    print("  🔥 VISUALIZACIONES COMPARATIVAS: SINTÉTICO VS REAL")
    print("=" * 80)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💾 Salida: {OUTPUT_DIR}")
    print()
    
    # Cargar métricas sintéticas
    print("📊 Cargando métricas sintéticas...")
    synthetic_metrics = load_all_metrics(SYNTHETIC_DIR)
    n_synthetic = sum(len(models) for models in synthetic_metrics.values())
    print(f"   ✅ {n_synthetic}/9 clasificaciones sintéticas cargadas")
    
    # Cargar métricas reales (si existen)
    real_metrics = None
    if REAL_DIR.exists():
        print("\n🎯 Cargando métricas reales...")
        real_metrics = load_all_metrics(REAL_DIR)
        n_real = sum(len(models) for models in real_metrics.values())
        print(f"   ✅ {n_real}/9 clasificaciones reales cargadas")
    else:
        print("\n⚠️  No hay clasificaciones reales aún")
        print(f"   Para comparar con datos reales, ejecuta los modelos y guarda en:")
        print(f"   {REAL_DIR}")
    
    # Generar visualizaciones
    print("\n🎨 Generando visualizaciones...")
    
    print("   1. Heatmap de Accuracy...")
    create_heatmap_accuracy(synthetic_metrics, real_metrics)
    
    print("   2. Heatmap de Confianza...")
    create_heatmap_confidence(synthetic_metrics, real_metrics)
    
    print("   3. Distribución de Clases...")
    create_class_distribution_heatmap(synthetic_metrics, real_metrics)
    
    print("   4. Matrices de Confusión...")
    create_confusion_matrices_comparison(synthetic_metrics, real_metrics)
    
    print("   5. Comparación de Precisión...")
    create_precision_comparison(synthetic_metrics, real_metrics)
    
    # Resumen
    print("\n" + "=" * 80)
    print("  ✅ VISUALIZACIONES COMPLETADAS")
    print("=" * 80)
    print(f"\n📁 Archivos generados en: {OUTPUT_DIR}")
    print("\n📊 Gráficos creados:")
    for file in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"   - {file.name}")
    
    print(f"\n💡 Abre las imágenes para comparar sintético vs real")
    if not real_metrics:
        print(f"   (Por ahora solo hay datos sintéticos)")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
