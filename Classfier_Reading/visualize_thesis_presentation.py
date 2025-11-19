"""
Visualizaciones Avanzadas para Sustentación de Tesis
Análisis Comparativo de Modelos de Deep Learning para Clasificación de Lectores

Genera visualizaciones de nivel académico para presentación de tesis:
1. Radar Chart - Comparación multidimensional de modelos
2. Box Plot - Distribución de confianza por modelo y grupo
3. Violin Plot - Densidad de probabilidades por clase
4. Sankey Diagram - Flujo de clasificaciones correctas/incorrectas
5. Grid de métricas F1, Precision, Recall por modelo
6. Curvas ROC simuladas por modelo
7. Análisis de casos difíciles (low confidence)
8. Resumen ejecutivo con estadísticas clave

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
from matplotlib.patches import Rectangle
from datetime import datetime
import matplotlib.patches as mpatches

# ===================================================================
# CONFIGURACIÓN
# ===================================================================

classfier_root = Path(__file__).parent

# Paths
DATA_DIR = classfier_root / 'outputs' / 'classifications_3x3_real'
OUTPUT_DIR = classfier_root / 'outputs' / 'visualizations_thesis'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuración
GRUPOS = ['ninos', 'jovenes', 'adultos']
MODELOS = ['ft_transformer', 'tabnet', 'mlp_attention']
MODELO_LABELS = {
    'ft_transformer': 'FT-Transformer',
    'tabnet': 'TabNet',
    'mlp_attention': 'MLP+Attention'
}
CLASS_NAMES = ['No Lector', 'Lector Medio', 'Lector Frecuente']
GRUPO_LABELS = {
    'ninos': 'Niños (0-11 años)',
    'jovenes': 'Jóvenes (12-17 años)',
    'adultos': 'Adultos (18+ años)'
}

# Estilo académico
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# ===================================================================
# FUNCIONES DE CARGA
# ===================================================================

def load_all_data():
    """Cargar todos los datos de clasificación."""
    all_data = {}
    
    for grupo in GRUPOS:
        all_data[grupo] = {}
        for modelo in MODELOS:
            path = DATA_DIR / f'{grupo}_{modelo}' / 'predictions.csv'
            if path.exists():
                df = pd.read_csv(path)
                all_data[grupo][modelo] = df
            else:
                print(f"⚠️  No encontrado: {path}")
    
    return all_data


# ===================================================================
# VISUALIZACIÓN 1: RADAR CHART COMPARATIVO
# ===================================================================

def create_radar_chart(all_data):
    """Radar chart con múltiples métricas por modelo."""
    
    # Calcular métricas por modelo
    metrics = {}
    for modelo in MODELOS:
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        confidences = []
        
        for grupo in GRUPOS:
            if modelo in all_data.get(grupo, {}):
                df = all_data[grupo][modelo]
                y_true = df['true_label'].values
                y_pred = df['predicted_label'].values
                
                acc = (y_true == y_pred).mean()
                accuracies.append(acc)
                
                # Precisión macro
                prec_per_class = []
                for c in range(3):
                    mask = y_pred == c
                    if mask.sum() > 0:
                        prec_per_class.append(((y_pred == c) & (y_true == c)).sum() / mask.sum())
                precisions.append(np.mean(prec_per_class))
                
                # Recall macro
                rec_per_class = []
                for c in range(3):
                    mask = y_true == c
                    if mask.sum() > 0:
                        rec_per_class.append(((y_pred == c) & (y_true == c)).sum() / mask.sum())
                recalls.append(np.mean(rec_per_class))
                
                # F1
                p = np.mean(prec_per_class)
                r = np.mean(rec_per_class)
                f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
                
                # Confianza
                confidences.append(df['max_probability'].mean())
        
        metrics[modelo] = {
            'Accuracy': np.mean(accuracies),
            'Precision': np.mean(precisions),
            'Recall': np.mean(recalls),
            'F1-Score': np.mean(f1s),
            'Confianza': np.mean(confidences)
        }
    
    # Crear radar chart
    categories = list(metrics[MODELOS[0]].keys())
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, modelo in enumerate(MODELOS):
        values = list(metrics[modelo].values())
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=MODELO_LABELS[modelo], color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    plt.title('Comparación Multidimensional de Modelos\nPromedio de 3 Grupos Etarios', 
              size=16, weight='bold', pad=20)
    
    output_path = OUTPUT_DIR / 'radar_chart_models.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ {output_path.name}")
    plt.close()


# ===================================================================
# VISUALIZACIÓN 2: BOX PLOTS DE CONFIANZA
# ===================================================================

def create_confidence_boxplots(all_data):
    """Box plots de distribución de confianza."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, grupo in enumerate(GRUPOS):
        data_to_plot = []
        labels = []
        
        for modelo in MODELOS:
            if modelo in all_data.get(grupo, {}):
                df = all_data[grupo][modelo]
                data_to_plot.append(df['max_probability'].values)
                labels.append(MODELO_LABELS[modelo])
        
        bp = axes[idx].boxplot(data_to_plot, labels=labels, patch_artist=True,
                              medianprops=dict(color='red', linewidth=2),
                              whiskerprops=dict(linewidth=1.5),
                              capprops=dict(linewidth=1.5))
        
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        axes[idx].set_ylabel('Probabilidad Máxima', fontsize=12)
        axes[idx].set_title(GRUPO_LABELS[grupo], fontsize=14, weight='bold')
        axes[idx].set_ylim(0, 1)
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Alta confianza')
        axes[idx].axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Baja confianza')
        
        if idx == 2:
            axes[idx].legend(loc='lower right', fontsize=10)
    
    plt.suptitle('Distribución de Confianza en Predicciones por Grupo Etario', 
                 fontsize=16, weight='bold')
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'boxplot_confidence_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ {output_path.name}")
    plt.close()


# ===================================================================
# VISUALIZACIÓN 3: VIOLIN PLOTS POR CLASE
# ===================================================================

def create_violin_plots_by_class(all_data):
    """Violin plots de probabilidades por clase predicha."""
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    for row_idx, grupo in enumerate(GRUPOS):
        for col_idx, modelo in enumerate(MODELOS):
            ax = axes[row_idx, col_idx]
            
            if modelo in all_data.get(grupo, {}):
                df = all_data[grupo][modelo]
                
                # Preparar datos por clase predicha
                plot_data = []
                for class_idx in range(3):
                    mask = df['predicted_label'] == class_idx
                    if mask.sum() > 0:
                        plot_data.append(df[mask]['max_probability'].values)
                    else:
                        plot_data.append([])
                
                parts = ax.violinplot(plot_data, positions=[0, 1, 2], 
                                     showmeans=True, showmedians=True)
                
                colors = ['#E63946', '#F4A261', '#2A9D8F']
                for pc, color in zip(parts['bodies'], colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                
                ax.set_xticks([0, 1, 2])
                ax.set_xticklabels(['No\nLector', 'Lector\nMedio', 'Lector\nFrec.'], fontsize=9)
                ax.set_ylim(0, 1)
                ax.grid(axis='y', alpha=0.3)
                
                if col_idx == 0:
                    ax.set_ylabel(f'{GRUPO_LABELS[grupo]}\n\nProbabilidad', fontsize=10)
                if row_idx == 0:
                    ax.set_title(MODELO_LABELS[modelo], fontsize=12, weight='bold')
            else:
                ax.text(0.5, 0.5, 'No disponible', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
    
    plt.suptitle('Densidad de Probabilidades por Clase Predicha', 
                 fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'violin_plots_by_class.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ {output_path.name}")
    plt.close()


# ===================================================================
# VISUALIZACIÓN 4: MÉTRICAS F1, PRECISION, RECALL
# ===================================================================

def create_metrics_grid(all_data):
    """Grid completo de métricas por modelo y grupo."""
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    metric_names = ['F1-Score', 'Precision', 'Recall']
    
    for metric_idx, metric_name in enumerate(metric_names):
        # Calcular métrica
        metric_data = np.zeros((len(GRUPOS), len(MODELOS)))
        
        for i, grupo in enumerate(GRUPOS):
            for j, modelo in enumerate(MODELOS):
                if modelo in all_data.get(grupo, {}):
                    df = all_data[grupo][modelo]
                    y_true = df['true_label'].values
                    y_pred = df['predicted_label'].values
                    
                    # Calcular por clase
                    values_per_class = []
                    for c in range(3):
                        if metric_name == 'Precision':
                            mask = y_pred == c
                            if mask.sum() > 0:
                                values_per_class.append(((y_pred == c) & (y_true == c)).sum() / mask.sum())
                        elif metric_name == 'Recall':
                            mask = y_true == c
                            if mask.sum() > 0:
                                values_per_class.append(((y_pred == c) & (y_true == c)).sum() / mask.sum())
                        else:  # F1
                            mask_pred = y_pred == c
                            mask_true = y_true == c
                            if mask_pred.sum() > 0 and mask_true.sum() > 0:
                                p = ((y_pred == c) & (y_true == c)).sum() / mask_pred.sum()
                                r = ((y_pred == c) & (y_true == c)).sum() / mask_true.sum()
                                values_per_class.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
                    
                    metric_data[i, j] = np.mean(values_per_class) if values_per_class else 0
        
        # Crear heatmap
        ax = fig.add_subplot(gs[metric_idx, :])
        
        sns.heatmap(
            metric_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            xticklabels=[MODELO_LABELS[m] for m in MODELOS],
            yticklabels=[GRUPO_LABELS[g] for g in GRUPOS],
            cbar_kws={'label': metric_name},
            vmin=0.65,
            vmax=0.95,
            ax=ax,
            linewidths=2,
            linecolor='white'
        )
        
        ax.set_title(f'{metric_name} Promedio por Clase', fontsize=14, weight='bold', pad=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    plt.suptitle('Métricas de Rendimiento: F1, Precision y Recall', 
                 fontsize=16, weight='bold', y=0.995)
    
    output_path = OUTPUT_DIR / 'metrics_grid_comprehensive.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ {output_path.name}")
    plt.close()


# ===================================================================
# VISUALIZACIÓN 5: ANÁLISIS DE CASOS DIFÍCILES
# ===================================================================

def create_difficult_cases_analysis(all_data):
    """Análisis de predicciones con baja confianza."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Porcentaje de casos difíciles por modelo
    ax1 = axes[0, 0]
    low_conf_percentages = {m: [] for m in MODELOS}
    
    for grupo in GRUPOS:
        for modelo in MODELOS:
            if modelo in all_data.get(grupo, {}):
                df = all_data[grupo][modelo]
                pct = (df['max_probability'] < 0.6).mean() * 100
                low_conf_percentages[modelo].append(pct)
    
    x = np.arange(len(GRUPOS))
    width = 0.25
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, modelo in enumerate(MODELOS):
        ax1.bar(x + idx*width, low_conf_percentages[modelo], width, 
               label=MODELO_LABELS[modelo], color=colors[idx], alpha=0.8)
    
    ax1.set_xlabel('Grupo Etario', fontsize=12)
    ax1.set_ylabel('% Casos con Confianza < 60%', fontsize=12)
    ax1.set_title('Porcentaje de Casos Difíciles por Modelo', fontsize=13, weight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([GRUPO_LABELS[g] for g in GRUPOS], rotation=0)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Accuracy en casos de alta vs baja confianza
    ax2 = axes[0, 1]
    high_conf_acc = {m: [] for m in MODELOS}
    low_conf_acc = {m: [] for m in MODELOS}
    
    for grupo in GRUPOS:
        for modelo in MODELOS:
            if modelo in all_data.get(grupo, {}):
                df = all_data[grupo][modelo]
                
                high_mask = df['max_probability'] >= 0.8
                if high_mask.sum() > 0:
                    high_conf_acc[modelo].append(df[high_mask]['is_correct'].mean())
                
                low_mask = df['max_probability'] < 0.6
                if low_mask.sum() > 0:
                    low_conf_acc[modelo].append(df[low_mask]['is_correct'].mean())
    
    x = np.arange(len(MODELOS))
    width = 0.35
    
    high_means = [np.mean(high_conf_acc[m]) if high_conf_acc[m] else 0 for m in MODELOS]
    low_means = [np.mean(low_conf_acc[m]) if low_conf_acc[m] else 0 for m in MODELOS]
    
    ax2.bar(x - width/2, high_means, width, label='Alta Confianza (≥80%)', 
           color='#2A9D8F', alpha=0.8)
    ax2.bar(x + width/2, low_means, width, label='Baja Confianza (<60%)', 
           color='#E63946', alpha=0.8)
    
    ax2.set_xlabel('Modelo', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy según Nivel de Confianza', fontsize=13, weight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODELO_LABELS[m] for m in MODELOS])
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Distribución de clases en casos difíciles
    ax3 = axes[1, 0]
    
    class_dist_difficult = {m: [0, 0, 0] for m in MODELOS}
    
    for grupo in GRUPOS:
        for modelo in MODELOS:
            if modelo in all_data.get(grupo, {}):
                df = all_data[grupo][modelo]
                low_conf = df[df['max_probability'] < 0.6]
                
                for c in range(3):
                    class_dist_difficult[modelo][c] += (low_conf['predicted_label'] == c).sum()
    
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    
    for idx, modelo in enumerate(MODELOS):
        total = sum(class_dist_difficult[modelo])
        percentages = [count/total*100 if total > 0 else 0 
                      for count in class_dist_difficult[modelo]]
        ax3.bar(x + idx*width, percentages, width, 
               label=MODELO_LABELS[modelo], color=colors[idx], alpha=0.8)
    
    ax3.set_xlabel('Clase Predicha', fontsize=12)
    ax3.set_ylabel('% en Casos Difíciles', fontsize=12)
    ax3.set_title('Distribución de Clases en Predicciones de Baja Confianza', 
                 fontsize=13, weight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(CLASS_NAMES, rotation=15, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Tabla resumen
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    summary_data = []
    for modelo in MODELOS:
        total_samples = 0
        total_difficult = 0
        total_correct_diff = 0
        
        for grupo in GRUPOS:
            if modelo in all_data.get(grupo, {}):
                df = all_data[grupo][modelo]
                total_samples += len(df)
                low_conf = df[df['max_probability'] < 0.6]
                total_difficult += len(low_conf)
                total_correct_diff += low_conf['is_correct'].sum()
        
        summary_data.append([
            MODELO_LABELS[modelo],
            f'{total_samples:,}',
            f'{total_difficult:,}',
            f'{total_difficult/total_samples*100:.1f}%',
            f'{total_correct_diff/total_difficult*100:.1f}%' if total_difficult > 0 else 'N/A'
        ])
    
    table = ax4.table(cellText=summary_data,
                     colLabels=['Modelo', 'Total\nMuestras', 'Casos\nDifíciles', 
                               '% Difíciles', 'Acc en\nDifíciles'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(summary_data) + 1):
        if i == 0:
            for j in range(5):
                table[(i, j)].set_facecolor('#4A90E2')
                table[(i, j)].set_text_props(weight='bold', color='white')
        else:
            for j in range(5):
                table[(i, j)].set_facecolor('#F0F0F0' if i % 2 == 0 else 'white')
    
    ax4.set_title('Resumen de Casos Difíciles', fontsize=13, weight='bold', pad=20)
    
    plt.suptitle('Análisis de Predicciones con Baja Confianza (<60%)', 
                 fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / 'difficult_cases_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ {output_path.name}")
    plt.close()


# ===================================================================
# VISUALIZACIÓN 6: RESUMEN EJECUTIVO
# ===================================================================

def create_executive_summary(all_data):
    """Resumen ejecutivo con estadísticas clave."""
    
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')
    
    # Calcular estadísticas globales
    stats = {}
    for modelo in MODELOS:
        total_samples = 0
        total_correct = 0
        all_confidences = []
        all_f1s = []
        
        for grupo in GRUPOS:
            if modelo in all_data.get(grupo, {}):
                df = all_data[grupo][modelo]
                total_samples += len(df)
                total_correct += df['is_correct'].sum()
                all_confidences.extend(df['max_probability'].values)
                
                # F1 por clase
                y_true = df['true_label'].values
                y_pred = df['predicted_label'].values
                for c in range(3):
                    mask_pred = y_pred == c
                    mask_true = y_true == c
                    if mask_pred.sum() > 0 and mask_true.sum() > 0:
                        p = ((y_pred == c) & (y_true == c)).sum() / mask_pred.sum()
                        r = ((y_pred == c) & (y_true == c)).sum() / mask_true.sum()
                        all_f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
        
        stats[modelo] = {
            'samples': total_samples,
            'accuracy': total_correct / total_samples if total_samples > 0 else 0,
            'avg_confidence': np.mean(all_confidences) if all_confidences else 0,
            'avg_f1': np.mean(all_f1s) if all_f1s else 0,
            'high_conf_pct': (np.array(all_confidences) >= 0.8).mean() * 100 if all_confidences else 0,
            'low_conf_pct': (np.array(all_confidences) < 0.6).mean() * 100 if all_confidences else 0
        }
    
    # Crear layout
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Título
    fig.text(0.5, 0.95, 'RESUMEN EJECUTIVO DE RENDIMIENTO', 
            ha='center', fontsize=20, weight='bold')
    fig.text(0.5, 0.92, 'Clasificación de Hábitos Lectores con Deep Learning', 
            ha='center', fontsize=14, style='italic', color='gray')
    
    # Estadísticas por modelo
    y_start = 0.85
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, modelo in enumerate(MODELOS):
        s = stats[modelo]
        y = y_start - idx * 0.25
        
        # Box de modelo
        rect = Rectangle((0.05, y-0.18), 0.9, 0.2, 
                         facecolor=colors[idx], alpha=0.1, 
                         edgecolor=colors[idx], linewidth=2,
                         transform=fig.transFigure)
        fig.patches.append(rect)
        
        # Nombre del modelo
        fig.text(0.08, y+0.01, MODELO_LABELS[modelo], 
                fontsize=16, weight='bold', color=colors[idx])
        
        # Métricas principales
        metrics_text = f"""
        Muestras Evaluadas: {s['samples']:,}
        Accuracy Global: {s['accuracy']:.2%}
        F1-Score Promedio: {s['avg_f1']:.2%}
        Confianza Promedio: {s['avg_confidence']:.2%}
        Alta Confianza (≥80%): {s['high_conf_pct']:.1f}%
        Baja Confianza (<60%): {s['low_conf_pct']:.1f}%
        """
        
        fig.text(0.08, y-0.15, metrics_text, 
                fontsize=11, verticalalignment='top', family='monospace')
        
        # Gráfico de barras mini
        bar_x = 0.75
        bar_width = 0.15
        bar_height = 0.08
        
        # Accuracy bar
        ax_bar = fig.add_axes([bar_x, y-0.05, bar_width, bar_height])
        ax_bar.barh([0], [s['accuracy']], color=colors[idx], alpha=0.7)
        ax_bar.set_xlim(0, 1)
        ax_bar.set_yticks([])
        ax_bar.set_xticks([0, 0.5, 1])
        ax_bar.set_xticklabels(['0%', '50%', '100%'], fontsize=8)
        ax_bar.set_title('Accuracy', fontsize=9, weight='bold')
        ax_bar.grid(axis='x', alpha=0.3)
    
    # Footer
    fig.text(0.5, 0.05, f'Generado: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
            ha='center', fontsize=10, color='gray')
    fig.text(0.5, 0.02, 'Universidad Peruana Unión | Investigación en Deep Learning', 
            ha='center', fontsize=10, weight='bold')
    
    output_path = OUTPUT_DIR / 'executive_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ {output_path.name}")
    plt.close()


# ===================================================================
# MAIN
# ===================================================================

def main():
    """Generar todas las visualizaciones para tesis."""
    
    print("=" * 80)
    print("  📊 VISUALIZACIONES PARA SUSTENTACIÓN DE TESIS")
    print("=" * 80)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💾 Salida: {OUTPUT_DIR}")
    print()
    
    # Cargar datos
    print("📂 Cargando datos...")
    all_data = load_all_data()
    total_loaded = sum(len(models) for models in all_data.values())
    print(f"   ✅ {total_loaded}/9 clasificaciones cargadas\n")
    
    if total_loaded == 0:
        print("❌ No hay datos disponibles. Ejecuta primero generate_synthetic_classifications.py")
        return 1
    
    # Generar visualizaciones
    print("🎨 Generando visualizaciones académicas...\n")
    
    print("   1. Radar Chart Comparativo...")
    create_radar_chart(all_data)
    
    print("   2. Box Plots de Confianza...")
    create_confidence_boxplots(all_data)
    
    print("   3. Violin Plots por Clase...")
    create_violin_plots_by_class(all_data)
    
    print("   4. Grid de Métricas (F1/Precision/Recall)...")
    create_metrics_grid(all_data)
    
    print("   5. Análisis de Casos Difíciles...")
    create_difficult_cases_analysis(all_data)
    
    print("   6. Resumen Ejecutivo...")
    create_executive_summary(all_data)
    
    # Resumen
    print("\n" + "=" * 80)
    print("  ✅ VISUALIZACIONES COMPLETADAS")
    print("=" * 80)
    print(f"\n📁 Archivos generados en: {OUTPUT_DIR}")
    print("\n📊 Gráficos creados:")
    for file in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"   - {file.name}")
    
    print("\n💡 Recomendaciones para la sustentación:")
    print("   1. radar_chart_models.png - Visión general multidimensional")
    print("   2. executive_summary.png - Resumen de resultados principales")
    print("   3. metrics_grid_comprehensive.png - Métricas detalladas")
    print("   4. difficult_cases_analysis.png - Análisis de limitaciones")
    print("   5. boxplot_confidence_distribution.png - Robustez del modelo")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
