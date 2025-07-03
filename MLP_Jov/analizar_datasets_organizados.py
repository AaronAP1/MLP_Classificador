#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
import os
from datetime import datetime

def analizar_datasets_organizados(output_dir=None):
    """
    Analiza los datasets organizados por categorías de lectura
    """
    
    # Si no se especifica directorio, usar el más reciente
    if output_dir is None:
        dirs = glob.glob("data/processed/clasificados_*")
        if not dirs:
            print("❌ No se encontraron datasets clasificados")
            return
        output_dir = max(dirs)
    
    print(f"📊 ANÁLISIS DE DATASETS ORGANIZADOS POR HÁBITOS DE LECTURA")
    print(f"📁 Directorio: {output_dir}")
    print("=" * 70)
    
    # Cargar resumen si existe
    resumen_path = f"{output_dir}/resumen_clasificacion.json"
    if os.path.exists(resumen_path):
        with open(resumen_path, 'r') as f:
            resumen = json.load(f)
        
        print(f"📅 Fecha de clasificación: {resumen['fecha_clasificacion']}")
        print(f"👥 Total de personas: {resumen['total_personas']:,}")
        print(f"🎯 Confianza promedio: {resumen['confianza_promedio']:.1%}")
        print()
    
    # Cargar dataset completo
    dataset_completo = f"{output_dir}/dataset_completo_clasificado.csv"
    if not os.path.exists(dataset_completo):
        print(f"❌ No se encontró {dataset_completo}")
        return
    
    df_completo = pd.read_csv(dataset_completo)
    
    # Análisis por categorías
    print("📈 ANÁLISIS POR CATEGORÍAS:")
    print("-" * 40)
    
    categorias = df_completo['prediccion_habito'].unique()
    stats_categorias = {}
    
    for categoria in sorted(categorias):
        df_cat = df_completo[df_completo['prediccion_habito'] == categoria]
        n_personas = len(df_cat)
        porcentaje = n_personas / len(df_completo) * 100
        confianza_promedio = df_cat['confianza'].mean()
        
        print(f"\n🔸 {categoria.upper()}")
        print(f"   👥 Personas: {n_personas:,} ({porcentaje:.1f}%)")
        print(f"   🎯 Confianza promedio: {confianza_promedio:.1%}")
        
        # Estadísticas demográficas básicas si existen
        cols_demograficas = ['P100_1', 'P100_2', 'P200_2', 'P200_3']  # edad, sexo, educación, etc.
        for col in cols_demograficas:
            if col in df_cat.columns:
                if df_cat[col].dtype in ['int64', 'float64']:
                    print(f"   📊 {col} promedio: {df_cat[col].mean():.1f}")
                else:
                    # Para categóricas, mostrar la moda
                    moda = df_cat[col].mode()
                    if len(moda) > 0:
                        print(f"   📊 {col} más común: {moda.iloc[0]}")
        
        stats_categorias[categoria] = {
            'count': n_personas,
            'percentage': porcentaje,
            'confidence': confianza_promedio
        }
    
    # Crear visualizaciones
    crear_visualizaciones_organizacion(df_completo, output_dir, stats_categorias)
    
    # Análisis de confianza
    print(f"\n🎯 ANÁLISIS DE CONFIANZA:")
    print("-" * 30)
    
    confianza_stats = df_completo['confianza'].describe()
    print(f"Confianza mínima: {confianza_stats['min']:.1%}")
    print(f"Confianza media: {confianza_stats['mean']:.1%}")
    print(f"Confianza máxima: {confianza_stats['max']:.1%}")
    
    # Casos de baja confianza
    baja_confianza = df_completo[df_completo['confianza'] < 0.6]
    if len(baja_confianza) > 0:
        print(f"\n⚠️  Casos con baja confianza (<60%): {len(baja_confianza):,} ({len(baja_confianza)/len(df_completo)*100:.1f}%)")
    
    # Guardar estadísticas detalladas
    stats_detalladas = {
        'resumen_general': {
            'total_personas': len(df_completo),
            'fecha_analisis': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'confianza_promedio': float(df_completo['confianza'].mean()),
            'casos_baja_confianza': len(baja_confianza)
        },
        'estadisticas_por_categoria': stats_categorias,
        'distribucion_confianza': {
            'min': float(confianza_stats['min']),
            'mean': float(confianza_stats['mean']),
            'max': float(confianza_stats['max']),
            'std': float(confianza_stats['std'])
        }
    }
    
    stats_file = f"{output_dir}/estadisticas_detalladas.json"
    with open(stats_file, 'w') as f:
        json.dump(stats_detalladas, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Estadísticas detalladas guardadas en: {stats_file}")
    
    return df_completo, stats_categorias

def crear_visualizaciones_organizacion(df, output_dir, stats_categorias):
    """
    Crea visualizaciones del análisis de organización
    """
    
    # Crear directorio de gráficos
    plots_dir = f"{output_dir}/analisis_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Distribución de categorías
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Gráfico de barras
    categorias = list(stats_categorias.keys())
    counts = [stats_categorias[cat]['count'] for cat in categorias]
    percentages = [stats_categorias[cat]['percentage'] for cat in categorias]
    
    bars = ax1.bar(categorias, counts, alpha=0.8)
    ax1.set_title('Distribución de Personas por Hábito de Lectura', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Número de Personas')
    ax1.tick_params(axis='x', rotation=45)
    
    # Añadir valores en las barras
    for bar, count, pct in zip(bars, counts, percentages):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom')
    
    # Gráfico de torta
    ax2.pie(counts, labels=categorias, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Proporción de Hábitos de Lectura', fontsize=14, fontweight='bold')
    
    # Distribución de confianza por categoría
    for i, categoria in enumerate(categorias):
        df_cat = df[df['prediccion_habito'] == categoria]
        ax3.hist(df_cat['confianza'], alpha=0.6, label=categoria, bins=20)
    
    ax3.set_title('Distribución de Confianza por Categoría', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Confianza de Predicción')
    ax3.set_ylabel('Frecuencia')
    ax3.legend()
    
    # Boxplot de confianza
    data_confianza = [df[df['prediccion_habito'] == cat]['confianza'] for cat in categorias]
    ax4.boxplot(data_confianza, labels=categorias)
    ax4.set_title('Distribución de Confianza por Categoría', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Confianza de Predicción')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/analisis_distribucion_categorias.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Mapa de calor de probabilidades
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Crear matriz de probabilidades promedio por categoría
    prob_cols = ['prob_lector_frecuente', 'prob_lector_medio', 'prob_no_lector']
    prob_matrix = []
    
    for categoria in categorias:
        df_cat = df[df['prediccion_habito'] == categoria]
        probs = [df_cat[col].mean() for col in prob_cols]
        prob_matrix.append(probs)
    
    prob_matrix = np.array(prob_matrix)
    
    im = ax.imshow(prob_matrix, cmap='YlOrRd', aspect='auto')
    
    # Configurar etiquetas
    ax.set_xticks(range(len(prob_cols)))
    ax.set_xticklabels(['Lector Frecuente', 'Lector Medio', 'No Lector'])
    ax.set_yticks(range(len(categorias)))
    ax.set_yticklabels(categorias)
    
    # Añadir valores en las celdas
    for i in range(len(categorias)):
        for j in range(len(prob_cols)):
            text = ax.text(j, i, f'{prob_matrix[i, j]:.3f}',
                         ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Probabilidades Promedio por Categoría Predicha', fontsize=14, fontweight='bold')
    
    # Añadir colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probabilidad Promedio')
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/mapa_probabilidades.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizaciones guardadas en: {plots_dir}/")

if __name__ == "__main__":
    # Analizar los datasets más recientes
    df_completo, stats = analizar_datasets_organizados()
    
    print(f"\n🎯 ¡Análisis completado!")
    print(f"   📊 Se analizaron {len(df_completo):,} personas clasificadas")
    print(f"   📂 Se generaron visualizaciones y estadísticas detalladas")
