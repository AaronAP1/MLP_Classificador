#!/usr/bin/env python3
"""
Script para generar gráficos de barras apiladas para análisis de datos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def crear_grafico_barras_apiladas(model_dir=None):
    """
    Crea un gráfico de barras apiladas similar al de la imagen.
    Muestra distribución de datos relevantes vs no relevantes por split.
    """
    
    # Si no se especifica directorio, usar el más reciente
    if model_dir is None:
        import glob
        model_dirs = glob.glob("outputs/models/202*")
        if not model_dirs:
            print("❌ No se encontró ningún modelo entrenado")
            return
        model_dir = max(model_dirs)
    
    print(f"📊 GENERANDO GRÁFICO DE BARRAS APILADAS")
    print(f"📁 Directorio: {model_dir}")
    print("=" * 50)
    
    # Crear directorio de análisis si no existe
    analysis_dir = os.path.join(model_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Datos simulados basados en tu dataset real
    # Tu dataset tiene ~14,021 muestras divididas 70/15/15
    total_samples = 14021
    train_samples = int(total_samples * 0.70)  # ~9,815
    test_samples = int(total_samples * 0.15)   # ~2,103
    
    # Simular distribución de relevancia
    # Asumiendo que ~75% de los datos son "relevantes" (tienen P517 o buenas variables de lectura)
    train_relevantes = int(train_samples * 0.75)
    train_no_relevantes = train_samples - train_relevantes
    
    test_relevantes = int(test_samples * 0.75)
    test_no_relevantes = test_samples - test_relevantes
    
    # Crear DataFrame para el gráfico
    data = {
        'split': ['test', 'train'],
        'relevante_0': [test_no_relevantes, train_no_relevantes],    # No relevantes (azul)
        'relevante_1': [test_relevantes, train_relevantes]          # Relevantes (naranja)
    }
    
    df = pd.DataFrame(data)
    
    # Configurar el gráfico
    plt.figure(figsize=(10, 8))
    
    # Definir colores exactos como en la imagen
    colors = ['#1f77b4', '#ff7f0e']  # Azul y naranja
    
    # Crear barras apiladas
    x_pos = np.arange(len(df['split']))
    
    # Barras base (no relevantes)
    p1 = plt.bar(x_pos, df['relevante_0'], color=colors[0], label='0')
    
    # Barras superiores (relevantes)
    p2 = plt.bar(x_pos, df['relevante_1'], bottom=df['relevante_0'], color=colors[1], label='1')
    
    # Personalizar el gráfico
    plt.xlabel('split', fontsize=12, fontweight='bold')
    plt.ylabel('nº eventos', fontsize=12, fontweight='bold')
    plt.title('Distribución de Datos Relevantes por Split', fontsize=14, fontweight='bold')
    
    # Configurar ejes
    plt.xticks(x_pos, df['split'])
    plt.ylim(0, max(df['relevante_0'] + df['relevante_1']) * 1.1)
    
    # Añadir leyenda
    plt.legend(title='relevante', loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Añadir valores en las barras
    for i, (base, top) in enumerate(zip(df['relevante_0'], df['relevante_1'])):
        # Texto para la parte azul (no relevantes)
        plt.text(i, base/2, f'{base:,}', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=10)
        
        # Texto para la parte naranja (relevantes)
        plt.text(i, base + top/2, f'{top:,}', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=10)
        
        # Texto total en la parte superior
        plt.text(i, base + top + 200, f'{base + top:,}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    # Añadir grid sutil
    plt.grid(True, alpha=0.3, axis='y')
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar el gráfico
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(analysis_dir, f"barras_apiladas_{timestamp}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"✅ Gráfico guardado en: {output_file}")
    
    # Mostrar el gráfico
    plt.show()
    
    # Imprimir estadísticas
    print(f"\n📊 ESTADÍSTICAS:")
    print(f"  Test - No relevantes: {test_no_relevantes:,} ({test_no_relevantes/test_samples*100:.1f}%)")
    print(f"  Test - Relevantes: {test_relevantes:,} ({test_relevantes/test_samples*100:.1f}%)")
    print(f"  Train - No relevantes: {train_no_relevantes:,} ({train_no_relevantes/train_samples*100:.1f}%)")
    print(f"  Train - Relevantes: {train_relevantes:,} ({train_relevantes/train_samples*100:.1f}%)")

def crear_grafico_distribucion_clases():
    """
    Crea un gráfico de barras apiladas para distribución de clases de lectura.
    """
    
    print(f"\n📊 GRÁFICO DE DISTRIBUCIÓN POR CLASES DE LECTURA")
    print("=" * 50)
    
    # Datos reales de la distribución de clases de tu modelo
    # Basado en los resultados mostrados anteriormente
    data_clases = {
        'split': ['test', 'train'],
        'no_lector': [470, 3132],           # Azul
        'lector_medio': [95, 632],          # Naranja  
        'lector_frecuente': [1539, 10257]  # Verde
    }
    
    df_clases = pd.DataFrame(data_clases)
    
    # Configurar el gráfico
    plt.figure(figsize=(10, 8))
    
    # Colores para las tres clases
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Azul, naranja, verde
    
    x_pos = np.arange(len(df_clases['split']))
    
    # Crear barras apiladas
    bottom1 = df_clases['no_lector']
    bottom2 = bottom1 + df_clases['lector_medio']
    
    p1 = plt.bar(x_pos, df_clases['no_lector'], color=colors[0], label='no_lector')
    p2 = plt.bar(x_pos, df_clases['lector_medio'], bottom=bottom1, color=colors[1], label='lector_medio')
    p3 = plt.bar(x_pos, df_clases['lector_frecuente'], bottom=bottom2, color=colors[2], label='lector_frecuente')
    
    # Personalizar
    plt.xlabel('split', fontsize=12, fontweight='bold')
    plt.ylabel('nº personas', fontsize=12, fontweight='bold')
    plt.title('Distribución de Clases de Hábitos de Lectura por Split', fontsize=14, fontweight='bold')
    
    plt.xticks(x_pos, df_clases['split'])
    plt.legend(title='Hábito de Lectura', loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Añadir valores en las barras
    for i in range(len(df_clases['split'])):
        no_lector = df_clases['no_lector'][i]
        lector_medio = df_clases['lector_medio'][i]
        lector_freq = df_clases['lector_frecuente'][i]
        
        # Texto en cada sección
        plt.text(i, no_lector/2, f'{no_lector:,}', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=10)
        
        plt.text(i, no_lector + lector_medio/2, f'{lector_medio:,}', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=10)
        
        plt.text(i, no_lector + lector_medio + lector_freq/2, f'{lector_freq:,}', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=10)
        
        # Total en la parte superior
        total = no_lector + lector_medio + lector_freq
        plt.text(i, total + 200, f'{total:,}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Guardar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outputs/plots/distribucion_clases_{timestamp}.png"
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"✅ Gráfico de clases guardado en: {output_file}")
    plt.show()

def main():
    """Función principal para generar ambos gráficos."""
    
    print("🎨 GENERADOR DE GRÁFICOS DE BARRAS APILADAS")
    print("=" * 60)
    
    # Gráfico 1: Relevancia de datos
    crear_grafico_barras_apiladas()
    
    # Gráfico 2: Distribución de clases
    crear_grafico_distribucion_clases()
    
    print(f"\n🎯 ¡Gráficos generados exitosamente!")

if __name__ == "__main__":
    main()
