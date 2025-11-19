"""
Visualización de Embeddings: Método Tradicional vs Deep Learning
=================================================================

Genera scatter plots tipo t-SNE/UMAP mostrando:
1. Cómo cada método separa las clases en el espacio latente
2. Comparación individual: mismas personas, diferentes clasificaciones
3. Análisis de correcciones: casos que DL mejoró vs tradicional

Similar a visualizaciones de embeddings de libros/documentos.

Universidad Peruana Unión - 2025
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from datetime import datetime
import matplotlib.patches as mpatches

# ===================================================================
# CONFIGURACIÓN
# ===================================================================

classfier_root = Path(__file__).parent

DATA_DIR_PROCESSED = classfier_root / 'data' / 'processed_fase2'
DATA_DIR_RAW = classfier_root / 'data' / 'unprocessed'
DL_PREDICTIONS_DIR = classfier_root / 'outputs' / 'classifications_3x3_real'
OUTPUT_DIR = classfier_root / 'outputs' / 'embeddings_comparison'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRUPOS = ['ninos', 'jovenes', 'adultos']
MODELOS_DL = ['ft_transformer', 'tabnet', 'mlp_attention']
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

CLASS_COLORS = ['#3498db', '#e74c3c', '#2ecc71']  # Azul, Rojo, Verde
plt.style.use('seaborn-v0_8-white')
sns.set_context("paper", font_scale=1.1)

# ===================================================================
# MÉTODO TRADICIONAL
# ===================================================================

def aplicar_clasificacion_tradicional(df_raw, grupo):
    """Aplica clasificación tradicional por reglas."""
    df = df_raw.copy()
    df['lectura_score'] = 0.0
    
    if grupo == "adultos":
        lectura_vars = ['P315_1', 'P315_2', 'P315_3', 'P315_4', 'P315_5', 'P315_6']
        frecuencia_var = 'P409'
    else:
        lectura_vars = ['P514_1', 'P514_2', 'P514_3']
        frecuencia_var = 'P516'
    
    for var in lectura_vars:
        if var in df.columns:
            mask = df[var].notna() & (df[var] > 0)
            df.loc[mask, 'lectura_score'] += 0.4 / len(lectura_vars)
    
    if 'P517' in df.columns:
        mask = df['P517'].notna()
        df.loc[mask, 'lectura_score'] += (df.loc[mask, 'P517'] - 1) / 4 * 0.4
    
    if frecuencia_var in df.columns:
        mask = df[frecuencia_var].notna()
        df.loc[mask, 'lectura_score'] += (df.loc[mask, frecuencia_var] - 1) / 5 * 0.2
    
    if 'P311' in df.columns:
        no_sabe_leer = df['P311'] == 0
        df.loc[no_sabe_leer, 'lectura_score'] = 0.0
    
    df['lectura_score'] = np.clip(df['lectura_score'], 0, 1)
    
    def score_to_category(score):
        if pd.isna(score) or score < 0.2:
            return 0
        elif score < 0.6:
            return 1
        else:
            return 2
    
    return df['lectura_score'].apply(score_to_category)

# ===================================================================
# CARGA DE DATOS
# ===================================================================

def cargar_y_preparar_datos(grupo):
    """Carga datos procesados, raw, y predicciones DL."""
    
    print(f"\n📂 Cargando datos de {grupo}...")
    
    # Test procesado
    test_path = DATA_DIR_PROCESSED / grupo / f'{grupo}_test.csv'
    if not test_path.exists():
        print(f"   ❌ No encontrado: {test_path}")
        return None
    
    df_test = pd.read_csv(test_path)
    print(f"   ✅ Test: {len(df_test):,} muestras, {len(df_test.columns)} features")
    
    # Raw completo
    raw_path = DATA_DIR_RAW / f'{grupo}.csv'
    if not raw_path.exists():
        print(f"   ❌ No encontrado: {raw_path}")
        return None
    
    df_raw = pd.read_csv(raw_path)
    print(f"   ✅ Raw: {len(df_raw):,} muestras")
    
    # Aplicar clasificación tradicional
    clasificacion_trad = aplicar_clasificacion_tradicional(df_raw, grupo)
    
    # Usar primeras N muestras como aproximación de test
    n_test = len(df_test)
    trad_test = clasificacion_trad.values[:n_test]
    
    # Ground truth
    y_true = df_test['target'].values
    
    # Features para embedding (solo numéricas)
    feature_cols = [col for col in df_test.columns 
                   if col not in ['target', 'index'] and df_test[col].dtype in ['int64', 'float64']]
    X_features = df_test[feature_cols].fillna(0).values
    
    print(f"   ✅ Features para embedding: {len(feature_cols)}")
    
    # Cargar predicciones DL
    dl_predictions = {}
    for modelo in MODELOS_DL:
        pred_path = DL_PREDICTIONS_DIR / f'{grupo}_{modelo}' / 'predictions.csv'
        if pred_path.exists():
            df_pred = pd.read_csv(pred_path)
            dl_predictions[modelo] = df_pred['predicted_label'].values
            print(f"   ✅ {MODELO_LABELS[modelo]}: {len(df_pred):,} predicciones")
    
    return {
        'grupo': grupo,
        'n_samples': n_test,
        'X_features': X_features,
        'y_true': y_true,
        'y_tradicional': trad_test,
        'dl_predictions': dl_predictions
    }

# ===================================================================
# REDUCCIÓN DIMENSIONAL
# ===================================================================

def reducir_dimensiones(X, metodo='tsne', random_state=42):
    """Reduce dimensiones con t-SNE o PCA."""
    
    print(f"   🔄 Reduciendo dimensiones con {metodo.upper()}...")
    
    if metodo == 'tsne':
        # Primero PCA a 50 dims, luego t-SNE
        if X.shape[1] > 50:
            pca = PCA(n_components=50, random_state=random_state)
            X_pca = pca.fit_transform(X)
            print(f"      PCA: {X.shape[1]} → 50 dims ({pca.explained_variance_ratio_.sum():.1%} varianza)")
        else:
            X_pca = X
        
        tsne = TSNE(n_components=2, random_state=random_state, 
                   perplexity=min(30, len(X)//4), max_iter=1000, verbose=0)
        X_2d = tsne.fit_transform(X_pca)
        
    else:  # PCA
        pca = PCA(n_components=2, random_state=random_state)
        X_2d = pca.fit_transform(X)
        print(f"      PCA 2D: {pca.explained_variance_ratio_.sum():.1%} varianza explicada")
    
    return X_2d

# ===================================================================
# VISUALIZACIÓN 1: EMBEDDING POR MÉTODO
# ===================================================================

def crear_scatter_por_metodo(datos):
    """Scatter plot: Ground Truth vs Tradicional vs Mejor DL."""
    
    grupo = datos['grupo']
    X = datos['X_features']
    y_true = datos['y_true']
    y_trad = datos['y_tradicional']
    
    # Encontrar mejor modelo DL
    mejor_modelo = None
    mejor_acc = 0
    for modelo, y_pred in datos['dl_predictions'].items():
        acc = (y_pred == y_true).mean()
        if acc > mejor_acc:
            mejor_acc = acc
            mejor_modelo = modelo
            y_dl = y_pred
    
    print(f"\n🎨 Creando scatter plots para {grupo}...")
    print(f"   Mejor modelo DL: {MODELO_LABELS[mejor_modelo]} ({mejor_acc:.1%} acc)")
    
    # Reducir dimensiones
    X_2d = reducir_dimensiones(X, metodo='tsne')
    
    # Crear figura
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Ground Truth
    ax = axes[0]
    for clase in range(3):
        mask = y_true == clase
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                  c=CLASS_COLORS[clase], label=CLASS_NAMES[clase],
                  alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    
    ax.set_title(f'Ground Truth\n{GRUPO_LABELS[grupo]}', fontsize=14, weight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.2)
    
    # Tradicional
    ax = axes[1]
    for clase in range(3):
        mask = y_trad == clase
        if mask.sum() > 0:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                      c=CLASS_COLORS[clase], label=CLASS_NAMES[clase],
                      alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    
    acc_trad = (y_trad == y_true).mean()
    ax.set_title(f'Método Tradicional (Reglas)\nAccuracy: {acc_trad:.1%}', 
                fontsize=14, weight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.2)
    
    # Deep Learning
    ax = axes[2]
    for clase in range(3):
        mask = y_dl == clase
        if mask.sum() > 0:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                      c=CLASS_COLORS[clase], label=CLASS_NAMES[clase],
                      alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    
    ax.set_title(f'{MODELO_LABELS[mejor_modelo]}\nAccuracy: {mejor_acc:.1%}', 
                fontsize=14, weight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.2)
    
    plt.suptitle(f'Visualización de Embeddings: Comparación de Métodos de Clasificación\n{GRUPO_LABELS[grupo]}', 
                 fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f'embedding_scatter_{grupo}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ {output_path.name}")
    plt.close()

# ===================================================================
# VISUALIZACIÓN 2: ANÁLISIS DE CORRECCIONES
# ===================================================================

def crear_scatter_correcciones(datos):
    """Scatter mostrando casos corregidos, empeorados, y sin cambio."""
    
    grupo = datos['grupo']
    X = datos['X_features']
    y_true = datos['y_true']
    y_trad = datos['y_tradicional']
    
    # Mejor DL
    mejor_modelo = None
    mejor_acc = 0
    for modelo, y_pred in datos['dl_predictions'].items():
        acc = (y_pred == y_true).mean()
        if acc > mejor_acc:
            mejor_acc = acc
            mejor_modelo = modelo
            y_dl = y_pred
    
    print(f"\n🎨 Creando análisis de correcciones para {grupo}...")
    
    # Reducir dimensiones
    X_2d = reducir_dimensiones(X, metodo='tsne')
    
    # Clasificar casos
    correctas_trad = y_trad == y_true
    correctas_dl = y_dl == y_true
    
    # 4 categorías
    ambos_correctos = correctas_trad & correctas_dl
    dl_mejora = (~correctas_trad) & correctas_dl  # Tradicional mal, DL bien
    dl_empeora = correctas_trad & (~correctas_dl)  # Tradicional bien, DL mal
    ambos_incorrectos = (~correctas_trad) & (~correctas_dl)
    
    print(f"   📊 Distribución de casos:")
    print(f"      Ambos correctos:     {ambos_correctos.sum():5,} ({ambos_correctos.mean():.1%})")
    print(f"      DL mejora:           {dl_mejora.sum():5,} ({dl_mejora.mean():.1%})")
    print(f"      DL empeora:          {dl_empeora.sum():5,} ({dl_empeora.mean():.1%})")
    print(f"      Ambos incorrectos:   {ambos_incorrectos.sum():5,} ({ambos_incorrectos.mean():.1%})")
    
    # Crear figura
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Panel izquierdo: Clasificación por resultado
    ax = axes[0]
    
    colors_resultado = {
        'ambos_correctos': '#2ecc71',      # Verde
        'dl_mejora': '#3498db',            # Azul
        'dl_empeora': '#e74c3c',           # Rojo
        'ambos_incorrectos': '#95a5a6'    # Gris
    }
    
    labels_resultado = {
        'ambos_correctos': f'Ambos Correctos ({ambos_correctos.sum():,})',
        'dl_mejora': f'DL Mejora ({dl_mejora.sum():,})',
        'dl_empeora': f'DL Empeora ({dl_empeora.sum():,})',
        'ambos_incorrectos': f'Ambos Incorrectos ({ambos_incorrectos.sum():,})'
    }
    
    # Plotear en orden: primero incorrectos (background), luego mejoras (highlight)
    for mask, key in [(ambos_incorrectos, 'ambos_incorrectos'),
                      (ambos_correctos, 'ambos_correctos'),
                      (dl_empeora, 'dl_empeora'),
                      (dl_mejora, 'dl_mejora')]:
        if mask.sum() > 0:
            size = 50 if key == 'dl_mejora' else 30
            alpha = 0.8 if key in ['dl_mejora', 'dl_empeora'] else 0.4
            zorder = 3 if key == 'dl_mejora' else 2 if key == 'dl_empeora' else 1
            
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                      c=colors_resultado[key], label=labels_resultado[key],
                      alpha=alpha, s=size, edgecolors='white', linewidth=0.5,
                      zorder=zorder)
    
    ax.set_title(f'Análisis de Correcciones: Tradicional → Deep Learning\n{GRUPO_LABELS[grupo]}', 
                fontsize=14, weight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.2)
    
    # Panel derecho: Ground truth con overlay de mejoras
    ax = axes[1]
    
    # Background: ground truth
    for clase in range(3):
        mask = y_true == clase
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                  c=CLASS_COLORS[clase], label=f'{CLASS_NAMES[clase]} (real)',
                  alpha=0.3, s=20, edgecolors='none')
    
    # Overlay: casos donde DL mejoró
    if dl_mejora.sum() > 0:
        ax.scatter(X_2d[dl_mejora, 0], X_2d[dl_mejora, 1],
                  c='none', s=100, edgecolors='blue', linewidths=2,
                  marker='o', label=f'Corregidos por DL ({dl_mejora.sum():,})',
                  zorder=3)
    
    # Overlay: casos donde DL empeoró
    if dl_empeora.sum() > 0:
        ax.scatter(X_2d[dl_empeora, 0], X_2d[dl_empeora, 1],
                  c='none', s=100, edgecolors='red', linewidths=2,
                  marker='X', label=f'Empeorados por DL ({dl_empeora.sum():,})',
                  zorder=3)
    
    mejora_neta = dl_mejora.sum() - dl_empeora.sum()
    mejora_pct = (mejor_acc - (y_trad == y_true).mean()) * 100
    
    ax.set_title(f'Ground Truth + Correcciones\nMejora Neta: +{mejora_neta:,} casos ({mejora_pct:+.1f}%)', 
                fontsize=14, weight='bold')
    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.legend(loc='best', fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.2)
    
    plt.suptitle(f'Análisis de Correcciones: ¿Qué Casos Mejoró Deep Learning?\n{MODELO_LABELS[mejor_modelo]} vs Método Tradicional', 
                 fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f'embedding_corrections_{grupo}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ {output_path.name}")
    plt.close()

# ===================================================================
# VISUALIZACIÓN 3: COMPARACIÓN 3 GRUPOS EN UNO
# ===================================================================

def crear_scatter_unificado(todos_datos):
    """Scatter unificado mostrando los 3 grupos."""
    
    print(f"\n🎨 Creando visualización unificada de 3 grupos...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    for idx, grupo in enumerate(GRUPOS):
        if grupo not in todos_datos:
            continue
        
        datos = todos_datos[grupo]
        X = datos['X_features']
        y_true = datos['y_true']
        y_trad = datos['y_tradicional']
        
        # Mejor DL
        mejor_modelo = None
        mejor_acc = 0
        for modelo, y_pred in datos['dl_predictions'].items():
            acc = (y_pred == y_true).mean()
            if acc > mejor_acc:
                mejor_acc = acc
                mejor_modelo = modelo
                y_dl = y_pred
        
        # Reducir dimensiones
        X_2d = reducir_dimensiones(X, metodo='tsne')
        
        # Row 0: Ground Truth
        ax = fig.add_subplot(gs[0, idx])
        for clase in range(3):
            mask = y_true == clase
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                      c=CLASS_COLORS[clase], label=CLASS_NAMES[clase],
                      alpha=0.6, s=25, edgecolors='white', linewidth=0.3)
        
        ax.set_title(f'{GRUPO_LABELS[grupo]}\nGround Truth', fontsize=12, weight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=10)
        if idx == 0:
            ax.set_ylabel('t-SNE 2', fontsize=10)
        if idx == 2:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.2)
        
        # Row 1: Comparación Trad vs DL
        ax = fig.add_subplot(gs[1, idx])
        
        correctas_trad = y_trad == y_true
        correctas_dl = y_dl == y_true
        dl_mejora = (~correctas_trad) & correctas_dl
        dl_empeora = correctas_trad & (~correctas_dl)
        sin_cambio_correcto = correctas_trad & correctas_dl
        sin_cambio_incorrecto = (~correctas_trad) & (~correctas_dl)
        
        # Background
        ax.scatter(X_2d[sin_cambio_incorrecto, 0], X_2d[sin_cambio_incorrecto, 1],
                  c='#95a5a6', alpha=0.2, s=15, edgecolors='none', label='Sin cambio (✗)')
        ax.scatter(X_2d[sin_cambio_correcto, 0], X_2d[sin_cambio_correcto, 1],
                  c='#2ecc71', alpha=0.3, s=20, edgecolors='white', linewidth=0.2, label='Sin cambio (✓)')
        
        # Highlights
        if dl_mejora.sum() > 0:
            ax.scatter(X_2d[dl_mejora, 0], X_2d[dl_mejora, 1],
                      c='#3498db', alpha=0.8, s=40, edgecolors='white', linewidth=0.5,
                      marker='*', label=f'DL Mejora ({dl_mejora.sum():,})')
        
        if dl_empeora.sum() > 0:
            ax.scatter(X_2d[dl_empeora, 0], X_2d[dl_empeora, 1],
                      c='#e74c3c', alpha=0.8, s=40, edgecolors='white', linewidth=0.5,
                      marker='X', label=f'DL Empeora ({dl_empeora.sum():,})')
        
        acc_trad = correctas_trad.mean()
        mejora = (mejor_acc - acc_trad) * 100
        
        ax.set_title(f'{MODELO_LABELS[mejor_modelo]}\nMejora: {mejora:+.1f}% ({acc_trad:.1%} → {mejor_acc:.1%})', 
                    fontsize=12, weight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=10)
        if idx == 0:
            ax.set_ylabel('t-SNE 2', fontsize=10)
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.2)
    
    plt.suptitle('Visualización Comparativa de Embeddings: Método Tradicional vs Deep Learning\nDistribución Espacial y Correcciones por Grupo Etario', 
                 fontsize=16, weight='bold', y=0.995)
    
    output_path = OUTPUT_DIR / 'embedding_unified_3_groups.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✅ {output_path.name}")
    plt.close()

# ===================================================================
# MAIN
# ===================================================================

def main():
    """Ejecutar análisis completo."""
    
    print("=" * 80)
    print("  🎨 VISUALIZACIÓN DE EMBEDDINGS: TRADICIONAL VS DEEP LEARNING")
    print("=" * 80)
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💾 Salida: {OUTPUT_DIR}")
    print()
    
    # Cargar datos
    todos_datos = {}
    for grupo in GRUPOS:
        datos = cargar_y_preparar_datos(grupo)
        if datos:
            todos_datos[grupo] = datos
    
    if not todos_datos:
        print("\n❌ No hay datos para visualizar")
        return 1
    
    # Generar visualizaciones
    print(f"\n{'='*80}")
    print("  🎨 GENERANDO VISUALIZACIONES")
    print(f"{'='*80}")
    
    # 1. Scatter por método (3 paneles) para cada grupo
    for grupo in GRUPOS:
        if grupo in todos_datos:
            crear_scatter_por_metodo(todos_datos[grupo])
    
    # 2. Análisis de correcciones para cada grupo
    for grupo in GRUPOS:
        if grupo in todos_datos:
            crear_scatter_correcciones(todos_datos[grupo])
    
    # 3. Visualización unificada de 3 grupos
    if len(todos_datos) >= 2:
        crear_scatter_unificado(todos_datos)
    
    # Resumen
    print(f"\n{'='*80}")
    print("  ✅ VISUALIZACIONES COMPLETADAS")
    print(f"{'='*80}\n")
    print(f"📁 Archivos generados en: {OUTPUT_DIR}\n")
    
    for file in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"   - {file.name}")
    
    print("\n💡 Interpretación:")
    print("   - Puntos cercanos = características similares")
    print("   - Colores = clases predichas")
    print("   - Estrellas azules = casos donde DL corrigió errores de Tradicional")
    print("   - Cruces rojas = casos donde DL empeoró")
    print("   - Mayor separación de colores = mejor clasificación")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
