"""
Generador de Clasificaciones Sintéticas 3×3

Este script genera datos clasificados sintéticos para probar el sistema de clasificación
sin necesidad de entrenar los modelos. Simula la salida de 9 clasificadores.

Genera:
- 9 carpetas (3 grupos × 3 modelos)
- 4 archivos CSV por clasificador:
  * predictions.csv: Predicciones con probabilidades
  * summary.csv: Resumen de distribución por clase
  * high_confidence.csv: Predicciones con alta confianza (>80%)
  * low_confidence.csv: Predicciones con baja confianza (<60%)

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
from datetime import datetime

# ===================================================================
# CONFIGURACIÓN
# ===================================================================

# Paths
classfier_root = Path(__file__).parent
DATA_DIR = classfier_root / 'data' / 'processed_fase2'
OUTPUT_DIR = classfier_root / 'outputs' / 'classifications_3x3'

# Grupos y modelos
GRUPOS = ['ninos', 'jovenes', 'adultos']
MODELOS = ['ft_transformer', 'tabnet', 'mlp_attention']

# Nombres de clases
CLASS_NAMES = {
    0: 'No Lector',
    1: 'Lector Medio',
    2: 'Lector Frecuente'
}

# Seed para reproducibilidad
SEED = 42
np.random.seed(SEED)

# ===================================================================
# FUNCIONES
# ===================================================================

def print_banner(text, char="="):
    """Imprimir banner decorativo."""
    print("\n" + char * 80)
    print(f"  {text}")
    print(char * 80)


def generate_predictions(n_samples, grupo, modelo):
    """
    Generar predicciones sintéticas basadas en características del grupo y modelo.
    
    Cada combinación grupo-modelo tiene diferentes características:
    - Accuracy base varía por modelo
    - Distribución de clases varía por grupo
    - Confianza varía por modelo
    """
    
    # Accuracies base por modelo (mejorado para jóvenes)
    if grupo == 'jovenes':
        model_acc = {
            'ft_transformer': 0.82,
            'tabnet': 0.84,
            'mlp_attention': 0.80
        }
    else:
        model_acc = {
            'ft_transformer': 0.85,
            'tabnet': 0.80,
            'mlp_attention': 0.78
        }
    
    # Distribución de clases por grupo (probabilidades)
    class_dist = {
        'ninos': [0.4, 0.35, 0.25],      # Más No Lectores
        'jovenes': [0.35, 0.35, 0.30],   # Más balanceado
        'adultos': [0.25, 0.3, 0.45]     # Más Lectores Frecuentes
    }
    
    # Nivel de confianza por modelo
    confidence_level = {
        'ft_transformer': 0.15,  # Alta confianza (std baja)
        'tabnet': 0.20,          # Media confianza
        'mlp_attention': 0.25    # Baja confianza (más variabilidad)
    }
    
    base_acc = model_acc[modelo]
    dist = class_dist[grupo]
    conf_std = confidence_level[modelo]
    
    # Generar clases verdaderas según distribución del grupo
    y_true = np.random.choice([0, 1, 2], size=n_samples, p=dist)
    
    # Generar predicciones (con accuracy definida)
    y_pred = y_true.copy()
    n_errors = int(n_samples * (1 - base_acc))
    error_indices = np.random.choice(n_samples, size=n_errors, replace=False)
    
    for idx in error_indices:
        # Cambiar a una clase incorrecta
        true_class = y_true[idx]
        other_classes = [c for c in [0, 1, 2] if c != true_class]
        y_pred[idx] = np.random.choice(other_classes)
    
    # Generar probabilidades
    probabilities = np.zeros((n_samples, 3))
    
    for i in range(n_samples):
        pred_class = y_pred[i]
        
        # Probabilidad alta para la clase predicha
        if y_pred[i] == y_true[i]:  # Predicción correcta
            main_prob = np.random.normal(0.85, conf_std)
        else:  # Predicción incorrecta
            main_prob = np.random.normal(0.60, conf_std)
        
        main_prob = np.clip(main_prob, 0.4, 0.99)
        
        # Distribuir resto entre otras clases
        remaining = 1.0 - main_prob
        other_probs = np.random.dirichlet([1, 1])
        other_probs = other_probs * remaining
        
        # Asignar probabilidades
        prob_vector = np.zeros(3)
        prob_vector[pred_class] = main_prob
        
        other_idx = 0
        for j in range(3):
            if j != pred_class:
                prob_vector[j] = other_probs[other_idx]
                other_idx += 1
        
        probabilities[i] = prob_vector
    
    return y_true, y_pred, probabilities


def create_classification_files(grupo, modelo, y_true, y_pred, probabilities, data_df):
    """Crear los 4 archivos CSV de salida."""
    
    output_path = OUTPUT_DIR / f'{grupo}_{modelo}'
    output_path.mkdir(parents=True, exist_ok=True)
    
    n_samples = len(y_pred)
    
    # 1. predictions.csv - Archivo principal con todas las predicciones
    predictions_df = data_df.copy()
    predictions_df['true_label'] = y_true
    predictions_df['predicted_label'] = y_pred
    predictions_df['predicted_class'] = [CLASS_NAMES[p] for p in y_pred]
    predictions_df['prob_no_lector'] = probabilities[:, 0]
    predictions_df['prob_lector_medio'] = probabilities[:, 1]
    predictions_df['prob_lector_frecuente'] = probabilities[:, 2]
    predictions_df['max_probability'] = probabilities.max(axis=1)
    predictions_df['is_correct'] = (y_true == y_pred)
    
    predictions_path = output_path / 'predictions.csv'
    predictions_df.to_csv(predictions_path, index=False, encoding='utf-8')
    
    # 2. summary.csv - Resumen de distribución
    summary_data = []
    for class_idx in [0, 1, 2]:
        class_name = CLASS_NAMES[class_idx]
        n_predicted = (y_pred == class_idx).sum()
        n_true = (y_true == class_idx).sum()
        n_correct = ((y_pred == class_idx) & (y_true == class_idx)).sum()
        
        precision = n_correct / n_predicted if n_predicted > 0 else 0
        recall = n_correct / n_true if n_true > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        summary_data.append({
            'class': class_name,
            'class_id': class_idx,
            'n_predicted': n_predicted,
            'n_true': n_true,
            'n_correct': n_correct,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
    
    # Añadir métricas globales
    accuracy = (y_pred == y_true).mean()
    summary_data.append({
        'class': 'OVERALL',
        'class_id': -1,
        'n_predicted': n_samples,
        'n_true': n_samples,
        'n_correct': (y_pred == y_true).sum(),
        'precision': accuracy,
        'recall': accuracy,
        'f1_score': accuracy
    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_path / 'summary.csv'
    summary_df.to_csv(summary_path, index=False, encoding='utf-8')
    
    # 3. high_confidence.csv - Predicciones con alta confianza (>80%)
    max_probs = probabilities.max(axis=1)
    high_conf_mask = max_probs > 0.80
    high_conf_df = predictions_df[high_conf_mask].copy()
    high_conf_path = output_path / 'high_confidence.csv'
    high_conf_df.to_csv(high_conf_path, index=False, encoding='utf-8')
    
    # 4. low_confidence.csv - Predicciones con baja confianza (<60%)
    low_conf_mask = max_probs < 0.60
    low_conf_df = predictions_df[low_conf_mask].copy()
    low_conf_path = output_path / 'low_confidence.csv'
    low_conf_df.to_csv(low_conf_path, index=False, encoding='utf-8')
    
    return {
        'grupo': grupo,
        'modelo': modelo,
        'n_samples': n_samples,
        'accuracy': float(accuracy),
        'high_confidence': high_conf_mask.sum(),
        'low_confidence': low_conf_mask.sum(),
        'files': {
            'predictions': str(predictions_path),
            'summary': str(summary_path),
            'high_confidence': str(high_conf_path),
            'low_confidence': str(low_conf_path)
        }
    }


def generate_all_classifications():
    """Generar clasificaciones para todas las combinaciones 3×3."""
    
    print_banner("🎲 GENERADOR DE CLASIFICACIONES SINTÉTICAS 3×3", "█")
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌱 Seed: {SEED}")
    print(f"📁 Salida: {OUTPUT_DIR}")
    
    all_results = []
    total_files = 0
    
    for grupo in GRUPOS:
        print_banner(f"📊 GRUPO: {grupo.upper()}", "=")
        
        # Cargar datos del grupo
        data_path = DATA_DIR / grupo / f'{grupo}_test.csv'
        
        if not data_path.exists():
            print(f"❌ No se encontró: {data_path}")
            continue
        
        data_df = pd.read_csv(data_path)
        
        # Separar features y target
        if 'target' in data_df.columns:
            y_true_real = data_df['target'].values
            X_df = data_df.drop(columns=['target'])
        else:
            print(f"⚠️  No hay columna 'target' en {data_path}, usando sintético")
            y_true_real = None
            X_df = data_df
        
        n_samples = len(data_df)
        print(f"   Registros: {n_samples:,}")
        
        for modelo in MODELOS:
            print(f"\n   🔹 Modelo: {modelo}")
            
            # Generar predicciones sintéticas
            y_true, y_pred, probabilities = generate_predictions(n_samples, grupo, modelo)
            
            # Crear archivos de clasificación
            result = create_classification_files(
                grupo, modelo, y_true, y_pred, probabilities, X_df
            )
            
            all_results.append(result)
            total_files += 4
            
            print(f"      ✅ Accuracy: {result['accuracy']:.4f}")
            print(f"      📊 Alta confianza: {result['high_confidence']:,} ({result['high_confidence']/n_samples*100:.1f}%)")
            print(f"      ⚠️  Baja confianza: {result['low_confidence']:,} ({result['low_confidence']/n_samples*100:.1f}%)")
    
    # Resumen final
    print_banner("📋 RESUMEN FINAL", "█")
    print(f"✅ Clasificaciones generadas: {len(all_results)}/9")
    print(f"📁 Archivos creados: {total_files}")
    print(f"💾 Directorio: {OUTPUT_DIR}")
    
    print("\n📊 ACCURACIES POR MODELO:")
    for modelo in MODELOS:
        modelo_results = [r for r in all_results if r['modelo'] == modelo]
        avg_acc = np.mean([r['accuracy'] for r in modelo_results])
        print(f"   {modelo:20s}: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    
    print("\n📊 ACCURACIES POR GRUPO:")
    for grupo in GRUPOS:
        grupo_results = [r for r in all_results if r['grupo'] == grupo]
        avg_acc = np.mean([r['accuracy'] for r in grupo_results])
        print(f"   {grupo.capitalize():20s}: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
    
    print("\n📂 ESTRUCTURA DE SALIDA:")
    print("   outputs/classifications_3x3/")
    for grupo in GRUPOS:
        for modelo in MODELOS:
            print(f"      ├── {grupo}_{modelo}/")
            print(f"      │   ├── predictions.csv        (Todas las predicciones)")
            print(f"      │   ├── summary.csv            (Resumen por clase)")
            print(f"      │   ├── high_confidence.csv    (>80% confianza)")
            print(f"      │   └── low_confidence.csv     (<60% confianza)")
    
    print("\n💡 USO:")
    print("   Los archivos generados simulan la salida del clasificador real.")
    print("   Puedes usarlos para:")
    print("   - Probar visualizaciones")
    print("   - Validar pipelines de análisis")
    print("   - Desarrollar reportes sin esperar entrenamiento")
    print("   - Comparar con resultados reales cuando estén listos")
    
    print_banner("✅ GENERACIÓN COMPLETADA", "█")


# ===================================================================
# MAIN
# ===================================================================

def main():
    """Ejecutar generación de clasificaciones sintéticas."""
    try:
        generate_all_classifications()
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
