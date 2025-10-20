"""
Script de Entrenamiento para JÓVENES (12-17 años)

Este script entrena y compara 3 arquitecturas de deep learning:
1. TabNet
2. MLP + Multi-Head Attention  
3. FT-Transformer

Grupo: Jóvenes (12-17 años)
Features: 139
Registros: 14,021
Distribución: DESBALANCEADA ⚠️
    - Clase 0: 80.00%
    - Clase 1: 0.21% (MINORITARIA CRÍTICA)
    - Clase 2: 19.79%

Estrategia: Class Weights [0.5, 50.0, 2.0]

Universidad Peruana Unión
Fecha: Octubre 2025
"""

import sys
import os
from pathlib import Path
import time
import random
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar path del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.Jovenes import (
    create_tabnet_for_jovenes,
    create_mlp_attention_for_jovenes,
    create_ft_transformer_for_jovenes
)


# ===================================================================
# CONFIGURACIÓN
# ===================================================================

CONFIG = {
    'seed': 42,
    'grupo': 'jovenes',
    'num_features': 139,
    'num_classes': 3,
    'batch_size': 128,
    'epochs': 200,
    'patience': 30,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Class weights para imbalance (80% / 0.21% / 19.79%)
    'class_weights': [0.5, 50.0, 2.0],  # Penalización fuerte para clase 1 (minoritaria)
    
    # Paths
    'data_dir': project_root / 'data' / 'processed' / 'jovenes',
    'output_dir': project_root / 'results' / 'jovenes',
    'models_dir': project_root / 'results' / 'jovenes' / 'models',
    'plots_dir': project_root / 'results' / 'jovenes' / 'plots',
}


# ===================================================================
# UTILIDADES
# ===================================================================

def set_seed(seed: int = 42):
    """Fijar seeds para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"✅ Seeds fijados: {seed}")


def load_data():
    """Cargar datos procesados de jóvenes."""
    print("\n" + "="*70)
    print("📂 CARGANDO DATOS DE JÓVENES")
    print("="*70)
    
    data_dir = CONFIG['data_dir']
    
    # Cargar CSVs
    train_df = pd.read_csv(data_dir / 'jovenes_train.csv')
    val_df = pd.read_csv(data_dir / 'jovenes_val.csv')
    test_df = pd.read_csv(data_dir / 'jovenes_test.csv')
    
    # Separar features y target
    X_train = train_df.drop(columns=['target']).values
    y_train = train_df['target'].values
    
    X_val = val_df.drop(columns=['target']).values
    y_val = val_df['target'].values
    
    X_test = test_df.drop(columns=['target']).values
    y_test = test_df['target'].values
    
    print(f"   Train: {X_train.shape[0]:,} registros")
    print(f"   Val:   {X_val.shape[0]:,} registros")
    print(f"   Test:  {X_test.shape[0]:,} registros")
    print(f"   Features: {X_train.shape[1]}")
    
    # Verificar distribución (CRÍTICO para imbalance)
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n   ⚠️  Distribución de clases (train) - DESBALANCEADA:")
    for cls, count in zip(unique, counts):
        pct = 100 * count / len(y_train)
        marker = "🔴" if pct < 1 else "🟡" if pct < 30 else "🟢"
        print(f"      {marker} Clase {cls}: {count:,} ({pct:.2f}%)")
    
    print(f"\n   🎯 Class Weights aplicados: {CONFIG['class_weights']}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_dataloaders(X_train, y_train, X_val, y_val, use_sampler=False):
    """
    Crear DataLoaders de PyTorch con opción de sampler balanceado.
    
    Args:
        use_sampler: Si True, usa WeightedRandomSampler para balancear batches
    """
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    if use_sampler:
        # Calcular pesos por muestra
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG['batch_size'],
            sampler=sampler,
            num_workers=0
        )
        print("   ✅ DataLoader con WeightedRandomSampler activado")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            num_workers=0
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'] * 2,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


def evaluate_model(model_name, y_true, y_pred, y_proba=None):
    """Calcular métricas de evaluación (F1-Macro prioritario para imbalance)."""
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    # Métricas por clase (CRÍTICO para verificar desempeño en clase minoritaria)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    for i in range(len(f1_per_class)):
        metrics[f'f1_class_{i}'] = f1_per_class[i]
        metrics[f'precision_class_{i}'] = precision_per_class[i]
        metrics[f'recall_class_{i}'] = recall_per_class[i]
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Graficar matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
                xticklabels=['Clase 0 (80%)', 'Clase 1 (0.21%)', 'Clase 2 (20%)'],
                yticklabels=['Clase 0 (80%)', 'Clase 1 (0.21%)', 'Clase 2 (20%)'])
    plt.title(f'Matriz de Confusión - {model_name} (Jóvenes)\n⚠️ Dataset Desbalanceado')
    plt.ylabel('Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Matriz de confusión guardada: {save_path}")


def save_results(results, filename='resultados_jovenes.json'):
    """Guardar resultados en JSON."""
    output_path = CONFIG['output_dir'] / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Resultados guardados: {output_path}")


# ===================================================================
# ENTRENAMIENTO DE MODELOS
# ===================================================================

def train_tabnet(X_train, y_train, X_val, y_val, X_test, y_test):
    """Entrenar TabNet para Jóvenes con class weights."""
    print("\n" + "="*70)
    print("🎯 MODELO 1/3: TabNet + Class Weights")
    print("="*70)
    
    set_seed(CONFIG['seed'])
    start_time = time.time()
    
    # Crear modelo
    model = create_tabnet_for_jovenes(input_dim=CONFIG['num_features'])
    
    # Configurar class weights
    model.set_class_weights(CONFIG['class_weights'])
    print(f"   🎯 Class weights configurados: {CONFIG['class_weights']}")
    
    print("\n🏋️  Entrenando TabNet...")
    
    # Entrenar
    history = model.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_epochs=CONFIG['epochs'],
        patience=CONFIG['patience'],
        batch_size=CONFIG['batch_size']
    )
    
    # Predicciones
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Evaluar
    metrics = evaluate_model('TabNet', y_test, y_pred, y_proba)
    
    # Tiempo
    training_time = time.time() - start_time
    metrics['training_time'] = training_time
    
    # Guardar modelo
    model_path = CONFIG['models_dir'] / 'tabnet_jovenes.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    
    # Matriz de confusión
    cm_path = CONFIG['plots_dir'] / 'confusion_matrix_tabnet.png'
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(y_test, y_pred, 'TabNet', cm_path)
    
    # Mostrar resultados
    print(f"\n📊 Resultados TabNet:")
    print(f"   Accuracy:       {metrics['accuracy']:.4f}")
    print(f"   F1-Score Macro: {metrics['f1_macro']:.4f} ⭐ (MÉTRICA PRINCIPAL)")
    print(f"   Precision:      {metrics['precision_macro']:.4f}")
    print(f"   Recall:         {metrics['recall_macro']:.4f}")
    print(f"\n   📌 Desempeño por clase:")
    print(f"      Clase 0 (80%):   F1={metrics['f1_class_0']:.4f}")
    print(f"      Clase 1 (0.21%): F1={metrics['f1_class_1']:.4f} 🔴 CRÍTICA")
    print(f"      Clase 2 (20%):   F1={metrics['f1_class_2']:.4f}")
    print(f"   Tiempo:         {training_time:.2f}s")
    
    return metrics, model


def train_mlp_attention(X_train, y_train, X_val, y_val, X_test, y_test):
    """Entrenar MLP+Attention para Jóvenes con class weights."""
    print("\n" + "="*70)
    print("🧠 MODELO 2/3: MLP + Multi-Head Attention + Class Weights")
    print("="*70)
    
    set_seed(CONFIG['seed'])
    start_time = time.time()
    
    # Crear modelo y trainer
    model, trainer = create_mlp_attention_for_jovenes(num_features=CONFIG['num_features'])
    
    # Configurar class weights
    trainer.set_class_weights(CONFIG['class_weights'])
    print(f"   🎯 Class weights configurados: {CONFIG['class_weights']}")
    
    # Crear DataLoaders (con sampler balanceado)
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, 
        use_sampler=True  # Activar sampler para mejor balance
    )
    
    print("\n🏋️  Entrenando MLP+Attention...")
    
    # Entrenar
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=CONFIG['epochs'],
        patience=CONFIG['patience'],
        verbose=True
    )
    
    # Predicciones
    y_pred = trainer.predict(X_test)
    y_proba = trainer.predict_proba(X_test)
    
    # Evaluar
    metrics = evaluate_model('MLP+Attention', y_test, y_pred, y_proba)
    
    # Tiempo
    training_time = time.time() - start_time
    metrics['training_time'] = training_time
    
    # Guardar modelo
    model_path = CONFIG['models_dir'] / 'mlp_attention_jovenes.pth'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(model_path))
    
    # Matriz de confusión
    cm_path = CONFIG['plots_dir'] / 'confusion_matrix_mlp.png'
    plot_confusion_matrix(y_test, y_pred, 'MLP+Attention', cm_path)
    
    # Mostrar resultados
    print(f"\n📊 Resultados MLP+Attention:")
    print(f"   Accuracy:       {metrics['accuracy']:.4f}")
    print(f"   F1-Score Macro: {metrics['f1_macro']:.4f} ⭐ (MÉTRICA PRINCIPAL)")
    print(f"   Precision:      {metrics['precision_macro']:.4f}")
    print(f"   Recall:         {metrics['recall_macro']:.4f}")
    print(f"\n   📌 Desempeño por clase:")
    print(f"      Clase 0 (80%):   F1={metrics['f1_class_0']:.4f}")
    print(f"      Clase 1 (0.21%): F1={metrics['f1_class_1']:.4f} 🔴 CRÍTICA")
    print(f"      Clase 2 (20%):   F1={metrics['f1_class_2']:.4f}")
    print(f"   Tiempo:         {training_time:.2f}s")
    
    return metrics, trainer


def train_ft_transformer(X_train, y_train, X_val, y_val, X_test, y_test):
    """Entrenar FT-Transformer para Jóvenes con class weights."""
    print("\n" + "="*70)
    print("🤖 MODELO 3/3: FT-Transformer + Class Weights")
    print("="*70)
    
    set_seed(CONFIG['seed'])
    start_time = time.time()
    
    # Crear modelo y trainer
    model, trainer = create_ft_transformer_for_jovenes(num_features=CONFIG['num_features'])
    
    # Configurar class weights
    trainer.set_class_weights(CONFIG['class_weights'])
    print(f"   🎯 Class weights configurados: {CONFIG['class_weights']}")
    
    # Crear DataLoaders (con sampler balanceado)
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val,
        use_sampler=True  # Activar sampler para mejor balance
    )
    
    print("\n🏋️  Entrenando FT-Transformer...")
    
    # Entrenar
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=CONFIG['epochs'],
        patience=CONFIG['patience'],
        verbose=True
    )
    
    # Predicciones
    y_pred = trainer.predict(X_test)
    y_proba = trainer.predict_proba(X_test)
    
    # Evaluar
    metrics = evaluate_model('FT-Transformer', y_test, y_pred, y_proba)
    
    # Tiempo
    training_time = time.time() - start_time
    metrics['training_time'] = training_time
    
    # Guardar modelo
    model_path = CONFIG['models_dir'] / 'ft_transformer_jovenes.pth'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(model_path))
    
    # Matriz de confusión
    cm_path = CONFIG['plots_dir'] / 'confusion_matrix_ft.png'
    plot_confusion_matrix(y_test, y_pred, 'FT-Transformer', cm_path)
    
    # Mostrar resultados
    print(f"\n📊 Resultados FT-Transformer:")
    print(f"   Accuracy:       {metrics['accuracy']:.4f}")
    print(f"   F1-Score Macro: {metrics['f1_macro']:.4f} ⭐ (MÉTRICA PRINCIPAL)")
    print(f"   Precision:      {metrics['precision_macro']:.4f}")
    print(f"   Recall:         {metrics['recall_macro']:.4f}")
    print(f"\n   📌 Desempeño por clase:")
    print(f"      Clase 0 (80%):   F1={metrics['f1_class_0']:.4f}")
    print(f"      Clase 1 (0.21%): F1={metrics['f1_class_1']:.4f} 🔴 CRÍTICA")
    print(f"      Clase 2 (20%):   F1={metrics['f1_class_2']:.4f}")
    print(f"   Tiempo:         {training_time:.2f}s")
    
    return metrics, trainer


def compare_models(results):
    """Comparar los 3 modelos (priorizando F1-Macro para imbalance)."""
    print("\n" + "="*70)
    print("📊 COMPARACIÓN DE MODELOS - JÓVENES (DATASET DESBALANCEADO)")
    print("="*70)
    
    # Crear DataFrame para comparación
    df = pd.DataFrame(results)
    df = df.set_index('model')
    
    print("\n📈 Tabla Comparativa:")
    cols_to_show = ['accuracy', 'f1_macro', 'f1_class_0', 'f1_class_1', 'f1_class_2', 
                    'precision_macro', 'recall_macro', 'training_time']
    print(df[cols_to_show].to_string())
    
    # Determinar mejor modelo por métrica (F1-Macro es prioritario)
    print("\n🏆 Mejor modelo por métrica:")
    print(f"   F1-Score Macro: {df['f1_macro'].idxmax()} ({df['f1_macro'].max():.4f}) ⭐ PRINCIPAL")
    print(f"   Accuracy:       {df['accuracy'].idxmax()} ({df['accuracy'].max():.4f})")
    print(f"   Precision:      {df['precision_macro'].idxmax()} ({df['precision_macro'].max():.4f})")
    print(f"   Recall:         {df['recall_macro'].idxmax()} ({df['recall_macro'].max():.4f})")
    
    # Análisis especial de clase minoritaria (Clase 1: 0.21%)
    print(f"\n🔴 Desempeño en Clase 1 (0.21% - CRÍTICA):")
    for model in df.index:
        f1_minority = df.loc[model, 'f1_class_1']
        print(f"   {model}: F1={f1_minority:.4f}")
    
    # Ranking general (priorizando F1-Macro)
    df['score_weighted'] = (
        df['f1_macro'] * 0.5 +  # 50% peso a F1-Macro
        df['accuracy'] * 0.2 +   # 20% peso a Accuracy
        df['precision_macro'] * 0.15 +  # 15% peso a Precision
        df['recall_macro'] * 0.15        # 15% peso a Recall
    )
    df_sorted = df.sort_values('score_weighted', ascending=False)
    
    print("\n🥇 Ranking General (ponderado por F1-Macro):")
    for i, (model, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"   {i}. {model}: {row['score_weighted']:.4f}")
    
    # Guardar tabla comparativa
    comparison_path = CONFIG['output_dir'] / 'comparacion_modelos_jovenes.csv'
    df.to_csv(comparison_path)
    print(f"\n✅ Tabla comparativa guardada: {comparison_path}")
    
    return df


# ===================================================================
# MAIN
# ===================================================================

def main():
    """Función principal."""
    print("\n" + "="*70)
    print("🎓 ENTRENAMIENTO DE MODELOS - JÓVENES (12-17 AÑOS)")
    print("⚠️  DATASET DESBALANCEADO: 80% / 0.21% / 19.79%")
    print("="*70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {CONFIG['device']}")
    print(f"Seed: {CONFIG['seed']}")
    print(f"Features: {CONFIG['num_features']}")
    print(f"Class Weights: {CONFIG['class_weights']}")
    
    # Cargar datos
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Entrenar los 3 modelos
    results = []
    
    try:
        # 1. TabNet
        metrics_tabnet, model_tabnet = train_tabnet(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        results.append(metrics_tabnet)
        
        # 2. MLP+Attention
        metrics_mlp, trainer_mlp = train_mlp_attention(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        results.append(metrics_mlp)
        
        # 3. FT-Transformer
        metrics_ft, trainer_ft = train_ft_transformer(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        results.append(metrics_ft)
        
        # Comparar modelos
        comparison_df = compare_models(results)
        
        # Guardar resultados finales
        final_results = {
            'grupo': 'jovenes',
            'fecha': datetime.now().isoformat(),
            'config': CONFIG,
            'class_weights': CONFIG['class_weights'],
            'distribucion': '80% / 0.21% / 19.79%',
            'modelos': results,
            'mejor_modelo': comparison_df['score_weighted'].idxmax(),
            'mejor_f1_macro': comparison_df['f1_macro'].idxmax()
        }
        save_results(final_results)
        
        print("\n" + "="*70)
        print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*70)
        print(f"\n🏆 Mejor modelo (score ponderado): {final_results['mejor_modelo']}")
        print(f"⭐ Mejor F1-Macro: {final_results['mejor_f1_macro']}")
        print(f"📁 Resultados guardados en: {CONFIG['output_dir']}")
        
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
