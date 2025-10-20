"""
Script de Entrenamiento para NIÑOS (0-11 años)

Este script entrena y compara 3 arquitecturas de deep learning:
1. TabNet
2. MLP + Multi-Head Attention  
3. FT-Transformer

Grupo: Niños (0-11 años)
Features: 131
Registros: 10,388
Distribución: Balanceada ✅

Características:
- Mismo seed aleatorio para reproducibilidad
- Mismo preprocesamiento
- Condiciones idénticas de entrenamiento
- Comparación justa entre arquitecturas

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
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar path del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.Ninos import (
    create_tabnet_for_ninos,
    create_mlp_attention_for_ninos,
    create_ft_transformer_for_ninos
)


# ===================================================================
# CONFIGURACIÓN
# ===================================================================

CONFIG = {
    'seed': 42,
    'grupo': 'ninos',
    'num_features': 131,
    'num_classes': 3,
    'batch_size': 128,
    'epochs': 200,
    'patience': 30,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Paths
    'data_dir': project_root / 'data' / 'processed' / 'ninos',
    'output_dir': project_root / 'results' / 'ninos',
    'models_dir': project_root / 'results' / 'ninos' / 'models',
    'plots_dir': project_root / 'results' / 'ninos' / 'plots',
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
    """Cargar datos procesados de niños."""
    print("\n" + "="*70)
    print("📂 CARGANDO DATOS DE NIÑOS")
    print("="*70)
    
    data_dir = CONFIG['data_dir']
    
    # Cargar CSVs
    train_df = pd.read_csv(data_dir / 'ninos_train.csv')
    val_df = pd.read_csv(data_dir / 'ninos_val.csv')
    test_df = pd.read_csv(data_dir / 'ninos_test.csv')
    
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
    
    # Verificar distribución
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n   Distribución de clases (train):")
    for cls, count in zip(unique, counts):
        pct = 100 * count / len(y_train)
        print(f"      Clase {cls}: {count:,} ({pct:.2f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_dataloaders(X_train, y_train, X_val, y_val):
    """Crear DataLoaders de PyTorch."""
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
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
    """Calcular métricas de evaluación."""
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
    }
    
    # Métricas por clase
    f1_per_class = f1_score(y_true, y_pred, average=None)
    for i, f1 in enumerate(f1_per_class):
        metrics[f'f1_class_{i}'] = f1
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Graficar matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Clase 0', 'Clase 1', 'Clase 2'],
                yticklabels=['Clase 0', 'Clase 1', 'Clase 2'])
    plt.title(f'Matriz de Confusión - {model_name} (Niños)')
    plt.ylabel('Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Matriz de confusión guardada: {save_path}")


def save_results(results, filename='resultados_ninos.json'):
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
    """Entrenar TabNet para Niños."""
    print("\n" + "="*70)
    print("🎯 MODELO 1/3: TabNet")
    print("="*70)
    
    set_seed(CONFIG['seed'])
    start_time = time.time()
    
    # Crear modelo
    model = create_tabnet_for_ninos(input_dim=CONFIG['num_features'])
    
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
    model_path = CONFIG['models_dir'] / 'tabnet_ninos.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    
    # Matriz de confusión
    cm_path = CONFIG['plots_dir'] / 'confusion_matrix_tabnet.png'
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(y_test, y_pred, 'TabNet', cm_path)
    
    # Mostrar resultados
    print(f"\n📊 Resultados TabNet:")
    print(f"   Accuracy:       {metrics['accuracy']:.4f}")
    print(f"   F1-Score Macro: {metrics['f1_macro']:.4f}")
    print(f"   Precision:      {metrics['precision_macro']:.4f}")
    print(f"   Recall:         {metrics['recall_macro']:.4f}")
    print(f"   Tiempo:         {training_time:.2f}s")
    
    return metrics, model


def train_mlp_attention(X_train, y_train, X_val, y_val, X_test, y_test):
    """Entrenar MLP+Attention para Niños."""
    print("\n" + "="*70)
    print("🧠 MODELO 2/3: MLP + Multi-Head Attention")
    print("="*70)
    
    set_seed(CONFIG['seed'])
    start_time = time.time()
    
    # Crear modelo y trainer
    model, trainer = create_mlp_attention_for_ninos(num_features=CONFIG['num_features'])
    
    # Crear DataLoaders
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val)
    
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
    model_path = CONFIG['models_dir'] / 'mlp_attention_ninos.pth'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(model_path))
    
    # Matriz de confusión
    cm_path = CONFIG['plots_dir'] / 'confusion_matrix_mlp.png'
    plot_confusion_matrix(y_test, y_pred, 'MLP+Attention', cm_path)
    
    # Mostrar resultados
    print(f"\n📊 Resultados MLP+Attention:")
    print(f"   Accuracy:       {metrics['accuracy']:.4f}")
    print(f"   F1-Score Macro: {metrics['f1_macro']:.4f}")
    print(f"   Precision:      {metrics['precision_macro']:.4f}")
    print(f"   Recall:         {metrics['recall_macro']:.4f}")
    print(f"   Tiempo:         {training_time:.2f}s")
    
    return metrics, trainer


def train_ft_transformer(X_train, y_train, X_val, y_val, X_test, y_test):
    """Entrenar FT-Transformer para Niños."""
    print("\n" + "="*70)
    print("🤖 MODELO 3/3: FT-Transformer")
    print("="*70)
    
    set_seed(CONFIG['seed'])
    start_time = time.time()
    
    # Crear modelo y trainer
    model, trainer = create_ft_transformer_for_ninos(num_features=CONFIG['num_features'])
    
    # Crear DataLoaders
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val)
    
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
    model_path = CONFIG['models_dir'] / 'ft_transformer_ninos.pth'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(str(model_path))
    
    # Matriz de confusión
    cm_path = CONFIG['plots_dir'] / 'confusion_matrix_ft.png'
    plot_confusion_matrix(y_test, y_pred, 'FT-Transformer', cm_path)
    
    # Mostrar resultados
    print(f"\n📊 Resultados FT-Transformer:")
    print(f"   Accuracy:       {metrics['accuracy']:.4f}")
    print(f"   F1-Score Macro: {metrics['f1_macro']:.4f}")
    print(f"   Precision:      {metrics['precision_macro']:.4f}")
    print(f"   Recall:         {metrics['recall_macro']:.4f}")
    print(f"   Tiempo:         {training_time:.2f}s")
    
    return metrics, trainer


def compare_models(results):
    """Comparar los 3 modelos y determinar el mejor."""
    print("\n" + "="*70)
    print("📊 COMPARACIÓN DE MODELOS - NIÑOS")
    print("="*70)
    
    # Crear DataFrame para comparación
    df = pd.DataFrame(results)
    df = df.set_index('model')
    
    print("\n📈 Tabla Comparativa:")
    print(df[['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'training_time']].to_string())
    
    # Determinar mejor modelo por métrica
    print("\n🏆 Mejor modelo por métrica:")
    print(f"   Accuracy:       {df['accuracy'].idxmax()} ({df['accuracy'].max():.4f})")
    print(f"   F1-Score Macro: {df['f1_macro'].idxmax()} ({df['f1_macro'].max():.4f})")
    print(f"   Precision:      {df['precision_macro'].idxmax()} ({df['precision_macro'].max():.4f})")
    print(f"   Recall:         {df['recall_macro'].idxmax()} ({df['recall_macro'].max():.4f})")
    
    # Ranking general (promedio de métricas)
    df['score_avg'] = (df['accuracy'] + df['f1_macro'] + df['precision_macro'] + df['recall_macro']) / 4
    df_sorted = df.sort_values('score_avg', ascending=False)
    
    print("\n🥇 Ranking General:")
    for i, (model, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"   {i}. {model}: {row['score_avg']:.4f}")
    
    # Guardar tabla comparativa
    comparison_path = CONFIG['output_dir'] / 'comparacion_modelos_ninos.csv'
    df.to_csv(comparison_path)
    print(f"\n✅ Tabla comparativa guardada: {comparison_path}")
    
    return df


# ===================================================================
# MAIN
# ===================================================================

def main():
    """Función principal."""
    print("\n" + "="*70)
    print("🎓 ENTRENAMIENTO DE MODELOS - NIÑOS (0-11 AÑOS)")
    print("="*70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {CONFIG['device']}")
    print(f"Seed: {CONFIG['seed']}")
    print(f"Features: {CONFIG['num_features']}")
    
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
            'grupo': 'ninos',
            'fecha': datetime.now().isoformat(),
            'config': CONFIG,
            'modelos': results,
            'mejor_modelo': comparison_df['score_avg'].idxmax()
        }
        save_results(final_results)
        
        print("\n" + "="*70)
        print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*70)
        print(f"\n🏆 Mejor modelo: {final_results['mejor_modelo']}")
        print(f"📁 Resultados guardados en: {CONFIG['output_dir']}")
        
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
