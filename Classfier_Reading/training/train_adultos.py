"""
Script de Entrenamiento para ADULTOS (18+ años)

Este script entrena y compara 3 arquitecturas de deep learning:
1. TabNet
2. MLP + Multi-Head Attention  
3. FT-Transformer

Grupo: Adultos (18+ años)
Features: 71
Registros: 40,559
Distribución: EXTREMADAMENTE DESBALANCEADA 🔴
    - Clase 0: 20.00%
    - Clase 1: 0.02% (MINORITARIA EXTREMA)
    - Clase 2: 79.98%

Estrategia: Class Weights EXTREMOS [2.0, 200.0, 0.5]

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

from src.models.Adultos import (
    create_tabnet_for_adultos,
    create_mlp_attention_for_adultos,
    create_ft_transformer_for_adultos
)


# ===================================================================
# CONFIGURACIÓN
# ===================================================================

CONFIG = {
    'seed': 42,
    'grupo': 'adultos',
    'num_features': 71,
    'num_classes': 3,
    'batch_size': 256,  # Batch más grande para dataset grande
    'epochs': 200,
    'patience': 30,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Class weights EXTREMOS para imbalance severo (20% / 0.02% / 80%)
    'class_weights': [2.0, 200.0, 0.5],  # Penalización EXTREMA para clase 1 (0.02%)
    
    # Opción alternativa: Considerar clasificación binaria (fusionar clases 0 y 1)
    'binary_fallback': False,  # Cambiar a True si la clase 1 es impredecible
    
    # Paths
    'data_dir': project_root / 'data' / 'processed' / 'adultos',
    'output_dir': project_root / 'results' / 'adultos',
    'models_dir': project_root / 'results' / 'adultos' / 'models',
    'plots_dir': project_root / 'results' / 'adultos' / 'plots',
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
    """Cargar datos procesados de adultos."""
    print("\n" + "="*70)
    print("📂 CARGANDO DATOS DE ADULTOS")
    print("="*70)
    
    data_dir = CONFIG['data_dir']
    
    # Cargar CSVs
    train_df = pd.read_csv(data_dir / 'adultos_train.csv')
    val_df = pd.read_csv(data_dir / 'adultos_val.csv')
    test_df = pd.read_csv(data_dir / 'adultos_test.csv')
    
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
    
    # Verificar distribución (CRÍTICO - la más desbalanceada)
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n   🔴 Distribución de clases (train) - EXTREMADAMENTE DESBALANCEADA:")
    for cls, count in zip(unique, counts):
        pct = 100 * count / len(y_train)
        if pct < 0.1:
            marker = "🔴🔴🔴"  # Clase casi imposible de predecir
        elif pct < 1:
            marker = "🔴🔴"
        elif pct < 30:
            marker = "🟡"
        else:
            marker = "🟢"
        print(f"      {marker} Clase {cls}: {count:,} ({pct:.4f}%)")
    
    print(f"\n   🎯 Class Weights EXTREMOS: {CONFIG['class_weights']}")
    print(f"   ⚠️  Advertencia: Clase 1 tiene solo ~0.02% de los datos")
    print(f"      → Puede ser difícil de predecir correctamente")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_dataloaders(X_train, y_train, X_val, y_val, use_sampler=True):
    """
    Crear DataLoaders de PyTorch con sampler balanceado OBLIGATORIO.
    
    Args:
        use_sampler: OBLIGATORIO True para adultos (imbalance extremo)
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
        # Calcular pesos por muestra (crítico para 0.02%)
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
        print("   ✅ WeightedRandomSampler OBLIGATORIO activado (imbalance extremo)")
    else:
        print("   ⚠️  ADVERTENCIA: Sampler desactivado - NO RECOMENDADO para adultos")
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
    """Calcular métricas de evaluación (F1-Macro CRÍTICO para imbalance extremo)."""
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    # Métricas por clase (CRÍTICO - verificar si clase 1 es predecible)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    for i in range(len(f1_per_class)):
        metrics[f'f1_class_{i}'] = f1_per_class[i]
        metrics[f'precision_class_{i}'] = precision_per_class[i]
        metrics[f'recall_class_{i}'] = recall_per_class[i]
    
    # Conteo de predicciones por clase (diagnóstico)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    metrics['pred_distribution'] = {int(cls): int(cnt) for cls, cnt in zip(unique_pred, counts_pred)}
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Graficar matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
                xticklabels=['Clase 0 (20%)', 'Clase 1 (0.02%)', 'Clase 2 (80%)'],
                yticklabels=['Clase 0 (20%)', 'Clase 1 (0.02%)', 'Clase 2 (80%)'])
    plt.title(f'Matriz de Confusión - {model_name} (Adultos)\n🔴 Dataset EXTREMADAMENTE Desbalanceado')
    plt.ylabel('Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Matriz de confusión guardada: {save_path}")


def save_results(results, filename='resultados_adultos.json'):
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
    """Entrenar TabNet para Adultos con class weights EXTREMOS."""
    print("\n" + "="*70)
    print("🎯 MODELO 1/3: TabNet + Class Weights EXTREMOS")
    print("="*70)
    
    set_seed(CONFIG['seed'])
    start_time = time.time()
    
    # Crear modelo
    model = create_tabnet_for_adultos(input_dim=CONFIG['num_features'])
    
    # Configurar class weights EXTREMOS
    model.set_class_weights(CONFIG['class_weights'])
    print(f"   🎯 Class weights EXTREMOS: {CONFIG['class_weights']}")
    print(f"      → Clase 1 (0.02%) tiene peso 200.0 (vs 2.0 y 0.5)")
    
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
    model_path = CONFIG['models_dir'] / 'tabnet_adultos.pkl'
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
    print(f"      Clase 0 (20%):    F1={metrics['f1_class_0']:.4f}")
    print(f"      Clase 1 (0.02%):  F1={metrics['f1_class_1']:.4f} 🔴🔴🔴 EXTREMA")
    print(f"      Clase 2 (80%):    F1={metrics['f1_class_2']:.4f}")
    print(f"\n   📊 Distribución de predicciones: {metrics['pred_distribution']}")
    print(f"   Tiempo:         {training_time:.2f}s")
    
    return metrics, model


def train_mlp_attention(X_train, y_train, X_val, y_val, X_test, y_test):
    """Entrenar MLP+Attention para Adultos con class weights EXTREMOS."""
    print("\n" + "="*70)
    print("🧠 MODELO 2/3: MLP + Multi-Head Attention + Class Weights EXTREMOS")
    print("="*70)
    
    set_seed(CONFIG['seed'])
    start_time = time.time()
    
    # Crear modelo y trainer
    model, trainer = create_mlp_attention_for_adultos(num_features=CONFIG['num_features'])
    
    # Configurar class weights EXTREMOS
    trainer.set_class_weights(CONFIG['class_weights'])
    print(f"   🎯 Class weights EXTREMOS: {CONFIG['class_weights']}")
    print(f"      → Clase 1 (0.02%) tiene peso 200.0 (vs 2.0 y 0.5)")
    
    # Crear DataLoaders (sampler OBLIGATORIO)
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val,
        use_sampler=True  # OBLIGATORIO para imbalance extremo
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
    model_path = CONFIG['models_dir'] / 'mlp_attention_adultos.pth'
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
    print(f"      Clase 0 (20%):    F1={metrics['f1_class_0']:.4f}")
    print(f"      Clase 1 (0.02%):  F1={metrics['f1_class_1']:.4f} 🔴🔴🔴 EXTREMA")
    print(f"      Clase 2 (80%):    F1={metrics['f1_class_2']:.4f}")
    print(f"\n   📊 Distribución de predicciones: {metrics['pred_distribution']}")
    print(f"   Tiempo:         {training_time:.2f}s")
    
    return metrics, trainer


def train_ft_transformer(X_train, y_train, X_val, y_val, X_test, y_test):
    """Entrenar FT-Transformer para Adultos con class weights EXTREMOS."""
    print("\n" + "="*70)
    print("🤖 MODELO 3/3: FT-Transformer + Class Weights EXTREMOS")
    print("="*70)
    
    set_seed(CONFIG['seed'])
    start_time = time.time()
    
    # Crear modelo y trainer
    model, trainer = create_ft_transformer_for_adultos(num_features=CONFIG['num_features'])
    
    # Configurar class weights EXTREMOS
    trainer.set_class_weights(CONFIG['class_weights'])
    print(f"   🎯 Class weights EXTREMOS: {CONFIG['class_weights']}")
    print(f"      → Clase 1 (0.02%) tiene peso 200.0 (vs 2.0 y 0.5)")
    
    # Crear DataLoaders (sampler OBLIGATORIO)
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val,
        use_sampler=True  # OBLIGATORIO para imbalance extremo
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
    model_path = CONFIG['models_dir'] / 'ft_transformer_adultos.pth'
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
    print(f"      Clase 0 (20%):    F1={metrics['f1_class_0']:.4f}")
    print(f"      Clase 1 (0.02%):  F1={metrics['f1_class_1']:.4f} 🔴🔴🔴 EXTREMA")
    print(f"      Clase 2 (80%):    F1={metrics['f1_class_2']:.4f}")
    print(f"\n   📊 Distribución de predicciones: {metrics['pred_distribution']}")
    print(f"   Tiempo:         {training_time:.2f}s")
    
    return metrics, trainer


def compare_models(results):
    """Comparar los 3 modelos (F1-Macro CRÍTICO para imbalance extremo)."""
    print("\n" + "="*70)
    print("📊 COMPARACIÓN DE MODELOS - ADULTOS (IMBALANCE EXTREMO)")
    print("="*70)
    
    # Crear DataFrame para comparación
    df = pd.DataFrame(results)
    df = df.set_index('model')
    
    print("\n📈 Tabla Comparativa:")
    cols_to_show = ['accuracy', 'f1_macro', 'f1_class_0', 'f1_class_1', 'f1_class_2',
                    'precision_macro', 'recall_macro', 'training_time']
    print(df[cols_to_show].to_string())
    
    # Determinar mejor modelo por métrica (F1-Macro es CRÍTICO)
    print("\n🏆 Mejor modelo por métrica:")
    print(f"   F1-Score Macro: {df['f1_macro'].idxmax()} ({df['f1_macro'].max():.4f}) ⭐⭐ CRÍTICO")
    print(f"   Accuracy:       {df['accuracy'].idxmax()} ({df['accuracy'].max():.4f})")
    print(f"   Precision:      {df['precision_macro'].idxmax()} ({df['precision_macro'].max():.4f})")
    print(f"   Recall:         {df['recall_macro'].idxmax()} ({df['recall_macro'].max():.4f})")
    
    # Análisis CRÍTICO de clase minoritaria extrema (Clase 1: 0.02%)
    print(f"\n🔴🔴🔴 Desempeño en Clase 1 (0.02% - EXTREMADAMENTE CRÍTICA):")
    for model in df.index:
        f1_minority = df.loc[model, 'f1_class_1']
        recall_minority = df.loc[model, 'recall_class_1']
        precision_minority = df.loc[model, 'precision_class_1']
        print(f"   {model}:")
        print(f"      F1={f1_minority:.4f} | Recall={recall_minority:.4f} | Precision={precision_minority:.4f}")
    
    # Advertencia si ningún modelo predice clase 1
    if df['f1_class_1'].max() < 0.01:
        print(f"\n   ⚠️  ADVERTENCIA CRÍTICA: Ningún modelo puede predecir Clase 1")
        print(f"      → Considerar clasificación binaria (fusionar clases 0 y 1)")
        print(f"      → O usar técnicas de oversampling (SMOTE, ADASYN)")
    
    # Ranking general (MÁXIMA prioridad a F1-Macro)
    df['score_weighted'] = (
        df['f1_macro'] * 0.6 +  # 60% peso a F1-Macro (MÁS que jovenes)
        df['accuracy'] * 0.15 +  # 15% peso a Accuracy
        df['precision_macro'] * 0.125 +  # 12.5% peso a Precision
        df['recall_macro'] * 0.125       # 12.5% peso a Recall
    )
    df_sorted = df.sort_values('score_weighted', ascending=False)
    
    print("\n🥇 Ranking General (ponderado FUERTEMENTE por F1-Macro):")
    for i, (model, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"   {i}. {model}: {row['score_weighted']:.4f}")
    
    # Guardar tabla comparativa
    comparison_path = CONFIG['output_dir'] / 'comparacion_modelos_adultos.csv'
    df.to_csv(comparison_path)
    print(f"\n✅ Tabla comparativa guardada: {comparison_path}")
    
    return df


# ===================================================================
# MAIN
# ===================================================================

def main():
    """Función principal."""
    print("\n" + "="*70)
    print("🎓 ENTRENAMIENTO DE MODELOS - ADULTOS (18+ AÑOS)")
    print("🔴 DATASET EXTREMADAMENTE DESBALANCEADO: 20% / 0.02% / 80%")
    print("="*70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {CONFIG['device']}")
    print(f"Seed: {CONFIG['seed']}")
    print(f"Features: {CONFIG['num_features']}")
    print(f"Class Weights EXTREMOS: {CONFIG['class_weights']}")
    print(f"\n⚠️  ADVERTENCIA: Clase 1 tiene solo 0.02% de los datos")
    print(f"   → Predicción extremadamente difícil")
    print(f"   → F1-Macro es la métrica más importante")
    
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
            'grupo': 'adultos',
            'fecha': datetime.now().isoformat(),
            'config': CONFIG,
            'class_weights': CONFIG['class_weights'],
            'distribucion': '20% / 0.02% / 79.98%',
            'warning': 'Clase 1 extremadamente minoritaria (0.02%)',
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
        
        # Recomendaciones finales
        print(f"\n💡 Recomendaciones para la tesis:")
        print(f"   1. Enfatizar F1-Macro como métrica principal")
        print(f"   2. Discutir dificultad de predecir clase minoritaria (0.02%)")
        print(f"   3. Comparar con enfoque binario si F1(clase 1) < 0.05")
        print(f"   4. Considerar SMOTE/ADASYN como trabajo futuro")
        
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
