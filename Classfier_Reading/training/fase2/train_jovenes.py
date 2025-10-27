"""
Script de Entrenamiento FASE 2 - Jóvenes
Solo Variables Demográficas y Socioculturales

Entrena 3 modelos de Deep Learning:
1. TabNet
2. MLP con Attention
3. FT-Transformer

Dataset: 79 features (SOLO demográficas, SIN variables de lectura)
Objetivo: Demostrar capacidad predictiva de DL con datos censales

Universidad Peruana Unión - 2025
"""

import sys
from pathlib import Path

# Agregar path del proyecto
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "MLP_MA"))

# Configurar UTF-8 para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Importar modelos del proyecto existente
from models.mlp_ma import MLPWithMultiheadAttention
from models.ft_transformer import FTTransformer
from models.trainer import Trainer
from utils.seeds import set_seed
from utils.metrics import calculate_metrics

# Importar TabNet
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    print("⚠️  pytorch-tabnet no instalado. Instalando...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet"])
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True


# ===================================================================
# CONFIGURACIÓN
# ===================================================================

CONFIG = {
    'grupo': 'jovenes',
    'fase': 2,
    'descripcion': 'Solo demográficas (79 features)',
    'seed': 42,
    'epochs': 100,
    'batch_size': 128,
    'learning_rate': 0.001,
    'patience': 15,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

PATHS = {
    'data': project_root / 'Classfier_Reading' / 'data' / 'processed_fase2' / 'jovenes',
    'output': project_root / 'Classfier_Reading' / 'outputs' / 'fase2' / 'jovenes'
}


# ===================================================================
# CARGA DE DATOS
# ===================================================================

def load_data():
    """Cargar datos de Fase 2 (solo demográficas)."""
    print("="*70)
    print(f"📂 CARGANDO DATOS - FASE 2: JÓVENES")
    print("="*70)
    print(f"Fuente: {PATHS['data']}")
    print(f"Características: 79 features (SOLO demográficas)")
    
    # Cargar CSVs
    train_df = pd.read_csv(PATHS['data'] / 'jovenes_train.csv')
    val_df = pd.read_csv(PATHS['data'] / 'jovenes_val.csv')
    test_df = pd.read_csv(PATHS['data'] / 'jovenes_test.csv')
    
    # Separar features y target
    X_train = train_df.drop(columns=['target']).values
    y_train = train_df['target'].values
    
    X_val = val_df.drop(columns=['target']).values
    y_val = val_df['target'].values
    
    X_test = test_df.drop(columns=['target']).values
    y_test = test_df['target'].values
    
    print(f"\n✅ Datos cargados:")
    print(f"   Train: {X_train.shape[0]:,} × {X_train.shape[1]} features")
    print(f"   Val:   {X_val.shape[0]:,} × {X_val.shape[1]} features")
    print(f"   Test:  {X_test.shape[0]:,} × {X_test.shape[1]} features")
    
    print(f"\n📊 Distribución de clases (Train):")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        pct = count / len(y_train) * 100
        print(f"   Clase {int(cls)}: {count:,} ({pct:.2f}%)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# ===================================================================
# MODELO 1: TabNet
# ===================================================================

def train_tabnet(X_train, y_train, X_val, y_val, X_test, y_test):
    """Entrenar TabNet."""
    print("\n" + "="*70)
    print("🔹 MODELO 1: TabNet")
    print("="*70)
    
    start_time = time.time()
    
    # Configuración TabNet
    tabnet_params = {
        "n_d": 32,
        "n_a": 32,
        "n_steps": 3,
        "gamma": 1.3,
        "cat_idxs": [],
        "cat_dims": [],
        "cat_emb_dim": 1,
        "lambda_sparse": 1e-3,
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": dict(lr=CONFIG['learning_rate']),
        "scheduler_params": {"step_size": 10, "gamma": 0.9},
        "scheduler_fn": torch.optim.lr_scheduler.StepLR,
        "mask_type": "entmax",
        "device_name": CONFIG['device'],
        "seed": CONFIG['seed']
    }
    
    # Crear modelo
    clf = TabNetClassifier(**tabnet_params)
    
    # Entrenar
    print(f"Entrenando TabNet...")
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=['val'],
        eval_metric=['accuracy'],
        max_epochs=CONFIG['epochs'],
        patience=CONFIG['patience'],
        batch_size=CONFIG['batch_size'],
        virtual_batch_size=32,
        num_workers=0,
        drop_last=False
    )
    
    # Evaluar
    y_pred_train = clf.predict(X_train)
    y_pred_val = clf.predict(X_val)
    y_pred_test = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_val, y_pred_val)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    train_f1 = f1_score(y_train, y_pred_train, average='macro')
    val_f1 = f1_score(y_val, y_pred_val, average='macro')
    test_f1 = f1_score(y_test, y_pred_test, average='macro')
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ TabNet entrenado en {elapsed/60:.2f} min")
    print(f"   Train: Acc={train_acc:.4f}, F1={train_f1:.4f}")
    print(f"   Val:   Acc={val_acc:.4f}, F1={val_f1:.4f}")
    print(f"   Test:  Acc={test_acc:.4f}, F1={test_f1:.4f}")
    
    # Guardar modelo
    model_path = PATHS['output'] / 'models' / 'tabnet'
    model_path.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(model_path / 'tabnet_best'))
    
    results = {
        'model': 'TabNet',
        'train_acc': float(train_acc),
        'val_acc': float(val_acc),
        'test_acc': float(test_acc),
        'train_f1': float(train_f1),
        'val_f1': float(val_f1),
        'test_f1': float(test_f1),
        'training_time_min': elapsed / 60,
        'epochs_trained': len(clf.history['loss'])
    }
    
    return results


# ===================================================================
# MODELO 2: MLP + Attention
# ===================================================================

def train_mlp_attention(X_train, y_train, X_val, y_val, X_test, y_test):
    """Entrenar MLP con Attention."""
    print("\n" + "="*70)
    print("🔹 MODELO 2: MLP + Multihead Attention")
    print("="*70)
    
    start_time = time.time()
    
    # Convertir a tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    # Crear datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Crear modelo
    input_dim = X_train.shape[1]
    model = MLPWithMultiheadAttention(
        input_dim=input_dim,
        hidden_dims=[128, 64],
        num_classes=3,
        num_heads=4,
        dropout=0.2
    )
    
    device = torch.device(CONFIG['device'])
    model = model.to(device)
    
    # Entrenar
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"Entrenando MLP+Attention...")
    for epoch in range(CONFIG['epochs']):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                outputs, _ = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
        
        val_acc = accuracy_score(y_val, val_preds)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Guardar modelo
            model_path = PATHS['output'] / 'models' / 'mlp_attention'
            model_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path / 'mlp_attention_best.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        if patience_counter >= CONFIG['patience']:
            print(f"   Early stopping en epoch {epoch+1}")
            break
    
    # Cargar mejor modelo
    model.load_state_dict(torch.load(model_path / 'mlp_attention_best.pth'))
    model.eval()
    
    # Evaluar
    def evaluate(X, y):
        X_t = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            outputs, _ = model(X_t)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average='macro')
        return acc, f1
    
    train_acc, train_f1 = evaluate(X_train, y_train)
    val_acc, val_f1 = evaluate(X_val, y_val)
    test_acc, test_f1 = evaluate(X_test, y_test)
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ MLP+Attention entrenado en {elapsed/60:.2f} min")
    print(f"   Train: Acc={train_acc:.4f}, F1={train_f1:.4f}")
    print(f"   Val:   Acc={val_acc:.4f}, F1={val_f1:.4f}")
    print(f"   Test:  Acc={test_acc:.4f}, F1={test_f1:.4f}")
    
    results = {
        'model': 'MLP+Attention',
        'train_acc': float(train_acc),
        'val_acc': float(val_acc),
        'test_acc': float(test_acc),
        'train_f1': float(train_f1),
        'val_f1': float(val_f1),
        'test_f1': float(test_f1),
        'training_time_min': elapsed / 60,
        'epochs_trained': epoch + 1
    }
    
    return results


# ===================================================================
# MODELO 3: FT-Transformer (Placeholder)
# ===================================================================

def train_ft_transformer(X_train, y_train, X_val, y_val, X_test, y_test):
    """Entrenar FT-Transformer (usando MLP como placeholder)."""
    print("\n" + "="*70)
    print("🔹 MODELO 3: FT-Transformer (Simplified)")
    print("="*70)
    print("⚠️  Usando MLP simplificado como placeholder para FT-Transformer")
    
    # Por ahora, usamos mismo código que MLP pero con nombre diferente
    return train_mlp_attention(X_train, y_train, X_val, y_val, X_test, y_test)


# ===================================================================
# MAIN
# ===================================================================

def main():
    """Ejecutar pipeline completo de entrenamiento."""
    print("\n" + "="*70)
    print("🚀 ENTRENAMIENTO FASE 2 - JÓVENES (SOLO DEMOGRÁFICAS)")
    print("="*70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {CONFIG['device']}")
    print(f"Seed: {CONFIG['seed']}")
    print(f"Features: 79 (demográficas + socioculturales)")
    print(f"Target: 3 clases (No lector / Medio / Frecuente)")
    
    # Set seed
    set_seed(CONFIG['seed'])
    
    # Crear directorio de salida
    PATHS['output'].mkdir(parents=True, exist_ok=True)
    
    # Cargar datos
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Entrenar modelos
    results = []
    
    try:
        tabnet_results = train_tabnet(X_train, y_train, X_val, y_val, X_test, y_test)
        results.append(tabnet_results)
    except Exception as e:
        print(f"\n❌ Error entrenando TabNet: {e}")
    
    try:
        mlp_results = train_mlp_attention(X_train, y_train, X_val, y_val, X_test, y_test)
        results.append(mlp_results)
    except Exception as e:
        print(f"\n❌ Error entrenando MLP+Attention: {e}")
    
    # Guardar resultados
    results_path = PATHS['output'] / 'results_fase2_jovenes.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': CONFIG,
            'dataset_info': {
                'n_train': len(X_train),
                'n_val': len(X_val),
                'n_test': len(X_test),
                'n_features': X_train.shape[1]
            },
            'results': results
        }, f, indent=4, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"\n📊 RESUMEN DE RESULTADOS:")
    for res in results:
        print(f"\n{res['model']}:")
        print(f"   Test Accuracy: {res['test_acc']:.4f} ({res['test_acc']*100:.2f}%)")
        print(f"   Test F1-Score: {res['test_f1']:.4f}")
        print(f"   Tiempo: {res['training_time_min']:.2f} min")
    
    print(f"\n📁 Resultados guardados en: {results_path}")
    print(f"📁 Modelos guardados en: {PATHS['output'] / 'models'}")


if __name__ == "__main__":
    main()
