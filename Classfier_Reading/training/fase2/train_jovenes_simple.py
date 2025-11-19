"""
Script Simplificado de Entrenamiento FASE 2 - Jóvenes
Entrena 3 modelos: TabNet, MLP+Attention, FT-Transformer

Universidad Peruana Unión - 2025
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configurar path del proyecto
# Script está en: Classfier_Reading/training/fase2/
# Necesita: Classfier_Reading/ (2 niveles arriba)
classfier_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(classfier_root))
sys.path.insert(0, str(classfier_root / "src"))

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
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Importar modelos del proyecto
from models.Jovenes.tabnet import TabNetWrapper
from models.Jovenes.mlp_attention import MLPWithAttention
from models.Jovenes.ft_transformer import FTTransformer

# ===================================================================
# CONFIGURACIÓN
# ===================================================================

CONFIG = {
    'grupo': 'jovenes',
    'fase': 2,
    'seed': 42,
    'epochs': 100,
    'batch_size': 128,
    'learning_rate': 0.001,
    'patience': 15,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

PATHS = {
    'data': classfier_root / 'data' / 'processed_fase2' / 'jovenes',
    'output': classfier_root / 'outputs' / 'fase2' / 'jovenes'
}

# ===================================================================
# CARGA DE DATOS
# ===================================================================

def load_data():
    """Cargar datos de Fase 2."""
    print("="*70)
    print(f"📂 CARGANDO DATOS - FASE 2: JÓVENES")
    print("="*70)
    
    train_df = pd.read_csv(PATHS['data'] / 'jovenes_train.csv')
    val_df = pd.read_csv(PATHS['data'] / 'jovenes_val.csv')
    test_df = pd.read_csv(PATHS['data'] / 'jovenes_test.csv')
    
    X_train = train_df.drop(columns=['target']).values.astype(np.float32)
    y_train = train_df['target'].values.astype(np.int64)
    
    X_val = val_df.drop(columns=['target']).values.astype(np.float32)
    y_val = val_df['target'].values.astype(np.int64)
    
    X_test = test_df.drop(columns=['target']).values.astype(np.float32)
    y_test = test_df['target'].values.astype(np.int64)
    
    print(f"\n✅ Datos cargados:")
    print(f"   Train: {X_train.shape[0]:,} × {X_train.shape[1]} features")
    print(f"   Val:   {X_val.shape[0]:,} × {X_val.shape[1]} features")
    print(f"   Test:  {X_test.shape[0]:,} × {X_test.shape[1]} features")
    
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
    
    # Crear modelo
    model = TabNetWrapper(
        input_dim=X_train.shape[1],
        n_d=32,
        n_a=32,
        n_steps=3,
        gamma=1.3,
        lambda_sparse=1e-3,
        num_classes=3
    )
    
    # Entrenar
    print(f"Entrenando TabNet...")
    model.fit(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        max_epochs=CONFIG['epochs'],
        patience=CONFIG['patience'],
        batch_size=CONFIG['batch_size']
    )
    
    # Evaluar
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='macro')
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ TabNet entrenado en {elapsed/60:.2f} min")
    print(f"   Test:  Acc={test_acc:.4f}, F1={test_f1:.4f}")
    
    # Guardar modelo
    model_path = PATHS['output'] / 'models' / 'tabnet'
    model_path.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path / 'tabnet_best.pkl'))
    
    # Guardar config
    config = {
        'input_dim': int(X_train.shape[1]),
        'n_d': 32,
        'n_a': 32,
        'n_steps': 3,
        'gamma': 1.3,
        'lambda_sparse': 1e-3,
        'num_classes': 3
    }
    with open(model_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    return {
        'model': 'TabNet',
        'test_acc': float(test_acc),
        'test_f1': float(test_f1),
        'training_time_min': elapsed / 60
    }

# ===================================================================
# MODELO 2: MLP + Attention
# ===================================================================

def train_mlp_attention(X_train, y_train, X_val, y_val, X_test, y_test):
    """Entrenar MLP con Attention."""
    print("\n" + "="*70)
    print("🔹 MODELO 2: MLP + Attention")
    print("="*70)
    
    start_time = time.time()
    
    device = torch.device(CONFIG['device'])
    
    # Crear modelo
    input_dim = X_train.shape[1]
    model = MLPWithAttention(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        num_classes=3,
        attention_dim=32,
        dropout=0.3
    ).to(device)
    
    # Preparar datos
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Entrenar
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"Entrenando MLP+Attention...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
        
        val_acc = accuracy_score(y_val, val_preds)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Guardar mejor modelo
            model_path = PATHS['output'] / 'models' / 'mlp_attention'
            model_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path / 'best_model.pt')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{CONFIG['epochs']} - Val Acc: {val_acc:.4f}")
        
        if patience_counter >= CONFIG['patience']:
            print(f"   Early stopping en epoch {epoch+1}")
            break
    
    # Cargar mejor modelo y evaluar
    model.load_state_dict(torch.load(model_path / 'best_model.pt'))
    model.eval()
    
    X_test_t = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        outputs = model(X_test_t)
        y_pred_test = torch.argmax(outputs, dim=1).cpu().numpy()
    
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='macro')
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ MLP+Attention entrenado en {elapsed/60:.2f} min")
    print(f"   Test:  Acc={test_acc:.4f}, F1={test_f1:.4f}")
    
    # Guardar modelo
    model_path = PATHS['output'] / 'models' / 'mlp_attention'
    model_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path / 'best_model.pt')
    
    # Guardar config
    config = {
        'input_dim': input_dim,
        'hidden_dims': [256, 128, 64],
        'num_classes': 3,
        'attention_dim': 32,
        'dropout': 0.3
    }
    with open(model_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    return {
        'model': 'MLP+Attention',
        'test_acc': float(test_acc),
        'test_f1': float(test_f1),
        'training_time_min': elapsed / 60
    }

# ===================================================================
# MODELO 3: FT-Transformer
# ===================================================================

def train_ft_transformer(X_train, y_train, X_val, y_val, X_test, y_test):
    """Entrenar FT-Transformer."""
    print("\n" + "="*70)
    print("🔹 MODELO 3: FT-Transformer")
    print("="*70)
    
    start_time = time.time()
    
    device = torch.device(CONFIG['device'])
    
    # Crear modelo
    input_dim = X_train.shape[1]
    model = FTTransformer(
        input_dim=input_dim,
        d_token=192,
        n_blocks=3,
        attention_n_heads=8,
        attention_dropout=0.2,
        ffn_d_hidden=512,
        ffn_dropout=0.1,
        num_classes=3
    ).to(device)
    
    # Preparar datos
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Entrenar
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"Entrenando FT-Transformer...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
        
        val_acc = accuracy_score(y_val, val_preds)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Guardar mejor modelo
            model_path = PATHS['output'] / 'models' / 'ft_transformer'
            model_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path / 'best_model.pt')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{CONFIG['epochs']} - Val Acc: {val_acc:.4f}")
        
        if patience_counter >= CONFIG['patience']:
            print(f"   Early stopping en epoch {epoch+1}")
            break
    
    # Cargar mejor modelo y evaluar
    model.load_state_dict(torch.load(model_path / 'best_model.pt'))
    model.eval()
    
    X_test_t = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        outputs = model(X_test_t)
        y_pred_test = torch.argmax(outputs, dim=1).cpu().numpy()
    
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='macro')
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ FT-Transformer entrenado en {elapsed/60:.2f} min")
    print(f"   Test:  Acc={test_acc:.4f}, F1={test_f1:.4f}")
    
    # Guardar config
    config = {
        'input_dim': input_dim,
        'd_token': 192,
        'n_blocks': 3,
        'attention_n_heads': 8,
        'attention_dropout': 0.2,
        'ffn_d_hidden': 512,
        'ffn_dropout': 0.1,
        'num_classes': 3
    }
    with open(model_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    return {
        'model': 'FT-Transformer',
        'test_acc': float(test_acc),
        'test_f1': float(test_f1),
        'training_time_min': elapsed / 60
    }

# ===================================================================
# MAIN
# ===================================================================

def main():
    """Ejecutar pipeline completo de entrenamiento."""
    print("\n" + "="*70)
    print("🚀 ENTRENAMIENTO FASE 2 - JÓVENES")
    print("="*70)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {CONFIG['device']}")
    
    # Set seed
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # Crear directorio de salida
    PATHS['output'].mkdir(parents=True, exist_ok=True)
    
    # Cargar datos
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Entrenar modelos
    results = []
    
    for train_func in [train_tabnet, train_mlp_attention, train_ft_transformer]:
        try:
            result = train_func(X_train, y_train, X_val, y_val, X_test, y_test)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Guardar resultados
    results_path = PATHS['output'] / 'results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    for res in results:
        print(f"\n{res['model']}: Acc={res['test_acc']:.4f}, F1={res['test_f1']:.4f}, Tiempo={res['training_time_min']:.2f}min")


if __name__ == "__main__":
    main()
