"""
Script principal para entrenar los 9 modelos baseline:
- 3 arquitecturas (MLP-MA, TabNet, FT-Transformer) 
- 3 grupos (adultos, jovenes, ninos)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import set_seed, load_config, get_logger


def load_data(config):
    """Carga datos procesados."""
    data_path = Path(config['data']['data_path'])
    
    # Cargar archivos
    train = pd.read_parquet(data_path / config['data']['train_file'])
    val = pd.read_parquet(data_path / config['data']['val_file'])
    test = pd.read_parquet(data_path / config['data']['test_file'])
    
    # Separar features y target
    X_train = train.drop('target', axis=1).values.astype(np.float32)
    y_train = train['target'].values.astype(np.int64)
    
    X_val = val.drop('target', axis=1).values.astype(np.float32)
    y_val = val['target'].values.astype(np.int64)
    
    X_test = test.drop('target', axis=1).values.astype(np.float32)
    y_test = test['target'].values.astype(np.int64)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_model(model_type, input_dim, config):
    """Crea modelo según tipo."""
    if model_type == "mlp_ma":
        from models.Adultos.mlp_ma import MLPWithAttention
        model = MLPWithAttention(
            input_dim=input_dim,
            hidden_dims=config['model']['hidden_dims'],
            attn_dim=config['model']['attn_dim'],
            dropout=config['model']['dropout']
        )
    elif model_type == "ft_transformer":
        from models.Adultos.ft_transformer import FTTransformerClassifier
        model = FTTransformerClassifier(
            num_features=input_dim,
            emb_dim=config['model']['d_token'],
            n_layers=config['model']['n_blocks'],
            n_heads=config['model']['attention_n_heads'],
            dropout=config['model']['attention_dropout'],
            dim_feedforward=config['model']['ffn_d_hidden']
        )
    else:
        raise ValueError(f"Modelo {model_type} no soportado")
    
    return model


def train_pytorch_model(model, train_data, val_data, config, logger):
    """Entrena modelo PyTorch (MLP-MA o FT-Transformer)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Usando device: {device}")
    
    model = model.to(device)
    
    # Datos
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Optimizer y loss
    criterion = nn.CrossEntropyLoss()
    
    if config['training']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
    
    # Scheduler
    if config['training']['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    else:
        scheduler = None
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            if config['model']['type'] in ['mlp_ma', 'ft_transformer']:
                outputs, _ = model(batch_x)
            else:
                outputs = model(batch_x)
            
            loss = criterion(outputs, batch_y)
            
            # Backward
            loss.backward()
            
            if config['training'].get('gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['gradient_clip']
                )
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                if config['model']['type'] in ['mlp_ma', 'ft_transformer']:
                    outputs, _ = model(batch_x)
                else:
                    outputs = model(batch_x)
                
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        # Logging
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{config['training']['epochs']} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
            )
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Guardar mejor modelo
            save_dir = Path(config['output']['save_dir'])
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                logger.info(f"Early stopping en epoch {epoch+1}")
                break
        
        if scheduler:
            scheduler.step()
    
    logger.info(f"Mejor Val Accuracy: {best_val_acc:.2f}%")
    
    # Cargar mejor modelo
    model.load_state_dict(torch.load(save_dir / 'best_model.pt'))
    
    return model


def train_tabnet_model(train_data, val_data, config, logger):
    """Entrena modelo TabNet."""
    from models.Adultos.tabnet import TabNetWrapper
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    model = TabNetWrapper(
        input_dim=X_train.shape[1],
        n_d=config['model']['n_d'],
        n_a=config['model']['n_a'],
        n_steps=config['model']['n_steps'],
        gamma=config['model']['gamma'],
        n_independent=config['model']['n_independent'],
        n_shared=config['model']['n_shared'],
        lambda_sparse=config['model']['lambda_sparse'],
        momentum=config['model']['momentum'],
        mask_type=config['model']['mask_type']
    )
    
    logger.info("Entrenando TabNet...")
    model.fit(
        X_train, y_train,
        X_val, y_val,
        max_epochs=config['training']['epochs'],
        patience=config['training']['early_stopping_patience'],
        batch_size=config['data']['batch_size']
    )
    
    # Guardar
    save_dir = Path(config['output']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(save_dir / 'tabnet_model'))
    
    return model


def evaluate_model(model, test_data, config, logger):
    """Evalúa modelo en test set."""
    X_test, y_test = test_data
    
    if config['model']['type'] == 'tabnet':
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        
        with torch.no_grad():
            X_test_tensor = torch.from_numpy(X_test).to(device)
            
            if config['model']['type'] in ['mlp_ma', 'ft_transformer']:
                outputs, _ = model(X_test_tensor)
            else:
                outputs = model(X_test_tensor)
            
            y_pred_proba = torch.softmax(outputs, dim=1).cpu().numpy()
            y_pred = outputs.argmax(dim=1).cpu().numpy()
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"\n{'='*50}")
    logger.info(f"RESULTADOS EN TEST SET")
    logger.info(f"{'='*50}")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"F1-Score (Macro): {f1_macro:.4f}")
    logger.info(f"F1-Score (Weighted): {f1_weighted:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['no_lector', 'lector_medio', 'lector_frecuente'])}")
    
    # Guardar resultados
    save_dir = Path(config['output']['save_dir'])
    results = {
        'accuracy': float(acc),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'predictions': y_pred.tolist(),
        'probabilities': y_pred_proba.tolist()
    }
    
    import json
    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return acc, f1_macro, f1_weighted


def main(config_path):
    """Pipeline principal."""
    # Cargar config
    config = load_config(config_path)
    
    # Setup
    set_seed(config['seed'])
    
    log_file = Path(config['output']['log_file'])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = get_logger(__name__, str(log_file))
    
    logger.info(f"Config: {config_path}")
    logger.info(f"Grupo: {config['data']['group']}")
    logger.info(f"Modelo: {config['model']['type']}")
    
    # Cargar datos
    logger.info("Cargando datos...")
    train_data, val_data, test_data = load_data(config)
    logger.info(f"Train: {train_data[0].shape}, Val: {val_data[0].shape}, Test: {test_data[0].shape}")
    
    # Entrenar
    logger.info("Iniciando entrenamiento...")
    
    if config['model']['type'] == 'tabnet':
        model = train_tabnet_model(train_data, val_data, config, logger)
    else:
        input_dim = train_data[0].shape[1]
        model = create_model(config['model']['type'], input_dim, config)
        model = train_pytorch_model(model, train_data, val_data, config, logger)
    
    # Evaluar
    logger.info("Evaluando en test set...")
    evaluate_model(model, test_data, config, logger)
    
    logger.info("✅ Entrenamiento completado!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path al archivo YAML de configuración')
    args = parser.parse_args()
    
    main(args.config)
