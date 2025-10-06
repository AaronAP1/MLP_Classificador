#!/usr/bin/env python3

import json
import os

# Cargar configs
with open('configs/config_jovenes.json', 'r') as f:
    config = json.load(f)
with open('configs/paths.json', 'r') as f:
    paths = json.load(f)

print("=== TEST ETL SIMPLIFICADO ===")
print(f"Config cargado: {config['data']['raw_csv']}")
print(f"Target: {config['etl']['target_col']}")
print(f"Features categóricas: {len(config['etl']['cat_cols'])}")
print(f"Features numéricas: {len(config['etl']['num_cols'])}")

# Intentar cargar ETL
try:
    print("\n=== Importando ETL ===")
    from utils.etl import preprocess_and_load
    print("✅ ETL importado correctamente")
    
    print("\n=== Ejecutando ETL ===")
    dl_tr, dl_va, dl_te, input_size, n_classes = preprocess_and_load(config, paths)
    
    print(f"✅ ETL completado:")
    print(f"  - Input size: {input_size}")
    print(f"  - Num clases: {n_classes}")
    print(f"  - Train batches: {len(dl_tr)}")
    print(f"  - Val batches: {len(dl_va)}")
    print(f"  - Test batches: {len(dl_te)}")
    
    # Probar un batch
    print(f"\n=== Probando primer batch ===")
    for X_batch, y_batch in dl_tr:
        print(f"  - X shape: {X_batch.shape}")
        print(f"  - y shape: {y_batch.shape}")
        print(f"  - X tiene NaN: {X_batch.isnan().sum().item()}")
        print(f"  - X tiene Inf: {X_batch.isinf().sum().item()}")
        print(f"  - y único: {y_batch.unique()}")
        print(f"  - X min/max: {X_batch.min().item():.3f} / {X_batch.max().item():.3f}")
        break
    
    print("✅ Test ETL exitoso!")
    
except Exception as e:
    print(f"❌ Error en ETL: {str(e)}")
    import traceback
    traceback.print_exc()
