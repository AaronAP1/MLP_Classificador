#!/usr/bin/env python3

import pandas as pd
import numpy as np

print("=== VERIFICACIÓN DATOS JÓVENES ===")

# Cargar solo primeras filas para inspección
try:
    df = pd.read_csv('data/raw/jovenes.csv', nrows=10)
    print(f"✅ CSV cargado: {df.shape}")
    print(f"Columnas: {list(df.columns)}")
    
    # Verificar columnas específicas del config
    cat_cols = ["P505","P506","P507","P508","P509","P201","P203","P206","P207","P208"]
    num_cols = ["FACTORFINAL_HOGAR","P210_A","P210_M","P310_N","P310_A"]
    target_col = "edad_total"
    
    print(f"\n=== Verificando columnas categóricas ===")
    for col in cat_cols:
        if col in df.columns:
            print(f"✅ {col}: {df[col].dtype}")
        else:
            print(f"❌ {col}: NO ENCONTRADA")
    
    print(f"\n=== Verificando columnas numéricas ===")
    for col in num_cols:
        if col in df.columns:
            print(f"✅ {col}: {df[col].dtype} - NaN: {df[col].isnull().sum()}")
        else:
            print(f"❌ {col}: NO ENCONTRADA")
            
    print(f"\n=== Verificando target ===")
    if target_col in df.columns:
        print(f"✅ {target_col}: {df[target_col].dtype}")
        print(f"  Valores únicos (muestra): {df[target_col].unique()}")
    else:
        print(f"❌ {target_col}: NO ENCONTRADA")
        
    # Cargar todo el dataset para más información
    print(f"\n=== Cargando dataset completo ===")
    df_full = pd.read_csv('data/raw/jovenes.csv')
    print(f"Dataset completo: {df_full.shape}")
    
    if target_col in df_full.columns:
        print(f"Target distribution:")
        print(df_full[target_col].value_counts().sort_index())
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
