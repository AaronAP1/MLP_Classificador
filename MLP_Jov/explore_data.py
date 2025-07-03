#!/usr/bin/env python3

import pandas as pd
import numpy as np

print("🔍 EXPLORACIÓN DEL DATASET JOVENES.CSV")
print("=" * 50)

# Cargar dataset
df = pd.read_csv('data/raw/jovenes.csv', low_memory=False)
print(f"📊 Dataset shape: {df.shape}")

# Buscar variables relacionadas con lectura
lectura_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['p3', 'p4', 'p5', 'lectur', 'libro', 'lee'])]
print(f"\n📚 Variables relacionadas con lectura (P3xx, P4xx, P5xx):")
for col in sorted(lectura_cols):
    non_null = df[col].notna().sum()
    unique_vals = df[col].nunique()
    print(f"  {col}: {non_null:,} registros ({non_null/len(df)*100:.1f}%) - {unique_vals} valores únicos")

# Explorar las variables P5xx más detalladamente
print(f"\n🎯 VARIABLES P5xx (Prácticas Lectoras):")
p5_cols = [col for col in df.columns if col.startswith('P5')]
for col in sorted(p5_cols)[:15]:  # Primeras 15
    if col in df.columns:
        non_null = df[col].notna().sum()
        if non_null > 1000:  # Solo variables con suficientes datos
            unique_vals = df[col].unique()[:10]  # Primeros 10 valores únicos
            print(f"  {col}: {non_null:,} registros - Valores: {unique_vals}")

# Variable target original P517
print(f"\n🎯 VARIABLE TARGET ORIGINAL (P517):")
if 'P517' in df.columns:
    p517_stats = df['P517'].value_counts().sort_index()
    print(f"P517 distribución:\n{p517_stats}")
    print(f"Total con P517: {df['P517'].notna().sum():,}")

# Explorar variables demográficas que podrían predecir hábitos
print(f"\n👥 VARIABLES DEMOGRÁFICAS:")
demo_cols = ['edad_total', 'P201', 'P203', 'P206', 'P207', 'P208']
for col in demo_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  {col}: {non_null:,} registros ({non_null/len(df)*100:.1f}%)")
        if col == 'edad_total':
            print(f"    Rango edades: {df[col].min():.0f} - {df[col].max():.0f}")
