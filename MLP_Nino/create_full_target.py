#!/usr/bin/env python3

import pandas as pd
import numpy as np

print("🔧 CREANDO NUEVO TARGET DE HÁBITOS DE LECTURA")
print("=" * 50)

# Cargar dataset
df = pd.read_csv('data/raw/jovenes.csv', low_memory=False)
print(f"📊 Dataset shape: {df.shape}")

# Estrategia: Crear un score compuesto de hábitos de lectura usando múltiples variables

# 1. Variable base: P517 (frecuencia de lectura directa)
df['lectura_score'] = 0.0

# 2. Si tiene P517, usar directamente (normalizado 0-1)
if 'P517' in df.columns:
    mask_p517 = df['P517'].notna()
    # P517: 1=nunca, 2=poco, 3=ocasional, 4=frecuente, 5=muy frecuente
    df.loc[mask_p517, 'lectura_score'] = (df.loc[mask_p517, 'P517'] - 1) / 4
    print(f"✅ P517 aplicado a {mask_p517.sum():,} registros")

# 3. Para el resto, usar variables P3xx (prácticas generales)
mask_no_p517 = ~mask_p517

# P315_x: Lee diferentes tipos de material (0=No, 1=Sí)
lectura_vars_p3 = ['P315_1', 'P315_2', 'P315_3', 'P315_4', 'P315_5', 'P315_6']
for var in lectura_vars_p3:
    if var in df.columns:
        # Sumar puntos por cada tipo de material que lee
        mask_valid = mask_no_p517 & df[var].notna()
        df.loc[mask_valid, 'lectura_score'] += df.loc[mask_valid, var] * 0.1

print(f"✅ Variables P315 aplicadas")

# P311: ¿Sabe leer? (fundamental)
if 'P311' in df.columns:
    mask_valid = mask_no_p517 & df['P311'].notna()
    # Si no sabe leer, score = 0
    df.loc[mask_valid & (df['P311'] == 0), 'lectura_score'] = 0.0
    print(f"✅ P311 (saber leer) aplicado")

# P409: Frecuencia de lectura alternativa
if 'P409' in df.columns:
    mask_valid = mask_no_p517 & df['P409'].notna()
    # P409: 1=nunca a 6=diario, normalizar
    df.loc[mask_valid, 'lectura_score'] += (df.loc[mask_valid, 'P409'] - 1) / 5 * 0.3
    print(f"✅ P409 aplicado a {mask_valid.sum():,} registros adicionales")

# 4. Normalizar score final entre 0-1
df['lectura_score'] = np.clip(df['lectura_score'], 0, 1)

# 5. Crear categorías finales
def score_to_category(score):
    if pd.isna(score) or score < 0.2:
        return "no_lector"
    elif score < 0.6:
        return "lector_medio"
    else:
        return "lector_frecuente"

df['habito_lectura_completo'] = df['lectura_score'].apply(score_to_category)

# 6. Análisis de resultados
print(f"\n📈 RESULTADOS:")
print(f"Total registros con score: {df['lectura_score'].notna().sum():,}")
print(f"Distribución de categorías:")
print(df['habito_lectura_completo'].value_counts())

print(f"\n📊 Distribución por score:")
print(f"Score 0.0-0.2 (no_lector): {((df['lectura_score'] >= 0) & (df['lectura_score'] < 0.2)).sum():,}")
print(f"Score 0.2-0.6 (lector_medio): {((df['lectura_score'] >= 0.2) & (df['lectura_score'] < 0.6)).sum():,}")
print(f"Score 0.6-1.0 (lector_frecuente): {(df['lectura_score'] >= 0.6).sum():,}")

# 7. Comparar con P517 original (validación)
if 'P517' in df.columns:
    print(f"\n🔍 VALIDACIÓN con P517 original:")
    comparison = df[df['P517'].notna()][['P517', 'habito_lectura_completo', 'lectura_score']]
    print(comparison.groupby(['P517', 'habito_lectura_completo']).size().unstack(fill_value=0))

# 8. Guardar resultado
df[['habito_lectura_completo', 'lectura_score']].to_csv('data/processed/target_lectura_completo.csv', index=False)
print(f"\n✅ Target guardado en: data/processed/target_lectura_completo.csv")
