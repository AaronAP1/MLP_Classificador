"""
Script para regenerar los datos con nueva lógica de target balanceada.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib


def create_balanced_target(df, grupo):
    """Crea target con distribución más balanceada."""
    print(f"\n🎯 Creando target balanceado para {grupo.upper()}")
    
    df = df.copy()
    df['lectura_score'] = 0.0
    
    # Configurar variables según grupo
    if grupo == "adultos":
        lectura_vars = ['P315_1', 'P315_2', 'P315_3', 'P315_4', 'P315_5', 'P315_6']
        frecuencia_var = 'P409'
        lee_var = 'P311'
    else:
        lectura_vars = ['P514_1', 'P514_2', 'P514_3'] if f'P514_1' in df.columns else []
        frecuencia_var = 'P515' if 'P515' in df.columns else 'P517'
        lee_var = None
    
    # Calcular score
    for var in lectura_vars:
        if var in df.columns:
            mask = df[var].notna()
            df.loc[mask, 'lectura_score'] += (df.loc[mask, var] / 2) * 0.15
    
    # Frecuencia (peso mayor)
    if frecuencia_var in df.columns:
        mask = df[frecuencia_var].notna()
        freq_norm = (df.loc[mask, frecuencia_var] - 1) / 4  # 1-5 → 0-1
        df.loc[mask, 'lectura_score'] += freq_norm * 0.6
    
    # Penalizar si no lee (adultos)
    if lee_var and lee_var in df.columns:
        mask = df[lee_var].notna()
        df.loc[mask & (df[lee_var] == 0), 'lectura_score'] *= 0.2
    
    df['lectura_score'] = np.clip(df['lectura_score'], 0, 1)
    
    # ✅ UMBRALES FIJOS Y RAZONABLES (no percentiles para evitar casos extremos)
    # Analizar distribución real
    valid_scores = df['lectura_score'].dropna()
    print(f"   Score stats: min={valid_scores.min():.3f}, mean={valid_scores.mean():.3f}, max={valid_scores.max():.3f}")
    print(f"   Percentiles: 25%={valid_scores.quantile(0.25):.3f}, 50%={valid_scores.quantile(0.5):.3f}, 75%={valid_scores.quantile(0.75):.3f}")
    
    # Umbrales más amplios para clase media
    def score_to_category(score):
        if pd.isna(score):
            return 0
        elif score < 0.3:  # Baja actividad lectora
            return 0  # No lector
        elif score < 0.7:  # Actividad moderada (rango amplio)
            return 1  # Lector medio
        else:              # Alta actividad
            return 2  # Lector frecuente
    
    df["target"] = df['lectura_score'].apply(score_to_category)
    
    # Mostrar distribución
    print(f"\n   📊 Distribución BALANCEADA:")
    for clase in [0, 1, 2]:
        count = (df["target"] == clase).sum()
        pct = count / len(df) * 100
        nombres = ["No lector", "Lector medio", "Lector frecuente"]
        print(f"      Clase {clase} ({nombres[clase]:20s}): {count:6,} ({pct:5.1f}%)")
    
    return df


def regenerate_group(grupo):
    """Regenera datos procesados con target balanceado."""
    print(f"\n{'='*70}")
    print(f"REGENERANDO: {grupo.upper()}")
    print(f"{'='*70}")
    
    # Leer datos raw
    df = pd.read_csv(f'data/unprocessed/{grupo}.csv', low_memory=False)
    print(f"Datos raw: {len(df):,} registros")
    
    # Crear target balanceado
    df = create_balanced_target(df, grupo)
    
    # Separar features y target
    feature_cols = [c for c in df.columns if c not in ['target', 'lectura_score', 'habito_lectura_completo']]
    X = df[feature_cols].copy()
    y = df['target'].copy()
    
    # ⚠️ ELIMINAR VARIABLES DE LECTURA QUE CAUSAN DATA LEAKAGE
    leakage_vars = []
    if grupo == "adultos":
        leakage_vars = ['P315_1', 'P315_2', 'P315_3', 'P315_4', 'P315_5', 'P315_6', 
                       'P409', 'P311']
    else:
        leakage_vars = ['P514_1', 'P514_2', 'P514_3', 'P515', 'P517']
    
    # Filtrar solo las que existen en X
    leakage_vars_in_X = [v for v in leakage_vars if v in X.columns]
    if leakage_vars_in_X:
        print(f"\n⚠️  Eliminando {len(leakage_vars_in_X)} variables que causan data leakage: {leakage_vars_in_X}")
        X = X.drop(columns=leakage_vars_in_X)
    
    # Eliminar filas con target NaN
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    print(f"\nDatos válidos: {len(X):,} registros")
    
    # Split 70/15/15 estratificado
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    val_size_adjusted = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    print(f"\nSplit:")
    print(f"   Train: {len(X_train):,}")
    print(f"   Val:   {len(X_val):,}")
    print(f"   Test:  {len(X_test):,}")
    
    # Aplicar SMOTE solo en train
    print(f"\n🔧 Aplicando SMOTE en train...")
    print(f"   Antes: {dict(y_train.value_counts().sort_index())}")
    
    # Seleccionar solo columnas numéricas para SMOTE
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train_num = X_train[numeric_cols].copy()
    
    # Eliminar columnas que son todas NaN primero
    cols_to_keep = X_train_num.columns[~X_train_num.isna().all()].tolist()
    X_train_num = X_train_num[cols_to_keep]
    
    # Imputar NaNs con la media antes de SMOTE
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train_num_transformed = imputer.fit_transform(X_train_num)
    
    # Las columnas que quedan después de la imputación
    X_train_num_imputed = pd.DataFrame(
        X_train_num_transformed,
        columns=X_train_num.columns
    )
    
    # Verificar si hay suficientes samples por clase
    min_samples = y_train.value_counts().min()
    if min_samples < 2:
        print(f"   ⚠️  Clase con muy pocos samples ({min_samples}), saltando SMOTE")
        X_train_resampled = X_train_num_imputed
        y_train_resampled = y_train.values
    else:
        k_neighbors = min(5, min_samples - 1)
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_num_imputed, y_train)
    
    print(f"   Después: {dict(pd.Series(y_train_resampled).value_counts().sort_index())}")
    
    # Crear DataFrames finales (usar cols_to_keep para consistencia)
    train_df = pd.DataFrame(X_train_resampled, columns=cols_to_keep)
    train_df['target'] = y_train_resampled
    
    # Imputar val y test también (usando las mismas columnas válidas)
    val_num = X_val[cols_to_keep].copy()
    val_df = pd.DataFrame(
        imputer.transform(val_num),
        columns=cols_to_keep
    )
    val_df['target'] = y_val.values
    
    test_num = X_test[cols_to_keep].copy()
    test_df = pd.DataFrame(
        imputer.transform(test_num),
        columns=cols_to_keep
    )
    test_df['target'] = y_test.values
    
    # Guardar
    output_dir = Path(f'data/processed/{grupo}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_parquet(output_dir / 'train.parquet', index=False)
    val_df.to_parquet(output_dir / 'val.parquet', index=False)
    test_df.to_parquet(output_dir / 'test.parquet', index=False)
    
    print(f"\n✅ Archivos guardados en {output_dir}")
    print(f"   - train.parquet: {len(train_df):,} registros")
    print(f"   - val.parquet:   {len(val_df):,} registros")
    print(f"   - test.parquet:  {len(test_df):,} registros")


if __name__ == "__main__":
    for grupo in ['adultos', 'jovenes', 'ninos']:
        try:
            regenerate_group(grupo)
        except Exception as e:
            print(f"\n❌ ERROR en {grupo}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("✅ REGENERACIÓN COMPLETADA")
    print(f"{'='*70}")
